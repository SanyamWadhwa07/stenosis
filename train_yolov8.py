"""
Train YOLOv8m on the stenosis detection dataset.
Saves best.pt and last.pt to PROJECT/MODEL_NAME/weights/.

Evaluates with the full COCO metric suite used in the paper
(arXiv 2503.01601) via pycocotools:
  mAP@[0.50:0.95], mAP50, mAP75, mAP(small/medium/large)
  AR@100, AR@300, AR@1000, AR(small/medium/large)
"""

import json
import pathlib
import time
from collections import defaultdict

import yaml

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "yolov8m"
MODEL_PT   = "yolov8m.pt"
IMG_SIZE   = 512
EPOCHS     = 100
BATCH      = 8
WORKERS    = 4
SEED       = 42
DEVICE     = 0

_HERE        = pathlib.Path(__file__).parent.resolve()
DATASET_ROOT = _HERE / "stenosis"
DATA_YAML    = str(DATASET_ROOT / "data_bbox.yaml")
PROJECT      = str(_HERE / "outputs")
TRAIN_IMAGES = str(DATASET_ROOT / "train" / "images")
VAL_IMAGES   = str(DATASET_ROOT / "val"   / "images")
VAL_LABELS   = str(DATASET_ROOT / "val"   / "labels")
VAL_JSON     = str(DATASET_ROOT / "val"   / "annotations" / "val.json")
# ──────────────────────────────────────────────────────────────────────────────


def fix_data_yaml():
    """Overwrite data_bbox.yaml with correct absolute paths."""
    pathlib.Path(DATA_YAML).parent.mkdir(parents=True, exist_ok=True)
    content = {
        "train": TRAIN_IMAGES,
        "val":   VAL_IMAGES,
        "nc":    1,
        "names": ["stenosis"],
    }
    pathlib.Path(DATA_YAML).write_text(yaml.dump(content, default_flow_style=False))
    print(f"[setup] data_bbox.yaml written to {DATA_YAML}")


def convert_coco_val_to_yolo():
    """Convert val/annotations/val.json → val/labels/*.txt (YOLO format).

    Idempotent: skips if val/labels/ already contains .txt files.
    Only annotations whose category name is 'stenosis' are written (class 0).
    """
    labels_dir = pathlib.Path(VAL_LABELS)
    labels_dir.mkdir(parents=True, exist_ok=True)

    existing = list(labels_dir.glob("*.txt"))
    if existing:
        print(f"[setup] Val labels already exist ({len(existing)} files), skipping conversion.")
        return

    print(f"[setup] Converting COCO val annotations → YOLO format …")
    with open(VAL_JSON) as f:
        coco = json.load(f)

    # Find the category_id for "stenosis" (don't hardcode — 26 categories exist)
    stenosis_id = next(
        c["id"] for c in coco["categories"] if c["name"].lower() == "stenosis"
    )

    img_map = {img["id"]: img for img in coco["images"]}

    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] == stenosis_id:
            anns_by_img[ann["image_id"]].append(ann)

    written = 0
    for img_id, img_info in img_map.items():
        fname = pathlib.Path(img_info["file_name"]).stem
        W = img_info["width"]
        H = img_info["height"]
        label_file = labels_dir / f"{fname}.txt"
        anns = anns_by_img.get(img_id, [])
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            wn = w / W
            hn = h / H
            lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        label_file.write_text("\n".join(lines))
        written += 1

    print(f"[setup] Converted {written} images → {labels_dir}")


def compute_coco_metrics() -> dict:
    """
    Run COCO-style evaluation on best.pt.
    Matches the exact metric suite reported in arXiv:2503.01601 (Table I).
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("[eval] pycocotools not found — skipping COCO metrics. "
              "Install with: pip install pycocotools")
        return {}

    from ultralytics import YOLO

    weights = str(pathlib.Path(PROJECT) / MODEL_NAME / "weights" / "best.pt")
    if not pathlib.Path(weights).exists():
        print(f"[eval] best.pt not found at {weights}, skipping COCO metrics.")
        return {}

    print(f"\n[eval] Computing COCO metrics on {weights} …")

    coco_gt = COCO(VAL_JSON)
    fname_to_id = {
        pathlib.Path(img["file_name"]).name: img["id"]
        for img in coco_gt.dataset["images"]
    }
    stenosis_cat = next(
        c["id"] for c in coco_gt.dataset["categories"]
        if c["name"].lower() == "stenosis"
    )

    model = YOLO(weights)
    preds = model.predict(
        source=VAL_IMAGES, imgsz=IMG_SIZE,
        conf=0.001, iou=0.65, verbose=False,
    )

    coco_preds = []
    for r in preds:
        fname  = pathlib.Path(r.path).name
        img_id = fname_to_id.get(fname)
        if img_id is None:
            continue
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box, score in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            coco_preds.append({
                "image_id":    img_id,
                "category_id": stenosis_cat,
                "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score":       float(score),
            })

    out_dir    = pathlib.Path(PROJECT) / MODEL_NAME
    preds_file = out_dir / "coco_preds.json"
    with open(preds_file, "w") as f:
        json.dump(coco_preds, f)

    if not coco_preds:
        print("[eval] No predictions — check conf threshold / model output.")
        return {}

    coco_dt = coco_gt.loadRes(str(preds_file))
    metrics: dict = {}

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    metrics.update({
        "mAP_0.50:0.95":  round(float(s[0]),  4),
        "mAP50":           round(float(s[1]),  4),
        "mAP75":           round(float(s[2]),  4),
        "mAP_small":       round(float(s[3]),  4),
        "mAP_medium":      round(float(s[4]),  4),
        "mAP_large":       round(float(s[5]),  4),
        "AR_maxDets_1":    round(float(s[6]),  4),
        "AR_maxDets_10":   round(float(s[7]),  4),
        "AR_maxDets_100":  round(float(s[8]),  4),
        "AR_small":        round(float(s[9]),  4),
        "AR_medium":       round(float(s[10]), 4),
        "AR_large":        round(float(s[11]), 4),
    })

    ev300 = COCOeval(coco_gt, coco_dt, "bbox")
    ev300.params.maxDets = [1, 10, 300]
    ev300.evaluate(); ev300.accumulate(); ev300.summarize()
    metrics["AR_maxDets_300"] = round(float(ev300.stats[8]), 4)

    ev1000 = COCOeval(coco_gt, coco_dt, "bbox")
    ev1000.params.maxDets = [1, 10, 1000]
    ev1000.evaluate(); ev1000.accumulate(); ev1000.summarize()
    metrics["AR_maxDets_1000"] = round(float(ev1000.stats[8]), 4)

    metrics_file = out_dir / "coco_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Metrics saved → {metrics_file}")
    return metrics


def print_summary(start_time: float):
    elapsed = int(time.time() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60

    metrics = compute_coco_metrics()

    print(f"\n{'='*60}")
    print(f"Model   : {MODEL_NAME}")
    print(f"Elapsed : {h}h {m}m {s}s")

    if metrics:
        print(f"\n--- COCO Metrics (paper suite, arXiv:2503.01601) ---")
        print(f"  mAP@[0.50:0.95]  : {metrics.get('mAP_0.50:0.95', 'N/A')}")
        print(f"  mAP50            : {metrics.get('mAP50',          'N/A')}")
        print(f"  mAP75            : {metrics.get('mAP75',          'N/A')}")
        print(f"  mAP (small)      : {metrics.get('mAP_small',      'N/A')}")
        print(f"  mAP (medium)     : {metrics.get('mAP_medium',     'N/A')}")
        print(f"  mAP (large)      : {metrics.get('mAP_large',      'N/A')}")
        print(f"  AR @ 100         : {metrics.get('AR_maxDets_100',  'N/A')}")
        print(f"  AR @ 300         : {metrics.get('AR_maxDets_300',  'N/A')}")
        print(f"  AR @ 1000        : {metrics.get('AR_maxDets_1000', 'N/A')}")
        print(f"  AR (small)       : {metrics.get('AR_small',        'N/A')}")
        print(f"  AR (medium)      : {metrics.get('AR_medium',       'N/A')}")
        print(f"  AR (large)       : {metrics.get('AR_large',        'N/A')}")

    weights_dir = pathlib.Path(PROJECT) / MODEL_NAME / "weights"
    print(f"\nWeights : {weights_dir}/best.pt")
    print(f"{'='*60}\n")


def main():
    start = time.time()
    print(f"\n[start] Training {MODEL_NAME}  —  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Fix dataset yaml
    fix_data_yaml()

    # 2. Convert val labels (COCO → YOLO)
    convert_coco_val_to_yolo()

    # 3. Create output directories
    pathlib.Path(PROJECT).mkdir(parents=True, exist_ok=True)

    # 4. Train
    from ultralytics import YOLO  # imported here so setup errors surface first

    model = YOLO(MODEL_PT)
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        project=PROJECT,
        name=MODEL_NAME,
        device=DEVICE,
        seed=SEED,
        exist_ok=True,
    )

    # 5. Summary
    print_summary(start)


if __name__ == "__main__":
    main()
