"""
Train RT-DETR-R50 on the stenosis detection dataset.
Uses Ultralytics ≥ 8.1 native RT-DETR support.
Saves best.pt and last.pt to PROJECT/MODEL_NAME/weights/.
"""

import csv
import json
import pathlib
import time
from collections import defaultdict

import yaml

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "rtdetr-r50"
MODEL_PT   = "rtdetr-r50.pt"
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


def print_summary(start_time: float):
    elapsed = int(time.time() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    print(f"\n{'='*60}")
    print(f"Model   : {MODEL_NAME}")
    print(f"Elapsed : {h}h {m}m {s}s")

    results_csv = pathlib.Path(PROJECT) / MODEL_NAME / "results.csv"
    if results_csv.exists():
        rows = list(csv.DictReader(results_csv.open()))
        map_col = next(
            (k for k in rows[0] if "mAP50" in k and "mAP50-95" not in k), None
        )
        if map_col and rows:
            best = max(rows, key=lambda r: float(r[map_col].strip() or 0))
            best_epoch = rows.index(best) + 1
            print(f"Best mAP50 : {float(best[map_col].strip()):.4f}  (epoch {best_epoch})")

    weights_dir = pathlib.Path(PROJECT) / MODEL_NAME / "weights"
    print(f"Weights : {weights_dir}/best.pt")
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
    # RT-DETR is natively supported by Ultralytics ≥ 8.1 via the YOLO() wrapper.
    # rtdetr-r50.pt uses a ResNet-50 backbone with a transformer detection head.
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
