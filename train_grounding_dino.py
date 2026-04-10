"""
Train Grounding DINO (ResNet-50 backbone) on the stenosis detection dataset.
Uses MMDetection v3.x — mirrors the paper (arXiv:2503.01601) exactly.

Config used by the paper: grounding_dino_r50_4xb2-12e_coco.py
We fine-tune it on ARCADE / stenosis with a single GPU.

Prerequisites:
    pip install -U openmim
    mim install mmengine mmcv mmdet
    # Grounding DINO also requires the BERT text encoder:
    pip install transformers

Evaluates with the full COCO metric suite from the paper:
  mAP@[0.50:0.95], mAP50, mAP75, mAP(small/medium/large)
  AR@100, AR@300, AR@1000, AR(small/medium/large)
"""

import json
import pathlib
import time

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "grounding-dino-r50"
IMG_SIZE    = 800          # paper default
EPOCHS      = 12           # 1x schedule
BATCH       = 2            # paper: 4xb2; adapt to 1 GPU
WORKERS     = 4
SEED        = 42
DEVICE      = "cuda:0"

# Text prompt used during inference (open-vocabulary detector)
TEXT_PROMPT = "stenosis"

_HERE        = pathlib.Path(__file__).parent.resolve()
DATASET_ROOT = _HERE / "stenosis"
TRAIN_IMAGES = str(DATASET_ROOT / "train" / "images")
VAL_IMAGES   = str(DATASET_ROOT / "val"   / "images")
TRAIN_JSON   = str(DATASET_ROOT / "train" / "annotations" / "train.json")
VAL_JSON     = str(DATASET_ROOT / "val"   / "annotations" / "val.json")
PROJECT      = str(_HERE / "outputs")
WORK_DIR     = str(pathlib.Path(PROJECT) / MODEL_NAME)

# Filtered JSONs — stenosis only, category_id remapped to 1
FILTERED_TRAIN_JSON = str(DATASET_ROOT / "train" / "annotations" / "train_stenosis.json")
FILTERED_VAL_JSON   = str(DATASET_ROOT / "val"   / "annotations" / "val_stenosis.json")
# ──────────────────────────────────────────────────────────────────────────────


def filter_coco_to_stenosis(src: str, dst: str):
    """
    Write a filtered COCO JSON containing only the 'stenosis' category
    with category_id remapped to 1.  Idempotent — skips if dst exists.
    """
    if pathlib.Path(dst).exists():
        print(f"[setup] Filtered JSON already exists: {dst}")
        return

    with open(src) as f:
        data = json.load(f)

    orig_id = next(
        c["id"] for c in data["categories"] if c["name"].lower() == "stenosis"
    )
    filtered_anns = [
        {**ann, "category_id": 1}
        for ann in data["annotations"]
        if ann["category_id"] == orig_id
    ]

    out = {
        "info":        data.get("info", {}),
        "licenses":    data.get("licenses", []),
        "images":      data["images"],
        "annotations": filtered_anns,
        "categories":  [{"id": 1, "name": "stenosis", "supercategory": "lesion"}],
    }
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"[setup] Filtered COCO JSON → {dst}  ({len(filtered_anns)} annotations)")


def build_config():
    """
    Generate a custom mmdet config extending the paper's Grounding DINO config
    and pointing all paths at the ARCADE stenosis dataset.
    Returns the path to the written config file.
    """
    try:
        import mmdet
    except ImportError:
        raise RuntimeError(
            "MMDetection not found.\n"
            "Install with:\n"
            "  pip install -U openmim\n"
            "  mim install mmengine mmcv mmdet\n"
            "  pip install transformers"
        )

    mmdet_root = pathlib.Path(mmdet.__file__).parent
    base_cfg   = (
        mmdet_root / "configs" / "grounding_dino"
        / "grounding_dino_r50_4xb2-12e_coco.py"
    )
    if not base_cfg.exists():
        raise FileNotFoundError(
            f"Base config not found: {base_cfg}\n"
            "Make sure mmdet is installed via mim (not plain pip)."
        )

    cfg_text = f"""
_base_ = ['{base_cfg.as_posix()}']

# ── Dataset ──────────────────────────────────────────────────────────────────
data_root = '{DATASET_ROOT.as_posix()}/'
metainfo  = dict(classes=('stenosis',), palette=[(220, 20, 60)])

# Grounding DINO uses a text encoder — tell it the class name as a caption
model = dict(
    bbox_head=dict(
        num_classes=1,
    ),
)

train_dataloader = dict(
    batch_size={BATCH},
    num_workers={WORKERS},
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotations/train_stenosis.json',
        data_prefix=dict(img='train/images/'),
        # Pass class name as the text prompt for grounded training
        return_classes=True,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers={WORKERS},
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotations/val_stenosis.json',
        data_prefix=dict(img='val/images/'),
        return_classes=True,
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/annotations/val_stenosis.json',
    metric='bbox',
    format_only=False,
    metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
                  'AR@100', 'AR@300', 'AR@1000', 'AR_s', 'AR_m', 'AR_l'],
)

test_dataloader  = val_dataloader
test_evaluator   = val_evaluator

# ── Training schedule ─────────────────────────────────────────────────────────
train_cfg = dict(max_epochs={EPOCHS}, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP'),
    logger=dict(type='LoggerHook', interval=50),
)

randomness = dict(seed={SEED})
work_dir   = '{WORK_DIR}'
"""

    cfg_path = pathlib.Path(WORK_DIR) / "custom_cfg.py"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(cfg_text)
    print(f"[setup] Config written → {cfg_path}")
    return str(cfg_path)


def train(cfg_path: str):
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg    = Config.fromfile(cfg_path)
    runner = Runner.from_cfg(cfg)
    runner.train()


def compute_coco_metrics() -> dict:
    """
    Run COCO-style evaluation on the best checkpoint using pycocotools,
    matching the full metric suite of the paper (arXiv:2503.01601).
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("[eval] pycocotools not found — skipping. Install: pip install pycocotools")
        return {}

    try:
        from mmdet.apis import DetInferencer
    except ImportError:
        print("[eval] mmdet not found — skipping inference-based evaluation.")
        return {}

    # Find best checkpoint
    work  = pathlib.Path(WORK_DIR)
    ckpts = sorted(work.glob("best_coco*.pth")) or sorted(work.glob("epoch_*.pth"))
    if not ckpts:
        print(f"[eval] No checkpoint found in {WORK_DIR}")
        return {}
    checkpoint = str(ckpts[-1])
    cfg_path   = str(work / "custom_cfg.py")
    print(f"\n[eval] Running inference with {checkpoint} …")

    # Grounding DINO requires a text prompt at inference time
    inferencer = DetInferencer(
        model=cfg_path, weights=checkpoint, device=DEVICE,
        texts=TEXT_PROMPT,
    )

    coco_gt = COCO(FILTERED_VAL_JSON)
    fname_to_id = {
        pathlib.Path(img["file_name"]).name: img["id"]
        for img in coco_gt.dataset["images"]
    }

    val_imgs   = sorted(pathlib.Path(VAL_IMAGES).glob("*"))
    coco_preds = []
    for img_path in val_imgs:
        result = inferencer(str(img_path), return_datasamples=True)
        pred   = result["predictions"][0]
        img_id = fname_to_id.get(img_path.name)
        if img_id is None:
            continue
        bboxes = pred.pred_instances.bboxes.cpu().numpy()  # xyxy
        scores = pred.pred_instances.scores.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(bboxes, scores):
            coco_preds.append({
                "image_id":    img_id,
                "category_id": 1,
                "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score":       float(score),
            })

    out_dir    = pathlib.Path(WORK_DIR)
    preds_file = out_dir / "coco_preds.json"
    with open(preds_file, "w") as f:
        json.dump(coco_preds, f)

    if not coco_preds:
        print("[eval] No predictions generated.")
        return {}

    coco_dt = coco_gt.loadRes(str(preds_file))
    metrics: dict = {}

    # Pass 1: standard COCO + AR@100
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

    # Pass 2: AR@300
    ev300 = COCOeval(coco_gt, coco_dt, "bbox")
    ev300.params.maxDets = [1, 10, 300]
    ev300.evaluate(); ev300.accumulate(); ev300.summarize()
    metrics["AR_maxDets_300"] = round(float(ev300.stats[8]), 4)

    # Pass 3: AR@1000
    ev1000 = COCOeval(coco_gt, coco_dt, "bbox")
    ev1000.params.maxDets = [1, 10, 1000]
    ev1000.evaluate(); ev1000.accumulate(); ev1000.summarize()
    metrics["AR_maxDets_1000"] = round(float(ev1000.stats[8]), 4)

    metrics_file = out_dir / "coco_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Metrics saved → {metrics_file}")
    return metrics


def print_summary(start_time: float, metrics: dict):
    elapsed = int(time.time() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60

    print(f"\n{'='*60}")
    print(f"Model   : {MODEL_NAME}")
    print(f"Elapsed : {h}h {m}m {s}s")

    if metrics:
        print("\n--- COCO Metrics (paper suite, arXiv:2503.01601) ---")
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

    print(f"\nWork dir : {WORK_DIR}")
    print(f"{'='*60}\n")


def main():
    start = time.time()
    print(f"\n[start] Training {MODEL_NAME}  —  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Filter COCO annotations to stenosis only
    filter_coco_to_stenosis(TRAIN_JSON, FILTERED_TRAIN_JSON)
    filter_coco_to_stenosis(VAL_JSON,   FILTERED_VAL_JSON)

    # 2. Build MMDetection config
    cfg_path = build_config()

    # 3. Train
    train(cfg_path)

    # 4. Evaluate + summary
    metrics = compute_coco_metrics()
    print_summary(start, metrics)


if __name__ == "__main__":
    main()
