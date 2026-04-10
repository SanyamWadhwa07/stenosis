# Stenosis Detection — Multi-Model Benchmarking Pipeline

A production-ready training and evaluation pipeline for **coronary stenosis detection** in angiography images, benchmarking 9 object detection models on the [ARCADE dataset](https://arcade.grand-challenge.org/).

All models are evaluated using the **identical COCO metric suite** from the reference paper (arXiv:2503.01601), enabling direct comparison.

---

## Models

### Our Models (Ultralytics)

| Model | Backbone | Type | Weight file |
|-------|----------|------|-------------|
| YOLO11m | CSP-DarkNet | One-stage | `yolo11m.pt` |
| YOLOv8m | CSP-DarkNet | One-stage | `yolov8m.pt` |
| YOLOv9m | GELAN | One-stage | `yolov9m.pt` |
| YOLOv10m | CSP-DarkNet | One-stage (NMS-free) | `yolov10m.pt` |
| RT-DETR-L | ResNet-18 | Transformer | `rtdetr-l.pt` |
| RT-DETR-X | ResNet-50 | Transformer | `rtdetr-x.pt` |

### Paper Baselines (MMDetection — arXiv:2503.01601)

| Model | Backbone | Framework | Paper mAP50 |
|-------|----------|-----------|-------------|
| YOLOv3 | DarkNet-53 | MMDetection | 0.254 |
| DINO-DETR | ResNet-50 | MMDetection | 0.228 |
| Grounding DINO | ResNet-50 | MMDetection | **0.259** |

---

## Evaluation Metrics

Every training script reports the **full COCO metric suite** (via pycocotools) to enable apples-to-apples comparison with the paper:

| Metric | Description |
|--------|-------------|
| mAP@[0.50:0.95] | Primary COCO metric |
| mAP50 | IoU threshold 0.50 |
| mAP75 | IoU threshold 0.75 (stricter) |
| mAP (small) | Objects < 32² px |
| mAP (medium) | Objects 32²–96² px |
| mAP (large) | Objects > 96² px |
| AR @ 100 | Average recall, max 100 detections |
| AR @ 300 | Average recall, max 300 detections |
| AR @ 1000 | Average recall, max 1000 detections |
| AR (small/medium/large) | Recall by object size |

Metrics are printed at the end of each run and saved to `outputs/<model>/coco_metrics.json`.

### Paper Results (Table I, ARCADE val set)

| Metric | DINO-DETR | YOLOv3 | Grounding DINO |
|--------|-----------|--------|----------------|
| mAP@[0.50:0.95] | 0.086 | 0.068 | 0.080 |
| mAP50 | 0.228 | 0.254 | **0.259** |
| mAP75 | **0.056** | 0.019 | 0.034 |
| mAP (small) | **0.198** | 0.126 | 0.168 |
| AR @ 100 | **0.526** | 0.180 | 0.416 |
| AR @ 300 | **0.621** | 0.180 | 0.469 |
| AR @ 1000 | **0.621** | 0.180 | 0.469 |
| AR (small) | **0.548** | 0.148 | 0.413 |
| AR (medium) | **0.734** | 0.229 | 0.555 |

---

## Dataset

- **Source:** ARCADE Challenge — Stenosis Detection task
- **Format:** COCO JSON (bboxes) + YOLO txt (auto-generated for Ultralytics)
- **Classes:** 1 — `stenosis`
- **Splits:** 1,001 train / 200 val / 300 test images
- **Resolution:** 512 × 512 PNG

---

## Training Configuration

Ultralytics models trained with identical hyperparameters:

```
Image size : 512 × 512
Epochs     : 100
Batch size : 8
Workers    : 4
Seed       : 42
Device     : GPU 0
```

MMDetection models use paper-matching configs (1× COCO schedule):

```
YOLOv3       : 273 epochs, img 608, batch 8
DINO-DETR    : 12 epochs,  img 800, batch 2
Grounding DINO: 12 epochs, img 800, batch 2
```

---

## Repository Structure

```
Stenosis/
├── app.py                      # Gradio inference UI
├── submit.sbatch               # SLURM job script — runs all 9 models
├── run_training.sh             # Training shell script (run inside container)
│
├── train_yolo11.py             # YOLO11m
├── train_yolov8.py             # YOLOv8m
├── train_yolov9.py             # YOLOv9m
├── train_yolov10.py            # YOLOv10m
├── train_rtdetr_r18.py         # RT-DETR-L
├── train_rtdetr_r50.py         # RT-DETR-X
│
├── train_yolov3.py             # YOLOv3-D53  (paper baseline, MMDetection)
├── train_dino_detr.py          # DINO-DETR-R50 (paper baseline, MMDetection)
├── train_grounding_dino.py     # Grounding DINO-R50 (paper baseline, MMDetection)
│
└── stenosis/
    ├── data_bbox.yaml          # Ultralytics dataset config (paths auto-fixed)
    ├── train/
    │   ├── images/
    │   └── annotations/
    │       ├── train.json                  # COCO format (26 classes)
    │       └── train_stenosis.json         # Auto-generated: stenosis only
    ├── val/
    │   ├── images/
    │   ├── labels/                         # Auto-generated YOLO txts
    │   └── annotations/
    │       ├── val.json
    │       └── val_stenosis.json
    └── test/
        ├── images/
        └── annotations/test.json
```

---

## Setup

### Ultralytics models
```bash
pip install ultralytics opencv-python-headless pyyaml pycocotools
```

### Paper baseline models (MMDetection)
```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" mmdet
pip install transformers    # required for Grounding DINO
```

---

## Training

### Run a single model
```bash
python train_yolo11.py          # or any train_*.py
```

### Run all 9 models sequentially (SLURM cluster)
```bash
sbatch submit.sbatch
```

Monitor:
```bash
squeue -u $USER
tail -f job_output_<JOBID>.txt
```

Each script automatically:
1. Sets up dataset paths / COCO annotation JSONs
2. Trains the model
3. Runs pycocotools COCO evaluation on `best.pt`
4. Prints + saves the full metric table to `outputs/<model>/coco_metrics.json`

Weights saved to:
```
outputs/<model_name>/weights/best.pt
outputs/<model_name>/weights/last.pt
```

---

## Inference — Gradio App

```bash
python app.py
# http://localhost:7860
```

Features:
- Model selector — switch between all models instantly
- Upload tab — run detection on any image
- Webcam tab — live real-time streaming inference
- Confidence & IoU sliders
- Weight status table — shows fine-tuned vs pretrained weights
- Per-detection output — confidence score and bounding box coordinates

Fine-tuned `best.pt` weights are loaded automatically when present in `outputs/`.

---

## SLURM Environment

| Setting | Value |
|---------|-------|
| Node | `dgxanode03` |
| Container | `nvidia+pytorch+25.04-py3.sqsh` |
| GPU | 1 × (via `--gres=gpu:1`) |
| CPUs | 8 |
| RAM | 32 GB |
| Wall time | 120 hours |
