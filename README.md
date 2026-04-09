# 🫀 Stenosis Detection — Multi-Model Benchmarking Pipeline

A production-ready training and inference pipeline for **coronary stenosis detection** in angiography images, benchmarking 6 state-of-the-art object detection models on the [ARCADE dataset](https://arcade.grand-challenge.org/).

---

## Models

| Model | Backbone | Type | Weight |
|-------|----------|------|--------|
| YOLO11m | CSP-DarkNet | One-stage | `yolo11m.pt` |
| YOLOv8m | CSP-DarkNet | One-stage | `yolov8m.pt` |
| YOLOv9m | GELAN | One-stage | `yolov9m.pt` |
| YOLOv10m | CSP-DarkNet | One-stage (NMS-free) | `yolov10m.pt` |
| RT-DETR-R18 | ResNet-18 | Transformer | `rtdetr-r18.pt` |
| RT-DETR-R50 | ResNet-50 | Transformer | `rtdetr-r50.pt` |

---

## Dataset

- **Source:** ARCADE Challenge — Stenosis Detection task
- **Format:** YOLO (bounding boxes)
- **Classes:** 1 — `stenosis`
- **Splits:** 1,001 train / 200 val / 300 test images
- **Resolution:** 512 × 512 PNG (trained at 1024 × 1024)

---

## Training Configuration

All models trained with **identical hyperparameters** for a fair benchmark:

```
Image size : 1024 × 1024
Epochs     : 100
Batch size : 8
Workers    : 4
Seed       : 42
Device     : GPU 0
```

---

## Repository Structure

```
stenosis/
├── app.py                  # Gradio inference UI (upload + live webcam)
├── submit.sbatch           # SLURM job script — runs all 6 models sequentially
├── train_yolo11.py         # YOLO11m training script
├── train_yolov8.py         # YOLOv8m training script
├── train_yolov9.py         # YOLOv9m training script
├── train_yolov10.py        # YOLOv10m training script
├── train_rtdetr_r18.py     # RT-DETR-R18 training script
├── train_rtdetr_r50.py     # RT-DETR-R50 training script
└── stenosis/
    ├── data_bbox.yaml      # Dataset config (paths auto-fixed at runtime)
    ├── train/images/       # Training images (not tracked in git)
    ├── val/images/         # Validation images (not tracked in git)
    └── test/images/        # Test images (not tracked in git)
```

---

## Setup

```bash
pip install ultralytics opencv-python-headless pyyaml gradio
```

---

## Training

### Run a single model
```bash
python train_yolo11.py
# or any of the other train_*.py scripts
```

### Run all 6 models sequentially (SLURM cluster)
```bash
sbatch submit.sbatch
```

Monitor:
```bash
squeue -u $USER
tail -f job_output_<JOBID>.txt
```

Each script automatically:
1. Fixes `data_bbox.yaml` paths to absolute cluster paths
2. Converts val COCO JSON annotations → YOLO format (idempotent)
3. Trains the model and saves `best.pt` + `last.pt`
4. Prints best mAP50, epoch, and elapsed time

Weights are saved to:
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
- **Model selector** — switch between all 6 models instantly
- **Upload tab** — run detection on any image
- **Webcam tab** — live real-time streaming inference
- **Confidence & IoU sliders** — tune thresholds at runtime
- **Weight status table** — shows which models have fine-tuned weights vs pretrained
- **Per-detection output** — confidence score and bounding box coordinates for each detection

> Fine-tuned `best.pt` weights are loaded automatically when present in `outputs/`.
> Falls back to Ultralytics pretrained weights if training hasn't run yet.

---

## SLURM Environment

| Setting | Value |
|---------|-------|
| Node | `dgxanode03` |
| Container | `nvidia+pytorch+25.04-py3.sqsh` |
| GPU | 1 × (via `--gres=gpu:1`) |
| CPUs | 8 |
| RAM | 32 GB |
| Wall time | 72 hours |
