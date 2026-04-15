# Stenosis Detection & QCA Analysis Pipeline

A production-ready pipeline for **coronary stenosis detection and quantitative analysis** in X-ray angiography images, benchmarking 9 object detection models on the [ARCADE dataset](https://arcade.grand-challenge.org/) and providing a full **Quantitative Coronary Angiography (QCA)** measurement suite with **QFR / FFR estimation**.

All models are evaluated using the **identical COCO metric suite** from the reference paper (arXiv:2503.01601).

---

## Models

### Ultralytics Models

| Model | Backbone | Type | Weight |
|-------|----------|------|--------|
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

## QCA Analysis Pipeline (`analyze_stenosis.py`)

After each detection bounding box is identified, the pipeline performs research-grade **Quantitative Coronary Angiography** — the same measurement methodology used in clinical systems (CAAS, QAngio XA).

### Pipeline Steps

```
Detected BBox
    │
    ▼
1. Crop ROI from original image
    │
    ▼
2. Greyscale + CLAHE
   (Contrast Limited Adaptive Histogram Equalisation)
   clipLimit=2.0, tileGridSize=8×8
    │
    ▼
3. Frangi Vesselness Filter  [Frangi et al., Med. Image Anal. 1998]
   Multi-scale (σ = 1–5 px), responds to tubular structures only
    │
    ▼
4. Otsu Threshold → Binary Mask
   Morphological closing (r=3) + opening (r=1) + largest component
    │
    ▼
5. Skeletonisation — Lee's algorithm  [skimage]
   Topology-preserving medial axis
    │
    ▼
6. Centerline Ordering
   Nearest-neighbour traversal from endpoint → ordered path
    │
    ▼
7. Tangent-based Perpendicular Radius Profile
   • Local tangent via PCA over ±7-point window
   • Perpendicular rays cast to vessel wall
   • radius[i] = (d₊ + d₋) / 2
    │
    ▼
8. QCA Metrics
   • MLD  (Minimal Lumen Diameter)
   • D_ref (mean of top-25% diameters — proximal + distal reference)
   • % DS  = (1 − MLD / D_ref) × 100
    │
    ▼
9. QFR / FFR Estimation  (see below)
```

### Outputs per Detection

| Output | Description |
|--------|-------------|
| Crop | Raw bounding-box crop |
| CLAHE | Contrast-enhanced greyscale |
| Frangi | Vesselness probability map |
| Binary Mask | Thresholded vessel region |
| Skeleton Overlay | Medial axis drawn on crop |
| Radius Overlay | Centerline + perpendicular ticks + MLD marker |
| Diameter Profile | Matplotlib plot: diameter vs. centerline position |

---

## QFR & FFR Estimation

### Clinical Context

| Term | Definition | Reference |
|------|-----------|-----------|
| **FFR** (Fractional Flow Reserve) | Wire-based gold standard: Pd/Pa during hyperaemia | De Bruyne et al., NEJM 2012 |
| **QFR** (Quantitative Flow Ratio) | Wire-free computational FFR from two angiographic views + TIMI frame count | Xu et al., EHJ 2019 (FAVOR II) |
| **qFFR** | Computational FFR variants using single-view or CT data | Nørgaard et al., JACC 2014 |

**Our estimate** — a 2-D single-plane proxy using the **simplified Gould-Gorlin pressure-drop model**:

```
FFR_est = 1 − 0.5 × (DS / (1 − DS))²
```

derived from Gould et al. (1974) under the assumption of normal resting coronary flow.

### Clinical Cut-offs Applied

| Zone | FFR_est | % DS (approx.) | Interpretation |
|------|---------|-----------------|----------------|
| Green | ≥ 0.80 | < 60% | Non-significant — deferral safe (DEFER, FAME) |
| Grey  | 0.75 – 0.80 | ~60–70% | Borderline — wire FFR / QFR recommended |
| Red   | < 0.75 | > 70% | Significant — revascularisation likely beneficial (FAME-2) |

### Limitations of the 2-D Estimate

Our estimate is a **rough proxy** — not a replacement for true QFR or FFR — because:

1. **Single-plane** — true QFR requires two angiographic views for 3-D vessel reconstruction.
2. **No flow velocity** — QFR uses TIMI frame count to calibrate flow; we assume normal resting flow.
3. **Lesion length not modelled** — the Gould formula is calibrated for focal stenoses; diffuse disease is underestimated.
4. **Tandem lesoses** — non-linear interaction between sequential lesions is not accounted for.

> For research and clinical decision support, the estimate should be interpreted alongside the % DS and diameter profile — not as a standalone FFR value.

---

## Evaluation Metrics (COCO Suite)

| Metric | Description |
|--------|-------------|
| mAP@[0.50:0.95] | Primary COCO metric |
| mAP50 | IoU threshold 0.50 |
| mAP75 | IoU threshold 0.75 (stricter) |
| mAP (small/medium/large) | By object size |
| AR @ 100 / 300 / 1000 | Average recall at max-detection counts |
| AR (small/medium/large) | Recall by object size |

Metrics saved to `outputs/<model>/coco_metrics.json`.

### Paper Results (Table I, ARCADE val set)

| Metric | DINO-DETR | YOLOv3 | Grounding DINO |
|--------|-----------|--------|----------------|
| mAP@[0.50:0.95] | 0.086 | 0.068 | 0.080 |
| mAP50 | 0.228 | 0.254 | **0.259** |
| mAP75 | **0.056** | 0.019 | 0.034 |
| mAP (small) | **0.198** | 0.126 | 0.168 |
| AR @ 100 | **0.526** | 0.180 | 0.416 |
| AR @ 300 | **0.621** | 0.180 | 0.469 |
| AR (medium) | **0.734** | 0.229 | 0.555 |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | ARCADE Challenge — Stenosis Detection task |
| Format | COCO JSON (bboxes) + auto-generated YOLO txt |
| Classes | 1 — `stenosis` |
| Splits | 1,001 train / 200 val / 300 test |
| Resolution | 512 × 512 PNG |

---

## Training Configuration

**Ultralytics models** — identical hyperparameters across all 6:

```
Image size : 512 × 512
Epochs     : 100
Batch      : 8
Workers    : 4
Seed       : 42
Device     : GPU 0
```

**MMDetection models** — paper-matching 1× COCO schedule:

```
YOLOv3         : 273 epochs, img 608, batch 8
DINO-DETR      : 12 epochs,  img 800, batch 2
Grounding DINO : 12 epochs,  img 800, batch 2
```

---

## Repository Structure

```
Stenosis/
├── app.py                      # Gradio UI (Detection + QCA Analysis tabs)
├── analyze_stenosis.py         # QCA pipeline + QFR/FFR estimation
├── submit.sbatch               # SLURM job script
├── run_training.sh             # Training shell (models 7–9 only: YOLOv3, DINO, G-DINO)
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
    ├── data_bbox.yaml
    ├── train/images/ + annotations/
    ├── val/images/ + annotations/
    └── test/images/ + annotations/
```

---

## Setup

### Ultralytics models
```bash
pip install ultralytics opencv-python-headless pyyaml pycocotools
```

### MMDetection models (paper baselines)
```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" mmdet
pip install transformers    # Grounding DINO BERT encoder
```

### QCA Analysis + Gradio UI
```bash
pip install gradio scikit-image scipy matplotlib opencv-python
```

---

## Training

```bash
# Single model
python train_yolo11.py

# Paper baselines only (cluster — models 7, 8, 9)
sbatch submit.sbatch
```

Weights saved to `outputs/<model_name>/weights/best.pt`.

---

## Gradio App

```bash
python app.py
# http://localhost:7860
```

### Tab 1 — Detection
- Model selector + confidence slider
- Ground-truth (green) vs. prediction overlay
- ARCADE test sample gallery

### Tab 2 — QCA Analysis
- Runs full QCA pipeline after detection
- Shows all 7 intermediate images per detection
- Reports QCA metrics table + QFR/FFR estimate + recommendation

---

## References

| Reference | Relevance |
|-----------|-----------|
| Frangi et al., MICCAI 1998 | Vesselness filter used in preprocessing |
| Gould et al., Circulation 1974 | Pressure-drop model for FFR estimation |
| De Bruyne et al., NEJM 2012 (FAME) | FFR cut-off validation (0.80) |
| Tonino et al., NEJM 2009 (FAME) | FFR-guided PCI outcome data |
| Pijls et al., NEJM 2007 (DEFER) | Deferral safety at FFR ≥ 0.80 |
| De Bruyne et al., NEJM 2014 (FAME-2) | Revascularisation benefit at FFR < 0.80 |
| Xu et al., EHJ 2019 (FAVOR II) | QFR validation vs. wire FFR |
| Marangoni et al., arXiv:2503.01601 | ARCADE benchmark paper |

---

## SLURM Environment

| Setting | Value |
|---------|-------|
| Node | `dgxanode03` |
| Container | `nvidia+pytorch+25.04-py3.sqsh` |
| GPU | 1 × A100 (via `--gres=gpu:1`) |
| CPUs | 8 |
| RAM | 32 GB |
| Wall time | 120 hours |
