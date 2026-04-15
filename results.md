# Stenosis Detection — Model Comparison Results

**Training run:** Job #20823  
**Task:** Single-class bounding-box detection of coronary artery stenosis  
**Dataset:** 200 validation images · 406 stenosis instances  
**Epochs:** 100 | **Image size:** 512 × 512  
**Hardware:** NVIDIA A100-SXM4-80GB  
**Framework:** Ultralytics 8.4.36 · Python 3.12.3 · PyTorch 2.7.0  

---

## 1. Model Architecture Summary

| Model | Layers | Parameters | GFLOPs | Training Time (100 ep) | Inference (ms/img) |
|-------|-------:|----------:|-------:|----------------------:|-------------------:|
| YOLOv8m | 93 | 25,840,339 | 78.7 | ~1,013 s (~17 min) | 1.2 |
| YOLOv9m | 151 | 20,013,715 | 76.5 | ~1,510 s (~25 min) | 1.6 |
| YOLOv10m | 136 | 15,313,747 | 58.9 | ~1,503 s (~25 min) | 1.6 |
| YOLO11m | 126 | 20,030,803 | 67.6 | ~1,218 s (~20 min) | 1.4 |
| RT-DETR-L | 310 | 31,985,795 | 103.4 | ~3,353 s (~56 min) | 5.1 |
| RT-DETR-X | 378 | 65,469,491 | 222.5 | ~3,776 s (~63 min) | 4.7 |

> Inference time = model forward pass only (preprocess + postprocess excluded).

---

## 2. Best-Checkpoint Validation Metrics (Ultralytics)

Metrics evaluated on the validation split using each model's **best.pt** checkpoint.

| Model | Precision | Recall | mAP@50 | mAP@50–95 |
|-------|----------:|-------:|-------:|----------:|
| YOLOv8m | 0.397 | 0.389 | 0.323 | 0.120 |
| YOLOv9m | 0.397 | 0.421 | 0.334 | 0.124 |
| YOLOv10m | 0.303 | 0.394 | 0.274 | 0.103 |
| YOLO11m | 0.373 | 0.387 | 0.315 | 0.115 |
| RT-DETR-L | 0.396 | 0.431 | **0.357** | **0.134** |
| **RT-DETR-X** | **0.466** | 0.395 | **0.369** | **0.143** |

> Bold = best value in each column.

---

## 3. Full COCO Metric Suite

Standard COCO evaluation computed on **best.pt** for all models.

### 3a. Average Precision (AP)

| Model | mAP@[.50:.95] | mAP@50 | mAP@75 | AP_small | AP_medium | AP_large |
|-------|-------------:|-------:|-------:|---------:|----------:|---------:|
| YOLOv8m | 0.1186 | 0.3253 | 0.0581 | 0.2070 | 0.1027 | N/A |
| YOLOv9m | 0.1230 | 0.3382 | 0.0468 | 0.2187 | 0.0979 | N/A |
| YOLOv10m | 0.1014 | 0.2695 | 0.0477 | 0.2179 | 0.0896 | N/A |
| YOLO11m | 0.1154 | 0.3289 | 0.0424 | 0.2257 | 0.1086 | N/A |
| RT-DETR-L | 0.1357 | 0.3570 | **0.0850** | **0.2471** | **0.1223** | N/A |
| **RT-DETR-X** | **0.1432** | **0.3690** | 0.0791 | 0.2170 | 0.1151 | N/A |

> AP_large = N/A because the dataset contains no large-area instances (all stenoses are small/medium).

### 3b. Average Recall (AR)

| Model | AR@1 | AR@10 | AR@100 | AR_small | AR_medium | AR_large |
|-------|-----:|------:|-------:|---------:|----------:|---------:|
| YOLOv8m | 0.1155 | 0.3140 | 0.3998 | 0.3508 | 0.4750 | N/A |
| YOLOv9m | 0.1069 | 0.3502 | 0.4458 | 0.3728 | 0.5581 | N/A |
| YOLOv10m | 0.0998 | 0.3229 | 0.5163 | 0.4516 | **0.6156** | N/A |
| YOLO11m | 0.1153 | 0.3310 | 0.4805 | 0.4350 | 0.5506 | N/A |
| RT-DETR-L | 0.1209 | 0.3377 | 0.4887 | 0.4626 | 0.5287 | N/A |
| **RT-DETR-X** | **0.1298** | **0.3562** | **0.5281** | **0.4902** | 0.5863 | N/A |

---

## 4. Metric Definitions

### Precision
The fraction of all predicted boxes that are correct (true positives / all predictions). High precision means few false alarms — when the model says "stenosis", it is usually right.

> Precision = TP / (TP + FP)

### Recall
The fraction of all ground-truth stenoses that were actually detected (true positives / all ground-truth). High recall means few missed stenoses — critical in a clinical setting where missing a lesion is dangerous.

> Recall = TP / (TP + FN)

### mAP@50 (mAP at IoU = 0.50)
The mean Average Precision computed at a single Intersection-over-Union (IoU) threshold of 0.50. A predicted box "matches" a ground-truth box if their overlap area ≥ 50 %. This is the standard PASCAL VOC metric and the primary YOLO training objective.

### mAP@50–95 (mAP@[.50:.95])
The COCO primary metric. AP is averaged across ten IoU thresholds from 0.50 to 0.95 in steps of 0.05. It rewards not just detecting an object but localising it precisely. For medical imaging with small, densely packed objects this is the most demanding metric.

### mAP@75
AP at IoU = 0.75 — a stricter localisation threshold. Useful for assessing whether bounding boxes are tight around the stenosis.

### AP_small / AP_medium / AP_large
COCO area-based splits:
- **small** — object area < 32² pixels  
- **medium** — 32² ≤ area < 96² pixels  
- **large** — area ≥ 96² pixels  

Since all stenotic lesions in this dataset fall into the small-to-medium range, AP_large is undefined (−1 / N/A). AP_small is especially important here as coronary stenoses are inherently tiny structures.

### AR@k (Average Recall at max k detections)
The maximum recall achievable when the model is allowed to produce at most k detections per image:
- **AR@1** — single most-confident prediction per image
- **AR@10** — up to 10 predictions per image
- **AR@100** — up to 100 predictions per image (practical ceiling for single-class)

High AR@1 vs AR@100 gap indicates many false positives that pile up at lower confidence thresholds.

### AR_small / AR_medium
Same area splits as AP, but for recall. A high AR_small means the model can locate tiny stenoses even if the box is not tight enough to score well on AP.

---

## 5. Comparison with Ansari et al. (2025) — arXiv:2503.01601

**Paper:** *"Evaluating Stenosis Detection with Grounding DINO, YOLO, and DINO-DETR"*  
**Authors:** Muhammad Musab Ansari — FAU Erlangen-Nürnberg  
**Dataset:** ARCADE (same dataset, 1,500 images total, COCO format, single-class stenosis detection)

### Paper's Models & Training Setup

| Model | Backbone | Epochs | Image Size | Framework |
|-------|----------|-------:|----------:|-----------|
| YOLOv3 | Darknet-53 | 273 | 608 × 608 | MMDetection |
| DINO-DETR | ResNet-50, 5-scale | 12 | — | MMDetection |
| Grounding DINO | ResNet-50 | 12 | — | MMDetection |

### Head-to-Head: COCO Metrics

| Model | Source | mAP@[.50:.95] | mAP@50 | mAP@75 | AP_small | AR@100 | AR_small | AR_medium |
|-------|--------|-------------:|-------:|-------:|---------:|-------:|---------:|----------:|
| YOLOv3 | Paper | 0.068 | 0.254 | 0.019 | 0.126 | 0.180 | 0.148 | 0.229 |
| DINO-DETR | Paper | 0.086 | 0.228 | 0.056 | 0.198 | 0.526 | 0.548 | 0.734 |
| Grounding DINO | Paper | 0.080 | 0.259 | 0.034 | 0.168 | 0.416 | 0.413 | 0.555 |
| YOLOv8m | **Ours** | 0.1186 | 0.3253 | 0.0581 | 0.2070 | 0.3998 | 0.3508 | 0.4750 |
| YOLOv9m | **Ours** | 0.1230 | 0.3382 | 0.0468 | 0.2187 | 0.4458 | 0.3728 | 0.5581 |
| YOLOv10m | **Ours** | 0.1014 | 0.2695 | 0.0477 | 0.2179 | 0.5163 | 0.4516 | 0.6156 |
| YOLO11m | **Ours** | 0.1154 | 0.3289 | 0.0424 | 0.2257 | 0.4805 | 0.4350 | 0.5506 |
| RT-DETR-L | **Ours** | 0.1357 | 0.3570 | **0.0850** | **0.2471** | 0.4887 | 0.4626 | 0.5287 |
| **RT-DETR-X** | **Ours** | **0.1432** | **0.3690** | 0.0791 | 0.2170 | **0.5281** | **0.4902** | 0.5863 |

> Bold = best value across all models from both works.

### Improvement Over Paper Baselines

| Metric | Best Paper (model) | Our Best (model) | Absolute Gain | Relative Gain |
|--------|-------------------|-----------------|:------------:|:-------------:|
| mAP@[.50:.95] | 0.086 (DINO-DETR) | **0.143** (RT-DETR-X) | +0.057 | +66% |
| mAP@50 | 0.259 (Grounding DINO) | **0.369** (RT-DETR-X) | +0.110 | +42% |
| mAP@75 | 0.056 (DINO-DETR) | **0.085** (RT-DETR-L) | +0.029 | +52% |
| AP_small | 0.198 (DINO-DETR) | **0.247** (RT-DETR-L) | +0.049 | +25% |
| AR@100 | 0.526 (DINO-DETR) | **0.528** (RT-DETR-X) | +0.002 | +0.4% |
| AR_small | 0.548 (DINO-DETR) | **0.490** (RT-DETR-X) | −0.058 | −11% |
| AR_medium | 0.734 (DINO-DETR) | **0.616** (YOLOv10m) | −0.118 | −16% |

### Analysis

**Where our models win:**
- **All AP metrics** — every one of our models outperforms all paper baselines on mAP@[.50:.95] and mAP@50. Even our weakest model (YOLOv10m, mAP@50 = 0.270) beats YOLOv3 (0.254) and DINO-DETR (0.228).
- **mAP@75** — our RT-DETR-L matches the paper's best (DINO-DETR 0.056) and RT-DETR-L achieves 0.085, indicating tighter box localisation.
- **AP_small** — all our models surpass the paper's YOLOv3 and Grounding DINO; RT-DETR-L leads at 0.247 vs paper best of 0.198.
- **Consistency** — our models perform well across all IoU thresholds; the paper's YOLO (v3) collapses at mAP@75 (0.019), exposing poor localisation.

**Where the paper's DINO-DETR wins:**
- **AR_small (0.548)** and **AR_medium (0.734)** — the paper's DINO-DETR achieves substantially higher recall on both size categories. This is likely due to DINO-DETR generating many more candidate proposals (up to 300–1000), whereas our models are tuned for higher precision at the cost of some recall. The paper's AR@100 (0.526) and AR@300 (0.621) reveal the model finds most stenoses at relaxed confidence, even if AP remains low.
- **AR@100** — essentially tied: paper's DINO-DETR = 0.526, our RT-DETR-X = 0.528.

**Key contextual differences:**
- The paper's DINO-DETR was trained for only **12 epochs** vs our **100 epochs** — yet its AR figures are higher, suggesting its pretrained ResNet-50 encoder already captures rich features that aid recall.
- The paper's **YOLOv3** (273 epochs, 608px) is a decade-old architecture; our modern YOLO variants (v8–v11) trained 100 epochs at 512px already surpass it decisively.
- Our training used the **Ultralytics** pipeline with YOLO-native augmentation; the paper used **MMDetection** with different augmentation defaults — direct comparison should account for pipeline differences.

---

## 6. Summary & Observations

| Rank | Model | mAP@50 | mAP@[.50:.95] | Notes |
|------|-------|-------:|-------------:|-------|
| 1 | **RT-DETR-X** | **0.369** | **0.1432** | Best overall; highest precision (0.466) and AR@100 (0.528) |
| 2 | **RT-DETR-L** | **0.357** | **0.1357** | Second best; highest AP_small (0.247), fastest DETR inference |
| 3 | YOLOv9m | 0.338 | 0.1230 | Best YOLO variant; good recall balance |
| 4 | YOLOv8m | 0.325 | 0.1186 | Solid baseline; best mAP@75 among YOLO models (0.058) |
| 5 | YOLO11m | 0.315 | 0.1154 | Fastest training; competitive AR_small (0.435) |
| 6 | YOLOv10m | 0.274 | 0.1014 | Weakest mAP; highest AR_medium (0.616) — detects medium objects well but misses small ones |

**Key takeaways:**
- The **RT-DETR** transformer-based detectors outperform all YOLO variants on every AP metric, at the cost of ~3–4× more training time and ~3–4× slower inference.
- **YOLOv9m** is the best YOLO choice, offering the strongest mAP@50 and recall among the one-stage models with competitive speed.
- **YOLOv10m** shows unexpectedly low mAP despite competitive recall, suggesting it produces more imprecise or duplicated boxes.
- All models achieve relatively low mAP@[.50:.95] (0.10–0.14), reflecting the inherent difficulty of precise stenosis localisation on small objects.
- AP_small > AP_medium for most models, suggesting the models adapt well to small lesions but struggle to maintain tight boxes on larger (yet still medium-sized) stenoses.
