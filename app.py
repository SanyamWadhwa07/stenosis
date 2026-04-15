"""
Stenosis Detection — Gradio Inference App
Tabs:
  1. Detection      — bounding-box inference with GT overlay
  2. QCA Analysis   — full QCA pipeline per detected box
"""

import json
import pathlib
import time
from functools import lru_cache

import cv2
import gradio as gr
import numpy as np

from analyze_stenosis import run_qca

_HERE   = pathlib.Path(__file__).parent.resolve()
OUTPUTS = _HERE / "outputs"

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "YOLO11m":  {"dir": "yolo11m",  "pretrained": "yolo11m.pt"},
    "YOLOv8m":  {"dir": "yolov8m",  "pretrained": "yolov8m.pt"},
    "YOLOv9m":  {"dir": "yolov9m",  "pretrained": "yolov9m.pt"},
    "YOLOv10m": {"dir": "yolov10m", "pretrained": "yolov10m.pt"},
    "RT-DETR-L":{"dir": "rtdetr-l", "pretrained": "rtdetr-l.pt"},
    "RT-DETR-X":{"dir": "rtdetr-x", "pretrained": "rtdetr-x.pt"},
}

# ── Sample images ─────────────────────────────────────────────────────────────
SAMPLE_DIR = _HERE / "stenosis" / "test" / "images"
SAMPLES    = [str(SAMPLE_DIR / f"{n}.png") for n in [1, 20, 50, 80, 120, 160]]

# ── Ground truth from COCO JSON ───────────────────────────────────────────────
_COCO_PATH = _HERE / "stenosis" / "test" / "annotations" / "test.json"
_STENOSIS_CAT_ID = 26   # category id for "stenosis" in this dataset

def _load_gt() -> dict[str, list[list[int]]]:
    """Returns {filename: [[x1,y1,x2,y2], ...]} for stenosis annotations."""
    with open(_COCO_PATH) as f:
        coco = json.load(f)
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    gt: dict[str, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != _STENOSIS_CAT_ID:
            continue
        fname = id2file[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        gt.setdefault(fname, []).append([int(x), int(y), int(x + w), int(y + h)])
    return gt

GT = _load_gt()


def _draw_gt(image_rgb: np.ndarray, fname: str) -> np.ndarray:
    """Draw ground-truth boxes (green) on a copy of the image."""
    out = image_rgb.copy()
    boxes = GT.get(fname, [])
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, "GT", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    if not boxes:
        cv2.putText(out, "No GT annotations", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
    return out


# ── Model loading ─────────────────────────────────────────────────────────────
def _weight_path(model_key: str) -> tuple[str, str]:
    cfg = MODELS[model_key]
    trained = OUTPUTS / cfg["dir"] / "weights" / "best.pt"
    if trained.exists():
        return str(trained), "fine-tuned"
    return cfg["pretrained"], "pretrained-only"


@lru_cache(maxsize=6)
def _load_model(model_key: str, weight_path: str):
    from ultralytics import YOLO
    return YOLO(weight_path)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_detection(image, model_key: str, conf: float, img_path: str):
    """
    image     : np.ndarray RGB from Gradio
    img_path  : hidden state holding the file path of the loaded image
                (set when a sample is clicked; empty for manual uploads)
    """
    if image is None:
        return None, None, "Upload or click a sample image first."

    t0 = time.perf_counter()
    weight_path, weight_label = _weight_path(model_key)
    model = _load_model(model_key, weight_path)
    results = model(image, conf=conf, iou=0.45, verbose=False)[0]
    elapsed = (time.perf_counter() - t0) * 1000

    pred_rgb = results.plot()[:, :, ::-1]   # BGR → RGB

    # Ground truth — derive filename from path if available
    fname = pathlib.Path(img_path).name if img_path else ""
    gt_rgb = _draw_gt(image, fname)

    n = len(results.boxes)
    confs = results.boxes.conf.cpu().numpy() if n > 0 else []
    det_lines = "\n".join(
        f"- Det {i}: conf={c:.2f}  [{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}]"
        for i, (b, c) in enumerate(zip(results.boxes.xyxy.cpu().numpy(), confs), 1)
    )
    gt_count = len(GT.get(fname, []))
    summary = (
        f"**Model:** {model_key} ({weight_label})  |  "
        f"**Predicted:** {n}  |  "
        f"**GT boxes:** {gt_count}  |  "
        f"**Inference:** {elapsed:.0f} ms\n\n"
        + (det_lines if det_lines else "_No stenosis detected above threshold._")
    )
    return gt_rgb, pred_rgb, summary


# ── QCA inference ─────────────────────────────────────────────────────────────
_MAX_QCA_BOXES = 3   # analyse at most this many detections per image

def run_qca_analysis(image, model_key: str, conf: float):
    """
    1. Run YOLO inference to get bounding boxes.
    2. For each box (up to _MAX_QCA_BOXES) run the full QCA pipeline.
    3. Return flat lists suitable for Gradio Gallery + Markdown.
    """
    if image is None:
        return None, None, None, None, None, None, None, None, [], "_Upload or click a sample image first._"

    weight_path, weight_label = _weight_path(model_key)
    model   = _load_model(model_key, weight_path)
    results = model(image, conf=conf, iou=0.45, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else []

    if len(boxes) == 0:
        msg = (
            f"**Model:** {model_key} ({weight_label})  |  "
            "**No stenosis detected above threshold.**\n\n"
            "_Lower the confidence slider and retry._"
        )
        return None, None, None, None, None, None, None, None, [], msg

    # Detection overview (annotated image)
    det_overview = results.plot()[:, :, ::-1]   # BGR → RGB

    # Per-box QCA — collect per-step gallery images
    # We show steps for the FIRST detected box prominently,
    # then append rows for subsequent boxes.
    all_crop     = []
    all_clahe    = []
    all_frangi   = []
    all_binary   = []
    all_skel     = []
    all_radius   = []
    all_plot     = []
    metrics_md   = []

    confs = results.boxes.conf.cpu().numpy()

    for i, (bbox, conf_score) in enumerate(zip(boxes[:_MAX_QCA_BOXES],
                                               confs[:_MAX_QCA_BOXES])):
        qca = run_qca(image, bbox.tolist())
        err = qca.get("error", "")

        all_crop.append(  qca.get("crop"))
        all_clahe.append( qca.get("clahe"))
        all_frangi.append(qca.get("frangi"))
        all_binary.append(qca.get("binary"))
        all_skel.append(  qca.get("skeleton_overlay"))
        all_radius.append(qca.get("radius_overlay"))
        all_plot.append(  qca.get("radius_plot"))

        m   = qca.get("metrics", {})
        qfr = m.get("qfr", {})
        if err:
            metrics_md.append(f"**Detection {i+1}** (conf={conf_score:.2f}): _{err}_")
        elif m:
            pct   = m.get("pct_diameter_stenosis", 0.0)
            mld   = m.get("mld_px", 0.0)
            ref   = m.get("ref_diameter_px", 0.0)
            ffr   = qfr.get("ffr_estimate", "N/A")
            ffr_lo = qfr.get("ffr_ci_low",  "")
            ffr_hi = qfr.get("ffr_ci_high", "")
            zone  = qfr.get("qfr_zone", "N/A")
            hcls  = qfr.get("hemodynamic_class", "N/A")
            rec   = qfr.get("recommendation", "")
            note  = qfr.get("note", "")

            ci_str = f" (95% CI: {ffr_lo}–{ffr_hi})" if ffr_lo else ""

            metrics_md.append(
                f"### Detection {i+1}  (model conf = {conf_score:.2f})\n\n"
                f"| QCA Metric | Value |\n"
                f"|---|---|\n"
                f"| MLD | {mld:.1f} px |\n"
                f"| Reference Diameter | {ref:.1f} px |\n"
                f"| **% Diameter Stenosis** | **{pct:.1f}%** |\n"
                f"| Est. FFR (Gould-Gorlin) | {ffr}{ci_str} |\n"
                f"| QFR Zone | {zone} |\n"
                f"| Haemodynamic Class | {hcls} |\n\n"
                f"**Recommendation:** {rec}\n\n"
                f"_⚠ {note}_"
            )
        else:
            metrics_md.append(
                f"**Detection {i+1}** (conf={conf_score:.2f}): Metrics unavailable."
            )

    # Gradio Gallery accepts list-of-images; pad to _MAX_QCA_BOXES so outputs
    # are always the same arity.
    def _pad(lst):
        while len(lst) < _MAX_QCA_BOXES:
            lst.append(None)
        return lst[:_MAX_QCA_BOXES]

    all_crop   = _pad(all_crop)
    all_clahe  = _pad(all_clahe)
    all_frangi = _pad(all_frangi)
    all_binary = _pad(all_binary)
    all_skel   = _pad(all_skel)
    all_radius = _pad(all_radius)
    all_plot   = _pad(all_plot)

    summary = (
        f"**Model:** {model_key} ({weight_label})  |  "
        f"**Detections:** {len(boxes)}  (showing up to {_MAX_QCA_BOXES})\n\n"
        + "\n\n".join(metrics_md)
    )

    # Pack everything into a single gallery list for the "all steps" view.
    # Order: [crop1, clahe1, frangi1, binary1, skel1, radius1, plot1,
    #         crop2, clahe2, ...] — labelled in captions.
    step_labels = ["Crop", "CLAHE", "Frangi", "Binary Mask",
                   "Skeleton", "Radius Overlay", "Diameter Profile"]
    gallery_items = []
    for i in range(_MAX_QCA_BOXES):
        imgs = [all_crop[i], all_clahe[i], all_frangi[i], all_binary[i],
                all_skel[i], all_radius[i], all_plot[i]]
        for img, lbl in zip(imgs, step_labels):
            if img is not None:
                gallery_items.append((img, f"Det {i+1} — {lbl}"))

    return (
        det_overview,       # detection_out
        all_crop[0],        # crop_out
        all_clahe[0],       # clahe_out
        all_frangi[0],      # frangi_out
        all_binary[0],      # binary_out
        all_skel[0],        # skel_out
        all_radius[0],      # radius_out
        all_plot[0],        # plot_out
        gallery_items,      # all_steps_gallery
        summary,            # metrics_out
    )


# ── UI ────────────────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    model_names = list(MODELS.keys())

    # Shared helper: load a sample image (called from .select, receives only evt)
    def _make_loader(samples_list):
        def _load(evt: gr.SelectData):
            path = samples_list[evt.index]
            from PIL import Image as PILImage
            img = np.array(PILImage.open(path).convert("RGB"))
            return img, path
        return _load

    with gr.Blocks(title="Stenosis Detection & QCA") as demo:
        gr.Markdown("## Stenosis Detection & QCA Analysis")

        # ── Tab 1: Detection ─────────────────────────────────────────────────
        with gr.Tab("Detection"):
            gr.Markdown(
                "Click a sample image (or upload your own), pick a model, hit **Detect**.\n"
                "Green = ground truth · Coloured = model prediction."
            )
            img_path_state = gr.State("")

            with gr.Row():
                with gr.Column(scale=1):
                    model_dd   = gr.Dropdown(choices=model_names, value=model_names[0],
                                             label="Model")
                    conf_slider = gr.Slider(0.10, 0.90, value=0.25, step=0.05,
                                            label="Confidence threshold")
                    run_btn    = gr.Button("Detect", variant="primary", size="lg")

                with gr.Column(scale=3):
                    img_input = gr.Image(label="Input image", type="numpy",
                                         sources=["upload"])
                    with gr.Row():
                        gt_out   = gr.Image(label="Ground Truth", type="numpy")
                        pred_out = gr.Image(label="Prediction",   type="numpy")
                    info_out = gr.Markdown()

            sample_gallery = gr.Gallery(
                value=SAMPLES,
                label="ARCADE test samples — click to load",
                columns=6, height=140, object_fit="contain", allow_preview=False,
            )
            sample_gallery.select(
                fn=_make_loader(SAMPLES),
                outputs=[img_input, img_path_state],
            )
            img_input.upload(fn=lambda _: "", inputs=[img_input],
                             outputs=[img_path_state])
            run_btn.click(
                fn=run_detection,
                inputs=[img_input, model_dd, conf_slider, img_path_state],
                outputs=[gt_out, pred_out, info_out],
            )

        # ── Tab 2: QCA Analysis ──────────────────────────────────────────────
        with gr.Tab("QCA Analysis"):
            gr.Markdown(
                "### Quantitative Coronary Angiography (QCA) Pipeline\n"
                "Runs full research-grade QCA on each detected stenosis:\n"
                "**Crop → CLAHE → Frangi filter → Binary mask → Skeleton → "
                "Tangent-radius profile → % Diameter Stenosis**"
            )

            qca_img_path_state = gr.State("")

            with gr.Row():
                with gr.Column(scale=1):
                    qca_model_dd    = gr.Dropdown(choices=model_names,
                                                  value=model_names[0],
                                                  label="Model")
                    qca_conf_slider = gr.Slider(0.10, 0.90, value=0.25, step=0.05,
                                                label="Confidence threshold")
                    qca_btn         = gr.Button("Detect + Analyse", variant="primary",
                                                size="lg")

                with gr.Column(scale=3):
                    qca_img_input = gr.Image(label="Input image", type="numpy",
                                             sources=["upload"])

            qca_sample_gallery = gr.Gallery(
                value=SAMPLES,
                label="ARCADE test samples — click to load",
                columns=6, height=140, object_fit="contain", allow_preview=False,
            )
            qca_sample_gallery.select(
                fn=_make_loader(SAMPLES),
                outputs=[qca_img_input, qca_img_path_state],
            )
            qca_img_input.upload(fn=lambda _: "", inputs=[qca_img_input],
                                 outputs=[qca_img_path_state])

            # ── Detection overview ─────────────────────────────────────────
            gr.Markdown("#### Detection Result")
            detection_out = gr.Image(label="Detected boxes", type="numpy")

            # ── Step-by-step for Detection 1 ──────────────────────────────
            gr.Markdown("#### QCA Pipeline — Detection 1")
            with gr.Row():
                crop_out   = gr.Image(label="1. Crop (raw)",       type="numpy")
                clahe_out  = gr.Image(label="2. CLAHE",            type="numpy")
                frangi_out = gr.Image(label="3. Frangi Vesselness", type="numpy")
                binary_out = gr.Image(label="4. Binary Mask",      type="numpy")
            with gr.Row():
                skel_out   = gr.Image(label="5. Skeleton Overlay",  type="numpy")
                radius_out = gr.Image(label="6. Radius Overlay",    type="numpy")
                plot_out   = gr.Image(label="7. Diameter Profile",  type="numpy")

            # ── All steps gallery (multi-detection) ───────────────────────
            gr.Markdown("#### All Detections — Step Gallery")
            all_steps_gallery = gr.Gallery(
                label="All steps for all detections",
                columns=7, height=220, object_fit="contain", allow_preview=True,
            )

            # ── Metrics ───────────────────────────────────────────────────
            gr.Markdown("#### QCA Metrics")
            metrics_out = gr.Markdown()

            qca_btn.click(
                fn=run_qca_analysis,
                inputs=[qca_img_input, qca_model_dd, qca_conf_slider],
                outputs=[
                    detection_out,
                    crop_out, clahe_out, frangi_out, binary_out,
                    skel_out, radius_out, plot_out,
                    all_steps_gallery,
                    metrics_out,
                ],
            )

    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
    )
