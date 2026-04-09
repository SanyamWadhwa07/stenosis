"""
Stenosis Detection — Gradio Inference App
==========================================
Select a model, upload an image (or use webcam), get annotated detections.

Trained weights are loaded from outputs/<model>/weights/best.pt if available,
otherwise falls back to the Ultralytics pretrained checkpoint.
"""

import pathlib
import time
from functools import lru_cache

import gradio as gr
import numpy as np
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE    = pathlib.Path(__file__).parent.resolve()
OUTPUTS  = _HERE / "outputs"

# ─── Model registry ───────────────────────────────────────────────────────────
MODELS = {
    "YOLO11m":    {"dir": "yolo11m",   "pretrained": "yolo11m.pt"},
    "YOLOv8m":    {"dir": "yolov8m",   "pretrained": "yolov8m.pt"},
    "YOLOv9m":    {"dir": "yolov9m",   "pretrained": "yolov9m.pt"},
    "YOLOv10m":   {"dir": "yolov10m",  "pretrained": "yolov10m.pt"},
    "RT-DETR-R18":{"dir": "rtdetr-r18","pretrained": "rtdetr-r18.pt"},
    "RT-DETR-R50":{"dir": "rtdetr-r50","pretrained": "rtdetr-r50.pt"},
}


def _weight_path(model_key: str) -> tuple[str, str]:
    """Return (weight_path, label) — prefers fine-tuned best.pt."""
    cfg = MODELS[model_key]
    trained = OUTPUTS / cfg["dir"] / "weights" / "best.pt"
    if trained.exists():
        return str(trained), "fine-tuned"
    return cfg["pretrained"], "pretrained"


@lru_cache(maxsize=6)
def _load_model(model_key: str, weight_path: str):
    """Cache loaded models so switching is instant on repeated calls."""
    from ultralytics import YOLO
    return YOLO(weight_path)


def _model_status_row() -> list[dict]:
    """Build a table showing which models have trained weights ready."""
    rows = []
    for name, cfg in MODELS.items():
        trained = OUTPUTS / cfg["dir"] / "weights" / "best.pt"
        rows.append({
            "Model": name,
            "Status": "✅ fine-tuned" if trained.exists() else "⚪ pretrained only",
            "Weight": str(trained) if trained.exists() else cfg["pretrained"],
        })
    return rows


# ─── Inference ────────────────────────────────────────────────────────────────
def run_detection(image, model_key: str, conf: float, iou: float):
    """
    Parameters
    ----------
    image      : np.ndarray (H, W, 3) RGB — from Gradio Image component
    model_key  : one of MODELS keys
    conf       : confidence threshold
    iou        : NMS IoU threshold

    Returns
    -------
    annotated  : np.ndarray  — image with drawn boxes
    info_md    : str         — markdown summary
    """
    if image is None:
        return None, "⚠️ Please upload or capture an image first."

    t0 = time.perf_counter()

    weight_path, weight_label = _weight_path(model_key)
    model = _load_model(model_key, weight_path)

    results = model(
        image,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    elapsed = (time.perf_counter() - t0) * 1000  # ms

    annotated = results.plot()  # BGR numpy array
    annotated_rgb = annotated[:, :, ::-1]  # → RGB for Gradio

    n = len(results.boxes)
    confs = results.boxes.conf.cpu().numpy() if n > 0 else []

    det_lines = ""
    for i, (box, c) in enumerate(zip(results.boxes.xyxy.cpu().numpy(), confs), 1):
        x1, y1, x2, y2 = box.astype(int)
        det_lines += f"- Detection {i}: conf={c:.3f}  box=[{x1},{y1},{x2},{y2}]\n"

    info_md = f"""
### Results — {model_key}
| Field | Value |
|-------|-------|
| Weights | {weight_label} |
| Detections | **{n}** |
| Inference | {elapsed:.1f} ms |
| Confidence threshold | {conf} |
| IoU threshold | {iou} |

{det_lines if det_lines else "_No stenosis detected above threshold._"}
"""
    return annotated_rgb, info_md


# ─── UI ───────────────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    model_names = list(MODELS.keys())

    with gr.Blocks(
        title="Stenosis Detection",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
# 🫀 Stenosis Detection
**Multi-model object detection benchmark**
Select a model, upload an angiography image (or use your webcam), and run inference.
Fine-tuned weights are loaded automatically when available.
"""
        )

        with gr.Row():
            # ── Left column ──────────────────────────────────────────────────
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=model_names,
                    value=model_names[0],
                    label="Model",
                    info="Fine-tuned weights used when available",
                )
                conf_slider = gr.Slider(
                    minimum=0.01, maximum=0.99, value=0.25, step=0.01,
                    label="Confidence threshold",
                )
                iou_slider = gr.Slider(
                    minimum=0.01, maximum=0.99, value=0.45, step=0.01,
                    label="NMS IoU threshold",
                )
                run_btn = gr.Button("Run Detection", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### Model weight status")
                status_table = gr.Dataframe(
                    value=_model_status_row(),
                    headers=["Model", "Status", "Weight"],
                    interactive=False,
                    wrap=True,
                )

            # ── Right column ─────────────────────────────────────────────────
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Upload image"):
                        img_input = gr.Image(
                            label="Input image",
                            type="numpy",
                            sources=["upload"],
                        )
                    with gr.Tab("Webcam (real-time)"):
                        webcam_input = gr.Image(
                            label="Webcam",
                            type="numpy",
                            sources=["webcam"],
                            streaming=True,
                        )

                img_output = gr.Image(label="Detections", type="numpy")
                info_out   = gr.Markdown()

        # ── Wiring ───────────────────────────────────────────────────────────
        # Upload tab — manual trigger
        run_btn.click(
            fn=run_detection,
            inputs=[img_input, model_dd, conf_slider, iou_slider],
            outputs=[img_output, info_out],
        )

        # Webcam tab — live streaming (fires on every frame)
        webcam_input.stream(
            fn=run_detection,
            inputs=[webcam_input, model_dd, conf_slider, iou_slider],
            outputs=[img_output, info_out],
        )

        gr.Markdown(
            """
---
**Dataset:** ARCADE Stenosis (1 class — stenosis) &nbsp;|&nbsp;
**Models:** YOLO11m · YOLOv8m · YOLOv9m · YOLOv10m · RT-DETR-R18 · RT-DETR-R50
"""
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
