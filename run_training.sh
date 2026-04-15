#!/bin/bash
# Executed INSIDE the container by submit.sbatch
# Do not run this directly — use: sbatch submit.sbatch

set -e

export CUDA_VISIBLE_DEVICES=0

# Ultralytics model.train() manages its own single-GPU training internally.
# It reads the RANK env var — anything != -1 triggers DistributedSampler
# which then crashes because init_process_group is never called.
# srun/PMI sets RANK=0, so we must override it.
export RANK=-1
export LOCAL_RANK=-1
export WORLD_SIZE=1

SCRIPTS_DIR=/workspace/stenosis/Stenosis

cd $SCRIPTS_DIR

echo ""
echo "============================================================"
echo "Installing dependencies ..."
echo "============================================================"
pip install --user --quiet opencv-python-headless
pip install --user --quiet ultralytics pyyaml pycocotools

# MMDetection (required for YOLOv3, DINO-DETR, Grounding DINO)
pip install --user --quiet -U openmim
python -m mim install --quiet mmengine
python -m mim install --quiet "mmcv>=2.0.0rc4,<2.2.0"
python -m mim install --quiet mmdet
# Grounding DINO needs the BERT text encoder
pip install --user --quiet transformers

nvidia-smi

# ── MMDetection models (paper baselines) ──────────────────────────────────────

echo ""
echo "============================================================"
echo "[7/9] YOLOv3-DarkNet53  (paper baseline)"
echo "============================================================"
python $SCRIPTS_DIR/train_yolov3.py

echo ""
echo "============================================================"
echo "[8/9] DINO-DETR-R50  (paper baseline)"
echo "============================================================"
python $SCRIPTS_DIR/train_dino_detr.py

echo ""
echo "============================================================"
echo "[9/9] Grounding DINO-R50  (paper baseline)"
echo "============================================================"
python $SCRIPTS_DIR/train_grounding_dino.py

echo ""
echo "============================================================"
echo "MODELS 7, 8, 9 DONE"
echo "============================================================"
