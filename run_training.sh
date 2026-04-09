#!/bin/bash
# Executed INSIDE the container by submit.sbatch
# Do not run this directly — use: sbatch submit.sbatch

set -e

export CUDA_VISIBLE_DEVICES=0

# srun sets SLURM_PROCID / PMI_RANK which the PyTorch container maps to
# RANK and WORLD_SIZE. Ultralytics reads RANK — anything != -1 triggers
# DistributedSampler which then crashes because init_process_group was
# never called. Force single-process mode explicitly.
export RANK=-1
export LOCAL_RANK=-1
export WORLD_SIZE=1

SCRIPTS_DIR=/workspace/stenosis/Stenosis

cd $SCRIPTS_DIR

echo ""
echo "============================================================"
echo "Installing dependencies ..."
echo "============================================================"
# Headless opencv must come first — the container has no libGL.so.1
pip install --user --quiet opencv-python-headless
pip install --user --quiet ultralytics pyyaml

echo ""
echo "============================================================"
echo "[1/6] YOLO11m"
echo "============================================================"
python $SCRIPTS_DIR/train_yolo11.py

echo ""
echo "============================================================"
echo "[2/6] YOLOv8m"
echo "============================================================"
python $SCRIPTS_DIR/train_yolov8.py

echo ""
echo "============================================================"
echo "[3/6] YOLOv9m"
echo "============================================================"
python $SCRIPTS_DIR/train_yolov9.py

echo ""
echo "============================================================"
echo "[4/6] YOLOv10m"
echo "============================================================"
python $SCRIPTS_DIR/train_yolov10.py

echo ""
echo "============================================================"
echo "[5/6] RT-DETR-R18"
echo "============================================================"
python $SCRIPTS_DIR/train_rtdetr_r18.py

echo ""
echo "============================================================"
echo "[6/6] RT-DETR-R50"
echo "============================================================"
python $SCRIPTS_DIR/train_rtdetr_r50.py

echo ""
echo "============================================================"
echo "ALL MODELS DONE"
echo "============================================================"
