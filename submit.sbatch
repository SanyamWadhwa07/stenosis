#!/bin/bash
#SBATCH --job-name=stenosis_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --nodelist=dgxanode03
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt

echo "------------------------------------------------------------"
echo "Job ID         : $SLURM_JOB_ID"
echo "Job Name       : $SLURM_JOB_NAME"
echo "Node List      : $SLURM_NODELIST"
echo "CPUs/Task      : $SLURM_CPUS_PER_TASK"
echo "Total Tasks    : $SLURM_NTASKS"
echo "Memory         : $SLURM_MEM_PER_NODE MB"
echo "Start Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"

srun --mpi=pmi2 \
     --container-image=/home/hpc_apps/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts=/home/harpreet:/workspace \
     bash -c '
        set -e

        export CUDA_VISIBLE_DEVICES=0

        SCRIPTS_DIR=/workspace/stenosis/Stenosis

        cd $SCRIPTS_DIR

        echo ""
        echo "============================================================"
        echo "Installing dependencies ..."
        echo "============================================================"
        # Install headless opencv FIRST so ultralytics doesn't pull in
        # opencv-python which requires libGL.so.1 (missing in this container)
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
'

echo "End Time       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"
