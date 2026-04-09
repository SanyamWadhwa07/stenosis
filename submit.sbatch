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

srun --container-image=/home/hpc_apps/nvidia+pytorch+25.04-py3.sqsh \
     --container-mounts=/home/harpreet:/workspace \
     bash /workspace/stenosis/Stenosis/run_training.sh

echo "End Time       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"
