#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/work/out
#SBATCH -o eval@%j.out

# Arguments
DATASET="${1}"
SPLIT="${2}"
CUT="${3}"
FRAMES="${4}"
MODEL="${5}"

# shellcheck source=/dev/null
source "${MT_SOURCE}/scripts/lisa/common/setup.sh" "gpu_short"

echo "=================================================================================================================="
echo "Evaluating experiment..."
echo "=================================================================================================================="
#python "-m" "torch.distributed.launch" "--nnodes=1" "--nproc_per_node=1" "${MT_SOURCE}/main.py" "eval_experiment" "--opts" "dataset:${DATASET}${SPLIT},cut:${CUT},locs:${FRAMES},model:${MODEL}"
python "${MT_SOURCE}/main.py" "eval_experiment" "--opts" "dataset:${DATASET}${SPLIT},cut:${CUT},frames:${FRAMES},model:${MODEL}"

echo "=================================================================================================================="
echo "Cleaning up..."
echo "=================================================================================================================="
mv "eval@${SLURM_JOB_ID}.out" "eval@${DATASET}_${CUT}_${FRAMES}_${MODEL}.out"

echo "=================================================================================================================="
echo "Ðone. All good."
echo "=================================================================================================================="
