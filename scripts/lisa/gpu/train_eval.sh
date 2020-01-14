#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/work/out
#SBATCH -o train_and_eval@%j.out

# Arguments
DATASET="${1}"
SPLIT="${2}"
CUT="${3}"
FRAMES="${4}"
MODEL="${5}"

# shellcheck source=/dev/null
source "${MT_SOURCE}/scripts/lisa/common/setup.sh" "gpu"
# shellcheck source=/dev/null
source "${HOME}/master-thesis/src/scripts/lisa/common/move_to_scratch.sh" "${DATASET}"

echo "=================================================================================================================="
echo "Running experiment..."
echo "=================================================================================================================="
python "-m" "torch.distributed.launch" "--nnodes=1" "--nproc_per_node=4" "${MT_SOURCE}/main.py" "run_experiment" "--opts" "dataset:${DATASET}${SPLIT},cut:${CUT},frames:${FRAMES},model:${MODEL}"
# shellcheck source=/dev/null
source "${MT_SOURCE}/scripts/lisa/common/move_to_home.sh"

echo "=================================================================================================================="
echo "Evaluating experiment..."
echo "=================================================================================================================="
python "-m" "torch.distributed.launch" "--nnodes=1" "--nproc_per_node=1" "${MT_SOURCE}/main.py" "eval_experiment" "--opts" "dataset:${DATASET}${SPLIT},cut:${CUT},frames:${FRAMES},model:${MODEL}"
# shellcheck source=/dev/null
source "${MT_SOURCE}/scripts/lisa/common/move_to_home.sh"

echo "=================================================================================================================="
echo "Cleaning up..."
echo "=================================================================================================================="
mv "train_and_eval@${SLURM_JOB_ID}.out" "train_and_eval@${DATASET}_${CUT}_${FRAMES}_${MODEL}.out"

echo "=================================================================================================================="
echo "Ðone. All good."
echo "=================================================================================================================="
