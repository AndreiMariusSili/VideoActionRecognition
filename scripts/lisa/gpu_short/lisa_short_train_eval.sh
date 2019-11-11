#!/bin/bash

#SBATCH -p gpu_short
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -t 00:05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/out
#SBATCH -o train_and_eval_%j.out

# Arguments
SPEC="${1}"
RESUME="${2}"


echo "========================="
echo "Setting up environmnet..."
echo "========================="
# shellcheck source=/dev/null
source "${HOME}/.bash_profile"
export MT_ROOT="${HOME}/master-thesis"
conda activate mt

# Run program
echo "=================="
echo "Running program..."
echo "=================="
cd "${MT_ROOT}/src" || :
python "-m" "torch.distributed.launch" "--nproc_per_node=4" "main.py" "run_model" "-o" "spec:${SPEC}" "-r" "${RESUME}"
python "-m" "torch.distributed.launch" "--nproc_per_node=4" "main.py" "evaluate_model" "-o" "spec:${SPEC}" "-r" "${RESUME}"


echo "=============="
echo "Cleaning up..."
echo "=============="
mv "train_and_eval@${SLURM_JOB_ID}.out" "train_and_eval@${SPEC}${RESUME}.out"


echo "==============="
echo "Ðone. All good."
echo "==============="