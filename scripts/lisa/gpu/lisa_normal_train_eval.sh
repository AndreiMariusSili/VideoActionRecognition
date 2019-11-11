#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/out
#SBATCH -o train_and_eval@%j.out

# Arguments
SPEC="${1}"
RESUME="${2}"


echo "========================="
echo "Setting up environmnet..."
echo "========================="
export MT_ROOT="${TMPDIR}/asili/master-thesis"
# shellcheck source=/dev/null
source "${HOME}/.bash_profile"
conda activate mt
mkdir -p "${TMPDIR}/asili/master-thesis/tar"
mkdir -p "${TMPDIR}/asili/master-thesis/src"
mkdir -p "${TMPDIR}/asili/master-thesis/out"
mkdir -p "${TMPDIR}/asili/master-thesis/data/dummy"
mkdir -p "${TMPDIR}/asili/master-thesis/data/full"
mkdir -p "${TMPDIR}/asili/master-thesis/runs"


echo "========================"
echo "Moving src to scratch..."
echo "========================"
tar "--totals" "-cf" "${HOME}/tar/src.tar" "-C" "${HOME}/master-thesis/src" "."
cp "${HOME}/tar/src.tar" "${TMPDIR}/asili/tar/src.tar"
tar "--totals" "-xf" "${TMPDIR}/asili/tar/src.tar" "-C" "${TMPDIR}/asili/master-thesis/src"


echo "========================="
echo "Moving data to scratch..."
echo "========================="
cp "${HOME}/tar/dummy.tar" "${TMPDIR}/asili/tar/dummy.tar"
tar "--totals" "-xf" "${TMPDIR}/asili/tar/dummy.tar" "-C" "${TMPDIR}/asili/master-thesis/data/dummy"


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


echo "======================"
echo "Moving runs to home..."
echo "======================"
rsync "-zvr" "${TMPDIR}/asili/master-thesis/runs/" "${HOME}/master-thesis/runs"
rsync "-zvr" "${TMPDIR}/asili/master-thesis/out/" "${HOME}/master-thesis/out"


echo "==============="
echo "Ðone. All good."
echo "==============="