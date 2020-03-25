#!/bin/bash

#SBATCH -p normal
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/work/out
#SBATCH -o select_subset@%j.out

# Arguments
SET="${1}"
NUM_CLASSES="${2}"

# shellcheck source=/dev/null
source "${MT_SOURCE}/scripts/lisa/common/setup.sh" "normal"

echo "=================================================================================================================="
echo "Selecting data subset..."
echo "=================================================================================================================="
python "${MT_SOURCE}/main.py" "select_subset" "-o" "set:${SET},num_classes:${NUM_CLASSES}"

echo "=================================================================================================================="
echo "Cleaning up..."
echo "=================================================================================================================="
mv "select_subset@${SLURM_JOB_ID}.out" "select_subset@${SET}.out"

echo "=================================================================================================================="
echo "Ðone. All good."
echo "=================================================================================================================="
