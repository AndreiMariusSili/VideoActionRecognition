#!/bin/bash

#SBATCH -p normal
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/work/out
#SBATCH -o prepro_set@%j.out

# Arguments
SET="${1}"
SPLIT="${2}"
JPEG="${3}"

# shellcheck source=/dev/null
source "${MT_SOURCE}/scripts/lisa/common/setup.sh" "normal"

echo "=================================================================================================================="
echo "Preprocessing data..."
echo "=================================================================================================================="
python "${MT_SOURCE}/main.py" "prepro_set" "-o" "set:${SET},split:${SPLIT},jpeg:${JPEG}"

echo "=================================================================================================================="
echo "Cleaning up..."
echo "=================================================================================================================="
mv "prepro_set@${SLURM_JOB_ID}.out" "prepro_set@${SET}_${SPLIT}_${JPEG}.out"

echo "=================================================================================================================="
echo "Ðone. All good."
echo "=================================================================================================================="
