#!/bin/bash

#SBATCH -p normal
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/out
#SBATCH -o prepro_set@%j.out

# Arguments
SET="${1}"
SPLIT="${2}"

# Setup
export MT_ROOT="${HOME}/master-thesis"
# shellcheck source=/dev/null
source "${HOME}/.bash_profile"
conda activate mt

# Run
echo "Running program..."
python "${MT_ROOT}/src/main.py" "prepro_set" "-o" "set:${SET},split:${SPLIT},jpeg:1"

# Cleanup
echo "Renaming output file..."
mv "prepro_set@${SLURM_JOB_ID}.out" "prepro_set@${SET}_${SPLIT}.out"


# Done
echo "Done. All good."