#!/bin/bash

#SBATCH -p short
#SBATCH -n 16
#SBATCH -t 00:05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/out
#SBATCH -o setup@%j.out

# Arguments
SET="${1}"

# Setup
export MT_ROOT="${HOME}/master-thesis"
# shellcheck source=/dev/null
source "${HOME}/.bash_profile"
conda activate mt

# Run
echo "Running program..."
python "${MT_ROOT}/src/main.py" "setup" "-o" "set:${SET}"

# Cleanup
echo "Renaming output file..."
mv "setup@${SLURM_JOB_ID}.out" "setup@${SET}.out"

# Done
echo "Done. All good."
