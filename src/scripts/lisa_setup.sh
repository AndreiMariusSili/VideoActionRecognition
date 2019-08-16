#!/bin/bash

#SBATCH -p normal
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=andrei.sili1994@gmail.com

# Source profile
source "${HOME}/.bash_profile"

# Activate environment
conda activate mt

# Set environment variables.
echo "Setting environment variables..."
export MT_ROOT="${HOME}/master-thesis"

# Run program
echo "Running program..."
python "${MT_ROOT}/src/main.py" "setup" "-o" "set:smth"