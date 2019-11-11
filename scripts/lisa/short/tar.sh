#!/bin/bash

#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 00:05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/out
#SBATCH -o tar.out

# Setup
export MT_ROOT="${HOME}/master-thesis"
# shellcheck source=/dev/null
source "${HOME}/.bash_profile"
conda activate mt

for w in "$@"; do
  if [ "${w}" == "src" ]; then
    echo "Tarring src folder..."
    tar "--totals" "-C" "${MT_ROOT}/src" "-cf" "${HOME}/tar/src.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "runs" ]; then
    echo "Tarring runs folder..."
    tar "--totals" "-C" "${MT_ROOT}/runs" "-cf" "${HOME}/tar/runs.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "dummy" ]; then
    echo "Tarring dummy folder..."
    tar "--totals" "-C" "${MT_ROOT}/data/dummy" "-cf" "${HOME}/tar/dummy.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "full" ]; then
    echo "Tarring full folder..."
    tar "--totals" "-C" "${MT_ROOT}/data/full" "-cf" "${HOME}/tar/full.tar" "."
    echo "Done."
  fi
done

# Done
echo "Done. All good."
