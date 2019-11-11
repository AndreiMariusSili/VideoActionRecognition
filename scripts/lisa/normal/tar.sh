#!/bin/bash

#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 12:00:00
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

  if [ "${w}" == "dummy/smth" ]; then
    echo "Tarring dummy smth folder..."
    tar "--totals" "-C" "${MT_ROOT}/data/dummy/smth" "-cf" "${HOME}/tar/data/dummy/smth.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "dummy/hmdb" ]; then
    echo "Tarring dummy hmdb folder..."
    tar "--totals" "-C" "${MT_ROOT}/data/dummy/hmdb" "-cf" "${HOME}/tar/data/dummy/hmdb.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "full/smth" ]; then
    echo "Tarring full smth folder..."
    tar "--totals" "-C" "${MT_ROOT}/data/full/smth" "-cf" "${HOME}/tar/data/full/smth.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "full/hmdb" ]; then
    echo "Tarring full hmdb folder..."
    tar "--totals" "-C" "${MT_ROOT}/data/full/hmdb" "-cf" "${HOME}/tar/data/full/hmdb.tar" "."
    echo "Done."
  fi

done

# Done
echo "Done. All good."