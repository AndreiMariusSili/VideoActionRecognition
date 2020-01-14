#!/bin/bash

#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 00:05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/work/out
#SBATCH -o tar.out

# shellcheck source=/dev/null
source "${MT_SOURCE}/scripts/lisa/common/setup.sh" "short"

echo "=================================================================================================================="
echo "Tarring folders..."
echo "=================================================================================================================="
for w in "$@"; do
  if [ "${w}" == "src" ]; then
    echo "Tarring src folder..."
    tar "--totals" "-C" "${MT_SOURCE}" "-cf" "${MT_WORK}tar/src.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "runs" ]; then
    echo "Tarring runs folder..."
    tar "--totals" "-C" "${MT_WORK}/runs" "-cf" "${MT_WORK}/tar/runs.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "smth" ]; then
    echo "Tarring smth folder..."
    tar "--totals" "--exclude" "./archive" "-C" "${MT_WORK}/data/smth" "-cf" "${MT_WORK}/tar/smth.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "hmdb" ]; then
    echo "Tarring hmdb folder..."
    tar "--totals" "--exclude" "./archive" "-C" "${MT_WORK}/data/hmdb" "-cf" "${MT_WORK}/tar/hmdb.tar" "."
    echo "Done."
  fi

done

echo "=================================================================================================================="
echo "Ðone. All good."
echo "=================================================================================================================="
