#!/bin/bash

#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrei.sili1994@gmail.com
#SBATCH -D /home/asili/master-thesis/work/out
#SBATCH -o tar.out

# shellcheck source=/dev/null
source "${MT_SOURCE}/scripts/lisa/common/setup.sh" "normal"

echo "=================================================================================================================="
echo "Tarring folders..."
echo "=================================================================================================================="
for w in "$@"; do
  if [ "${w}" == "src" ]; then
    echo "Tarring src folder..."
    mkdir "-p" "${MT_WORK}/tar"
    tar "--totals" "-C" "${MT_SOURCE}" "-cf" "${MT_WORK}/tar/src.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "hmdb" ]; then
    echo "Tarring hmdb folder..."
    mkdir "-p" "${MT_WORK}/tar/data"
    tar "--totals" "--exclude" "./archive" "-C" "${MT_WORK}/data
    /hmdb" "-cf" "${MT_WORK}/tar/hmdb.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "smth" ]; then
    echo "Tarring smth folder..."
    mkdir "-p" "${MT_WORK}/tar/data"
    tar "--totals" "--exclude" "./archive" "-C" "${MT_WORK}/data/smth" "-cf" "${MT_WORK}/tar/smth.tar" "."
    echo "Done."
  fi

  if [ "${w}" == "runs" ]; then
    echo "Tarring runs folder..."
    mkdir "-p" "${MT_WORK}/tar"
    tar "--totals" "-C" "${MT_WORK}/runs" "-cf" "${MT_WORK}/tar/runs.tar" "."
    echo "Done."
  fi

done

echo "=================================================================================================================="
echo "Done. All good."
echo "=================================================================================================================="
