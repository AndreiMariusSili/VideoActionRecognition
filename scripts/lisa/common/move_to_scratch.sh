DATASET="${1}"

echo "=================================================================================================================="
echo "Moving src to scratch..."
echo "=================================================================================================================="
tar "--totals" "-cf" "${HOME}/master-thesis/work/tar/src.tar" "-C" "${HOME}/master-thesis/src" "."
cp "${HOME}/master-thesis/work/tar/src.tar" "${MT_WORK}/tar/src.tar"
tar "--totals" "-xf" "${MT_WORK}/tar/src.tar" "-C" "${MT_SOURCE}"

echo "=================================================================================================================="
echo "Moving data to scratch..."
echo "=================================================================================================================="
cp "${HOME}/master-thesis/work/tar/${DATASET}.tar" "${MT_WORK}/tar/${DATASET}.tar"
tar "--totals" "-xf" "${MT_WORK}/tar/${DATASET}.tar" "-C" "${MT_WORK}/data/${DATASET}"
