echo "=================================================================================================================="
echo "Moving runs to home..."
echo "=================================================================================================================="
rsync "-zvr" "${MT_WORK}/runs/" "${HOME}/master-thesis/work/runs"