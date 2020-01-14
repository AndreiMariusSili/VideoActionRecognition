PARTITION="${1}"

echo "=================================================================================================================="
echo "Setting up environmnet..."
echo "=================================================================================================================="

set -e

if [ "${PARTITION}" = "short" ] || [ "${PARTITION}" = "gpu_short" ] || [ "${PARTITION}" = "normal" ]; then
  export MT_SOURCE="${HOME}/master-thesis/src"
  export MT_WORK="${HOME}/master-thesis/work"
elif [ "${PARTITION}" = "gpu" ]; then
  export MT_SOURCE="${TMPDIR}/asili/master-thesis/src"
  export MT_WORK="${TMPDIR}/asili/master-thesis/work"
  mkdir -p "${TMPDIR}/asili/master-thesis/src"
  mkdir -p "${TMPDIR}/asili/master-thesis/work/tar"
  mkdir -p "${TMPDIR}/asili/master-thesis/work/runs"
  mkdir -p "${TMPDIR}/asili/master-thesis/work/data/hmdb"
  mkdir -p "${TMPDIR}/asili/master-thesis/work/data/smth"
else
  echo "Unknown partition." 1>&2
  exit 1
fi
