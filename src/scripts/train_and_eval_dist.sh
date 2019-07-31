#!/usr/bin/env bash

SPEC=$1
NUM_GPU=$2
RESUME=$3

cd ../ &&
"${MT_ENV}"/bin/python -m torch.distributed.launch --nproc_per_node="${NUM_GPU}" main.py run_model -o spec:"${SPEC}" -r "${RESUME}" &&
"${MT_ENV}"/bin/python -m torch.distributed.launch --nproc_per_node="${NUM_GPU}" main.py evaluate_model -o spec:"${SPEC}" -r "${RESUME}" &&
gsutil -m rsync -r -x ".*__pycache__.*" /home/Play/master-thesis/runs/ gs://mt-buffer/runs/
