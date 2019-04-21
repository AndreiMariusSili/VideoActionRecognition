#!/usr/bin/env bash

JOB=$1
SPEC=$2
NUM_GPU=$3

cd ../ &&
${MT_ENV}/bin/python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} main.py ${JOB} -o spec:${SPEC}