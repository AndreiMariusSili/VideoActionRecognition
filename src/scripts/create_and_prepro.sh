#!/usr/bin/env bash

SET=$1
SAMPLE=$2

cd ../ &&
"${MT_ENV}"/bin/python main.py create_dummy_set -o set:${SET},sample:"${SAMPLE}" &&
"${MT_ENV}"/bin/python main.py prepro_set -o set:"${SET}"