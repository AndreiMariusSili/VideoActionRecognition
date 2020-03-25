# VideoActionRecognition

This repo holds the code for my master thesis in Artificial Intelligence at the University of Amsterdam. The paper is 
made publicly available [here](https://arxiv.org).

## Abstract

In  this  paper,  we  investigate  3  challenges  in  video  processing:high  computational cost,low data availability, 
and high intra-class variability, proposing ways of mitigating each of them. First, to mitigate computational cost, we 
propose an efficient sequence model, the Time-Aligned ResNet, based on the Time-Aligned DenseNet, that grows linearly 
with  frame  sampling  frequency,  achieving  significant  performance  gains compared to its predecessor.  Second, we 
seek to mitigate the problem of low data availability by designing multi-task models and training routines to extract 
richer information from existing data, although we do not arrive at a formulation that out-performs vanilla 
classification models.   Finally,  we try to mitigate the problem of intra-class variations by proposing a class of 
stochastic models.  While we observe some improvement in generalisation power, these are not substantial enough to 
out-weigh the increase in computational cost.

## Requirements

* Ubuntu 18.04 or similar
* Cuda 10.1
* Python 3.7
* Miniconda 4.8.1

Some other linux libraries might be needed for open-cv, like libjpeg, or libpng, depending on your Linux distribution. 
Install on a per-need basis. 

## Setup

Create a conda environment from the `environment.yml` file.

```shell script
conda install -f env.yml
conda activate mt
```

Set 2 environment variables: 

* the path to the source code
* the path where experiment assets will be stored.

```shell script
export ML_SOURCEDIR="path/to/sourcecode"
exprot ML_WORKDIR="path/to/experiments"
```

Experiments are run on 2 datasets: Human Motion Database, and Something-Something-v2. You need to download the the .tar 
files from the links below and place them under the specified path relative to `ML_WORKDIR`.  

* [Human Motion Database](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) - `./data/hmdb`
* [Something-Something-v2](https://20bn.com/datasets/something-something) - `./data/smth`

## Pre-processing Data

Several pre-processing steps need to be run before training. For the Human Motion Database data set, run the following 
setup all necessary folders, extract videos from .tar files, split data, extract .jpeg frames from videos, etc.

* `python main.py setup --opts=set:hmdb`
* `python main.py prepro_set --opts=set:hmdb,split:1,jpeg:yes`

For Something-Something-v2, run the following to do all of the above but also select a subset of the data.

* `python main.py setup --opts=set:smth`
* `python main.py select_subset --opts=set:smth,num_classes:51`
* `python main.py prepro_set --opts=set:smth,split:1,jpeg:yes`

## Running Experiments

Everything should be set now. You can run each of the experiments via the scripts under `./experiments`. This will run a
training routine, and an evaluation routine at the end.

```shell script
sh ./experiments/experiment_1.sh
sh ./experiments/experiment_2.sh
sh ./experiments/experiment_3.sh
``` 

TensorBoard Logs will be saved to `${MT_WORKDIR}/runs`. You can view them as follows.
```shell script
tensorboard --logdir=${MT_WORKDIR}/runs --bind_all
```

## Visualising model inputs and outputs.

A visualiser tool is provided for each class of models. Check out `visualise.py` for usage examples.
