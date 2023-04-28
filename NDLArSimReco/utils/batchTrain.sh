#!/bin/bash

#SBATCH --partition=ml
#
#SBATCH --job-name=LNDS-train
#SBATCH --output=logs/output-%j.txt
#SBATCH --error=logs/output-%j.txt
#
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=20g
#SBATCH --gpus=a100:1
#
#SBATCH --time=24:00:00

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/larcv2_ub20.04-cuda11.3-cudnn8-pytorch1.10.0-larndsim-2022-11-03.sif
ROOT_PATH=/sdf/home/d/dougl215/studies/NDLArSimReco

MANIFEST=$1

COMMAND="/usr/bin/python3 $ROOT_PATH/NDLArSimReco/train.py -m $MANIFEST"
singularity exec --nv -B /sdf,/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} ${COMMAND}
