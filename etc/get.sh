#!/bin/sh

#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -o /home/bkl46/mult/test.out
#SBATCH -J dist_test          # Job name


nvidia-smi


python gpulook.py
