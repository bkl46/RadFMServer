#!/bin/sh

#SBATCH -p gpu               # Partition (queue) name
#SBATCH -c 48                # Number of CPU cores
#SBATCH --gres=gpu:4         # Request 2 GPUs
#SBATCH --mem=128G            # Memory
#SBATCH -t 24:00:00          # Time limit (24 hours)
#SBATCH -J gpu_test          # Job name
#SBATCH -o /home/bkl46/mult/logs/run.out      # Standard output file


source /home/bkl46/sdlevenv/bin/activate

python3 -m pip install flask

nvidia-smi

python server.py
