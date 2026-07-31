#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --nodelist=gnode01
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --job-name=gpu_test
#SBATCH --output=%j_gputest.log

cd ~/scfm-toolkit
bash test_gpu.sh
