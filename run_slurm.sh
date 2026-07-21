#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --nodelist=gnode01
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --job-name=scgpt
#SBATCH --output=%j.log

cd ~/scgpt-toolkit
bash run.sh
