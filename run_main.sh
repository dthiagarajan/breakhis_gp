#!/bin/bash
#SBATCH -J pbe_delta    # Job name
#SBATCH -o pbe_delta.%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e pbe_delta.%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 4   # Total number of CPU cores requrested
#SBATCH --mem=100000    # CPU Memory pool for all cores
#SBATCH --partition=default_gpu --gres=gpu:4 --nodelist=tripods-compute01


if [ ! -d /scratch/datasets/BreaKHis_v1/ ]; then
        cp -R /share/nikola/export/dt372/BreaKHis_v1/ /scratch/datasets
fi

if [ ! -d /scratch/datasets/models/ ]; then
        mkdir /scratch/datasets/models
fi

if [ ! -d /scratch/datasets/models/BreaKHis_v1/ ]; then
        mkdir /scratch/datasets/models/BreaKHis_v1
fi

python3 main.py --base_dir /scratch/datasets/ 
