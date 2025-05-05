#!/bin/bash
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --time=24:00:00
#SBATCH --output=PFC.out
#SBATCH --error=PFC.err
##SBATCH --mem=[MEMORY_NEEDED_FOR_JOB]
##SBATCH --mail-user@domain.edu

SBATCH --job-name=PFC

cd $SLURM_SUBMIT_DIR
 
module load cuda92/toolkit
module load cuda92/fft

nvcc -lcufft ${SLURM_JOB_NAME}.cu -O3 -o $SLURM_JOB_NAME
./$SLURM_JOB_NAME

