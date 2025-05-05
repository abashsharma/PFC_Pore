#!/bin/bash
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --partition=gpuq
#SBATCH --time=10:00:00
#SBATCH --account=PHYS
#SBATCH --output=PFC.out
#SBATCH --error=PFC.err
##SBATCH --mem=[MEMORY_NEEDED_FOR_JOB]
##SBATCH --mail-user=@memphis.edu

##SBATCH --job-name=PFC


#################################################
# [SOMETHING]                                   #
#-----------------------------------------------#
# Please replace anything in [] brackes with    #
# what you want.                                #
# Example:                                      #
# #SBATCH --partition=[PARTITION/QUEUE]         #
# Becomes:                                      #
# #SBATCH --partition=computeq                  #
#################################################



#################################################
# --partition=[PARTITION/QUEUE]                 #
#-----------------------------------------------#
# For this script we are assuming:              #
#   gpuq: 40 cores, 192 GB mem, 2 v100 GPUs     #
#################################################
# --ntasks=[NTASKS] and --nodes=[NNODES]        #
#-----------------------------------------------#
# Number of threads and nodes needed per job.   #
# Note that there are only 2 GPUs available per #
# node. Multiple programs and users may use each#
# card at one time, but you may run into        #
# performance issues. To mitigate this, use more#
# --ntasks to reduce overlap.                   #
#################################################

cd $SLURM_SUBMIT_DIR
 
#################################################
# modules                                       #
#-----------------------------------------------#
# Any modules you need can be found with        #
# 'module avail'. If you compile something with #
# a particular compiler using a module, you     #
# probably want to call that module here. You   #
# might need one of the cuda modules.           #
#################################################
module load cuda92/toolkit
module load cuda92/fft

 
#################################################
# Run your executable here                      #
#################################################
#[EXECUTABLE] [OPTIONS]
#nvcc -lcufft Binary.cu -O3 -o pfc
nvcc -lcufft ${SLURM_JOB_NAME}.cu -O3 -o $SLURM_JOB_NAME
./$SLURM_JOB_NAME

