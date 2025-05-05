# PFC_Pore
Phase Field Crystal Modeling of materials in the presence of a pore


#This Model solves a PFC crystal model descibed in this paper:
https://iopscience.iop.org/article/10.1088/1361-651X/ac3dd2

The code can simply be compiled with nvcc, and is written for CUDA92 (for higher CUDA versions with backwards compatibility):

Compile as:
nvcc -lcufft Cuda_pore.cu -O3 -o PFC

A sample sbatch job submission to run on the HPC (needs GPU) is included as:

