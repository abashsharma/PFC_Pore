# Phase Field Crystal Modeling
Phase Field Crystal (PFC) modeling is a computational method used to study the dynamics of materials at the atomic scale. It combines the flexibility of phase field models, which are commonly used to simulate macroscopic processes like solidification, with the ability to describe atomic structures and dynamics. PFC models represent materials as a continuous field that evolves over time, allowing for the simulation of phenomena like crystallization, defect formation, and phase transitions. This approach is particularly useful for understanding the formation of complex microstructures and is applied in areas such as materials science, solid-state physics, and nanotechnology.

# PFC with a Pore
The present code solves an effective model to perform Phase Field Crystal Modeling of materials in the presence of a nanoscale pore. This Model solves a PFC model descibed in this paper:\
https://iopscience.iop.org/article/10.1088/1361-651X/ac3dd2

# Usage
The code can simply be compiled with nvcc, and is written for CUDA92 (for higher CUDA versions with backwards compatibility):
**Compile as:**
nvcc -lcufft Cuda_pore.cu -O3 -o PFC

A sample sbatch job submission to run on the HPC (needs GPU) is included as 'slurmSubmitGPU.sh'

**Note**
Please reach out to me if you need the value of the parameters for the model.\
(i.e for this function in Cuda_pore.cu):\
load_parameters(N, psi0, r, dn, dt, chi, phi0, tnuc, Rmax, nuc, seeds, num_seeds, r0, r1,xi,yi);



**Bonus:**
heatMap_scales.py is a python script that lets visualize the heat map of output coardinates.

