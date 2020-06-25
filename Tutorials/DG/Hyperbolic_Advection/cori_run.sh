#!/bin/bash

#SBATCH --account=mp111
#SBATCH --qos=debug
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=haswell

srun ./main2d.intel.haswell.DEBUG.MPI.ex inputs
