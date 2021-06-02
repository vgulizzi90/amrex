#!/bin/bash

export LD_LIBRARY_PATH=/home/vgulizzi/RACE/Projects/2021/amrex/Tutorials/dG/Pardiso/:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1

mpirun -n 1 ./main2d.gnu.DEBUG.TPROF.MPI.ex inputs