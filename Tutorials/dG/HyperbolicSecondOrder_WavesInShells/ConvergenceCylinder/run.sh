#!/bin/bash

export LD_LIBRARY_PATH=/home/vgulizzi/RACE/Projects/2021/amrex/Tutorials/dG/Pardiso/:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=8

mpirun -n 1 ./main2d.gnu.TPROF.MPI.OMP.ex inputs