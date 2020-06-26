#!/bin/bash

#BSUB -P CSC308
#BSUB -J ICS_periodic_BCS_periodic
#BSUB -W 0:10
#BSUB -nnodes 1

date

jsrun -r 2 -a 1 -g 1 -c 21 ./main2d.gnu.MPI.ex inputs
