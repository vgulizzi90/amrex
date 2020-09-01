#!/bin/bash

#BSUB -P CSC308
#BSUB -J ICS_periodic_BCS_periodic
#BSUB -W 6:00
#BSUB -nnodes 64

p="p1"
N_MPI_RANKS="64"
RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_ElasticWaves"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH/n_MPI_ranks_$N_MPI_RANKS/$p"

INPUT="inputs"
EXE="./main3d.gnu.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 64 -a 1 -c 1 $EXE $INPUT



