#!/bin/bash

#BSUB -P CSC308
#BSUB -J Problem_one_layer
#BSUB -W 0:30
#BSUB -nnodes 1

p="p3"
N_MPI_RANKS="8"
RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_ElasticWaves"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH/n_MPI_ranks_$N_MPI_RANKS/$p"

INPUT="inputs_one_layer"
EXE="./main2d.gnu.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 1 -a 8 -c 8 $EXE $INPUT



