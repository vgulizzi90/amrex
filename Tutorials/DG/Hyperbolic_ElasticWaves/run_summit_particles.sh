#!/bin/bash

#BSUB -P CSC308
#BSUB -J Particles
#BSUB -W 0:05
#BSUB -nnodes 1

p="p3"
N_MPI_RANKS="16"
RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_ElasticWaves"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH/n_MPI_ranks_$N_MPI_RANKS/$p"

INPUT="inputs_particles"
EXE="./main2d.gnu.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 1 -a 16 -c 16 $EXE $INPUT



