#!/bin/bash

#BSUB -P CSC308
#BSUB -J ICS_periodic_BCS_periodic
#BSUB -W 1:00
#BSUB -nnodes 1

RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_ElasticWaves/"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH"

INPUT="inputs"
EXE="./main2d.gnu.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 1 -a 4 -c 4 $EXE $INPUT



