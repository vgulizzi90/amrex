#!/bin/bash

#BSUB -P CSC308
#BSUB -J ICS_periodic_BCS_periodic
#BSUB -W 2:00
#BSUB -nnodes 4

RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_Advection/"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH"

INPUT="inputs"
EXE="./main3d.gnu.MPI.ex"
OUTPUT="hp-Convergence.txt"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 4 -a 4 -c 4 $EXE $INPUT

cp $OUTPUT $HOME_DIR

