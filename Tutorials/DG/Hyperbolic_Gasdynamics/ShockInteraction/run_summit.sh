#!/bin/bash

#BSUB -P CSC308
#BSUB -J m1024x1024
#BSUB -W 12:00
#BSUB -nnodes 92

RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_Gasdynamics/ShockInteraction"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH"

INPUT="inputs_two_levels"
EXE="./main2d.gnu.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 92 -a 42 -c 42 $EXE $INPUT