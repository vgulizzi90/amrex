#!/bin/bash

#BSUB -P CSC308
#BSUB -J PROBLEM_THREE_LEVELS
#BSUB -W 2:00
#BSUB -nnodes 1

RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_Gasdynamics/ShockInteraction"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH"

INPUT="inputs_three_levels"
EXE="./main3d.gnu.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 1 -a 42 -c 42 $EXE $INPUT