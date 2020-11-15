#!/bin/bash

#BSUB -P CSC308
#BSUB -J m2048x1024_TH49
#BSUB -W 2:00
#BSUB -nnodes 4

RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_Gasdynamics/DoubleMachReflection_IMFV"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH"

INPUT="inputs"
EXE="./main2d.gnu.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 4 -a 32 -c 32 $EXE $INPUT