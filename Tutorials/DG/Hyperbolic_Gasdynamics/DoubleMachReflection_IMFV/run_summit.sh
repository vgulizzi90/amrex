#!/bin/bash

#BSUB -P CSC308
#BSUB -J m1024x512_TH49
#BSUB -W 2:00
#BSUB -nnodes 1

RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_Gasdynamics/DoubleMachReflection_IMFV"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH"

INPUT="inputs"
EXE="./main2d.gnu.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 1 -a 32 -c 32 $EXE $INPUT