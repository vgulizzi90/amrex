#!/bin/bash

#BSUB -P CSC308
#BSUB -J m512x512x512
#BSUB -W 12:00
#BSUB -nnodes 98

RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_Gasdynamics/SodsTube"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH"

INPUT="inputs"
EXE="./main3d.gnu.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
jsrun --nrs 98 -a 42 -c 42 $EXE $INPUT