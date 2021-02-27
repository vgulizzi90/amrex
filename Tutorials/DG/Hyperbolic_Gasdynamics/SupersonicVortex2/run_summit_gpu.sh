#!/bin/bash

#BSUB -P CSC308
#BSUB -J m64x64
#BSUB -W 0:10
#BSUB -nnodes 1

RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_Gasdynamics/SupersonicVortex2"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$MEMBERWORK/csc308/$RACE_PATH"

INPUT="inputs"
EXE="./main2d.gnu.MPI.CUDA.ex"
JSRUN="jsrun --nrs 1 -a 1 -g 1 -c 1"

SMPIARGS+=" --smpiargs="-disable_gpu_hooks -x PAMI_DISABLE_CUDA_HOOK=1""

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

date
$JSRUN $SMPIARGS $EXE $INPUT