#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J PROBLEM_2D
#SBATCH --mail-user=vgulizzi@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 00:01:00

RACE_PATH="RACE/Projects/2021/amrex/Tutorials/dG/Hyperbolic_WavesInSolids/SingleDomain"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$SCRATCH/$RACE_PATH"

module load impi

INPUT="inputs"
EXE="./main2d.intel.haswell.DEBUG.TPROF.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

#run the application:
srun -n 1 -c 1 --cpu_bind=cores $EXE $INPUT
