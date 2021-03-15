#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J NACA0012_supersonic
#SBATCH --mail-user=vgulizzi@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00

RACE_PATH="RACE/Projects/2020/amrex/Tutorials/DG/Hyperbolic_Gasdynamics/BodyInFreeStream"
HOME_DIR="$HOME/$RACE_PATH"
WORK_DIR="$SCRATCH/$RACE_PATH"

module load impi

INPUT="inputs_NACA0012"
EXE="./main2d.intel.haswell.MPI.ex"

cp $INPUT $WORK_DIR
cp $EXE $WORK_DIR
cd $WORK_DIR

#run the application:
srun -n 32 -c 2 --cpu_bind=cores $EXE $INPUT