# PATH TO AMREX #######################################################
AMREX_HOME ?= ../../../../
#######################################################################

COMP = gnu
CXXSTD = c++17
LDFLAGS += -L/home/vgulizzi/intel/compilers_and_libraries/linux/mkl/lib/intel64
LIBRARIES += -lblas -llapack

DEBUG = TRUE
TINY_PROFILE = TRUE
USE_MPI = TRUE
USE_CUDA = FALSE
USE_OMP = TRUE

# LINAER SOLVERS ======================================================
USE_PARDISO = TRUE
LDFLAGS += -L../../Pardiso
LIBRARIES += -lpardiso600-GNU800-X86-64 -lgfortran -fopenmp -lpthread -lm

USE_HYPRE = FALSE
#HYPRE_DIR = ../../../../../../../Extern/hypre/src/hypre
# =====================================================================

DIM = 2

include ./Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package

#include $(AMREX_HOME)/Src/LinearSolvers/MLMG/Make.package
#include $(AMREX_HOME)/Src/Extern/HYPRE/Make.package

include $(AMREX_HOME)/Src/dG/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
