# PATH TO AMREX #######################################################
machine := naphta

ifeq ($(machine),naphta)
AMREX_HOME ?= /home/vgulizzi/RACE/Projects/2019/amrex
endif
ifeq ($(machine),cori)
AMREX_HOME ?= /global/homes/v/vgulizzi/RACE/Projects/2019/amrex
endif
#######################################################################

COMP = gnu
DEBUG = TRUE
USE_MPI = TRUE
USE_CUDA = FALSE

DIM = 2

include ./Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package

include $(AMREX_HOME)/Src/DG/Make.package
include $(AMREX_HOME)/Src/DG/Algoim/Make.package
include $(AMREX_HOME)/Src/DG/LinAlg/blas/Make.package
include $(AMREX_HOME)/Src/DG/LinAlg/lapack/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
