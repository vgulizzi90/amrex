# PATH TO AMREX #######################################################
AMREX_HOME ?= ../../../../
#######################################################################

COMP = gnu
CXXSTD = c++17

DEBUG = TRUE
USE_MPI = TRUE
USE_CUDA = FALSE

DIM = 2

include ./Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package

include $(AMREX_HOME)/Src/DG/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
