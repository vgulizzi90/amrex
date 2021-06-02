CXX = g++

CFLAGS = -v -O0 -g -Wall -ansi -pedantic

PARDISO_DIR = ../
MKL_DIR = /home/vgulizzi/intel/compilers_and_libraries/linux/mkl/lib/intel64

PARDISO_LIB = -lpardiso600-GNU800-X86-64
MKL_LIBS = -lblas -llapack
ADDITIONAL_LIBS = -lgfortran -fopenmp -lpthread -lm

execs=pardiso_sym.ex

all:
	$(MAKE) $(execs)

$(execs): main.cpp
	$(CXX) $(CFLAGS) -o $@ $^ -L$(PARDISO_DIR) $(PARDISO_LIB) -L$(MKL_LIBS) $(MKL_LIBS) $(ADDITIONAL_LIBS)

clean:
	rm -rf $(objs) $(execs)

.PHONY: clean