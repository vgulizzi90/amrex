all:
	$(MAKE) linalg_test.out

src = ./lapack/ilaenv.cpp ./lapack/dlahr2.cpp ./lapack/dlarfb.cpp ./lapack/dlaexc.cpp ./lapack/dlatrd.cpp \
	./lapack/dgetf2.cpp ./lapack/dgetrf.cpp ./lapack/dgetrs.cpp ./lapack/dtrti2.cpp ./lapack/dtrtri.cpp ./lapack/dgetri.cpp \
	./lapack/dpotf2.cpp ./lapack/dpotrf.cpp ./lapack/dpotrs.cpp ./lapack/dgebal.cpp ./lapack/dgehd2.cpp ./lapack/dgehrd.cpp \
	./lapack/dorg2r.cpp ./lapack/dorgqr.cpp ./lapack/dorghr.cpp ./lapack/dtrexc.cpp ./lapack/dorm2r.cpp ./lapack/dormqr.cpp \
	./lapack/dormhr.cpp ./lapack/dtrevc.cpp ./lapack/dgebak.cpp ./lapack/dlasrt.cpp ./lapack/dsterf.cpp ./lapack/dorg2l.cpp \
	./lapack/dorgql.cpp ./lapack/dorgtr.cpp ./lapack/dsteqr.cpp ./lapack/dsytd2.cpp ./lapack/dsytrd.cpp ./lapack/dsyev.cpp \
	./lapack/dlaqr5.cpp ./lapack/dlaqr2.cpp ./lapack/dlaqr4.cpp ./lapack/dlaqr3.cpp ./lapack/dlaqr0.cpp ./lapack/dhseqr.cpp \
	./lapack/dgeev.cpp

objs = $(patsubst %.cpp, %.o, $(src))

clean:
	rm -rf $(objs)

realclean:
	rm -rf $(objs)
	rm linalg_test.out

%.o: %.cpp
	g++ -O3 -Wall -c $< -o $@

linalg_test.out: AMReX_linalg_test.cpp $(objs)
	g++ -O3 -Wall -std=c++14 -o $@ $^ -L/usr/lib/lapack -L./blas -L./lapack -llapack -lblas

.PHONY: clean realclean