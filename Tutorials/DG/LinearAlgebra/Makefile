all:
	$(MAKE) linalg_tests.ex

src = 

objs = $(patsubst %.cpp, %.o, $(src))

clean:
	rm -rf $(objs)

realclean:
	rm -rf $(objs)
	rm linalg_tests.ex

%.o: %.cpp
	g++ -O3 -Wall -c $< -o $@

linalg_tests.ex: linalg_tests.cpp $(objs)
	g++ -O3 -Wall -std=c++14 -o $@ $^ -L/usr/lib/lapack -L./blas -L./lapack -llapack -lblas

.PHONY: clean realclean