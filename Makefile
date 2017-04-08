all:
	mpic++ -std=c++14 -O3 -Wall -o Binomial-three-test main.cpp

test: all
	mpiexec -n 7 ./Binomial-three-test
	mpiexec -n 20 ./Binomial-three-test

clear:
	rm Binomial-three-test



