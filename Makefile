CPPFLAGS=-std=c++14 -O3

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CLEANFILES=bin/* build/*

default: main dot_microbenchmark

all: main dot_microbenchmark

main: main.cpp
	-$(PETSC_CXXCOMPILE_SINGLE) -c main.cpp -o main.o
	-${CLINKER} main.o ${PETSC_KSP_LIB} -o main

dot_microbenchmark: dot_microbenchmark.cpp
	-$(PETSC_CXXCOMPILE_SINGLE) -c dot_microbenchmark.cpp -o dot_microbenchmark.o
	-${CLINKER} dot_microbenchmark.o ${PETSC_KSP_LIB} -o dot_microbenchmark
