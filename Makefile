CPPFLAGS=-std=c++14 -O3

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CLEANFILES=bin/* build/*

default: main dot_microbenchmark gpu-bug

all: main dot_microbenchmark gpu-bug

main: main.cpp
	-$(PETSC_CXXCOMPILE_SINGLE) -c main.cpp -o main.o
	-${CLINKER} main.o ${PETSC_KSP_LIB} -lmpi_cxx -o main

main: gpu-bug.cpp
	-$(PETSC_CXXCOMPILE_SINGLE) -c gpu-bug.cpp -o gpu-bug.o
	-${CLINKER} gpu-bug.o ${PETSC_KSP_LIB} -lmpi_cxx -o gpu-bug

dot_microbenchmark: dot_microbenchmark.cpp
	-$(PETSC_CXXCOMPILE_SINGLE) -c dot_microbenchmark.cpp -o dot_microbenchmark.o
	-${CLINKER} dot_microbenchmark.o ${PETSC_KSP_LIB} -lmpi_cxx -o dot_microbenchmark
