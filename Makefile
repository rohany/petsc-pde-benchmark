CPPFLAGS=-std=c++14 -O3

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CLEANFILES=bin/* build/*

default: main

all: main

main: main.cpp
	-$(PETSC_CXXCOMPILE_SINGLE) -c main.cpp -o main.o
	-${CLINKER} main.o ${PETSC_KSP_LIB} -lmpi_cxx -o main
