# PETSc PDE CG Solver Benchmark

This repository contains source code and data generators for a simple PETSc benchmark
that solves a simple partial differential equation.

## Build instructions for PETSc

First, create a fresh conda environment with `scipy`, `cython` and `numpy`. Install PETSc using
```
./configure --with-cuda=1 --with-petsc4py --prefix=petsc-install --with-cuda-dir=/usr/local/cuda-11.0/ CXXFLAGS="-O3" COPTFLAGS="-O3" CXXOPTFLAGS="-O3" --with-debugging=0 --with-64-bit-indices
make
make install
```

To compile the benchmark code, run
```
OMPI_CC=gcc-8 OMPI_CXX=g++-8 PETSC_DIR=<petsc/install/dir/ make main
```

## Run instructions

To run on CPUs, use:
```
mpirun -np 20 --bind-to core ./main -ksp_type cg -ksp_max_it 100 -pc_type none -ksp_atol 1e-10 -ksp_rtol 1e-10
```

To run on GPUs, use:
```
mpirun -np 1 --bind-to core ./main -ksp_type cg -ksp_max_it 100 -pc_type none -ksp_atol 1e-10 -ksp_rtol 1e-10 -vec_type cuda -mat_type aijcusparse -use_gpu_aware_mpi 0
```

I don't think the machine that I have access to right now has a GPU aware MPI, but future machines might.