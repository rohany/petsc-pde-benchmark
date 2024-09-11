# PETSc PDE CG Solver Benchmark

This repository contains source code and data generators for a simple PETSc benchmark
that solves a simple partial differential equation.

It also contains a small SpMV and SpMM microbenchmark, runnable in a similar manner as the PDE benchmark.

## Build instructions for PETSc

First, create a fresh conda environment with `scipy`, `cython` and `numpy`. Install PETSc using
```
./configure --with-cuda=1 --prefix=petsc-install --with-cuda-dir=<PATH/TO/CUDA/> CXXFLAGS="-O3" COPTFLAGS="-O3" CXXOPTFLAGS="-O3" FOPTFLAGS="-O3" --download-fblaslapack=1 --with-debugging=0 --with-64-bit-indices
make
make install
```

To compile the benchmark code, run
```
OMPI_CC=gcc-8 OMPI_CXX=g++-8 PETSC_DIR=<petsc/install/dir/> make main
```

## Run instructions

To run on CPUs, use:
```
mpirun -np 20 --bind-to core ./main -ksp_type cg -ksp_max_it 100 -pc_type none -ksp_atol 1e-100 -ksp_rtol 1e-100 -ksp_divtol 1e+100
```

To run on GPUs, use:
```
mpirun -np 1 --bind-to core ./main -ksp_type cg -ksp_max_it 100 -pc_type none -ksp_atol 1e-100 -ksp_rtol 1e-100 -ksp_divtol 1e+100 -vec_type cuda -mat_type aijcusparse -use_gpu_aware_mpi 0
```

On the Summit supercomputer, use the following command for GPUs:
```
jsrun -n 1 -g 1 -c 1 -b rs --smpiargs="-gpu" ./main -ksp_type cg -ksp_max_it 100 -pc_type none -ksp_atol 1e-100 -ksp_rtol 1e-100 -ksp_divtol 1e+100 -vec_type cuda -mat_type aijcusparse
```
