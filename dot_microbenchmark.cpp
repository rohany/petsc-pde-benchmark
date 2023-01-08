#include "mpi.h"
#include "petscmat.h"
#include "petscvec.h"
#include "petsc.h"
#include "petscsys.h"
#include "petsctime.h"

#include <cuda_runtime_api.h>

int main(int argc, char** argv) {
  PetscInt ierr;
  PetscInitialize(&argc, &argv, (char *)0, "Dot MicroBenchmark");
  PetscInt n = 100;
  PetscInt iters = 25;
  PetscInt nnz_per_row = 11;
  PetscInt k = 4;
  PetscBool spmm = PETSC_FALSE;
  PetscBool gpu = PETSC_FALSE;
  PetscBool nSet, iSet, spmmSet, gpuSet, nnzSet, kSet;
  ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, &nSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-i", &iters, &iSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-k", &k, &kSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-nnz-per-row", &nnz_per_row, &nnzSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-gpu", &gpu, &gpuSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-spmm", &spmm, &spmmSet); CHKERRQ(ierr);


  Mat A;
  PetscInt rowStart, rowEnd;
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
  MatSetType(A, MATMPIAIJ);
  MatSetFromOptions(A);
  // I don't understand how preallocation works. So, I just overallocate
  // both of these and my program stops hanging!
  MatSeqAIJSetPreallocation(A, 2 * nnz_per_row, NULL);
  MatMPIAIJSetPreallocation(A, 2 * nnz_per_row, NULL, 2 * nnz_per_row, NULL);
  MatGetOwnershipRange(A, &rowStart, &rowEnd);
  int inserted = 0;
  for (auto r = rowStart; r < rowEnd; r++) {
    for (int j = r - (nnz_per_row / 2); j < r + (nnz_per_row / 2); j++) {
      if (j >= 0 && j < n) {
        MatSetValue(A, r, j, 1.0, INSERT_VALUES);
	inserted++;
      }
    }
  }
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  if (spmm) {
    Mat x, y;
    if (gpu) {
      MatCreateDenseCUDA(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, k, NULL, &x);
      MatCreateDenseCUDA(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, k, NULL, &y);
    } else {
      MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, k, NULL, &x);
      MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, k, NULL, &y);
    }
    MatZeroEntries(x);
    MatZeroEntries(y);

    // Warmup set of matmult operations to pull all data onto the GPU.
    {
      for (int i = 0; i < 5; i++) {
	MatMatMult(A, x, MAT_REUSE_MATRIX, PETSC_DEFAULT, &y);
      }
    }

    PetscLogStage logStage;
    PetscLogStageRegister("BENCHMARK", &logStage);
    PetscLogDouble start, end;
    PetscTime(&start);
    PetscLogStagePush(logStage);
    for (int i = 0; i < iters; i++) {
      MatMatMult(A, x, MAT_REUSE_MATRIX, PETSC_DEFAULT, &y);
    }
    if (gpu) {
      cudaDeviceSynchronize();
    }
    PetscLogStagePop();
    PetscTime(&end);
    auto time = (end - start);
    PetscPrintf(PETSC_COMM_WORLD, "Achieved %f iterations / sec. \n", iters / time);
  } else {
    Vec x, y;
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    VecSetSizes(x, PETSC_DECIDE, n);
    VecSetFromOptions(x);
    VecSet(x, 1.0);
    ierr = VecCreate(PETSC_COMM_WORLD, &y); CHKERRQ(ierr);
    VecSetSizes(y, PETSC_DECIDE, n);
    VecSetFromOptions(y);
    VecSet(y, 0.0);
    
    // Warmup set of matmult operations to pull all data onto the GPU.
    {
      for (int i = 0; i < 5; i++) {
        MatMult(A, x, y);
      }
    }

    PetscLogStage logStage;
    PetscLogStageRegister("BENCHMARK", &logStage);
    PetscLogDouble start, end;
    PetscTime(&start);
    PetscLogStagePush(logStage);
    for (int i = 0; i < iters; i++) {
      MatMult(A, x, y);
    }
    if (gpu) {
      cudaDeviceSynchronize();
    }
    PetscLogStagePop();
    PetscTime(&end);
    auto time = (end - start);
    PetscPrintf(PETSC_COMM_WORLD, "Achieved %f iterations / sec. \n", iters / time);
  }
  PetscFinalize();
}
