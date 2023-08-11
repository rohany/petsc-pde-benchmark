#include "petscmat.h"
#include "petscvec.h"
#include "petsc.h"

int main(int argc, char** argv) {
  PetscInt ierr;
  PetscInitialize(&argc, &argv, (char *)0, "GPU bug");

  PetscInt numRows = 1;
  PetscInt numCols = PetscInt(INT_MAX) * 2;

  Mat A;
  PetscInt rowStart, rowEnd;
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, numRows, numCols);
  MatSetType(A, MATMPIAIJ);
  MatSetFromOptions(A);

  MatSetValue(A, 0, 0, 1.0, INSERT_VALUES);
  MatSetValue(A, 0, numCols - 1, 1.0, INSERT_VALUES);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  Vec b;
  ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
  VecSetSizes(b, PETSC_DECIDE, numCols);
  VecSetFromOptions(b);
  VecSet(b, 0.0);
  VecSetValue(b, 0, 42.0, INSERT_VALUES);
  VecSetValue(b, numCols - 1, 58.0, INSERT_VALUES);
  VecAssemblyBegin(b);
  VecAssemblyEnd(b);
  
  Vec x;
  ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
  VecSetSizes(x, PETSC_DECIDE, numRows);
  VecSetFromOptions(x);
  VecSet(x, 0.0);

  MatMult(A, b, x);
  PetscScalar result;
  VecSum(x, &result);
  PetscPrintf(PETSC_COMM_WORLD, "Result of mult: %f\n", result);
  PetscFinalize();
}
