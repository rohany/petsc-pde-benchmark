#include "mpi.h"
#include "petscmat.h"
#include "petscvec.h"
#include "petsc.h"
#include "petscsys.h"
#include "petsctime.h"
#include "petscksp.h"
#include <math.h>

#define PI 3.14159265

int main(int argc, char** argv) {
  PetscInt ierr;
  PetscInitialize(&argc, &argv, (char *)0, "PDE Benchmark");
  PetscInt nx = 101;
  PetscInt ny = 101;
  PetscBool nxSet, nySet;
  ierr = PetscOptionsGetInt(NULL, NULL, "-nx", &nx, &nxSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-ny", &ny, &nySet); CHKERRQ(ierr);

  auto xmin = 0.0;
  auto xmax = 1.0;
  auto ymin = -0.5;
  auto ymax = 0.5;
  auto lx = xmax - xmin;
  auto ly = ymax - ymin;
  auto dx = lx / (nx-1);
  auto dy = ly / (ny-1);

  auto a = 1.0 / (dx * dx);
  auto g = 1.0 / (dy * dy);
  auto c = -2.0*a - 2.0*g;

  // Create the stencil matrix A.
  Mat A;
  PetscInt rowStart, rowEnd;
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, (nx - 2) * (nx - 2), (ny - 2) * (ny - 2));
  MatSetType(A, MATMPIAIJ);
  MatSetFromOptions(A);
  // I don't understand how preallocation works. So, I just overallocate
  // both of these and my program stops hanging!
  MatSeqAIJSetPreallocation(A, 10, NULL);
  MatMPIAIJSetPreallocation(A, 10, NULL, 10, NULL);
  MatGetOwnershipRange(A, &rowStart, &rowEnd);
  for (auto r = rowStart; r < rowEnd; r++) {
    // We need to set 5 points on the stencil:
    // a * p_{i-1},j + c * p_i,j + a * p_{i+1},j + g * p_i,{j-1} + g * p_i, j+1.
    auto i = (r % (ny - 2)) + 1;
    auto j = (r / (ny - 2)) + 1;
    if (j - 1 > 0) {
      MatSetValue(A, r, r - (ny - 2), g, INSERT_VALUES);
    }
    if (i - 1 > 0) {
      MatSetValue(A, r, r - 1, a, INSERT_VALUES);
    }
    MatSetValue(A, r, r, c, INSERT_VALUES);
    if (j + 1 < (ny - 1)) {
      MatSetValue(A, r, r + (ny - 2), g, INSERT_VALUES);
    }
    if (i + 1 < (nx - 1)) {
      MatSetValue(A, r, r + 1, a, INSERT_VALUES);
    }
  }
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  // Code to dump the constructed matrix.
  // PetscViewer viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
  // PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE);
  // MatView(A, viewer);

  // Create the solution vector b.
  Vec b;
  ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
  VecSetSizes(b, PETSC_DECIDE, (nx - 2) * (ny - 2));
  VecSetFromOptions(b);
  PetscInt lo, hi;
  VecGetOwnershipRange(b, &lo, &hi);
  for (auto idx = lo; idx < hi; idx++) {
    auto i = (idx % (ny - 2)) + 1;
    auto j = (idx / (ny - 2)) + 1;
    auto x = ((xmax - xmin) / (nx - 1)) * i + xmin;
    auto y = ((ymax - ymin) / (ny - 1)) * j + ymin;
    auto value = (sin(PI * x) * cos(PI * y)) + (sin(5.0 * PI * x) * cos(5.0 * PI * y));
    VecSetValue(b, idx, value, INSERT_VALUES);
  }
  VecAssemblyBegin(b);
  VecAssemblyEnd(b);
  // To dump the solution vector.
  // VecView(b, viewer);
  
  // Initialize the output vector x.
  Vec x;
  ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
  VecSetSizes(x, PETSC_DECIDE, (nx - 2) * (ny - 2));
  VecSetFromOptions(x);
  VecSet(x, 0.0);
  
  PetscLogStage logStage;
  PetscLogStageRegister("BENCHMARK", &logStage);
  // Warmup set of matmult operations to pull all data onto the GPU.
  {
    for (int i = 0; i < 5; i++) {
      MatMult(A, b, x);
      VecPointwiseMult(x, x, x);
      PetscReal dummy;
      VecNorm(x, NORM_1, &dummy);
    }
  }
   
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, A, A);
  KSPSetFromOptions(ksp);
  PetscLogDouble start, end;
  PetscTime(&start);
  PetscLogStagePush(logStage);
  KSPSolve(ksp, b, x);
  PetscLogStagePop();
  PetscTime(&end);
  auto time = (end - start);
  PetscInt iters;
  KSPGetIterationNumber(ksp, &iters);
  PetscPrintf(PETSC_COMM_WORLD, "Achieved %f iterations / sec. \n", iters / time);
  PetscPrintf(PETSC_COMM_WORLD, "Iters: %d\n", iters);
  PetscFinalize();
}
