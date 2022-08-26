#include "mpi.h"
#include "petscmat.h"
#include "petscvec.h"
#include "petsc.h"
#include "petscsys.h"
#include "petsctime.h"
#include "petscksp.h"

int loadMatrixFromFile(Mat* A, const char* filename) {
  auto ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
  MatSetFromOptions(*A);
  PetscViewer viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERBINARY);
  PetscViewerFileSetMode(viewer, FILE_MODE_READ);
  PetscViewerFileSetName(viewer, filename);
  MatLoad(*A, viewer);
  return 0;
}

int loadVecFromFile(Vec* b, const char* filename) {
  auto ierr = VecCreate(PETSC_COMM_WORLD, b); CHKERRQ(ierr);
  VecSetFromOptions(*b);
  PetscViewer viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERBINARY);
  PetscViewerFileSetMode(viewer, FILE_MODE_READ);
  PetscViewerFileSetName(viewer, filename);
  VecLoad(*b, viewer);
  return 0;
}

int main(int argc, char** argv) {
  PetscInt ierr;
  PetscInitialize(&argc, &argv, (char *)0, "PDE Benchmark");
  char dataPath[PETSC_MAX_PATH_LEN]; PetscBool dataPathSet;
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-datadir", dataPath, PETSC_MAX_PATH_LEN-1, &dataPathSet); CHKERRQ(ierr);
  Mat A; Vec b, x;
  std::string path;
  if (dataPathSet) {
    path = std::string(dataPath);
  } else {
    path = "./";
  }
  if (path[path.size() - 1] != '/') {
    path = path + "/";
  }
  ierr = loadMatrixFromFile(&A, (path + "A.dat").c_str()); CHKERRQ(ierr);
  ierr = loadVecFromFile(&b, (path + "bflat.dat").c_str()); CHKERRQ(ierr);
  ierr = loadVecFromFile(&x, (path + "x.dat").c_str()); CHKERRQ(ierr);
  
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
