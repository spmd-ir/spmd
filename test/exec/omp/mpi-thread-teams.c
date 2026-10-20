// RUN: %spmd-verify compile intermediates=%t.d -o %t %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.d/02-lowered.mlir
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS mpi-thread-teams
//
// MPI+OpenMP: every rank starts a team of four, and the ranks combine what
// their threads computed.  Outside the team the thread number is raised on
// MPI's process team -- the team the raiser folds the query's into -- and is
// still omp_get_thread_num() to OpenMP.
//
// IR: llvm.call @MPI_Init_thread(
// IR: llvm.call @omp_get_thread_num()
// IR: omp.parallel num_threads(
// IR: llvm.call @MPI_Allreduce(

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int provided, rank, size;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (omp_get_thread_num() != 0 || omp_get_num_threads() != 1)
    abort();

  int perThread[4] = {0, 0, 0, 0};
  #pragma omp parallel num_threads(4)
  { perThread[omp_get_thread_num()] = rank + omp_get_thread_num(); }
  int local = 0;
  for (int t = 0; t < 4; ++t)
    local += perThread[t];

  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  // Each rank r contributes 4r + (0 + 1 + 2 + 3).
  if (global != 4 * (size * (size - 1) / 2) + 6 * size)
    abort();
  if (rank == 0)
    printf("PASS mpi-thread-teams\n");
  MPI_Finalize();
  return 0;
}
