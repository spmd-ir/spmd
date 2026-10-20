// RUN: %spmd-verify compile intermediates=%t.d -o %t %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.d/02-lowered.mlir
// RUN: %run %t | FileCheck %s
//
// CHECK: PASS orphaned-queries
//
// Queries in a function a parallel region calls ask the team the function runs
// in.  The raiser gives them spmd.getParallelComm, which does not fold outside
// a region; it is the current team all the same.
//
// IR: omp.parallel
// IR: llvm.call @omp_get_thread_num()

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int seen[4];

static void work(void) { seen[omp_get_thread_num()] = omp_get_num_threads(); }

int main(void) {
  #pragma omp parallel num_threads(4)
  work();
  for (int t = 0; t < 4; ++t)
    if (seen[t] != 4)
      abort();
  printf("PASS orphaned-queries\n");
  return 0;
}
