// RUN: %spmd-verify compile intermediates=%t.d -o %t %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.d/02-lowered.mlir
// RUN: %run %t | FileCheck %s
//
// CHECK: PASS single-critical-masked
//
// A single runs once, a critical section once per thread without losing an
// update, and master and masked only on the thread they name.  The raiser
// gives single a barrier of its own and keeps the program's explicit one;
// lowering folds the first back into the single.
//
// IR: omp.single {
// IR: omp.barrier
// IR: omp.critical

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int count, master, masked;

int main(void) {
  #pragma omp parallel num_threads(4)
  {
    #pragma omp single
    { count += 100; }
    #pragma omp barrier
    #pragma omp critical
    { count += 1; }
    #pragma omp master
    { master += 1; }
    #pragma omp masked filter(2)
    { masked += omp_get_thread_num(); }
  }
  if (count != 104 || master != 1 || masked != 2)
    abort();
  printf("PASS single-critical-masked\n");
  return 0;
}
