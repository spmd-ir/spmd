// RUN: %spmd-verify compile intermediates=%t.d -o %t %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.d/02-lowered.mlir
// RUN: %run %t | FileCheck %s
//
// CHECK: PASS thread-numbers
//
// Every thread of a team of four sees its own number and the team's size, and
// outside the team the program is thread 0 of 1.  The queries are the calls
// the program made, lowered from the SPMD operations they were raised to.
//
// IR: omp.parallel num_threads(
// IR: llvm.call @omp_get_thread_num()
// IR: llvm.call @omp_get_num_threads()

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int size[4] = {0, 0, 0, 0};
  if (omp_get_thread_num() != 0 || omp_get_num_threads() != 1)
    abort();
  #pragma omp parallel num_threads(4)
  {
    int t = omp_get_thread_num();
    size[t] = omp_get_num_threads();
  }
  for (int t = 0; t < 4; ++t)
    if (size[t] != 4)
      abort();
  printf("PASS thread-numbers\n");
  return 0;
}
