// RUN: %spmd-verify compile intermediates=%t.d -o %t %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.d/02-lowered.mlir
// RUN: %run %t | FileCheck %s
//
// CHECK: PASS nested-teams
//
// A team inside a team: the inner queries ask the inner team, and the outer
// thread number read before the inner region is still the outer one.
//
// IR: omp.parallel num_threads(
// IR: omp.parallel num_threads(

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int seen[4][2];

int main(void) {
  omp_set_max_active_levels(2);
  #pragma omp parallel num_threads(4)
  {
    int outer = omp_get_thread_num();
    #pragma omp parallel num_threads(2)
    { seen[outer][omp_get_thread_num()] = 10 * outer + omp_get_num_threads(); }
  }
  for (int o = 0; o < 4; ++o)
    for (int i = 0; i < 2; ++i)
      if (seen[o][i] != 10 * o + 2)
        abort();
  printf("PASS nested-teams\n");
  return 0;
}
