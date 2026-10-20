// RUN: %spmd-verify compile intermediates=%t.d -o %t %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.d/02-lowered.mlir
// RUN: %run %t | FileCheck %s
//
// CHECK: PASS worksharing-loops
//
// Worksharing loops with and without nowait, and schedules.  Clang normalises
// each loop to one whose upper bound is inclusive; before -omp-to-spmd kept
// that, the last iteration was lost and a[999] stayed 0.  A sections construct
// is a worksharing loop over the sections to clang, and comes back as one.
// (An explicit `omp barrier` right after a nowait loop would come back as the
// loop without nowait, which waits at the same point.)
//
// IR: omp.wsloop schedule(static
// IR: inclusive
// IR: omp.wsloop nowait schedule(dynamic

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000
int a[N];
int x, y;

int main(void) {
  #pragma omp parallel num_threads(4)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < N; ++i)
      a[i] = i;
    #pragma omp for schedule(dynamic, 7) nowait
    for (int i = 0; i < N; ++i)
      a[i] += 1;
    #pragma omp sections
    {
      #pragma omp section
      { x = 1; }
      #pragma omp section
      { y = 2; }
    }
  }
  for (int i = 0; i < N; ++i)
    if (a[i] != i + 1)
      abort();
  if (x != 1 || y != 2)
    abort();
  printf("PASS worksharing-loops\n");
  return 0;
}
