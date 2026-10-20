// RUN: spmd-opt %s -test-omp-lowering-table | FileCheck %s
//
// Every OpenMP call -lower-spmd-to-omp can emit, as it derives it from the
// CACTS map the raiser reads (lib/spmd/passes/CACTS/ompSpmdEntries.cpp) and the
// USIS API description (includesProgramModels/omp/api.json).  Both routines
// take nothing: the communicator the raiser injects is the current team, and
// the result is the return value.  The directives are not calls and not here.

// CHECK:      omp_get_num_threads <- spmd.getSizeOfComm
// CHECK-NEXT:     return: result #0
// CHECK-NEXT:     $0 comm: injected
// CHECK-NEXT: omp_get_thread_num <- spmd.getRankInComm
// CHECK-NEXT:     return: result #0
// CHECK-NEXT:     $0 comm: injected
// CHECK-NOT:  <-
module {
}
