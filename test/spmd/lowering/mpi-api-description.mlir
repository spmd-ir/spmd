// RUN: env -u MPI_JSON_PATH spmd-opt %s -lower-spmd | FileCheck %s
//
// The MPI lowering derives its calls from the same API description the raiser
// reads, so it needs MPI_JSON_PATH -- but only when there is something to
// lower.  -lower-spmd checks every model of its table, MPI included, whatever
// module it is given, so a module of another model must not fail because an
// API description it never reads is missing.

// CHECK-LABEL: func.func @openmp_only
// CHECK:         omp.barrier
func.func @openmp_only(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 5 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
