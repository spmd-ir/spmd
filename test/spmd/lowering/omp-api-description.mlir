// RUN: env -u OMP_JSON_PATH spmd-opt %s -lower-spmd-to-omp 2>&1 | FileCheck %s --implicit-check-not='"spmd.' --implicit-check-not=ERROR
//
// Only OpenMP's runtime queries are derived from the API description; the
// directives are not calls.  A module without a query lowers without it, and
// does not complain about it either.  omp-api-description-missing.mlir is the
// module with a query.

// CHECK-LABEL: func.func @directives_only
// CHECK:         omp.barrier
func.func @directives_only() {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %b = "spmd.barrier"(%world) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
