// RUN: env -u SHMEM_JSON_PATH -u NVSHMEM_JSON_PATH spmd-opt %s -lower-spmd -o %t 2> %t.err
// RUN: FileCheck %s --implicit-check-not='"spmd.' --input-file=%t
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty --input-file=%t.err
//
// The default table lowers every model to itself, OpenSHMEM and NVSHMEM
// included, so -lower-spmd runs their checks and lowerings on every module.  A
// module with nothing tagged for either does not need their API descriptions,
// and lowers as if the models did not exist -- without so much as reading
// them, so nothing complains that the description is missing.

// QUIET-NOT: JSON_PATH

// CHECK-LABEL: func.func @mpi_only
// CHECK:         llvm.call @MPI_Barrier(
func.func @mpi_only() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
