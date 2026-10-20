// RUN: env -u SHMEM_JSON_PATH SPMD_IR_EXITCODE=0 spmd-opt %s -lower-spmd-to-shmem 2>&1 | FileCheck %s
//
// An operation that does need an OpenSHMEM call says what is missing, rather
// than failing to legalize with no reason given: with no API description there
// are no patterns at all.  driver-default-table-no-shmem-description.mlir is the
// module that does not need it.
//
// SPMD_IR_EXITCODE=0 is what makes the failure readable here rather than fatal
// to the RUN line; lit.cfg.py unsets it for every other test.

// CHECK: no OpenSHMEM API description; set SHMEM_JSON_PATH
func.func @shmem_barrier() {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
