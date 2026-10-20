// RUN: env -u NVSHMEM_JSON_PATH SPMD_IR_EXITCODE=0 spmd-opt %s -lower-spmd-to-nvshmem 2>&1 | FileCheck %s
//
// An operation that does need an NVSHMEM call says what is missing, rather
// than failing to legalize with no reason given: with no API description there
// are no patterns at all.
//
// SPMD_IR_EXITCODE=0 is what makes the failure readable here rather than fatal
// to the RUN line; lit.cfg.py unsets it for every other test.

// CHECK: lower-spmd-to-nvshmem: no NVSHMEM API description; set NVSHMEM_JSON_PATH to includesProgramModels/mynvshmem/api.json
func.func @nvshmem_barrier() {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
