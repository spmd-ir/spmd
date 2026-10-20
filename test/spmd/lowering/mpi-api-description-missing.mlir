// RUN: env -u MPI_JSON_PATH SPMD_IR_EXITCODE=0 spmd-opt %s -lower-spmd-to-mpi 2>&1 | FileCheck %s
//
// An operation that does need an MPI call says what is missing, rather than
// failing to legalize with no reason given: with no API description there are
// no patterns at all.
//
// SPMD_IR_EXITCODE=0 is what makes the failure readable here rather than fatal
// to the RUN line; lit.cfg.py unsets it for every other test, where it would
// hide a failure instead of being the thing under test.

// CHECK: no MPI API description; set MPI_JSON_PATH
func.func @mpi_barrier() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
