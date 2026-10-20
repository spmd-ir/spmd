// RUN: env -u OMP_JSON_PATH SPMD_IR_EXITCODE=0 spmd-opt %s -lower-spmd-to-omp 2>&1 | FileCheck %s
//
// A thread-number query needs the call the API description describes, and says
// so, rather than failing to legalize with no reason given.
//
// SPMD_IR_EXITCODE=0 is what makes the failure readable here rather than fatal
// to the RUN line; lit.cfg.py unsets it for every other test, where it would
// hide a failure instead of being the thing under test.

// CHECK: no OpenMP API description; set OMP_JSON_PATH to includesProgramModels/omp/api.json
func.func @query() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %rank, %e = "spmd.getRankInComm"(%world) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
  return %rank : i32
}
