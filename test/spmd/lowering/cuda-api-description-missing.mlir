// RUN: env -u CUDA_JSON_PATH SPMD_IR_EXITCODE=0 spmd-opt %s -lower-spmd-to-cuda 2>&1 | FileCheck %s
//
// The calls are derived from the same API description the raiser reads, so a
// module with a CUDA operation needs it and says so, rather than failing to
// legalize with no reason given.
//
// SPMD_IR_EXITCODE=0 is what makes the failure readable here rather than fatal
// to the RUN line; lit.cfg.py unsets it for every other test, where it would
// hide a failure instead of being the thing under test.

// CHECK: no CUDA API description; set CUDA_JSON_PATH to includesProgramModels/cuda/api.json
func.func @memset(%dst: !llvm.ptr, %value: i32, %n: i64) {
  %e = "spmd.memset"(%dst, %value, %n) <{usedModel = 4 : i32}> : (!llvm.ptr, i32, i64) -> !spmd.error
  return
}
