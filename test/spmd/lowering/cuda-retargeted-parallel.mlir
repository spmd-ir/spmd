// RUN: spmd-opt %s -lower-spmd-to-cuda | FileCheck %s --implicit-check-not='"spmd.'
//
// CUDA's map has no init and no finalize, so -convert-apiCalls-to-spmd never
// wraps a process-level spmd.parallel around a CUDA program of its own.  A
// retarget can hand this pass one all the same -- -spmd-retarget-model and
// -lower-spmd's models= move every operation of a model, the process-level
// parallel among them -- and the structure patterns lower it, splicing the
// region in and dropping the barrier that matches the region itself.
//
// So it must not be reported as an operation CUDA has no call for: it is not a
// call at all.  The check used to say it was, and the module failed before the
// patterns ever ran.

// The communicator the parallel took is MPI's and stays, as any other model's
// operation does; it is dead once the region has been spliced in.
// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         "spmd.commWorld"() <{usedModel = 0 : i32}>
// CHECK-NOT:     spmd.parallel
// CHECK:         llvm.call @cudaMemset(
// CHECK-NOT:     spmd.barrier
// CHECK:         llvm.return
llvm.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %code = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 0 : i32, usedModel = 4 : i32}> ({
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %bytes = llvm.mlir.constant(256 : i64) : i64
    %device = llvm.alloca %one x i8 : (i32) -> !llvm.ptr
    %e = "spmd.memset"(%device, %c0, %bytes) <{usedModel = 4 : i32}> : (!llvm.ptr, i32, i64) -> !spmd.error
    %b = "spmd.barrier"(%world) <{isBlocking = true, isUniqueMatching = true, usedModel = 4 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  llvm.return %code : i32
}
