// RUN: spmd-opt %s -lower-spmd-to-cuda | FileCheck %s --implicit-check-not='"spmd.'
//
// The two CUDA runtime routines -convert-apiCalls-to-spmd raises, lowered back
// to the calls they were raised from.  Every argument is an operand the
// operation already carries in the C type the prototype wants, so the lowering
// passes them through in prototype order and the cudaError_t is the result.
//
// The declarations go to the start of the module, each in front of the one
// before it, so they are checked without an order.

// CHECK-DAG: llvm.func @cudaMemcpy(!llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK-DAG: llvm.func @cudaMemset(!llvm.ptr, i32, i64) -> i32

// A copy whose status the program reads.
// CHECK-LABEL: func.func @memcpy_checked
// CHECK:         %[[RC:.*]] = llvm.call @cudaMemcpy(%arg0, %arg1, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK:         return %[[RC]] : i32
func.func @memcpy_checked(%dst: !llvm.ptr, %src: !llvm.ptr) -> i32 {
  %n = arith.constant 4096 : i64
  %h2d = arith.constant 1 : i32
  %e = "spmd.memcpy"(%dst, %src, %n, %h2d) <{usedModel = 4 : i32}> : (!llvm.ptr, !llvm.ptr, i64, i32) -> !spmd.error
  %rc = "spmd.castSPMDErrorToI32"(%e) : (!spmd.error) -> i32
  return %rc : i32
}

// The status is the call's return value whether or not the program reads it;
// an unread one leaves the call standing, because cudaMemset writes memory.
// CHECK-LABEL: func.func @memset_unchecked
// CHECK:         llvm.call @cudaMemset(%arg0, %{{.*}}, %{{.*}}) : (!llvm.ptr, i32, i64) -> i32
// CHECK:         return
func.func @memset_unchecked(%dst: !llvm.ptr) {
  %zero = arith.constant 0 : i32
  %n = arith.constant 1024 : i64
  %e = "spmd.memset"(%dst, %zero, %n) <{usedModel = 4 : i32}> : (!llvm.ptr, i32, i64) -> !spmd.error
  return
}

// A buffer the raiser saw through a memref view reaches the runtime as the
// address the view starts at, as it does for every other model.
// CHECK-LABEL: func.func @memcpy_memref
// CHECK:         llvm.call @cudaMemcpy(%arg0, %arg1,
func.func @memcpy_memref(%dst: !llvm.ptr, %src: !llvm.ptr) {
  %n = arith.constant 64 : i64
  %d2h = arith.constant 2 : i32
  %dstView = "enzymexla.pointer2memref"(%dst) : (!llvm.ptr) -> memref<1xi8>
  %srcView = "enzymexla.pointer2memref"(%src) : (!llvm.ptr) -> memref<1xi8>
  %e = "spmd.memcpy"(%dstView, %srcView, %n, %d2h) <{usedModel = 4 : i32}> : (memref<1xi8>, memref<1xi8>, i64, i32) -> !spmd.error
  return
}

// The direction is the i32 the program computed; the lowering emits no
// direction of its own, so one that is not a constant passes through
// untouched.  Only a constant is checked, in cuda-inexpressible.mlir.
// CHECK-LABEL: func.func @memcpy_computed_kind
// CHECK:         llvm.call @cudaMemcpy(%arg0, %arg1, %arg2, %arg3)
func.func @memcpy_computed_kind(%dst: !llvm.ptr, %src: !llvm.ptr, %n: i64, %kind: i32) {
  %e = "spmd.memcpy"(%dst, %src, %n, %kind) <{usedModel = 4 : i32}> : (!llvm.ptr, !llvm.ptr, i64, i32) -> !spmd.error
  return
}

// An operation of another model is not this pass's, and neither is the cast of
// its result: only the CUDA ones become calls.
// CHECK-LABEL: func.func @cuda_and_mpi
// CHECK:         llvm.call @cudaMemset(
// CHECK:         "spmd.commWorld"() <{usedModel = 0 : i32}>
// CHECK:         "spmd.barrier"(%{{.*}}) <{isBlocking = true, usedModel = 0 : i32}>
// CHECK:         "spmd.castSPMDErrorToI32"
func.func @cuda_and_mpi(%dst: !llvm.ptr) -> i32 {
  %zero = arith.constant 0 : i32
  %n = arith.constant 8 : i64
  %e0 = "spmd.memset"(%dst, %zero, %n) <{usedModel = 4 : i32}> : (!llvm.ptr, i32, i64) -> !spmd.error
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %rc = "spmd.castSPMDErrorToI32"(%e1) : (!spmd.error) -> i32
  return %rc : i32
}
