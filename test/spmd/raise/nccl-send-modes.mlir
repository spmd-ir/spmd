// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s
//
// spmd.send requires $isBuffered and $isReadyMode, which the MPI raiser sets
// and the NCCL raiser did not, so every raised ncclSend was an operation that
// failed to verify.  NCCL has one send mode, the standard one.

// CHECK-LABEL: llvm.func @send
// CHECK:         "spmd.send"({{.*}}) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 3 : i32}>
llvm.func @send(%comm: !llvm.ptr, %stream: !llvm.ptr, %buf: !llvm.ptr, %peer: i32) {
  %n = arith.constant 4 : i64
  %int = arith.constant 2 : i32
  %r = llvm.call @ncclSend(%buf, %n, %int, %peer, %comm, %stream) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.return
}
llvm.func @ncclSend(!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
