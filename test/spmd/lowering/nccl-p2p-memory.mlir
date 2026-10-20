// RUN: spmd-opt %s -lower-spmd-to-nccl | FileCheck %s --implicit-check-not='"spmd.' --implicit-check-not=ncclGroup
//
// Point-to-point and memory.  A send or receive on its own is its own group in
// NCCL, so nothing brackets it; the operations of a group are in
// nccl-groups.mlir.  NCCL has no tag, and the raiser gives the operation a tag
// of 0, so that is the only one there is a call for (nccl-inexpressible.mlir).
//
// ncclMemAlloc and ncclMemFree need no communicator.  spmd.malloc and
// spmd.free carry the world communicator because the dialect's allocations
// are collective in general, and it is not passed on.

// CHECK-LABEL: func.func @send_recv
// CHECK:         %[[C:.*]] = builtin.unrealized_conversion_cast %arg0 : !spmd.comm to !llvm.ptr
// CHECK:         %[[I8:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         llvm.call @ncclSend(%arg2, %arg3, %[[I8]], %arg4, %[[C]], %arg1) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.call @ncclRecv(%arg2, %arg3, %[[I8]], %arg4, %[[C]], %arg1) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
func.func @send_recv(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64, %peer: i32) {
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si8, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %tag, %s) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> !spmd.error
  %e1 = "spmd.recv"(%comm, %buf, %n, %dt, %peer, %tag, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> !spmd.error
  return
}

// The address comes back through the first argument.
// CHECK-LABEL: func.func @malloc_free
// CHECK:         %[[SLOT:.*]] = llvm.alloca %{{.*}} x !llvm.ptr
// CHECK:         llvm.call @ncclMemAlloc(%[[SLOT]], %arg0) : (!llvm.ptr, i64) -> i32
// CHECK:         %[[BUF:.*]] = llvm.load %[[SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @ncclMemFree(%[[BUF]]) : (!llvm.ptr) -> i32
// CHECK:         return %[[BUF]] : !llvm.ptr
func.func @malloc_free(%size: i64) -> !llvm.ptr {
  %world = "spmd.commWorld"() <{usedModel = 3 : i32}> : () -> !spmd.comm
  %buf, %e0 = "spmd.malloc"(%world, %size) <{usedModel = 3 : i32}> : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  %e1 = "spmd.free"(%world, %buf) <{usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr) -> !spmd.error
  return %buf : !llvm.ptr
}
