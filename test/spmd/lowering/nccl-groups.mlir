// RUN: spmd-opt %s -split-input-file -lower-spmd-to-nccl | FileCheck %s --implicit-check-not='"spmd.' --implicit-check-not='llvm.call @ncclGroup'
//
// Groups.  The raiser turns ncclGroupStart, the calls between and ncclGroupEnd
// into operations with a request each, stored in an array, and an spmd.waitAll
// of the array where ncclGroupEnd was.  NCCL has no requests, so the lowering
// turns that back: every operation with a request opens a group of its own and
// every request the wait completes closes one.  NCCL counts nested groups and
// runs all of them at the outermost ncclGroupEnd, which is where the wait is,
// so the calls run together, as the program had them.
//
// roundtrip's lowering claimed to bracket the blocking operations in a group
// and bracketed none; a send and a receive of a pair would each have blocked
// for the other.

// CHECK-LABEL: func.func @send_recv_pair
// CHECK:         llvm.call @ncclGroupStart() : () -> i32
// CHECK-NEXT:    llvm.call @ncclSend(
// CHECK:         %[[REQ0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         llvm.call @ncclGroupStart() : () -> i32
// CHECK-NEXT:    llvm.call @ncclRecv(
// CHECK:         llvm.call @ncclGroupEnd() : () -> i32
// CHECK-NEXT:    %[[E:.*]] = llvm.call @ncclGroupEnd() : () -> i32
// CHECK:         return %[[E]] : i32
func.func @send_recv_pair(%comm: !spmd.comm, %stream: !llvm.ptr, %sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64, %peer: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %tag = arith.constant 0 : i32
  %reqs = memref.alloca() : memref<2x!spmd.req>
  %dt = "spmd.constDatatype"() <{typeAttr = f32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %r0, %e0 = "spmd.send"(%comm, %sbuf, %n, %dt, %peer, %tag, %s) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> (!spmd.req, !spmd.error)
  memref.store %r0, %reqs[%c0] : memref<2x!spmd.req>
  %r1, %e1 = "spmd.recv"(%comm, %rbuf, %n, %dt, %peer, %tag, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> (!spmd.req, !spmd.error)
  memref.store %r1, %reqs[%c1] : memref<2x!spmd.req>
  %e2 = "spmd.waitAll"(%c2_i32, %reqs) <{usedModel = 3 : i32}> : (i32, memref<2x!spmd.req>) -> !spmd.error
  %ret = "spmd.castSPMDErrorToI32"(%e2) : (!spmd.error) -> i32
  return %ret : i32
}

// -----

// One operation and a wait of its request: a group of one, as a group the
// raiser made of a single call is.  A communicator query inside a group runs
// immediately, which NCCL allows.
// CHECK-LABEL: func.func @one_collective
// CHECK:         llvm.call @ncclGroupStart() : () -> i32
// CHECK-NEXT:    llvm.call @ncclAllGather(
// CHECK:         llvm.call @ncclCommCount(
// CHECK:         llvm.call @ncclGroupEnd() : () -> i32
// CHECK-NOT:     ncclGroupEnd
// CHECK:         return
func.func @one_collective(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %r, %e0 = "spmd.allgather"(%comm, %buf, %n, %dt, %buf, %n, %dt, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, !spmd.stream) -> (!spmd.req, !spmd.error)
  %size, %e1 = "spmd.getSizeOfComm"(%comm) <{usedModel = 3 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %e2 = "spmd.wait"(%r) <{usedModel = 3 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// -----

// Where the raiser could not pair a group's brackets -- a group around a loop,
// say -- it leaves them as spmd.groupStart and spmd.groupEnd and the calls
// between them as they are, which lowers one to one.
// CHECK-LABEL: func.func @explicit_brackets
// CHECK:         llvm.call @ncclGroupStart() : () -> i32
// CHECK:         llvm.call @ncclSend(
// CHECK:         llvm.call @ncclGroupEnd() : () -> i32
func.func @explicit_brackets(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64, %peer: i32) {
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %e0 = "spmd.groupStart"() <{usedModel = 3 : i32}> : () -> !spmd.error
  %e1 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %tag, %s) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> !spmd.error
  %e2 = "spmd.groupEnd"() <{usedModel = 3 : i32}> : () -> !spmd.error
  return
}

// -----

// A wait of no requests completes nothing and closes no group.  The check
// accepted it and the pattern refused it, so the conversion failed after
// rewriting part of the module.
// CHECK-LABEL: func.func @wait_all_none
// CHECK:         %[[OK:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         return %[[OK]] : i32
func.func @wait_all_none(%reqs: memref<1x!spmd.req>) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %e = "spmd.waitAll"(%c0_i32, %reqs) <{usedModel = 3 : i32}> : (i32, memref<1x!spmd.req>) -> !spmd.error
  %ret = "spmd.castSPMDErrorToI32"(%e) : (!spmd.error) -> i32
  return %ret : i32
}
