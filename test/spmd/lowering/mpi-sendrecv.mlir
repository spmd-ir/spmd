// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// spmd.sendrecv carries the request as an optional result, the same shape
// spmd.send and spmd.recv have.  The blocking form lowers to MPI_Sendrecv and
// passes MPI_STATUS_IGNORE for the last parameter; the non-blocking form lowers
// to MPI_Isendrecv, where that same parameter is the request out-pointer.
//
// MPI_Sendrecv_replace raises to this op with sendBuf and recvBuf the same
// *value*, and that is what tells the two apart on the way back down:
// MPI_Sendrecv/MPI_Isendrecv do not tolerate an aliased send and receive
// buffer (no internal temporary, unlike the _replace forms), so whenever the
// two are still one value it lowers back to MPI_Sendrecv_replace /
// MPI_Isendrecv_replace instead, never to the plain form.

// CHECK-LABEL: func.func @blocking
// CHECK:         llvm.call @MPI_Sendrecv(
// CHECK-NOT:     llvm.call @MPI_Isendrecv(
func.func @blocking(%out: !llvm.ptr, %in: !llvm.ptr, %peer: i32) {
  %stride = arith.constant 1 : i64
  %count = arith.constant 1 : i64
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.sendrecv"(%comm, %out, %count, %dt, %stride, %peer, %tag, %in, %count, %dt, %stride, %peer, %tag) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32) -> !spmd.error
  return
}

// The non-blocking form hands back a real request, so the wait that completes
// it has something to wait on.
// CHECK-LABEL: func.func @nonblocking
// CHECK:         llvm.call @MPI_Isendrecv(
// CHECK:         llvm.load
// CHECK:         llvm.call @MPI_Wait(
func.func @nonblocking(%out: !llvm.ptr, %in: !llvm.ptr, %peer: i32) {
  %stride = arith.constant 1 : i64
  %count = arith.constant 1 : i64
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e = "spmd.sendrecv"(%comm, %out, %count, %dt, %stride, %peer, %tag, %in, %count, %dt, %stride, %peer, %tag) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32) -> (!spmd.req, !spmd.error)
  %e2 = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// A blocking sendrecv that still carries a request result gets
// MPI_REQUEST_NULL (0x2c000000) for it, the same convention send and recv use.
// CHECK-LABEL: func.func @blocking_with_request_result
// CHECK:         llvm.call @MPI_Sendrecv(
// CHECK:         llvm.mlir.constant(738197504 : i32)
func.func @blocking_with_request_result(%out: !llvm.ptr, %in: !llvm.ptr, %peer: i32) -> !spmd.req {
  %stride = arith.constant 1 : i64
  %count = arith.constant 1 : i64
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e = "spmd.sendrecv"(%comm, %out, %count, %dt, %stride, %peer, %tag, %in, %count, %dt, %stride, %peer, %tag) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32) -> (!spmd.req, !spmd.error)
  return %req : !spmd.req
}

// sendBuf and recvBuf the same value -- what MPI_Sendrecv_replace raises to --
// gets the _replace form back, not the plain one: MPI_Sendrecv would read and
// overwrite the one buffer with no guarantee the peer has read the old value
// first (test/exec/mpi/test-sendrecv.c caught this losing the exchanged value
// across nodes).
// CHECK-LABEL: func.func @blocking_aliased
// CHECK:         llvm.call @MPI_Sendrecv_replace(
// CHECK-NOT:     llvm.call @MPI_Sendrecv(
func.func @blocking_aliased(%buf: !llvm.ptr, %peer: i32) {
  %stride = arith.constant 1 : i64
  %count = arith.constant 1 : i64
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = i32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.sendrecv"(%comm, %buf, %count, %dt, %stride, %peer, %tag, %buf, %count, %dt, %stride, %peer, %tag) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32) -> !spmd.error
  return
}

// The same, non-blocking.
// CHECK-LABEL: func.func @nonblocking_aliased
// CHECK:         llvm.call @MPI_Isendrecv_replace(
// CHECK-NOT:     llvm.call @MPI_Isendrecv(
func.func @nonblocking_aliased(%buf: !llvm.ptr, %peer: i32) {
  %stride = arith.constant 1 : i64
  %count = arith.constant 1 : i64
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = i32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e = "spmd.sendrecv"(%comm, %buf, %count, %dt, %stride, %peer, %tag, %buf, %count, %dt, %stride, %peer, %tag) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32) -> (!spmd.req, !spmd.error)
  %e2 = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}
