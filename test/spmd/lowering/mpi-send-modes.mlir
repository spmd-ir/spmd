// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// Buffered and ready mode are attributes on spmd.send, so MPI_Bsend / MPI_Rsend
// and their non-blocking forms come back as themselves rather than collapsing
// onto MPI_Send.  Synchronous mode is not modelled by the dialect, so MPI_Ssend
// round-trips as MPI_Send - that is a dialect limitation, not a lowering one.

// CHECK-LABEL: func.func @standard
// CHECK:         llvm.call @MPI_Send(
func.func @standard(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @buffered
// CHECK:         llvm.call @MPI_Bsend(
func.func @buffered(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag) <{isBlocking = true, isBuffered = true, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @ready
// CHECK:         llvm.call @MPI_Rsend(
func.func @ready(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag) <{isBlocking = true, isBuffered = false, isReadyMode = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @nonblocking_buffered
// CHECK:         llvm.call @MPI_Ibsend(
// CHECK:         llvm.call @MPI_Wait(
func.func @nonblocking_buffered(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag) <{isBlocking = false, isBuffered = true, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %e2 = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// A blocking send that still carries a request result gets MPI_REQUEST_NULL
// (0x2c000000) for it: the handle is an int, not a pointer.
// CHECK-LABEL: func.func @blocking_with_request_result
// CHECK:         llvm.call @MPI_Send(
// CHECK:         llvm.mlir.constant(738197504 : i32)
func.func @blocking_with_request_result(%buf: !llvm.ptr) -> !spmd.req {
  %count = arith.constant 1 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  return %req : !spmd.req
}
