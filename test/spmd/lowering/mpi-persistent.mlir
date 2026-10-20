// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// Persistent and partitioned point-to-point: spmd.sendInit / spmd.recvInit pick
// their MPI initiator from the buffered / ready-mode attributes and from the
// presence of a partition count, and spmd.start / spmd.pReady drive them.
// MPI_Pready takes its request by value; every other one of these calls takes
// it by address, which is why the request handle goes through an alloca.

// CHECK-LABEL: func.func @send_init_standard
// CHECK:         llvm.call @MPI_Send_init(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[REQ:[0-9]+]])
// CHECK:         llvm.load %[[REQ]]
// CHECK:         llvm.call @MPI_Start(
// CHECK:         llvm.call @MPI_Request_free(
func.func @send_init_standard(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e0 = "spmd.sendInit"(%comm, %buf, %count, %dt, %dest, %tag) <{isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %e1 = "spmd.start"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %e2 = "spmd.reqFree"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @send_init_buffered
// CHECK:         llvm.call @MPI_Bsend_init(
func.func @send_init_buffered(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e0 = "spmd.sendInit"(%comm, %buf, %count, %dt, %dest, %tag) <{isBuffered = true, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @send_init_ready
// CHECK:         llvm.call @MPI_Rsend_init(
func.func @send_init_ready(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e0 = "spmd.sendInit"(%comm, %buf, %count, %dt, %dest, %tag) <{isBuffered = false, isReadyMode = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  return
}

// A partition count selects the partitioned initiator, whose element count is an
// MPI_Count (i64) rather than an int, and which takes an MPI_Info argument.
// MPI_INFO_NULL is 0x1c000000.
// CHECK-LABEL: func.func @psend_init
// CHECK:         %[[INFO:.*]] = llvm.mlir.constant(469762048 : i32)
// CHECK:         llvm.call @MPI_Psend_init(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[INFO]], %{{.*}}) : (!llvm.ptr, i32, i64, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Pready(
func.func @psend_init(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %parts = arith.constant 4 : i32
  %part = arith.constant 0 : i32
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e0 = "spmd.sendInit"(%comm, %buf, %parts, %count, %dt, %dest, %tag) <{isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i32, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %e1 = "spmd.start"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %e2 = "spmd.pReady"(%part, %req) <{usedModel = 0 : i32}> : (i32, !spmd.req) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @recv_init
// CHECK:         llvm.call @MPI_Recv_init(
// CHECK:         llvm.call @MPI_Wait(
func.func @recv_init(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %src = arith.constant 0 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e0 = "spmd.recvInit"(%comm, %buf, %count, %dt, %src, %tag) <{usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %e1 = "spmd.start"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %e2 = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @precv_init
// CHECK:         llvm.call @MPI_Precv_init(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i32, i64, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
func.func @precv_init(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %parts = arith.constant 4 : i32
  %src = arith.constant 0 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e0 = "spmd.recvInit"(%comm, %buf, %parts, %count, %dt, %src, %tag) <{usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i32, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  return
}
