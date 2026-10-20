// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// Element counts are MPI_Counts (i64) in the dialect and ints in MPI's classic
// calls.  A count that provably fits an int is passed as one; a call with any
// other count becomes its MPI 4 large-count form, which takes every count as an
// MPI_Count.  The lowering used to truncate, which sends fewer elements than
// the program asked for once a count reaches 2^31.

// The raiser widens an int count with a zero extension; the int goes straight
// back in.
// CHECK-LABEL: func.func @widened_int
// CHECK:         llvm.call @MPI_Send(%arg0, %arg1, %{{.*}}) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
func.func @widened_int(%buf: !llvm.ptr, %n: i32) {
  %count = arith.extui %n : i32 to i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A constant in int range is an int.
// CHECK-LABEL: func.func @small_constant
// CHECK:         %[[COUNT:.*]] = llvm.trunc %{{.*}} : i64 to i32
// CHECK:         llvm.call @MPI_Send(%arg0, %[[COUNT]], %{{.*}}) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
func.func @small_constant(%buf: !llvm.ptr) {
  %count = arith.constant 4 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// 2^31 does not fit, and neither does an i64 nothing is known about.
// CHECK-LABEL: func.func @large_constant
// CHECK:         llvm.call @MPI_Send_c(%arg0, %{{.*}}, %{{.*}}) : (!llvm.ptr, i64, i32, i32, i32, i32) -> i32
func.func @large_constant(%buf: !llvm.ptr) {
  %count = arith.constant 2147483648 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @unknown_count
// CHECK:         %[[REQ:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         llvm.call @MPI_Irecv_c(%arg0, %arg1, %{{.*}}, %[[REQ]]) : (!llvm.ptr, i64, i32, i32, i32, i32, !llvm.ptr) -> i32
func.func @unknown_count(%buf: !llvm.ptr, %count: i64) {
  %src = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e = "spmd.recv"(%comm, %buf, %count, %dt, %src, %tag) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  return
}

// One count that may not fit is enough: every count of the call becomes an
// MPI_Count, the one that would have fitted included.
// CHECK-LABEL: func.func @mixed_counts
// CHECK:         llvm.call @MPI_Sendrecv_c(%arg0, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %arg1, %arg3, %{{.*}}) : (!llvm.ptr, i64, i32, i32, i32, !llvm.ptr, i64, i32, i32, i32, i32, !llvm.ptr) -> i32
func.func @mixed_counts(%out: !llvm.ptr, %in: !llvm.ptr, %n: i32, %big: i64) {
  %small = arith.extui %n : i32 to i64
  %stride = arith.constant 1 : i64
  %peer = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.sendrecv"(%comm, %out, %small, %dt, %stride, %peer, %tag, %in, %big, %dt, %stride, %peer, %tag) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32) -> !spmd.error
  return
}
