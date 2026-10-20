// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// Matched probe/receive.  !spmd.msg is an int handle at the MPICH ABI:
// spmd.mProbe reads back the MPI_Message MPI wrote through its pointer
// argument, and spmd.mRecv / spmd.imRecv hand MPI the handle by address again.

// Without a status result the probe passes MPI_STATUS_IGNORE, which MPICH
// spells as (MPI_Status *)1.
// CHECK-LABEL: func.func @mprobe_mrecv
// CHECK:         %[[IGN:.*]] = llvm.mlir.constant(1 : i64)
// CHECK:         %[[IGNPTR:.*]] = llvm.inttoptr %[[IGN]]
// CHECK:         llvm.call @MPI_Mprobe(%{{.*}}, %{{.*}}, %{{.*}}, %[[MSGPTR:[0-9]+]], %[[IGNPTR]])
// CHECK:         %[[MSG:.*]] = llvm.load %[[MSGPTR]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[MSG]], %[[TMP:[0-9]+]]
// CHECK:         llvm.call @MPI_Mrecv(%{{.*}}, %{{.*}}, %{{.*}}, %[[TMP]], %{{.*}})
func.func @mprobe_mrecv(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %src = arith.constant 0 : i32
  %tag = arith.constant 7 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %msg, %e0 = "spmd.mProbe"(%comm, %src, %tag) <{usedModel = 0 : i32}> : (!spmd.comm, i32, i32) -> (!spmd.msg, !spmd.error)
  %e1 = "spmd.mRecv"(%buf, %count, %dt, %msg) <{isBlocking = true, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, !spmd.msg) -> !spmd.error
  return
}

// A status result means the status needs real storage: MPICH's MPI_Status is
// five ints, and the same address is what the status result becomes.
// CHECK-LABEL: func.func @mprobe_imrecv
// CHECK:         %[[STATUS:.*]] = llvm.alloca %{{.*}} x !llvm.array<5 x i32>
// CHECK:         llvm.call @MPI_Mprobe(%{{.*}}, %{{.*}}, %{{.*}}, %[[MSGPTR:[0-9]+]], %[[STATUS]])
// CHECK:         %[[MSG:.*]] = llvm.load %[[MSGPTR]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[MSG]], %[[TMP:[0-9]+]]
// CHECK:         llvm.call @MPI_Imrecv(%{{.*}}, %{{.*}}, %{{.*}}, %[[TMP]], %[[REQ:[0-9]+]])
// CHECK:         llvm.load %[[REQ]]
// CHECK:         llvm.call @MPI_Wait(
func.func @mprobe_imrecv(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %src = arith.constant 0 : i32
  %tag = arith.constant 7 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %msg, %st, %e0 = "spmd.mProbe"(%comm, %src, %tag) <{usedModel = 0 : i32}> : (!spmd.comm, i32, i32) -> (!spmd.msg, !spmd.status, !spmd.error)
  %req, %e1 = "spmd.imRecv"(%buf, %count, %dt, %msg) <{isBlocking = false, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, !spmd.msg) -> (!spmd.req, !spmd.error)
  %e2 = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}
