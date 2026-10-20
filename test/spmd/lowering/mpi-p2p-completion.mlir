// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// Point-to-point receives and non-blocking sends, and the completion calls on
// one request.  A blocking receive passes MPI_STATUS_IGNORE, which MPICH
// spells (MPI_Status *)1; MPI_Wait and MPI_Test take the request by address
// and write a status only when the op has a status result to put it in.

// CHECK-LABEL: func.func @recv
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[INT:.*]] = llvm.mlir.constant(1275069445 : i32) : i32
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[IGNORE:.*]] = llvm.inttoptr %[[ONE]] : i64 to !llvm.ptr
// CHECK:         llvm.call @MPI_Recv(%arg0, %{{.*}}, %[[INT]], %arg2, %arg3, %[[WORLD]], %[[IGNORE]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Irecv(%arg0, %{{.*}}, %[[INT]], %arg2, %arg3, %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @recv(%buf: !llvm.ptr, %n: i32, %src: i32, %tag: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.recv"(%world, %buf, %count, %dt, %src, %tag) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %req, %e1 = "spmd.recv"(%world, %buf, %count, %dt, %src, %tag) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @isend
// CHECK:         llvm.call @MPI_Isend(%arg0, %{{.*}}, %{{.*}}, %arg2, %arg3, %{{.*}}, %[[REQ:[0-9]+]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
// CHECK:         llvm.call @MPI_Irsend(%arg0, %{{.*}}, %{{.*}}, %arg2, %arg3, %{{.*}}, %[[RREQ:[0-9]+]]) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[RREQ]] : !llvm.ptr -> i32
func.func @isend(%buf: !llvm.ptr, %n: i32, %dest: i32, %tag: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %count = arith.extui %n : i32 to i64
  %r0, %e0 = "spmd.send"(%world, %buf, %count, %dt, %dest, %tag) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %r1, %e1 = "spmd.send"(%world, %buf, %count, %dt, %dest, %tag) <{isBlocking = false, isBuffered = false, isReadyMode = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  return
}

// A status result is five ints of stack storage, and that storage is what the
// status result becomes.
// CHECK-LABEL: func.func @wait_with_status
// CHECK:         %[[STATUS:.*]] = llvm.alloca %{{.*}} x !llvm.array<5 x i32>
// CHECK:         %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[STATUS]] : !llvm.ptr to !spmd.status
// CHECK:         llvm.store %[[REQ:[0-9]+]], %[[REQPTR:[0-9]+]] : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Wait(%[[REQPTR]], %[[STATUS]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         return %[[RESULT]] : !spmd.status
func.func @wait_with_status(%req: !spmd.req) -> !spmd.status {
  %st, %e = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> (!spmd.status, !spmd.error)
  return %st : !spmd.status
}

// The flag MPI writes is an int; the op's flag is that int compared with zero.
// CHECK-LABEL: func.func @test
// CHECK:         %[[STATUS:.*]] = llvm.alloca %{{.*}} x !llvm.array<5 x i32>
// CHECK:         %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[STATUS]] : !llvm.ptr to !spmd.status
// CHECK:         llvm.store %[[REQ:[0-9]+]], %[[REQPTR:[0-9]+]] : i32, !llvm.ptr
// CHECK:         %[[IGNORE:.*]] = llvm.inttoptr
// CHECK:         llvm.call @MPI_Test(%[[REQPTR]], %[[FLAGPTR:[0-9]+]], %[[IGNORE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[FLAG:.*]] = llvm.load %[[FLAGPTR]] : !llvm.ptr -> i32
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[SET:.*]] = llvm.icmp "ne" %[[FLAG]], %[[ZERO]] : i32
// CHECK:         llvm.store %[[REQ]], %[[REQPTR2:[0-9]+]] : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Test(%[[REQPTR2]], %[[FLAGPTR2:[0-9]+]], %[[STATUS]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.store %[[REQ]], %[[REQPTR3:[0-9]+]] : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Request_free(%[[REQPTR3]]) : (!llvm.ptr) -> i32
// CHECK:         return %[[SET]], %{{.*}}, %[[RESULT]] : i1, i1, !spmd.status
func.func @test(%req: !spmd.req) -> (i1, i1, !spmd.status) {
  %f0, %e0 = "spmd.test"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> (i1, !spmd.error)
  %f1, %st, %e1 = "spmd.test"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> (i1, !spmd.status, !spmd.error)
  %e2 = "spmd.reqFree"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return %f0, %f1, %st : i1, i1, !spmd.status
}
