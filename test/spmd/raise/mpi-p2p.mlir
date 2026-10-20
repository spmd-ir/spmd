// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// Point-to-point MPI calls: every send mode, blocking and nonblocking, and the
// receives. What has to survive the raise is the envelope (communicator, peer,
// tag), the buffer as a memref of the storage the call names, the count
// widened to i64, and the mode as attributes:
//  - buffered (MPI_Bsend) and ready (MPI_Rsend) modes are recorded,
//  - synchronous mode (MPI_Ssend) is not: it raises like MPI_Send, which the
//    standard permits an implementation to be;
//  - a nonblocking call yields a request, written back into the MPI_Request
//    the program passed.
// The peer is 3, the tag 7 and the count 5, so a swap shows.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK:         %[[W:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
// CHECK:         %[[BUF:.*]] = llvm.alloca %c5_i32 x i32
// CHECK:         %[[REQ:.*]] = llvm.alloca %c1_i32 x i32
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %c1_i64 = arith.constant 1 : i64
  // MPI_INT and MPI_COMM_WORLD in the bundled MPICH ABI; MPI_STATUS_IGNORE.
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %ignore = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %buf = llvm.alloca %c5 x i32 : (i32) -> !llvm.ptr
  %req = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %i = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[M0:.*]] = "enzymexla.pointer2memref"(%[[BUF]]) : (!llvm.ptr) -> memref<5xi32>
  // CHECK: "spmd.send"(%[[W]], %[[M0]], %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}>
  // CHECK-SAME: (!spmd.comm, memref<5xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %0 = llvm.call @MPI_Send(%buf, %c5, %int, %c3, %c7, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  // CHECK: "spmd.send"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = true, isBuffered = true, isReadyMode = false
  %1 = llvm.call @MPI_Bsend(%buf, %c5, %int, %c3, %c7, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  // CHECK: "spmd.send"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = true, isBuffered = false, isReadyMode = false
  %2 = llvm.call @MPI_Ssend(%buf, %c5, %int, %c3, %c7, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  // CHECK: "spmd.send"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = true, isBuffered = false, isReadyMode = true
  %3 = llvm.call @MPI_Rsend(%buf, %c5, %int, %c3, %c7, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32

  // CHECK: %[[M4:.*]] = "enzymexla.pointer2memref"(%[[BUF]]) : (!llvm.ptr) -> memref<5xi32>
  // CHECK: "spmd.recv"(%[[W]], %[[M4]], %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: (!spmd.comm, memref<5xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %4 = llvm.call @MPI_Recv(%buf, %c5, %int, %c3, %c7, %world, %ignore) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

  // CHECK: %[[R6:.*]], %{{.*}} = "spmd.send"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = false, isBuffered = false, isReadyMode = false
  // CHECK-SAME: -> (!spmd.req, !spmd.error)
  // CHECK: %[[S6:.*]] = "enzymexla.pointer2memref"(%[[REQ]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: %[[C6:.*]] = builtin.unrealized_conversion_cast %[[R6]] : !spmd.req to i32
  // CHECK: memref.store %[[C6]], %[[S6]][%c0]
  %6 = llvm.call @MPI_Isend(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: %[[R7:.*]], %{{.*}} = "spmd.send"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = false, isBuffered = true, isReadyMode = false
  // CHECK: %[[C7:.*]] = builtin.unrealized_conversion_cast %[[R7]] : !spmd.req to i32
  // CHECK: memref.store %[[C7]]
  %7 = llvm.call @MPI_Ibsend(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: %[[R8:.*]], %{{.*}} = "spmd.send"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = false, isBuffered = false, isReadyMode = false
  // CHECK: %[[C8:.*]] = builtin.unrealized_conversion_cast %[[R8]] : !spmd.req to i32
  // CHECK: memref.store %[[C8]]
  %8 = llvm.call @MPI_Issend(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: %[[R9:.*]], %{{.*}} = "spmd.send"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = false, isBuffered = false, isReadyMode = true
  // CHECK: %[[C9:.*]] = builtin.unrealized_conversion_cast %[[R9]] : !spmd.req to i32
  // CHECK: memref.store %[[C9]]
  %9 = llvm.call @MPI_Irsend(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: %[[R10:.*]], %{{.*}} = "spmd.recv"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: -> (!spmd.req, !spmd.error)
  // CHECK: %[[C10:.*]] = builtin.unrealized_conversion_cast %[[R10]] : !spmd.req to i32
  // CHECK: memref.store %[[C10]]
  %10 = llvm.call @MPI_Irecv(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

  %f = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Send(!llvm.ptr, i32, i32, i32, i32, i32) -> i32
llvm.func @MPI_Bsend(!llvm.ptr, i32, i32, i32, i32, i32) -> i32
llvm.func @MPI_Ssend(!llvm.ptr, i32, i32, i32, i32, i32) -> i32
llvm.func @MPI_Rsend(!llvm.ptr, i32, i32, i32, i32, i32) -> i32
llvm.func @MPI_Recv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Isend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Ibsend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Issend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Irsend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Irecv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
