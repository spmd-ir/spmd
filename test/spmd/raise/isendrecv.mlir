// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// MPI_Isendrecv and MPI_Isendrecv_replace return a request, like MPI_Isend. Two
// things kept them from raising:
//  - spmd.sendrecv had no $req result, so the op built for MPI_Isendrecv
//    failed verification with "requires one result";
//  - MPI_Isendrecv_replace was derived from "MPI_Sendrecv_Replace", a name not
//    in the table. StringMap::lookup handed back an empty entry, and raising
//    the call crashed spmd-opt.
// The request has to reach the MPI_Request slot, where MPI_Wait reads it back.

// CHECK-LABEL: llvm.func @main
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  // MPI_INT and MPI_COMM_WORLD in the bundled MPICH ABI; MPI_STATUS_IGNORE.
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %ignore = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %sbuf = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %rbuf = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %req = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[R1:.*]], %{{.*}} = "spmd.sendrecv"
  // CHECK-SAME: isBlocking = false
  // CHECK-SAME: -> (!spmd.req, !spmd.error)
  // CHECK: %[[C1:.*]] = builtin.unrealized_conversion_cast %[[R1]] : !spmd.req to i32
  // CHECK: memref.store %[[C1]]
  // CHECK: "spmd.wait"
  %1 = llvm.call @MPI_Isendrecv(%sbuf, %c1_i32, %int, %c1_i32, %c0_i32, %rbuf, %c1_i32, %int, %c0_i32, %c0_i32, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  %2 = llvm.call @MPI_Wait(%req, %ignore) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[R2:.*]], %{{.*}} = "spmd.sendrecv"
  // CHECK-SAME: isBlocking = false
  // CHECK-SAME: -> (!spmd.req, !spmd.error)
  // CHECK: %[[C2:.*]] = builtin.unrealized_conversion_cast %[[R2]] : !spmd.req to i32
  // CHECK: memref.store %[[C2]]
  // CHECK: "spmd.wait"
  %3 = llvm.call @MPI_Isendrecv_replace(%sbuf, %c1_i32, %int, %c1_i32, %c0_i32, %c0_i32, %c0_i32, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  %4 = llvm.call @MPI_Wait(%req, %ignore) : (!llvm.ptr, !llvm.ptr) -> i32

  %5 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Isendrecv(!llvm.ptr, i32, i32, i32, i32, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Isendrecv_replace(!llvm.ptr, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
