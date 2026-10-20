// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// spmd.sendrecv declares its send and receive counts I64, like every other
// communication op, while MPI's C binding passes them as int. The sendrecv
// entries were the only ones without the i32 -> i64 widening, so both calls
// produced an op that failed verification ("operand #2 must be 64-bit
// signless integer, but got 'i32'").

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
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: "spmd.sendrecv"
  // CHECK-SAME: isBlocking = true
  // CHECK-SAME: (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i64, i32, i32, memref<1xi32>, i64, !spmd.datatype, i64, i32, i32) -> !spmd.error
  %1 = llvm.call @MPI_Sendrecv(%sbuf, %c1_i32, %int, %c1_i32, %c0_i32, %rbuf, %c1_i32, %int, %c0_i32, %c0_i32, %world, %ignore) : (!llvm.ptr, i32, i32, i32, i32, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

  // CHECK: "spmd.sendrecv"
  // CHECK-SAME: isBlocking = true
  // CHECK-SAME: (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i64, i32, i32, memref<1xi32>, i64, !spmd.datatype, i64, i32, i32) -> !spmd.error
  %2 = llvm.call @MPI_Sendrecv_replace(%sbuf, %c1_i32, %int, %c1_i32, %c0_i32, %c0_i32, %c0_i32, %world, %ignore) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

  %3 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Sendrecv(!llvm.ptr, i32, i32, i32, i32, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Sendrecv_replace(!llvm.ptr, i32, i32, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
