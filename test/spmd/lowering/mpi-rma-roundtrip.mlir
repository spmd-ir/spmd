// RUN: spmd-opt %s -convert-apiCalls-to-spmd -lower-spmd-to-mpi | FileCheck %s --implicit-check-not='"spmd.'
//
// A one-sided transfer raised by -convert-apiCalls-to-spmd lowers back to the
// call it was raised from.  It is the round trip $isBlocking used to break:
// MPI marks MPI_Put non-blocking although it hands out no request, so the
// raised spmd.put carries isBlocking = false and has no request result.

// CHECK-DAG: llvm.func @MPI_Win_create(!llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Put(!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
// CHECK-DAG: llvm.func @MPI_Win_free(!llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         %[[INFO:.*]] = llvm.mlir.constant(469762048 : i32) : i32
// CHECK:         llvm.call @MPI_Win_create(%{{.*}}, %{{.*}}, %{{.*}}, %[[INFO]], %{{.*}}, %{{.*}})
// CHECK:         llvm.call @MPI_Put(
// CHECK:         llvm.call @MPI_Win_free(
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  // MPI_INT, MPI_COMM_WORLD and MPI_INFO_NULL in the bundled MPICH ABI.
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %info = arith.constant 469762048 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %base = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %win = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %e0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %e1 = llvm.call @MPI_Win_create(%base, %c8_i64, %c1_i32, %info, %world, %win) : (!llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
  %w = llvm.load %win : !llvm.ptr -> i32
  %e2 = llvm.call @MPI_Put(%base, %c1_i32, %int, %c0_i32, %c1_i64, %c1_i32, %int, %w) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
  %e3 = llvm.call @MPI_Win_free(%win) : (!llvm.ptr) -> i32
  %e4 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Win_create(!llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Put(!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
llvm.func @MPI_Win_free(!llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
