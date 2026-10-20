// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s
//
// $isAtomic is what tells OpenSHMEM's atomic single-element store and fetch
// (shmem_<T>_atomic_set, shmem_<T>_atomic_fetch) from its transfers of a
// buffer.  MPI_Put and MPI_Get are transfers of a buffer, and used to raise
// atomic, so an MPI program lowered to OpenSHMEM could only take the one-element
// atomic fetch for its gets: a get of several elements was rejected, and one of
// MPI_CHAR, which has no atomic fetch, failed outright.  The accumulating calls
// stay atomic; MPI makes them atomic per element.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.put"
// CHECK-SAME:    isAtomic = false
// CHECK:         "spmd.get"
// CHECK-SAME:    isAtomic = false
// CHECK:         "spmd.accumulate"
// CHECK-SAME:    isAtomic = true
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i64 = arith.constant 0 : i64
  %c4_i32 = arith.constant 4 : i32
  %c16_i64 = arith.constant 16 : i64
  // MPI_INT, MPI_SUM, MPI_COMM_WORLD and MPI_INFO_NULL in the bundled MPICH ABI.
  %int = arith.constant 1275069445 : i32
  %sum = arith.constant 1476395011 : i32
  %world = arith.constant 1140850688 : i32
  %info = arith.constant 469762048 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %base = llvm.alloca %c1_i32 x !llvm.ptr : (i32) -> !llvm.ptr
  %win = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %buf = llvm.alloca %c4_i32 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Win_allocate(%c16_i64, %c1_i32, %info, %world, %base, %win) : (i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  %w = llvm.load %win : !llvm.ptr -> i32
  %2 = llvm.call @MPI_Put(%buf, %c4_i32, %int, %c0_i32, %c0_i64, %c4_i32, %int, %w) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
  %3 = llvm.call @MPI_Get(%buf, %c4_i32, %int, %c0_i32, %c0_i64, %c4_i32, %int, %w) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
  %4 = llvm.call @MPI_Accumulate(%buf, %c4_i32, %int, %c0_i32, %c0_i64, %c4_i32, %int, %sum, %w) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32) -> i32
  %5 = llvm.call @MPI_Win_free(%win) : (!llvm.ptr) -> i32
  %6 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Win_allocate(i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Put(!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
llvm.func @MPI_Get(!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
llvm.func @MPI_Accumulate(!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32) -> i32
llvm.func @MPI_Win_free(!llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
