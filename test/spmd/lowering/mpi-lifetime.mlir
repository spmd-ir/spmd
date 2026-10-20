// RUN: spmd-opt %s -convert-apiCalls-to-spmd -lower-spmd-to-mpi | FileCheck %s
//
// Lifetime markers are the program's, and survive the round trip.  roundtrip's
// lowering deleted every llvm.intr.lifetime.start/end in the module after the
// conversion, because markers once ended up using the base address of
// MPI_Win_allocate before the call that defines it.  The raiser no longer
// substitutes that result for the program's variable: it stores it into the
// variable, and the markers keep naming the variable's own alloca, which
// dominates them.  The lowering replaces spmd operations in place and cannot
// move a use in front of its definition, so there is nothing to strip, and the
// output verifies with the markers where the program put them.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:       ^bb1:
// CHECK:         %[[BASE:.*]] = llvm.alloca %{{.*}} x !llvm.ptr
// CHECK:         %[[WIN:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         llvm.call @MPI_Init(
// CHECK:         llvm.intr.lifetime.start %[[BASE]] : !llvm.ptr
// CHECK:         llvm.intr.lifetime.start %[[WIN]] : !llvm.ptr
// CHECK:         llvm.call @MPI_Win_allocate(
// CHECK:         llvm.load %[[BASE]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @MPI_Win_free(
// CHECK:         llvm.intr.lifetime.end %[[WIN]] : !llvm.ptr
// CHECK:         llvm.intr.lifetime.end %[[BASE]] : !llvm.ptr
// CHECK:         llvm.call @MPI_Finalize()
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c64_i64 = arith.constant 64 : i64
  // MPI_INFO_NULL and MPI_COMM_WORLD in the bundled MPICH ABI.
  %info = arith.constant 469762048 : i32
  %world = arith.constant 1140850688 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %base = llvm.alloca %c1_i32 x !llvm.ptr : (i32) -> !llvm.ptr
  %win = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  llvm.intr.lifetime.start %base : !llvm.ptr
  llvm.intr.lifetime.start %win : !llvm.ptr
  %1 = llvm.call @MPI_Win_allocate(%c64_i64, %c4_i32, %info, %world, %base, %win) : (i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  %p = llvm.load %base : !llvm.ptr -> !llvm.ptr
  llvm.store %c1_i32, %p : i32, !llvm.ptr
  %2 = llvm.call @MPI_Win_free(%win) : (!llvm.ptr) -> i32
  llvm.intr.lifetime.end %win : !llvm.ptr
  llvm.intr.lifetime.end %base : !llvm.ptr
  %3 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Win_allocate(i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Win_free(!llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
