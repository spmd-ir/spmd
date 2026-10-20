// RUN: spmd-opt %s -convert-apiCalls-to-spmd -lower-spmd-to-mpi | FileCheck %s --implicit-check-not='"spmd.'
//
// Raising and lowering compose: an MPI program raised by
// -convert-apiCalls-to-spmd lowers back to the calls it was raised from, with
// the same predefined handles, and the spmd.parallel the raiser wraps around
// main is gone.  What is left besides the calls - the memref views of the
// program's variables and the casts between them and the handles - is the
// raiser's, and -spmd-reconcile-unrealized-cast and the memref lowering remove
// it.

// CHECK-DAG: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Isend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK-DAG: llvm.func @MPI_Finalize() -> i32

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:       ^bb1:
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[INT:.*]] = llvm.mlir.constant(1275069445 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:         llvm.call @MPI_Init(%[[NULL]], %[[NULL]])
// CHECK:         llvm.call @MPI_Comm_rank(%[[WORLD]], %{{.*}})
// CHECK:         llvm.call @MPI_Isend(%{{.*}}, %{{.*}}, %[[INT]], %{{.*}}, %{{.*}}, %[[WORLD]], %{{.*}})
// CHECK:         llvm.call @MPI_Wait(%{{.*}}, %{{.*}})
// CHECK:         llvm.call @MPI_Allreduce(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[SUM]], %[[WORLD]])
// CHECK:         llvm.call @MPI_Finalize()
// CHECK:         llvm.return
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  // MPI_INT, MPI_SUM and MPI_COMM_WORLD in the bundled MPICH ABI;
  // MPI_STATUS_IGNORE.
  %int = arith.constant 1275069445 : i32
  %sum = arith.constant 1476395011 : i32
  %world = arith.constant 1140850688 : i32
  %ignore = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %rank = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %buf = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %out = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %req = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Comm_rank(%world, %rank) : (i32, !llvm.ptr) -> i32
  %2 = llvm.call @MPI_Isend(%buf, %c1_i32, %int, %c0_i32, %c0_i32, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  %3 = llvm.call @MPI_Wait(%req, %ignore) : (!llvm.ptr, !llvm.ptr) -> i32
  %4 = llvm.call @MPI_Allreduce(%buf, %out, %c1_i32, %int, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %5 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
llvm.func @MPI_Isend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Finalize() -> i32
