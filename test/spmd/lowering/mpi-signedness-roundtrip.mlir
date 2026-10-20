// RUN: spmd-opt %s -convert-apiCalls-to-spmd -cse -lower-spmd-to-mpi | FileCheck %s --implicit-check-not='"spmd.'
//
// A maximum and a minimum over unsigned datatypes raise and lower back to the
// same datatypes.  The raiser used to record MPI_UNSIGNED as the signless i32
// it records MPI_INT as, and the lowering took MPI_INT for it, so rank 0's
// 0x80000000 and rank 1's 1 had the maximum 1.  test/exec/mpi/unsigned-max-min.c
// runs that program.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK-DAG:     %[[UNSIGNED:.*]] = llvm.mlir.constant(1275069446 : i32) : i32
// CHECK-DAG:     %[[ULONG:.*]] = llvm.mlir.constant(1275070472 : i32) : i32
// CHECK-DAG:     %[[UCHAR:.*]] = llvm.mlir.constant(1275068674 : i32) : i32
// CHECK-DAG:     %[[MAX:.*]] = llvm.mlir.constant(1476395009 : i32) : i32
// CHECK-DAG:     %[[MIN:.*]] = llvm.mlir.constant(1476395010 : i32) : i32
// CHECK:         llvm.call @MPI_Allreduce(%{{.*}}, %{{.*}}, %{{.*}}, %[[UNSIGNED]], %[[MAX]], %{{.*}})
// CHECK:         llvm.call @MPI_Allreduce(%{{.*}}, %{{.*}}, %{{.*}}, %[[UNSIGNED]], %[[MIN]], %{{.*}})
// CHECK:         llvm.call @MPI_Allreduce(%{{.*}}, %{{.*}}, %{{.*}}, %[[ULONG]], %[[MAX]], %{{.*}})
// CHECK:         llvm.call @MPI_Allreduce(%{{.*}}, %{{.*}}, %{{.*}}, %[[UCHAR]], %[[MIN]], %{{.*}})
// CHECK:         llvm.return
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_MIN and
  // MPI_COMM_WORLD in the bundled MPICH ABI.
  %unsigned = arith.constant 1275069446 : i32
  %ulong = arith.constant 1275070472 : i32
  %uchar = arith.constant 1275068674 : i32
  %max = arith.constant 1476395009 : i32
  %min = arith.constant 1476395010 : i32
  %world = arith.constant 1140850688 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %buf = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr
  %out = llvm.alloca %c1_i32 x i64 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Allreduce(%buf, %out, %c1_i32, %unsigned, %max, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %2 = llvm.call @MPI_Allreduce(%buf, %out, %c1_i32, %unsigned, %min, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %3 = llvm.call @MPI_Allreduce(%buf, %out, %c1_i32, %ulong, %max, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %4 = llvm.call @MPI_Allreduce(%buf, %out, %c1_i32, %uchar, %min, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %5 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Finalize() -> i32
