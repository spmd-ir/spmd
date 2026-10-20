// RUN: spmd-opt %s -convert-apiCalls-to-spmd -lower-spmd="models=mpi=shmem" | FileCheck %s --implicit-check-not='"spmd.' --implicit-check-not="@MPI_"
//
// An MPI program translated to OpenSHMEM: raised, retargeted by the driver and
// lowered with OpenSHMEM's calls.  The predefined handles come out in
// OpenSHMEM's form -- MPI_COMM_WORLD is the SHMEM_TEAM_WORLD variable, which
// the rank and size queries need not even pass -- and MPI's reductions over
// its datatypes are the typed reductions of the same element type, signedness
// included: MPI_UNSIGNED's maximum is shmem_uint_max_reduce, not the
// shmem_int_max_reduce that compares differently.  MPI_BYTE says nothing about
// signedness, and its bitwise and is the same either way.  A bcast's
// one buffer is both of OpenSHMEM's.  The world barrier is shmem_sync_all.
//
// The translation is exactly as good as OpenSHMEM's view of the program's
// memory: every buffer here comes from malloc, and OpenSHMEM only reaches the
// symmetric heap.  That is the program's business, as it is in OpenSHMEM.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         llvm.call @shmem_init() : () -> ()
// CHECK:         llvm.call @shmem_my_pe() : () -> i32
// CHECK:         llvm.call @shmem_n_pes() : () -> i32
// CHECK:         llvm.call @shmem_int_sum_reduce(%{{.*}}, %[[OUT:.*]], %[[BUF:.*]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @shmem_double_max_reduce(%{{.*}}, %[[OUT]], %[[BUF]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @shmem_uint_max_reduce(%{{.*}}, %[[OUT]], %[[BUF]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @shmem_ulong_min_reduce(%{{.*}}, %[[OUT]], %[[BUF]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @shmem_int8_and_reduce(%{{.*}}, %[[OUT]], %[[BUF]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @shmem_int_broadcast(%{{.*}}, %[[BUF]], %[[BUF]], %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK:         llvm.call @shmem_sync_all() : () -> ()
// CHECK:         llvm.call @shmem_finalize() : () -> ()
// CHECK:         llvm.return
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c16_i64 = arith.constant 16 : i64
  // MPI_INT, MPI_DOUBLE, MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_BYTE, MPI_SUM,
  // MPI_MAX, MPI_MIN, MPI_BAND and MPI_COMM_WORLD in the bundled MPICH ABI.
  %int = arith.constant 1275069445 : i32
  %double = arith.constant 1275070475 : i32
  %unsigned = arith.constant 1275069446 : i32
  %ulong = arith.constant 1275070472 : i32
  %byte = arith.constant 1275068685 : i32
  %sum = arith.constant 1476395011 : i32
  %max = arith.constant 1476395009 : i32
  %min = arith.constant 1476395010 : i32
  %band = arith.constant 1476395014 : i32
  %world = arith.constant 1140850688 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %rank = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %size = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %buf = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
  %out = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Comm_rank(%world, %rank) : (i32, !llvm.ptr) -> i32
  %2 = llvm.call @MPI_Comm_size(%world, %size) : (i32, !llvm.ptr) -> i32
  %3 = llvm.call @MPI_Allreduce(%buf, %out, %c4_i32, %int, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %4 = llvm.call @MPI_Allreduce(%buf, %out, %c1_i32, %double, %max, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %8 = llvm.call @MPI_Allreduce(%buf, %out, %c4_i32, %unsigned, %max, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %9 = llvm.call @MPI_Allreduce(%buf, %out, %c1_i32, %ulong, %min, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %10 = llvm.call @MPI_Allreduce(%buf, %out, %c16_i32, %byte, %band, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  %5 = llvm.call @MPI_Bcast(%buf, %c4_i32, %int, %c0_i32, %world) : (!llvm.ptr, i32, i32, i32, i32) -> i32
  %6 = llvm.call @MPI_Barrier(%world) : (i32) -> i32
  %7 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
llvm.func @MPI_Comm_size(i32, !llvm.ptr) -> i32
llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Bcast(!llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Barrier(i32) -> i32
llvm.func @MPI_Finalize() -> i32
