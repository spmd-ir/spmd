// RUN: spmd-opt %s -convert-apiCalls-to-spmd -lower-spmd | FileCheck %s --implicit-check-not='"spmd.'
//
// Raising and lowering compose for a program that runs NCCL under MPI, the way
// NCCL programs are written: MPI starts the processes, rank 0's unique id goes
// to the others by MPI_Bcast, and ncclCommInitRank makes the communicator.
// -lower-spmd lowers both models, and the calls come back with the same
// datatypes, reductions, communicator and stream, and the send and receive of
// the group back in one group.
//
// The raiser keeps neither the size nor the id nor the rank of
// ncclCommInitRank, so the lowered program bootstraps NCCL on its own and the
// program's ncclGetUniqueId and MPI_Bcast of its own id stay as they were,
// unused.  The request array the raiser made for the group stays too, written
// and never read; neither is this lowering's to remove.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         llvm.call @MPI_Init(
// CHECK:         llvm.call @MPI_Bcast(
// CHECK:         llvm.call @cudaStreamCreate(%[[STREAMSLOT:[0-9]+]])
// CHECK:         llvm.call @__spmd_nccl_bootstrap() : () -> i32
// CHECK:         %[[STREAM:.*]] = llvm.load %[[STREAMSLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @ncclGroupStart() : () -> i32
// CHECK-NEXT:    llvm.call @ncclSend(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[COMM:[0-9]+]], %[[STREAM]])
// CHECK:         llvm.call @ncclGroupStart() : () -> i32
// CHECK-NEXT:    llvm.call @ncclRecv(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[COMM]], %[[STREAM]])
// CHECK:         llvm.call @ncclGroupEnd() : () -> i32
// CHECK-NEXT:    llvm.call @ncclGroupEnd() : () -> i32
// CHECK:         llvm.call @ncclAllReduce(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[COMM]], %[[STREAM]])
// CHECK:         llvm.call @cudaStreamSynchronize(%[[STREAM]])
// CHECK:         llvm.call @ncclCommDestroy(%[[COMM]])
// CHECK:         llvm.call @MPI_Finalize()

// CHECK-LABEL: llvm.func internal @__spmd_nccl_bootstrap() -> i32
// CHECK:         llvm.call @MPI_Comm_rank(
// CHECK:         llvm.call @ncclGetUniqueId(
// CHECK:         llvm.call @MPI_Bcast(
// CHECK:         llvm.call @ncclCommInitRank(
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c128_i32 = arith.constant 128 : i32
  %c4_i64 = arith.constant 4 : i64
  %world = arith.constant 1140850688 : i32
  %byte = arith.constant 1275068685 : i32
  %ncclFloat = arith.constant 7 : i32
  %ncclMax = arith.constant 2 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %sizeSlot = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %rankSlot = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %id = llvm.alloca %c1_i32 x !llvm.struct<(array<128 x i8>)> : (i32) -> !llvm.ptr
  %commSlot = llvm.alloca %c1_i32 x !llvm.ptr : (i32) -> !llvm.ptr
  %streamSlot = llvm.alloca %c1_i32 x !llvm.ptr : (i32) -> !llvm.ptr
  %sbuf = llvm.alloca %c1_i32 x !llvm.array<4 x f32> : (i32) -> !llvm.ptr
  %rbuf = llvm.alloca %c1_i32 x !llvm.array<4 x f32> : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Comm_size(%world, %sizeSlot) : (i32, !llvm.ptr) -> i32
  %2 = llvm.call @MPI_Comm_rank(%world, %rankSlot) : (i32, !llvm.ptr) -> i32
  %3 = llvm.call @ncclGetUniqueId(%id) : (!llvm.ptr) -> i32
  %4 = llvm.call @MPI_Bcast(%id, %c128_i32, %byte, %c0_i32, %world) : (!llvm.ptr, i32, i32, i32, i32) -> i32
  %sizeView = "enzymexla.pointer2memref"(%sizeSlot) : (!llvm.ptr) -> memref<1xi32>
  %size = memref.load %sizeView[%c0] : memref<1xi32>
  %rankView = "enzymexla.pointer2memref"(%rankSlot) : (!llvm.ptr) -> memref<1xi32>
  %rank = memref.load %rankView[%c0] : memref<1xi32>
  %5 = llvm.call @cudaStreamCreate(%streamSlot) : (!llvm.ptr) -> i32
  %6 = llvm.call @ncclCommInitRank(%commSlot, %size, %id, %rank) : (!llvm.ptr, i32, !llvm.ptr, i32) -> i32
  %comm = llvm.load %commSlot : !llvm.ptr -> !llvm.ptr
  %stream = llvm.load %streamSlot : !llvm.ptr -> !llvm.ptr
  %7 = llvm.call @ncclGroupStart() : () -> i32
  %8 = llvm.call @ncclSend(%sbuf, %c4_i64, %ncclFloat, %c0_i32, %comm, %stream) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  %9 = llvm.call @ncclRecv(%rbuf, %c4_i64, %ncclFloat, %c0_i32, %comm, %stream) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  %10 = llvm.call @ncclGroupEnd() : () -> i32
  %11 = llvm.call @ncclAllReduce(%sbuf, %rbuf, %c4_i64, %ncclFloat, %ncclMax, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  %12 = llvm.call @cudaStreamSynchronize(%stream) : (!llvm.ptr) -> i32
  %13 = llvm.call @ncclCommDestroy(%comm) : (!llvm.ptr) -> i32
  %14 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Comm_size(i32, !llvm.ptr) -> i32
llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
llvm.func @MPI_Bcast(!llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Finalize() -> i32
llvm.func @cudaStreamCreate(!llvm.ptr) -> i32
llvm.func @cudaStreamSynchronize(!llvm.ptr) -> i32
llvm.func @ncclGetUniqueId(!llvm.ptr) -> i32
llvm.func @ncclCommInitRank(!llvm.ptr, i32, !llvm.ptr, i32) -> i32
llvm.func @ncclGroupStart() -> i32
llvm.func @ncclGroupEnd() -> i32
llvm.func @ncclSend(!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclRecv(!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclAllReduce(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclCommDestroy(!llvm.ptr) -> i32
