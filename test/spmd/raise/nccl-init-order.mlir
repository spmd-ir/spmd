// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s
//
// ncclCommInitRank initialises NCCL and hands out the world communicator at
// once; the raiser splits it into spmd.init and spmd.commWorld.  The init has
// to come first: the communicator does not exist before it, and the NCCL
// lowering reads it from where the init leaves it.  The raiser used to put the
// init after the communicator it creates.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.init"(%{{.*}}) <{usedModel = 0 : i32}>
// CHECK:         %[[SIZE:.*]], %{{.*}} = "spmd.getSizeOfComm"
// CHECK:         "spmd.init"(%{{.*}}) <{usedModel = 3 : i32}>
// CHECK-NEXT:    %[[COMM:.*]] = "spmd.commWorld"() <{usedModel = 3 : i32}>
// CHECK:         "spmd.getSizeOfComm"(%{{.*}}) <{usedModel = 3 : i32}>
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c128_i32 = arith.constant 128 : i32
  // MPI_COMM_WORLD and MPI_BYTE in the bundled MPICH ABI.
  %world = arith.constant 1140850688 : i32
  %byte = arith.constant 1275068685 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %sizeSlot = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %rankSlot = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %id = llvm.alloca %c1_i32 x !llvm.struct<(array<128 x i8>)> : (i32) -> !llvm.ptr
  %commSlot = llvm.alloca %c1_i32 x !llvm.ptr : (i32) -> !llvm.ptr
  %countSlot = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Comm_size(%world, %sizeSlot) : (i32, !llvm.ptr) -> i32
  %2 = llvm.call @MPI_Comm_rank(%world, %rankSlot) : (i32, !llvm.ptr) -> i32
  %3 = llvm.call @ncclGetUniqueId(%id) : (!llvm.ptr) -> i32
  %4 = llvm.call @MPI_Bcast(%id, %c128_i32, %byte, %c0_i32, %world) : (!llvm.ptr, i32, i32, i32, i32) -> i32
  // Read the way reactant-clang leaves a variable: through a memref view.
  // That is the form the raiser traces the size of the world communicator
  // through.
  %sizeView = "enzymexla.pointer2memref"(%sizeSlot) : (!llvm.ptr) -> memref<1xi32>
  %size = memref.load %sizeView[%c0] : memref<1xi32>
  %rankView = "enzymexla.pointer2memref"(%rankSlot) : (!llvm.ptr) -> memref<1xi32>
  %rank = memref.load %rankView[%c0] : memref<1xi32>
  %5 = llvm.call @ncclCommInitRank(%commSlot, %size, %id, %rank) : (!llvm.ptr, i32, !llvm.ptr, i32) -> i32
  %comm = llvm.load %commSlot : !llvm.ptr -> !llvm.ptr
  %6 = llvm.call @ncclCommCount(%comm, %countSlot) : (!llvm.ptr, !llvm.ptr) -> i32
  %7 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Comm_size(i32, !llvm.ptr) -> i32
llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
llvm.func @MPI_Bcast(!llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Finalize() -> i32
llvm.func @ncclGetUniqueId(!llvm.ptr) -> i32
llvm.func @ncclCommInitRank(!llvm.ptr, i32, !llvm.ptr, i32) -> i32
llvm.func @ncclCommCount(!llvm.ptr, !llvm.ptr) -> i32
