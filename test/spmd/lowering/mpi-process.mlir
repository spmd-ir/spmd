// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// Process and communicator management.  MPI hands back a rank, a size and a
// new communicator through pointer arguments; each gets stack storage that is
// read back as the op's result.  MPI_Comm_free takes its communicator by
// address.

// CHECK-LABEL: func.func @process
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         llvm.call @MPI_Comm_rank(%[[WORLD]], %[[RANKPTR:[0-9]+]]) : (i32, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[RANK:.*]] = llvm.load %[[RANKPTR]] : !llvm.ptr -> i32
// CHECK:         llvm.call @MPI_Comm_size(%[[WORLD]], %[[SIZEPTR:[0-9]+]]) : (i32, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[SIZE:.*]] = llvm.load %[[SIZEPTR]] : !llvm.ptr -> i32
// CHECK:         llvm.call @MPI_Comm_split(%[[WORLD]], %[[RANK]], %[[SIZE]], %[[SPLITPTR:[0-9]+]]) : (i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[SPLIT:.*]] = llvm.load %[[SPLITPTR]] : !llvm.ptr -> i32
// CHECK:         llvm.call @MPI_Comm_dup(%[[SPLIT]], %[[DUPPTR:[0-9]+]]) : (i32, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[DUP:.*]] = llvm.load %[[DUPPTR]] : !llvm.ptr -> i32
// CHECK:         llvm.store %[[DUP]], %[[FREEPTR:[0-9]+]] : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Comm_free(%[[FREEPTR]]) : (!llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Abort(%[[WORLD]], %arg0) : (i32, i32) -> i32
// CHECK:         llvm.call @MPI_Finalize() : () -> i32
// CHECK:         return %[[RANK]], %[[SIZE]] : i32, i32
func.func @process(%code: i32) -> (i32, i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %rank, %e0 = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %size, %e1 = "spmd.getSizeOfComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %split, %e2 = "spmd.commSplit"(%world, %rank, %size) <{usedModel = 0 : i32}> : (!spmd.comm, i32, i32) -> (!spmd.comm, !spmd.error)
  %dup, %e3 = "spmd.commDup"(%split) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %e4 = "spmd.commDestroy"(%dup) <{usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e5 = "spmd.abort"(%world, %code) <{usedModel = 0 : i32}> : (!spmd.comm, i32) -> !spmd.error
  %e6 = "spmd.finalize"() <{usedModel = 0 : i32}> : () -> !spmd.error
  return %rank, %size : i32, i32
}
