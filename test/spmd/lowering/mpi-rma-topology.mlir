// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// PSCW window synchronisation, Cartesian topology, and the lock-type constant.
// !spmd.win, !spmd.comm and !spmd.lock are all plain integer handles at the
// MPICH ABI, so they lower to i32 - a window that arrives as a block argument
// keeps an explicit cast, since only the ops this pass rewrites get converted.

// CHECK-LABEL: func.func @pscw
// CHECK:         %[[WIN:.*]] = builtin.unrealized_conversion_cast %arg0 : !spmd.win to i32
// CHECK:         llvm.call @MPI_Win_start(%arg1, %{{.*}}, %[[WIN]]) : (i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Win_complete(%[[WIN]]) : (i32) -> i32
func.func @pscw(%win: !spmd.win, %group: i32) {
  %assert = arith.constant 0 : i32
  %e0 = "spmd.winStart"(%group, %assert, %win) <{usedModel = 0 : i32}> : (i32, i32, !spmd.win) -> !spmd.error
  %e1 = "spmd.winComplete"(%win) <{usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
  return
}

// MPI_LOCK_EXCLUSIVE is 234, MPI_LOCK_SHARED is 235.
// CHECK-LABEL: func.func @lock_types
// CHECK:         %[[EXCL:.*]] = llvm.mlir.constant(234 : i32)
// CHECK:         llvm.call @MPI_Win_lock(%[[EXCL]],
// CHECK:         %[[SHARED:.*]] = llvm.mlir.constant(235 : i32)
// CHECK:         llvm.call @MPI_Win_lock(%[[SHARED]],
// CHECK:         llvm.call @MPI_Win_unlock(
func.func @lock_types(%win: !spmd.win) {
  %rank = arith.constant 0 : i32
  %assert = arith.constant 0 : i32
  %excl = "spmd.constLockTypeOp"() <{stringAttr = "EXCLUSIVE", usedModel = 0 : i32}> : () -> !spmd.lock
  %e0 = "spmd.lock"(%rank, %win, %excl, %assert) <{usedModel = 0 : i32}> : (i32, !spmd.win, !spmd.lock, i32) -> !spmd.error
  %shared = "spmd.constLockTypeOp"() <{stringAttr = "SHARED", usedModel = 0 : i32}> : () -> !spmd.lock
  %e1 = "spmd.lock"(%rank, %win, %shared, %assert) <{usedModel = 0 : i32}> : (i32, !spmd.win, !spmd.lock, i32) -> !spmd.error
  %e2 = "spmd.unlock"(%rank, %win) <{usedModel = 0 : i32}> : (i32, !spmd.win) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @topology
// CHECK:         llvm.call @MPI_Dims_create(%{{.*}}, %{{.*}}, %arg0) : (i32, i32, !llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Cart_create(%{{.*}}, %{{.*}}, %arg0, %arg1, %{{.*}}, %[[CART:[0-9]+]]) : (i32, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK:         %[[COMM:.*]] = llvm.load %[[CART]]
// CHECK:         llvm.call @MPI_Barrier(%[[COMM]])
func.func @topology(%dims: !llvm.ptr, %periods: !llvm.ptr) {
  %nproc = arith.constant 4 : i32
  %ndims = arith.constant 2 : i32
  %reorder = arith.constant 0 : i32
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.dimsCreate"(%nproc, %ndims, %dims) <{usedModel = 0 : i32}> : (i32, i32, !llvm.ptr) -> !spmd.error
  %cart, %e1 = "spmd.cartCreate"(%comm, %ndims, %dims, %periods, %reorder) <{usedModel = 0 : i32}> : (!spmd.comm, i32, !llvm.ptr, !llvm.ptr, i32) -> (!spmd.comm, !spmd.error)
  %e2 = "spmd.barrier"(%cart) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
