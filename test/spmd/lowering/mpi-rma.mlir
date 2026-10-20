// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// Memory allocation and one-sided communication.  MPI_Alloc_mem and
// MPI_Win_allocate hand back the base address through a pointer, and every
// allocation takes MPI_INFO_NULL (0x1c000000).  The one-sided calls take the
// displacement as an MPI_Aint (i64) and the counts as ints, and the R-forms
// append the request.

// CHECK-LABEL: func.func @memory
// CHECK:         %[[BASEPTR:.*]] = llvm.alloca %{{.*}} x !llvm.ptr
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[INFO:.*]] = llvm.mlir.constant(469762048 : i32) : i32
// CHECK:         llvm.call @MPI_Alloc_mem(%arg0, %[[INFO]], %[[BASEPTR]]) : (i64, i32, !llvm.ptr) -> i32
// CHECK:         %[[BASE:.*]] = llvm.load %[[BASEPTR]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @MPI_Free_mem(%[[BASE]]) : (!llvm.ptr) -> i32
func.func @memory(%size: i64) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %mem, %e0 = "spmd.malloc"(%world, %size) <{usedModel = 0 : i32}> : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  %e1 = "spmd.free"(%world, %mem) <{usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @windows
// CHECK:         %[[BASEPTR:.*]] = llvm.alloca %{{.*}} x !llvm.ptr
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[INFO:.*]] = llvm.mlir.constant(469762048 : i32) : i32
// CHECK:         llvm.call @MPI_Win_allocate(%arg1, %arg2, %[[INFO]], %[[WORLD]], %[[BASEPTR]], %[[AWINPTR:[0-9]+]]) : (i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[BASE:.*]] = llvm.load %[[BASEPTR]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[AWIN:.*]] = llvm.load %[[AWINPTR]] : !llvm.ptr -> i32
// CHECK:         %[[INFO2:.*]] = llvm.mlir.constant(469762048 : i32) : i32
// CHECK:         llvm.call @MPI_Win_create(%arg0, %arg1, %arg2, %[[INFO2]], %[[WORLD]], %[[CWINPTR:[0-9]+]]) : (!llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK:         %[[CWIN:.*]] = llvm.load %[[CWINPTR]] : !llvm.ptr -> i32
// CHECK:         llvm.call @MPI_Win_fence(%arg3, %[[AWIN]]) : (i32, i32) -> i32
// CHECK:         llvm.call @MPI_Win_lock_all(%arg3, %[[AWIN]]) : (i32, i32) -> i32
// CHECK:         llvm.call @MPI_Win_unlock_all(%[[AWIN]]) : (i32) -> i32
// CHECK:         llvm.call @MPI_Win_flush(%arg4, %[[AWIN]]) : (i32, i32) -> i32
// CHECK:         llvm.call @MPI_Win_flush_local(%arg4, %[[AWIN]]) : (i32, i32) -> i32
// CHECK:         llvm.call @MPI_Win_flush_all(%[[AWIN]]) : (i32) -> i32
// CHECK:         llvm.call @MPI_Win_flush_local_all(%[[AWIN]]) : (i32) -> i32
// CHECK:         llvm.store %[[CWIN]], %[[FREEPTR:[0-9]+]] : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Win_free(%[[FREEPTR]]) : (!llvm.ptr) -> i32
// CHECK:         return %[[BASE]] : !llvm.ptr
func.func @windows(%base: !llvm.ptr, %size: i64, %unit: i32, %assert: i32, %rank: i32) -> !llvm.ptr {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %abase, %awin, %e0 = "spmd.winAlloc"(%world, %size, %unit) <{usedModel = 0 : i32}> : (!spmd.comm, i64, i32) -> (!llvm.ptr, !spmd.win, !spmd.error)
  %cwin, %e1 = "spmd.winCreate"(%world, %base, %size, %unit) <{usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, i32) -> (!spmd.win, !spmd.error)
  %e2 = "spmd.fence"(%awin, %assert) <{usedModel = 0 : i32}> : (!spmd.win, i32) -> !spmd.error
  %e3 = "spmd.lockAll"(%awin, %assert) <{usedModel = 0 : i32}> : (!spmd.win, i32) -> !spmd.error
  %e4 = "spmd.unlockAll"(%awin) <{usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
  %e5 = "spmd.flush"(%rank, %awin) <{isLocal = false, usedModel = 0 : i32}> : (i32, !spmd.win) -> !spmd.error
  %e6 = "spmd.flush"(%rank, %awin) <{isLocal = true, usedModel = 0 : i32}> : (i32, !spmd.win) -> !spmd.error
  %e7 = "spmd.flushAll"(%awin) <{isLocal = false, usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
  %e8 = "spmd.flushAll"(%awin) <{isLocal = true, usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
  %e9 = "spmd.winFree"(%cwin) <{usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
  return %abase : !llvm.ptr
}

// CHECK-LABEL: func.func @put_get
// CHECK:         %[[WIN:.*]] = builtin.unrealized_conversion_cast %arg5 : !spmd.win to i32
// CHECK:         %[[INT:.*]] = llvm.mlir.constant(1275069445 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Put(%arg0, %{{.*}}, %[[INT]], %arg3, %arg4, %{{.*}}, %[[INT]], %[[WIN]]) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Rput(%arg0, %{{.*}}, %[[INT]], %arg3, %arg4, %{{.*}}, %[[INT]], %[[WIN]], %[[REQ:[0-9]+]]) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Get(%arg0, %{{.*}}, %[[INT]], %arg3, %arg4, %{{.*}}, %[[INT]], %[[WIN]]) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Rget(%arg0, %{{.*}}, %[[INT]], %arg3, %arg4, %{{.*}}, %[[INT]], %[[WIN]], %{{.*}}) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Accumulate(%arg0, %{{.*}}, %[[INT]], %arg3, %arg4, %{{.*}}, %[[INT]], %[[SUM]], %[[WIN]]) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Raccumulate(%arg0, %{{.*}}, %[[INT]], %arg3, %arg4, %{{.*}}, %[[INT]], %[[SUM]], %[[WIN]], %{{.*}}) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Get_accumulate(%arg0, %{{.*}}, %[[INT]], %arg1, %{{.*}}, %[[INT]], %arg3, %arg4, %{{.*}}, %[[INT]], %[[SUM]], %[[WIN]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Rget_accumulate(%arg0, %{{.*}}, %[[INT]], %arg1, %{{.*}}, %[[INT]], %arg3, %arg4, %{{.*}}, %[[INT]], %[[SUM]], %[[WIN]], %{{.*}}) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Fetch_and_op(%arg0, %arg1, %[[INT]], %arg3, %arg4, %[[SUM]], %[[WIN]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i64, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Compare_and_swap(%arg0, %arg2, %arg1, %[[INT]], %arg3, %arg4, %[[WIN]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i64, i32) -> i32
func.func @put_get(%origin: !llvm.ptr, %result: !llvm.ptr, %compare: !llvm.ptr, %rank: i32, %disp: i64, %win: !spmd.win, %n: i32) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %one = arith.constant 1 : i64
  %e0 = "spmd.put"(%origin, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one) <{isAtomic = true, isBlocking = true, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %r1, %e1 = "spmd.put"(%origin, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> (!spmd.req, !spmd.error)
  %e2 = "spmd.get"(%origin, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one) <{isAtomic = true, isBlocking = true, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %r3, %e3 = "spmd.get"(%origin, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> (!spmd.req, !spmd.error)
  %e4 = "spmd.accumulate"(%origin, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one, %op) <{isAtomic = true, isBlocking = true, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64, !spmd.op) -> !spmd.error
  %r5, %e5 = "spmd.accumulate"(%origin, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one, %op) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64, !spmd.op) -> (!spmd.req, !spmd.error)
  %e6 = "spmd.getAccumulate"(%origin, %count, %dt, %one, %result, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one, %op) <{isAtomic = true, isBlocking = true, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, !llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64, !spmd.op) -> !spmd.error
  %r7, %e7 = "spmd.getAccumulate"(%origin, %count, %dt, %one, %result, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one, %op) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, !llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64, !spmd.op) -> (!spmd.req, !spmd.error)
  %e8 = "spmd.fetchAndOp"(%origin, %result, %dt, %op, %rank, %win, %disp) <{isAtomic = true, isBlocking = true, usedModel = 0 : i32}> : (!llvm.ptr, !llvm.ptr, !spmd.datatype, !spmd.op, i32, !spmd.win, i64) -> !spmd.error
  %e9 = "spmd.compareAndSwap"(%origin, %compare, %result, %dt, %rank, %win, %disp) <{isAtomic = true, isBlocking = true, usedModel = 0 : i32}> : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !spmd.datatype, i32, !spmd.win, i64) -> !spmd.error
  return
}
