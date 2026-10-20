// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// One-sided MPI beyond MPI_Put/Get/Accumulate and the window constructors
// (mpi-rma-not-atomic.mlir, lowering/mpi-rma-roundtrip.mlir): passive-target
// locks, flushes, the fence and start/complete of active target, and the
// request-returning and read-modify-write transfers.
//
// A transfer names an origin buffer (8 ints), for the read-modify-write calls
// a result buffer (16 ints) and for compare-and-swap a compare buffer (4
// ints). The origin count is 2, the result count 4, the target count 5, the
// target rank 3 and the displacement 11, so a swap shows. The raiser adds a
// stride of 1 after each count.
//
// A transfer returns before the data has moved (USIS: blocking = false), and
// accumulating ones are atomic per element, which the op records.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK-DAG:     %[[EXCL:.*]] = "spmd.constLockTypeOp"() <{stringAttr = "EXCLUSIVE", usedModel = 0 : i32}>
// CHECK-DAG:     %[[SHARED:.*]] = "spmd.constLockTypeOp"() <{stringAttr = "SHARED", usedModel = 0 : i32}>
// CHECK:         %[[H:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c5 = arith.constant 5 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %c11 = arith.constant 11 : i64
  %c64 = arith.constant 64 : i64
  // MPI_INT, MPI_COMM_WORLD, MPI_INFO_NULL, MPI_SUM, MPI_LOCK_EXCLUSIVE and
  // MPI_LOCK_SHARED in the bundled MPICH ABI, and some group handle.
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %infonull = arith.constant 469762048 : i32
  %sum = arith.constant 1476395011 : i32
  %group = arith.constant 1207959553 : i32
  %excl = arith.constant 234 : i32
  %shared = arith.constant 235 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %obuf = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr
  %rbuf = llvm.alloca %c16 x i32 : (i32) -> !llvm.ptr
  %cbuf = llvm.alloca %c4 x i32 : (i32) -> !llvm.ptr
  %base = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  %winp = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %req = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %init = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %alloc = llvm.call @MPI_Win_allocate(%c64, %c4, %infonull, %world, %base, %winp) : (i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  %win = llvm.load %winp : !llvm.ptr -> i32

  // Passive target: lock(rank, win, lock type, assert).
  // CHECK: %[[W0:.*]] = builtin.unrealized_conversion_cast %[[H]] : i32 to !spmd.win
  // CHECK: "spmd.lock"(%c3_i32, %[[W0]], %[[EXCL]], %c0_i32) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (i32, !spmd.win, !spmd.lock, i32) -> !spmd.error
  %0 = llvm.call @MPI_Win_lock(%excl, %c3, %c0, %win) : (i32, i32, i32, i32) -> i32
  // CHECK: "spmd.unlock"(%c3_i32, %{{.*}}) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (i32, !spmd.win) -> !spmd.error
  %1 = llvm.call @MPI_Win_unlock(%c3, %win) : (i32, i32) -> i32
  // CHECK: "spmd.lock"(%c3_i32, %{{.*}}, %[[SHARED]], %c0_i32)
  %2 = llvm.call @MPI_Win_lock(%shared, %c3, %c0, %win) : (i32, i32, i32, i32) -> i32
  // CHECK: "spmd.flush"(%c3_i32, %{{.*}}) <{isLocal = false, usedModel = 0 : i32}>
  %3 = llvm.call @MPI_Win_flush(%c3, %win) : (i32, i32) -> i32
  // CHECK: "spmd.flush"(%c3_i32, %{{.*}}) <{isLocal = true, usedModel = 0 : i32}>
  %4 = llvm.call @MPI_Win_flush_local(%c3, %win) : (i32, i32) -> i32
  // CHECK: "spmd.unlock"(%c3_i32
  %5 = llvm.call @MPI_Win_unlock(%c3, %win) : (i32, i32) -> i32
  // CHECK: "spmd.lockAll"(%{{.*}}, %c0_i32) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.win, i32) -> !spmd.error
  %6 = llvm.call @MPI_Win_lock_all(%c0, %win) : (i32, i32) -> i32
  // CHECK: "spmd.flushAll"(%{{.*}}) <{isLocal = false, usedModel = 0 : i32}>
  %7 = llvm.call @MPI_Win_flush_all(%win) : (i32) -> i32
  // CHECK: "spmd.flushAll"(%{{.*}}) <{isLocal = true, usedModel = 0 : i32}>
  %8 = llvm.call @MPI_Win_flush_local_all(%win) : (i32) -> i32

  // CHECK: %[[R9:.*]], %{{.*}} = "spmd.put"(%{{.*}}, %c2_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c11_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (memref<8xi32>, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q9:.*]] = builtin.unrealized_conversion_cast %[[R9]] : !spmd.req to i32
  // CHECK: memref.store %[[Q9]]
  %9 = llvm.call @MPI_Rput(%obuf, %c2, %int, %c3, %c11, %c5, %int, %win, %req) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: %[[R10:.*]], %{{.*}} = "spmd.get"(%{{.*}}, %c2_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c11_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (memref<16xi32>, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q10:.*]] = builtin.unrealized_conversion_cast %[[R10]] : !spmd.req to i32
  // CHECK: memref.store %[[Q10]]
  %10 = llvm.call @MPI_Rget(%rbuf, %c2, %int, %c3, %c11, %c5, %int, %win, %req) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: %[[R11:.*]], %{{.*}} = "spmd.accumulate"(%{{.*}}, %c2_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c11_i64, %c5_i64, %{{.*}}, %c1_i64, %{{.*}}) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (memref<8xi32>, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64, !spmd.op) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q11:.*]] = builtin.unrealized_conversion_cast %[[R11]] : !spmd.req to i32
  // CHECK: memref.store %[[Q11]]
  %11 = llvm.call @MPI_Raccumulate(%obuf, %c2, %int, %c3, %c11, %c5, %int, %sum, %win, %req) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32, !llvm.ptr) -> i32

  // CHECK: "spmd.getAccumulate"(%{{.*}}, %c2_i64, %{{.*}}, %c1_i64, %{{.*}}, %c4_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c11_i64, %c5_i64, %{{.*}}, %c1_i64, %{{.*}}) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (memref<8xi32>, i64, !spmd.datatype, i64, memref<16xi32>, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64, !spmd.op) -> !spmd.error
  %12 = llvm.call @MPI_Get_accumulate(%obuf, %c2, %int, %rbuf, %c4, %int, %c3, %c11, %c5, %int, %sum, %win) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32) -> i32
  // CHECK: %[[R13:.*]], %{{.*}} = "spmd.getAccumulate"(%{{.*}}, %c2_i64, %{{.*}}, %c1_i64, %{{.*}}, %c4_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c11_i64, %c5_i64, %{{.*}}, %c1_i64, %{{.*}}) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q13:.*]] = builtin.unrealized_conversion_cast %[[R13]] : !spmd.req to i32
  // CHECK: memref.store %[[Q13]]
  %13 = llvm.call @MPI_Rget_accumulate(%obuf, %c2, %int, %rbuf, %c4, %int, %c3, %c11, %c5, %int, %sum, %win, %req) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.compareAndSwap"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %c3_i32, %{{.*}}, %c11_i64) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (memref<8xi32>, memref<4xi32>, memref<16xi32>, !spmd.datatype, i32, !spmd.win, i64) -> !spmd.error
  %14 = llvm.call @MPI_Compare_and_swap(%obuf, %cbuf, %rbuf, %int, %c3, %c11, %win) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i64, i32) -> i32
  // CHECK: "spmd.fetchAndOp"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %c3_i32, %{{.*}}, %c11_i64) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (memref<8xi32>, memref<16xi32>, !spmd.datatype, !spmd.op, i32, !spmd.win, i64) -> !spmd.error
  %15 = llvm.call @MPI_Fetch_and_op(%obuf, %rbuf, %int, %c3, %c11, %sum, %win) : (!llvm.ptr, !llvm.ptr, i32, i32, i64, i32, i32) -> i32

  // CHECK: "spmd.unlockAll"(%{{.*}}) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.win) -> !spmd.error
  %16 = llvm.call @MPI_Win_unlock_all(%win) : (i32) -> i32

  // Active target.
  // CHECK: "spmd.fence"(%{{.*}}, %c0_i32) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.win, i32) -> !spmd.error
  %17 = llvm.call @MPI_Win_fence(%c0, %win) : (i32, i32) -> i32
  // The group stays the program's integer handle.
  // CHECK: "spmd.winStart"(%c1207959553_i32, %c0_i32, %{{.*}}) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (i32, i32, !spmd.win) -> !spmd.error
  %18 = llvm.call @MPI_Win_start(%group, %c0, %win) : (i32, i32, i32) -> i32
  // CHECK: "spmd.winComplete"(%{{.*}}) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.win) -> !spmd.error
  %19 = llvm.call @MPI_Win_complete(%win) : (i32) -> i32

  %fin = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Win_allocate(i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Win_lock(i32, i32, i32, i32) -> i32
llvm.func @MPI_Win_unlock(i32, i32) -> i32
llvm.func @MPI_Win_flush(i32, i32) -> i32
llvm.func @MPI_Win_flush_local(i32, i32) -> i32
llvm.func @MPI_Win_lock_all(i32, i32) -> i32
llvm.func @MPI_Win_flush_all(i32) -> i32
llvm.func @MPI_Win_flush_local_all(i32) -> i32
llvm.func @MPI_Rput(!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Rget(!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Raccumulate(!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Get_accumulate(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32) -> i32
llvm.func @MPI_Rget_accumulate(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i64, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Compare_and_swap(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i64, i32) -> i32
llvm.func @MPI_Fetch_and_op(!llvm.ptr, !llvm.ptr, i32, i32, i64, i32, i32) -> i32
llvm.func @MPI_Win_unlock_all(i32) -> i32
llvm.func @MPI_Win_fence(i32, i32) -> i32
llvm.func @MPI_Win_start(i32, i32, i32) -> i32
llvm.func @MPI_Win_complete(i32) -> i32
llvm.func @MPI_Finalize() -> i32
