// RUN: spmd-opt %s -convert-apiCalls-to-spmd -cse | FileCheck %s --implicit-check-not="llvm.call @nvshmem_"
//
// NVSHMEM's host-side setup, symmetric memory, synchronisation, teams and
// team collectives, tagged usedModel = 2. NVSHMEM follows OpenSHMEM
// (shmem-management.mlir, shmem-collectives.mlir), with one difference in the
// program's representation: a team is an int, and NVSHMEM_TEAM_WORLD is 0.
//
// A team that is not a constant stops the raiser (MR 16's description,
// finding 11), so none is here.
//
// The raiser materialises a world communicator for every use; -cse, which
// spmd-verify runs after raising too, folds them into one, the world the
// parallel region runs over.

// CHECK-LABEL: llvm.func @main
// CHECK:         %[[OUTER:.*]] = "spmd.commWorld"() <{usedModel = 2 : i32}>
// CHECK:         "spmd.parallel"(%[[OUTER]]) <{isProcessLevel = true, {{.*}}threadLevel = 1 : i32, usedModel = 2 : i32}> ({
// CHECK:           %[[LEVEL:.*]] = "spmd.constThreadLevel"() <{level = "FUNNELED", usedModel = 2 : i32}>
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c8 = arith.constant 8 : i32
  %c9 = arith.constant 9 : i32
  %c16 = arith.constant 16 : i32
  %c5_i64 = arith.constant 5 : i64
  %c64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  // NVSHMEM_THREAD_FUNNELED, and NVSHMEM_TEAM_WORLD.
  %funneled = arith.constant 1 : i32
  %world = arith.constant 0 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %provided = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %newteam = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr

  // CHECK: %[[GOT:.*]], %{{.*}} = "spmd.init"(%[[LEVEL]]) <{usedModel = 2 : i32}>
  // CHECK: %[[GOTINT:.*]] = "spmd.castThreadLevelToI32"(%[[GOT]])
  // CHECK: memref.store %[[GOTINT]]
  %init = llvm.call @nvshmem_init_thread(%funneled, %provided) : (i32, !llvm.ptr) -> i32
  // CHECK: %[[DEST:.*]] = llvm.alloca %c8_i32 x i32
  // CHECK: %[[SRC:.*]] = llvm.alloca %c16_i32 x i32
  %dest = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr
  %src = llvm.alloca %c16 x i32 : (i32) -> !llvm.ptr
  // CHECK: %[[ME:.*]], %{{.*}} = "spmd.getRankInComm"(%[[OUTER]]) <{usedModel = 2 : i32}>
  // CHECK: %[[N:.*]], %{{.*}} = "spmd.getSizeOfComm"(%[[OUTER]]) <{usedModel = 2 : i32}>
  %me = llvm.call @nvshmem_my_pe() : () -> i32
  %n = llvm.call @nvshmem_n_pes() : () -> i32
  // CHECK: %[[P:.*]], %{{.*}} = "spmd.malloc"(%[[OUTER]], %c64_i64) <{usedModel = 2 : i32}>
  // CHECK: "spmd.free"(%[[OUTER]], %[[P]]) <{usedModel = 2 : i32}>
  %p = llvm.call @nvshmem_malloc(%c64) : (i64) -> !llvm.ptr
  llvm.call @nvshmem_free(%p) : (!llvm.ptr) -> ()

  // CHECK: "spmd.fence"(%c0_i32) <{usedModel = 2 : i32}>
  llvm.call @nvshmem_barrier_all() : () -> ()
  // CHECK: "spmd.barrier"(%[[OUTER]]) <{isBlocking = true, usedModel = 2 : i32}>
  llvm.call @nvshmem_sync_all() : () -> ()
  // CHECK: "spmd.flushAll"() <{isLocal = false, usedModel = 2 : i32}>
  llvm.call @nvshmem_quiet() : () -> ()

  // CHECK: %[[TME:.*]], %{{.*}} = "spmd.getRankInComm"(%[[OUTER]]) <{usedModel = 2 : i32}>
  // CHECK: %[[TN:.*]], %{{.*}} = "spmd.getSizeOfComm"(%[[OUTER]]) <{usedModel = 2 : i32}>
  %tme = llvm.call @nvshmem_team_my_pe(%world) : (i32) -> i32
  %tn = llvm.call @nvshmem_team_n_pes(%world) : (i32) -> i32
  // The new team is written back in the program's int form.
  // CHECK: %[[TEAM:.*]], %{{.*}} = "spmd.commSplitStrided"(%[[OUTER]], %c1_i32, %c2_i32, %c3_i32) <{usedModel = 2 : i32}>
  // CHECK: %[[TI:.*]] = builtin.unrealized_conversion_cast %[[TEAM]] : !spmd.comm to i32
  // CHECK: memref.store %[[TI]]
  %split = llvm.call @nvshmem_team_split_strided(%world, %c1, %c2, %c3, %null, %c0_i64, %newteam) : (i32, i32, i32, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
  // CHECK: "spmd.barrier"(%[[OUTER]]) <{isBlocking = true, usedModel = 2 : i32}>
  %sync = llvm.call @nvshmem_team_sync(%world) : (i32) -> i32
  // CHECK: "spmd.commDestroy"(%[[OUTER]]) <{usedModel = 2 : i32}>
  llvm.call @nvshmem_team_destroy(%world) : (i32) -> ()

  // Collectives: destination before source, as in OpenSHMEM; nelems 5, root 3.
  // CHECK: "spmd.bcast"(%[[OUTER]], %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %c3_i32) <{isBlocking = true, usedModel = 2 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %bcast = llvm.call @nvshmem_int_broadcast(%world, %dest, %src, %c5_i64, %c3) : (i32, !llvm.ptr, !llvm.ptr, i64, i32) -> i32
  // CHECK: "spmd.allgather"(%[[OUTER]], %{{.*}}, %c5_i64, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}) <{isBlocking = true, usedModel = 2 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, memref<8xi32>, i64, !spmd.datatype) -> !spmd.error
  %fcollect = llvm.call @nvshmem_int_fcollect(%world, %dest, %src, %c5_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: "spmd.alltoall"(%[[OUTER]], %{{.*}}, %c5_i64, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}) <{isBlocking = true, usedModel = 2 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, memref<8xi32>, i64, !spmd.datatype) -> !spmd.error
  %alltoall = llvm.call @nvshmem_int_alltoall(%world, %dest, %src, %c5_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  // Team reductions, the operation from the name: sum over int, the bitwise
  // and over uint32 (NVSHMEM has it for unsigned and fixed-width integers
  // only), and max over half.
  // CHECK: %[[SUM:.*]] = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 2 : i32}>
  // CHECK: "spmd.allreduce"(%[[OUTER]], %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %[[SUM]]) <{isBlocking = true, usedModel = 2 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %sum = llvm.call @nvshmem_int_sum_reduce(%world, %dest, %src, %c5_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: %[[UI32:.*]] = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 2 : i32}>
  // CHECK: %[[AND:.*]] = "spmd.constOpType"() <{stringAttr = "and", usedModel = 2 : i32}>
  // CHECK: "spmd.allreduce"(%[[OUTER]], %{{.*}}, %{{.*}}, %c5_i64, %[[UI32]], %[[AND]]) <{isBlocking = true, usedModel = 2 : i32}>
  %and = llvm.call @nvshmem_uint32_and_reduce(%world, %dest, %src, %c5_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: %[[F16:.*]] = "spmd.constDatatype"() <{typeAttr = f16, usedModel = 2 : i32}>
  // CHECK: %[[MAX:.*]] = "spmd.constOpType"() <{stringAttr = "max", usedModel = 2 : i32}>
  // CHECK: "spmd.allreduce"(%[[OUTER]], %{{.*}}, %{{.*}}, %c5_i64, %[[F16]], %[[MAX]]) <{isBlocking = true, usedModel = 2 : i32}>
  %max = llvm.call @nvshmem_half_max_reduce(%world, %dest, %src, %c5_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: "spmd.abort"(%[[OUTER]], %c9_i32) <{usedModel = 2 : i32}>
  llvm.call @nvshmem_global_exit(%c9) : (i32) -> ()
  // CHECK: "spmd.finalize"() <{usedModel = 2 : i32}>
  llvm.call @nvshmem_finalize() : () -> ()

  // CHECK: %[[S1:.*]] = arith.addi %[[ME]], %[[N]] : i32
  // CHECK: %[[S2:.*]] = arith.addi %[[TME]], %[[TN]] : i32
  // CHECK: %[[S3:.*]] = arith.addi %[[S1]], %[[S2]] : i32
  // CHECK: "spmd.barrier"(%[[OUTER]]) <{isBlocking = true, isUniqueMatching = true, usedModel = 2 : i32}>
  // CHECK: "spmd.parallel_yield"(%[[S3]]) : (i32) -> ()
  %s1 = arith.addi %me, %n : i32
  %s2 = arith.addi %tme, %tn : i32
  %s3 = arith.addi %s1, %s2 : i32
  llvm.return %s3 : i32
}
llvm.func @nvshmem_init_thread(i32, !llvm.ptr) -> i32
llvm.func @nvshmem_my_pe() -> i32
llvm.func @nvshmem_n_pes() -> i32
llvm.func @nvshmem_malloc(i64) -> !llvm.ptr
llvm.func @nvshmem_free(!llvm.ptr)
llvm.func @nvshmem_barrier_all()
llvm.func @nvshmem_sync_all()
llvm.func @nvshmem_quiet()
llvm.func @nvshmem_team_my_pe(i32) -> i32
llvm.func @nvshmem_team_n_pes(i32) -> i32
llvm.func @nvshmem_team_split_strided(i32, i32, i32, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
llvm.func @nvshmem_team_sync(i32) -> i32
llvm.func @nvshmem_team_destroy(i32)
llvm.func @nvshmem_int_broadcast(i32, !llvm.ptr, !llvm.ptr, i64, i32) -> i32
llvm.func @nvshmem_int_fcollect(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_int_sum_reduce(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_uint32_and_reduce(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_half_max_reduce(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_int_alltoall(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_global_exit(i32)
llvm.func @nvshmem_finalize()
