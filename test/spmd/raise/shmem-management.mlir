// RUN: spmd-opt %s -convert-apiCalls-to-spmd -cse | FileCheck %s --implicit-check-not="llvm.call @shmem_"
//
// OpenSHMEM setup, symmetric memory, synchronisation and teams. Every
// operation is tagged usedModel = 1. A call without a team works on the world:
// the raiser supplies spmd.commWorld for it.
//
// Synchronisation is mapped by what each call guarantees:
//  - shmem_sync_all is a barrier over all PEs,
//  - shmem_barrier_all is a barrier that also completes all communication
//    issued before it, which is what a fence is,
//  - shmem_quiet only completes communication: a flush of everything.
//
// The raiser materialises a world communicator for every use; -cse, which
// spmd-verify runs after raising too, folds them into one, the world the
// parallel region runs over.

// CHECK-LABEL: llvm.func @main
// CHECK:         %[[OUTER:.*]] = "spmd.commWorld"() <{usedModel = 1 : i32}>
// CHECK:         "spmd.parallel"(%[[OUTER]]) <{isProcessLevel = true, {{.*}}threadLevel = 2 : i32, usedModel = 1 : i32}> ({
// CHECK-DAG:     %[[LEVEL:.*]] = "spmd.constThreadLevel"() <{level = "SERIALIZED", usedModel = 1 : i32}>
// CHECK-DAG:     %[[PROVIDED:.*]] = llvm.alloca %c1_i32 x i32
// CHECK-DAG:     %[[NEWTEAM:.*]] = llvm.alloca %c1_i32 x !llvm.ptr
llvm.mlir.global external @SHMEM_TEAM_WORLD() : !llvm.ptr
llvm.func @main() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c9 = arith.constant 9 : i32
  %c64 = arith.constant 64 : i64
  %c128 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  // SHMEM_THREAD_SERIALIZED
  %serialized = arith.constant 2 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %provided = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %newteam = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr

  // CHECK: %[[GOT:.*]], %{{.*}} = "spmd.init"(%[[LEVEL]]) <{usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.threadLevel) -> (!spmd.threadLevel, !spmd.error)
  // CHECK: %[[GOTINT:.*]] = "spmd.castThreadLevelToI32"(%[[GOT]])
  // CHECK: %[[PS:.*]] = "enzymexla.pointer2memref"(%[[PROVIDED]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: memref.store %[[GOTINT]], %[[PS]][%c0] : memref<1xi32>
  %init = llvm.call @shmem_init_thread(%serialized, %provided) : (i32, !llvm.ptr) -> i32

  // The rank and size are the call's return value.
  // CHECK: %[[ME:.*]], %{{.*}} = "spmd.getRankInComm"(%[[OUTER]]) <{usedModel = 1 : i32}>
  // CHECK: %[[N:.*]], %{{.*}} = "spmd.getSizeOfComm"(%[[OUTER]]) <{usedModel = 1 : i32}>
  %me = llvm.call @shmem_my_pe() : () -> i32
  %n = llvm.call @shmem_n_pes() : () -> i32
  // CHECK: %[[P:.*]], %{{.*}} = "spmd.malloc"(%[[OUTER]], %c64_i64) <{usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  %p = llvm.call @shmem_malloc(%c64) : (i64) -> !llvm.ptr
  // CHECK: %[[Q:.*]], %{{.*}} = "spmd.realloc"(%[[OUTER]], %[[P]], %c128_i64) <{usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, !llvm.ptr, i64) -> (!llvm.ptr, !spmd.error)
  %q = llvm.call @shmem_realloc(%p, %c128) : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK: "spmd.free"(%[[OUTER]], %[[Q]]) <{usedModel = 1 : i32}>
  llvm.call @shmem_free(%q) : (!llvm.ptr) -> ()
  // CHECK: "spmd.barrier"(%[[OUTER]]) <{isBlocking = true, usedModel = 1 : i32}>
  llvm.call @shmem_sync_all() : () -> ()
  // CHECK: "spmd.fence"(%c0_i32) <{usedModel = 1 : i32}>
  // CHECK-SAME: : (i32) -> !spmd.error
  llvm.call @shmem_barrier_all() : () -> ()
  // CHECK: "spmd.flushAll"() <{isLocal = false, usedModel = 1 : i32}>
  // CHECK-SAME: : () -> !spmd.error
  llvm.call @shmem_quiet() : () -> ()

  // A team is a communicator. SHMEM_TEAM_WORLD, loaded from its global, is
  // the world; a new team is written back through the pointer the program
  // passed, and read from there it is the team itself again.
  // CHECK: %[[TME:.*]], %{{.*}} = "spmd.getRankInComm"(%[[OUTER]]) <{usedModel = 1 : i32}>
  // CHECK: %[[TN:.*]], %{{.*}} = "spmd.getSizeOfComm"(%[[OUTER]]) <{usedModel = 1 : i32}>
  %wa = llvm.mlir.addressof @SHMEM_TEAM_WORLD : !llvm.ptr
  %w = llvm.load %wa : !llvm.ptr -> !llvm.ptr
  %tme = llvm.call @shmem_team_my_pe(%w) : (!llvm.ptr) -> i32
  %tn = llvm.call @shmem_team_n_pes(%w) : (!llvm.ptr) -> i32
  // Start 1, stride 2, size 3; the configuration is not modelled.
  // CHECK: %[[TEAM:.*]], %{{.*}} = "spmd.commSplitStrided"(%[[OUTER]], %c1_i32, %c2_i32, %c3_i32) <{usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, i32, i32, i32) -> (!spmd.comm, !spmd.error)
  // CHECK: %[[TS:.*]] = "enzymexla.pointer2memref"(%[[NEWTEAM]]) : (!llvm.ptr) -> memref<1x!llvm.ptr>
  // CHECK: %[[TP:.*]] = builtin.unrealized_conversion_cast %[[TEAM]] : !spmd.comm to !llvm.ptr
  // CHECK: memref.store %[[TP]], %[[TS]][%c0] : memref<1x!llvm.ptr>
  %s = llvm.call @shmem_team_split_strided(%w, %c1, %c2, %c3, %null, %c0_i64, %newteam) : (!llvm.ptr, i32, i32, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
  %t = llvm.load %newteam : !llvm.ptr -> !llvm.ptr
  // CHECK: "spmd.barrier"(%[[TEAM]]) <{isBlocking = true, usedModel = 1 : i32}>
  %ts = llvm.call @shmem_team_sync(%t) : (!llvm.ptr) -> i32
  // CHECK: "spmd.commDestroy"(%[[TEAM]]) <{usedModel = 1 : i32}>
  llvm.call @shmem_team_destroy(%t) : (!llvm.ptr) -> ()
  // CHECK: "spmd.abort"(%[[OUTER]], %c9_i32) <{usedModel = 1 : i32}>
  llvm.call @shmem_global_exit(%c9) : (i32) -> ()

  // CHECK: "spmd.finalize"() <{usedModel = 1 : i32}>
  llvm.call @shmem_finalize() : () -> ()

  // CHECK: %[[S1:.*]] = arith.addi %[[ME]], %[[N]] : i32
  // CHECK: %[[S2:.*]] = arith.addi %[[TME]], %[[TN]] : i32
  // CHECK: %[[S3:.*]] = arith.addi %[[S1]], %[[S2]] : i32
  // CHECK: "spmd.barrier"(%[[OUTER]]) <{isBlocking = true, isUniqueMatching = true, usedModel = 1 : i32}>
  // CHECK: "spmd.parallel_yield"(%[[S3]]) : (i32) -> ()
  %s1 = arith.addi %me, %n : i32
  %s2 = arith.addi %tme, %tn : i32
  %s3 = arith.addi %s1, %s2 : i32
  llvm.return %s3 : i32
}
llvm.func @shmem_init_thread(i32, !llvm.ptr) -> i32
llvm.func @shmem_my_pe() -> i32
llvm.func @shmem_n_pes() -> i32
llvm.func @shmem_malloc(i64) -> !llvm.ptr
llvm.func @shmem_realloc(!llvm.ptr, i64) -> !llvm.ptr
llvm.func @shmem_free(!llvm.ptr)
llvm.func @shmem_sync_all()
llvm.func @shmem_barrier_all()
llvm.func @shmem_quiet()
llvm.func @shmem_team_my_pe(!llvm.ptr) -> i32
llvm.func @shmem_team_n_pes(!llvm.ptr) -> i32
llvm.func @shmem_team_split_strided(!llvm.ptr, i32, i32, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
llvm.func @shmem_team_sync(!llvm.ptr) -> i32
llvm.func @shmem_team_destroy(!llvm.ptr)
llvm.func @shmem_global_exit(i32)
llvm.func @shmem_finalize()
