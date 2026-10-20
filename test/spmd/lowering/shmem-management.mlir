// RUN: spmd-opt %s -lower-spmd-to-shmem | FileCheck %s --implicit-check-not='"spmd.'
//
// Process and team management, memory and synchronisation.  OpenSHMEM returns
// ranks, sizes and symmetric addresses rather than writing them through a
// pointer, and most of its routines return nothing, so their spmd.error is
// OpenSHMEM's success.

// CHECK-LABEL: func.func @init_finalize
// CHECK:         llvm.call @shmem_init() : () -> ()
// CHECK:         llvm.call @shmem_finalize() : () -> ()
func.func @init_finalize() {
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 1 : i32}> : () -> !spmd.threadLevel
  %e0 = "spmd.init"(%single) <{usedModel = 1 : i32}> : (!spmd.threadLevel) -> !spmd.error
  %e1 = "spmd.finalize"() <{usedModel = 1 : i32}> : () -> !spmd.error
  return
}

// The granted level is an out-parameter, read back as the operation's result.
// CHECK-LABEL: func.func @init_thread
// CHECK:         %[[SLOT:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         %[[MULTIPLE:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         llvm.call @shmem_init_thread(%[[MULTIPLE]], %[[SLOT]]) : (i32, !llvm.ptr) -> i32
// CHECK:         %[[LEVEL:.*]] = llvm.load %[[SLOT]] : !llvm.ptr -> i32
// CHECK:         return %[[LEVEL]] : i32
func.func @init_thread() -> i32 {
  %multiple = "spmd.constThreadLevel"() <{level = "MULTIPLE", usedModel = 1 : i32}> : () -> !spmd.threadLevel
  %provided, %e = "spmd.init"(%multiple) <{usedModel = 1 : i32}> : (!spmd.threadLevel) -> (!spmd.threadLevel, !spmd.error)
  %level = "spmd.castThreadLevelToI32"(%provided) : (!spmd.threadLevel) -> i32
  return %level : i32
}

// The world team needs no argument; any other team is passed.  The rank and
// the size are the return values.
// CHECK-LABEL: func.func @rank_size
// CHECK:         %[[TEAM:.*]] = builtin.unrealized_conversion_cast %arg0 : !spmd.comm to !llvm.ptr
// CHECK:         %[[R0:.*]] = llvm.call @shmem_my_pe() : () -> i32
// CHECK:         %[[S0:.*]] = llvm.call @shmem_n_pes() : () -> i32
// CHECK:         %[[R1:.*]] = llvm.call @shmem_team_my_pe(%[[TEAM]]) : (!llvm.ptr) -> i32
// CHECK:         %[[S1:.*]] = llvm.call @shmem_team_n_pes(%[[TEAM]]) : (!llvm.ptr) -> i32
// CHECK:         return %[[R0]], %[[S0]], %[[R1]], %[[S1]]
func.func @rank_size(%team: !spmd.comm) -> (i32, i32, i32, i32) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %r0, %e0 = "spmd.getRankInComm"(%world) <{usedModel = 1 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %s0, %e1 = "spmd.getSizeOfComm"(%world) <{usedModel = 1 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %r1, %e2 = "spmd.getRankInComm"(%team) <{usedModel = 1 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %s1, %e3 = "spmd.getSizeOfComm"(%team) <{usedModel = 1 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  return %r0, %s0, %r1, %s1 : i32, i32, i32, i32
}

// SHMEM_TEAM_WORLD is an extern variable holding a pointer, so the team is
// loaded from it.  No configuration is passed: a null config and an empty
// mask.  The new team comes back through the last argument, and the status is
// the call's return value.  Sandia's shmem_team_destroy returns an int, which
// the specification USIS describes does not.
// CHECK-LABEL: func.func @split
// CHECK:         %[[NEW:.*]] = llvm.alloca %{{.*}} x !llvm.ptr
// CHECK:         %[[ADDR:.*]] = llvm.mlir.addressof @SHMEM_TEAM_WORLD : !llvm.ptr
// CHECK:         %[[WORLD:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:         %[[MASK:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[STATUS:.*]] = llvm.call @shmem_team_split_strided(%[[WORLD]], %arg0, %arg1, %arg2, %[[NULL]], %[[MASK]], %[[NEW]]) : (!llvm.ptr, i32, i32, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         %[[TEAM:.*]] = llvm.load %[[NEW]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @shmem_team_destroy(%[[TEAM]]) : (!llvm.ptr) -> i32
// CHECK:         return %[[STATUS]] : i32
func.func @split(%start: i32, %stride: i32, %size: i32) -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %team, %e0 = "spmd.commSplitStrided"(%world, %start, %stride, %size) <{usedModel = 1 : i32}> : (!spmd.comm, i32, i32, i32) -> (!spmd.comm, !spmd.error)
  %e1 = "spmd.commDestroy"(%team) <{usedModel = 1 : i32}> : (!spmd.comm) -> !spmd.error
  %code = "spmd.castSPMDErrorToI32"(%e0) : (!spmd.error) -> i32
  return %code : i32
}

// shmem_align takes the alignment first, where spmd.malloc has the size first.
// CHECK-LABEL: func.func @memory
// CHECK:         %[[A:.*]] = llvm.call @shmem_malloc(%arg0) : (i64) -> !llvm.ptr
// CHECK:         %[[B:.*]] = llvm.call @shmem_align(%arg1, %arg0) : (i64, i64) -> !llvm.ptr
// CHECK:         %[[C:.*]] = llvm.call @shmem_realloc(%[[A]], %arg0) : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK:         llvm.call @shmem_free(%[[C]]) : (!llvm.ptr) -> ()
// CHECK:         return %[[B]] : !llvm.ptr
func.func @memory(%n: i64, %align: i64) -> !llvm.ptr {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %a, %e0 = "spmd.malloc"(%world, %n) <{usedModel = 1 : i32}> : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  %b, %e1 = "spmd.malloc"(%world, %n, %align) <{usedModel = 1 : i32}> : (!spmd.comm, i64, i64) -> (!llvm.ptr, !spmd.error)
  %c, %e2 = "spmd.realloc"(%world, %a, %n) <{usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, i64) -> (!llvm.ptr, !spmd.error)
  %e3 = "spmd.free"(%world, %c) <{usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr) -> !spmd.error
  return %b : !llvm.ptr
}

// CHECK-LABEL: func.func @abort
// CHECK:         llvm.call @shmem_global_exit(%arg0) : (i32) -> ()
func.func @abort(%code: i32) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %e = "spmd.abort"(%world, %code) <{usedModel = 1 : i32}> : (!spmd.comm, i32) -> !spmd.error
  return
}

// A barrier on the world team is shmem_sync_all, on any other team
// shmem_team_sync.  OpenSHMEM's fence, flush and ordering primitives take no
// window and apply to everything the PE has outstanding.
// CHECK-LABEL: func.func @synchronisation
// CHECK:         %[[TEAM:.*]] = builtin.unrealized_conversion_cast %arg0 : !spmd.comm to !llvm.ptr
// CHECK:         llvm.call @shmem_sync_all() : () -> ()
// CHECK:         llvm.call @shmem_team_sync(%[[TEAM]]) : (!llvm.ptr) -> i32
// CHECK:         llvm.call @shmem_barrier_all() : () -> ()
// CHECK:         llvm.call @shmem_quiet() : () -> ()
// CHECK:         llvm.call @shmem_fence() : () -> ()
func.func @synchronisation(%team: !spmd.comm) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i32
  %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.barrier"(%team) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm) -> !spmd.error
  %e2 = "spmd.fence"(%c0) <{usedModel = 1 : i32}> : (i32) -> !spmd.error
  %e3 = "spmd.flushAll"() <{isLocal = false, usedModel = 1 : i32}> : () -> !spmd.error
  %e4 = "spmd.fenceOrdering"() <{usedModel = 1 : i32}> : () -> !spmd.error
  return
}

// A team the program keeps in an index is as wide as the pointer it is.  The
// check that reports one kept in an int used to ask the index for a bit width
// it does not have.
// CHECK-LABEL: func.func @team_in_index
// CHECK:         llvm.call @shmem_team_my_pe(
func.func @team_in_index(%handle: index) -> i32 {
  %team = builtin.unrealized_conversion_cast %handle : index to !spmd.comm
  %rank, %e = "spmd.getRankInComm"(%team) <{usedModel = 1 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  return %rank : i32
}
