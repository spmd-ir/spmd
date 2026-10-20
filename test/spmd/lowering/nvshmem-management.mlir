// RUN: spmd-opt %s -lower-spmd-to-nvshmem | FileCheck %s --implicit-check-not='"spmd.'
//
// Process and team management, memory and synchronisation, as
// shmem-management.mlir has them for OpenSHMEM.  The calls are the same with
// an nvshmem_ prefix; the ABI is not.  A team is an int32_t, and the
// predefined teams are the constants NVSHMEM_TEAM_WORLD (0) and
// NVSHMEMX_TEAM_NODE (2), so nothing is loaded from a variable.
//
// nvshmem_init, nvshmem_init_thread and nvshmem_finalize are what the program
// called, and what the raiser raised, so they are what the lowering emits;
// nvshmem-host-abi.mlir maps them to what NVIDIA's library exports.

// CHECK-LABEL: func.func @init_finalize
// CHECK:         llvm.call @nvshmem_init() : () -> ()
// CHECK:         llvm.call @nvshmem_finalize() : () -> ()
func.func @init_finalize() {
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 2 : i32}> : () -> !spmd.threadLevel
  %e0 = "spmd.init"(%single) <{usedModel = 2 : i32}> : (!spmd.threadLevel) -> !spmd.error
  %e1 = "spmd.finalize"() <{usedModel = 2 : i32}> : () -> !spmd.error
  return
}

// The granted level is an out-parameter, read back as the operation's result;
// the status is the call's return value.
// CHECK-LABEL: func.func @init_thread
// CHECK:         %[[SLOT:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         %[[MULTIPLE:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         %[[STATUS:.*]] = llvm.call @nvshmem_init_thread(%[[MULTIPLE]], %[[SLOT]]) : (i32, !llvm.ptr) -> i32
// CHECK:         %[[LEVEL:.*]] = llvm.load %[[SLOT]] : !llvm.ptr -> i32
// CHECK:         return %[[LEVEL]], %[[STATUS]] : i32, i32
func.func @init_thread() -> (i32, i32) {
  %multiple = "spmd.constThreadLevel"() <{level = "MULTIPLE", usedModel = 2 : i32}> : () -> !spmd.threadLevel
  %provided, %e = "spmd.init"(%multiple) <{usedModel = 2 : i32}> : (!spmd.threadLevel) -> (!spmd.threadLevel, !spmd.error)
  %level = "spmd.castThreadLevelToI32"(%provided) : (!spmd.threadLevel) -> i32
  %code = "spmd.castSPMDErrorToI32"(%e) : (!spmd.error) -> i32
  return %level, %code : i32, i32
}

// The world team needs no argument; any other team is passed, as the int32_t
// it is, and spmd.commNode is NVSHMEMX_TEAM_NODE.
// CHECK-LABEL: func.func @rank_size
// CHECK:         %[[TEAM:.*]] = builtin.unrealized_conversion_cast %arg0 : !spmd.comm to i32
// CHECK:         %[[NODE:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[R0:.*]] = llvm.call @nvshmem_my_pe() : () -> i32
// CHECK:         %[[S0:.*]] = llvm.call @nvshmem_n_pes() : () -> i32
// CHECK:         %[[R1:.*]] = llvm.call @nvshmem_team_my_pe(%[[TEAM]]) : (i32) -> i32
// CHECK:         %[[S1:.*]] = llvm.call @nvshmem_team_n_pes(%[[TEAM]]) : (i32) -> i32
// CHECK:         %[[R2:.*]] = llvm.call @nvshmem_team_my_pe(%[[NODE]]) : (i32) -> i32
// CHECK:         return %[[R0]], %[[S0]], %[[R1]], %[[S1]], %[[R2]]
func.func @rank_size(%team: !spmd.comm) -> (i32, i32, i32, i32, i32) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %node = "spmd.commNode"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %r0, %e0 = "spmd.getRankInComm"(%world) <{usedModel = 2 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %s0, %e1 = "spmd.getSizeOfComm"(%world) <{usedModel = 2 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %r1, %e2 = "spmd.getRankInComm"(%team) <{usedModel = 2 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %s1, %e3 = "spmd.getSizeOfComm"(%team) <{usedModel = 2 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %r2, %e4 = "spmd.getRankInComm"(%node) <{usedModel = 2 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  return %r0, %s0, %r1, %s1, %r2 : i32, i32, i32, i32, i32
}

// NVSHMEM_TEAM_WORLD is the constant 0.  No configuration is passed: a null
// config and an empty mask.  The new team comes back through the last
// argument as an int32_t, and the status is the call's return value.
// CHECK-LABEL: func.func @split
// CHECK:         %[[NEW:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:         %[[MASK:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[STATUS:.*]] = llvm.call @nvshmem_team_split_strided(%[[WORLD]], %arg0, %arg1, %arg2, %[[NULL]], %[[MASK]], %[[NEW]]) : (i32, i32, i32, i32, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK:         %[[TEAM:.*]] = llvm.load %[[NEW]] : !llvm.ptr -> i32
// CHECK:         llvm.call @nvshmem_team_destroy(%[[TEAM]]) : (i32) -> ()
// CHECK:         return %[[STATUS]] : i32
func.func @split(%start: i32, %stride: i32, %size: i32) -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %team, %e0 = "spmd.commSplitStrided"(%world, %start, %stride, %size) <{usedModel = 2 : i32}> : (!spmd.comm, i32, i32, i32) -> (!spmd.comm, !spmd.error)
  %e1 = "spmd.commDestroy"(%team) <{usedModel = 2 : i32}> : (!spmd.comm) -> !spmd.error
  %code = "spmd.castSPMDErrorToI32"(%e0) : (!spmd.error) -> i32
  return %code : i32
}

// Symmetric memory is nvshmem_malloc's, on the GPU.  NVSHMEM has no aligned
// allocation and no realloc in the API the raiser reads
// (nvshmem-inexpressible.mlir).
// CHECK-LABEL: func.func @memory
// CHECK:         %[[A:.*]] = llvm.call @nvshmem_malloc(%arg0) : (i64) -> !llvm.ptr
// CHECK:         %[[B:.*]] = llvm.call @nvshmem_malloc(%arg0) : (i64) -> !llvm.ptr
// CHECK:         llvm.call @nvshmem_free(%[[A]]) : (!llvm.ptr) -> ()
// CHECK:         return %[[B]] : !llvm.ptr
func.func @memory(%n: i64) -> !llvm.ptr {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %a, %e0 = "spmd.malloc"(%world, %n) <{usedModel = 2 : i32}> : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  %b, %e1 = "spmd.malloc"(%world, %n) <{usedModel = 2 : i32}> : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  %e3 = "spmd.free"(%world, %a) <{usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr) -> !spmd.error
  return %b : !llvm.ptr
}

// CHECK-LABEL: func.func @abort
// CHECK:         llvm.call @nvshmem_global_exit(%arg0) : (i32) -> ()
func.func @abort(%code: i32) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %e = "spmd.abort"(%world, %code) <{usedModel = 2 : i32}> : (!spmd.comm, i32) -> !spmd.error
  return
}

// A barrier on the world team is nvshmem_sync_all, on any other team
// nvshmem_team_sync, which returns a status.  The fence, flush and ordering
// primitives take no window and apply to everything the PE has outstanding.
// CHECK-LABEL: func.func @synchronisation
// CHECK:         %[[TEAM:.*]] = builtin.unrealized_conversion_cast %arg0 : !spmd.comm to i32
// CHECK:         llvm.call @nvshmem_sync_all() : () -> ()
// CHECK:         %[[STATUS:.*]] = llvm.call @nvshmem_team_sync(%[[TEAM]]) : (i32) -> i32
// CHECK:         llvm.call @nvshmem_barrier_all() : () -> ()
// CHECK:         llvm.call @nvshmem_quiet() : () -> ()
// CHECK:         llvm.call @nvshmem_fence() : () -> ()
// CHECK:         return %[[STATUS]] : i32
func.func @synchronisation(%team: !spmd.comm) -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i32
  %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.barrier"(%team) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm) -> !spmd.error
  %e2 = "spmd.fence"(%c0) <{usedModel = 2 : i32}> : (i32) -> !spmd.error
  %e3 = "spmd.flushAll"() <{isLocal = false, usedModel = 2 : i32}> : () -> !spmd.error
  %e4 = "spmd.fenceOrdering"() <{usedModel = 2 : i32}> : () -> !spmd.error
  %code = "spmd.castSPMDErrorToI32"(%e1) : (!spmd.error) -> i32
  return %code : i32
}

// A team the program keeps in an int is the int32_t it is: unlike an
// OpenSHMEM team, it fits a variable declared for an MPI handle.
// CHECK-LABEL: func.func @team_in_int
// CHECK:         llvm.call @nvshmem_team_my_pe(%arg0) : (i32) -> i32
func.func @team_in_int(%handle: i32) -> i32 {
  %team = builtin.unrealized_conversion_cast %handle : i32 to !spmd.comm
  %rank, %e = "spmd.getRankInComm"(%team) <{usedModel = 2 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  return %rank : i32
}
