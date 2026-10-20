// RUN: spmd-opt %s -lower-spmd-to-nccl | FileCheck %s --implicit-check-not='usedModel = 3'
//
// A module that starts OpenSHMEM and no MPI bootstraps NCCL over OpenSHMEM:
// the rank and the size are shmem_my_pe and shmem_n_pes, and rank 0's unique
// id goes to the others by shmem_broadcastmem on SHMEM_TEAM_WORLD.  Both ids
// are globals, which OpenSHMEM makes symmetric.  The broadcast is not in place,
// and the root passes the id it made: whether shmem_broadcastmem allows one
// buffer as source and destination, or writes the root's destination, is not
// something the lowering has to know.

// CHECK-DAG:   llvm.mlir.global internal @__spmd_nccl_unique_id_received() {{.*}} : !llvm.struct<(array<128 x i8>)>

// CHECK-LABEL: llvm.func internal @__spmd_nccl_bootstrap() -> i32
// CHECK:         %[[RANK:.*]] = llvm.call @shmem_my_pe() : () -> i32
// CHECK:         %[[SIZE:.*]] = llvm.call @shmem_n_pes() : () -> i32
// CHECK:         %[[ISROOT:.*]] = llvm.icmp "eq" %[[RANK]]
// CHECK:         llvm.cond_br %[[ISROOT]]
// CHECK:         llvm.call @ncclGetUniqueId(
// CHECK:         %[[ID:.*]] = llvm.mlir.addressof @__spmd_nccl_unique_id : !llvm.ptr
// CHECK:         %[[ROOT:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[RECV:.*]] = llvm.mlir.addressof @__spmd_nccl_unique_id_received : !llvm.ptr
// CHECK:         %[[TEAMADDR:.*]] = llvm.mlir.addressof @SHMEM_TEAM_WORLD : !llvm.ptr
// CHECK:         %[[TEAM:.*]] = llvm.load %[[TEAMADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[BYTES:.*]] = llvm.mlir.constant(128 : i64) : i64
// CHECK:         llvm.call @shmem_broadcastmem(%[[TEAM]], %[[RECV]], %[[ID]], %[[BYTES]], %[[ROOT]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK:         %[[MINE:.*]] = llvm.select %[[ISROOT]], %[[ID]], %[[RECV]] : i1, !llvm.ptr
// CHECK:         llvm.call @ncclCommInitRank(%{{.*}}, %[[SIZE]], %[[MINE]], %[[RANK]])
func.func @main() {
  %shmemSingle = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 1 : i32}> : () -> !spmd.threadLevel
  %e0 = "spmd.init"(%shmemSingle) <{usedModel = 1 : i32}> : (!spmd.threadLevel) -> !spmd.error
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 3 : i32}> : () -> !spmd.threadLevel
  %e1 = "spmd.init"(%single) <{usedModel = 3 : i32}> : (!spmd.threadLevel) -> !spmd.error
  return
}
