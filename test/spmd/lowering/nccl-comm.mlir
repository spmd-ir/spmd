// RUN: spmd-opt %s -split-input-file -lower-spmd-to-nccl | FileCheck %s --implicit-check-not='usedModel = 3'
//
// Initialisation and communicators.  NCCL has no world communicator: every NCCL
// program makes one with ncclCommInitRank from a unique id that rank 0 gets
// from ncclGetUniqueId and sends to the others over another model.  The raiser
// keeps none of that -- it makes the call an spmd.init and an spmd.commWorld --
// so spmd.init calls a bootstrap the lowering adds, which does it over the
// module's MPI, and leaves the communicator in a global spmd.commWorld loads.
//
// roundtrip's lowering passed ncclCommInitRank a null id, a size of 1 and rank
// 0, and lowered spmd.commWorld to a null pointer, so the communicator it made
// was never used.

// The bootstrap: rank and size from MPI_COMM_WORLD, the id from rank 0 over
// MPI_Bcast as 128 MPI_BYTEs, and the id passed to ncclCommInitRank by value,
// as the x86-64 C ABI passes a 128-byte struct: a pointer marked byval.
// CHECK-DAG:   llvm.mlir.global internal @__spmd_nccl_unique_id() {{.*}} : !llvm.struct<(array<128 x i8>)>
// CHECK-DAG:   llvm.mlir.global internal @__spmd_nccl_comm_world() {{.*}} : !llvm.ptr
// CHECK-DAG:   llvm.func @ncclCommInitRank(!llvm.ptr, i32, !llvm.ptr {llvm.byval = !llvm.struct<(array<128 x i8>)>}, i32) -> i32

// CHECK-LABEL: func.func @init_and_query
// CHECK:         %[[E:.*]] = llvm.call @__spmd_nccl_bootstrap() : () -> i32
// CHECK:         %[[ADDR:.*]] = llvm.mlir.addressof @__spmd_nccl_comm_world : !llvm.ptr
// CHECK:         %[[WORLD:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @ncclCommUserRank(%[[WORLD]], %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.call @ncclCommCount(%[[WORLD]], %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.call @ncclCommCuDevice(%[[WORLD]], %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         return %{{.*}}, %{{.*}}, %{{.*}}, %[[E]] :

// CHECK-LABEL: llvm.func internal @__spmd_nccl_bootstrap() -> i32
// CHECK:         %[[RSLOT:.*]] = llvm.alloca
// CHECK:         %[[SSLOT:.*]] = llvm.alloca
// CHECK:         %[[MPIWORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         llvm.call @MPI_Comm_rank(%[[MPIWORLD]], %[[RSLOT]]) : (i32, !llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Comm_size(%[[MPIWORLD]], %[[SSLOT]]) : (i32, !llvm.ptr) -> i32
// CHECK:         %[[RANK:.*]] = llvm.load %[[RSLOT]] : !llvm.ptr -> i32
// CHECK:         %[[SIZE:.*]] = llvm.load %[[SSLOT]] : !llvm.ptr -> i32
// CHECK:         %[[ISROOT:.*]] = llvm.icmp "eq" %[[RANK]], %{{.*}} : i32
// CHECK:         llvm.cond_br %[[ISROOT]], ^[[ROOT:.*]], ^[[SHARE:.*]]
// CHECK:       ^[[ROOT]]:
// CHECK:         %[[ID0:.*]] = llvm.mlir.addressof @__spmd_nccl_unique_id : !llvm.ptr
// CHECK:         llvm.call @ncclGetUniqueId(%[[ID0]]) : (!llvm.ptr) -> i32
// CHECK:         llvm.br ^[[SHARE]]
// CHECK:       ^[[SHARE]]:
// CHECK:         %[[ID:.*]] = llvm.mlir.addressof @__spmd_nccl_unique_id : !llvm.ptr
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[W:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[BYTES:.*]] = llvm.mlir.constant(128 : i32) : i32
// CHECK:         %[[BYTE:.*]] = llvm.mlir.constant(1275068685 : i32) : i32
// CHECK:         llvm.call @MPI_Bcast(%[[ID]], %[[BYTES]], %[[BYTE]], %[[ZERO]], %[[W]]) : (!llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK:         %[[COMM:.*]] = llvm.mlir.addressof @__spmd_nccl_comm_world : !llvm.ptr
// CHECK:         %[[R:.*]] = llvm.call @ncclCommInitRank(%[[COMM]], %[[SIZE]], %[[ID]], %[[RANK]]) : (!llvm.ptr, i32, !llvm.ptr {llvm.byval = !llvm.struct<(array<128 x i8>)>}, i32) -> i32
// CHECK:         llvm.return %[[R]] : i32
func.func @init_and_query() -> (i32, i32, i32, i32) {
  %mpiSingle = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  %e0 = "spmd.init"(%mpiSingle) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 3 : i32}> : () -> !spmd.threadLevel
  %e1 = "spmd.init"(%single) <{usedModel = 3 : i32}> : (!spmd.threadLevel) -> !spmd.error
  %world = "spmd.commWorld"() <{usedModel = 3 : i32}> : () -> !spmd.comm
  %rank, %e2 = "spmd.getRankInComm"(%world) <{usedModel = 3 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %size, %e3 = "spmd.getSizeOfComm"(%world) <{usedModel = 3 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %device, %e4 = "spmd.getDeviceInComm"(%world) <{usedModel = 3 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %ret = "spmd.castSPMDErrorToI32"(%e1) : (!spmd.error) -> i32
  return %rank, %size, %device, %ret : i32, i32, i32, i32
}

// -----

// A split takes the parent, the colour and the key in that order and a null
// configuration, and writes the new communicator through a pointer.
// CHECK-LABEL: func.func @split_and_destroy
// CHECK:         %[[NEW:.*]] = llvm.alloca %{{.*}} x !llvm.ptr
// CHECK:         %[[PARENT:.*]] = builtin.unrealized_conversion_cast %arg0 : !spmd.comm to !llvm.ptr
// CHECK:         %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:         llvm.call @ncclCommSplit(%[[PARENT]], %arg1, %arg2, %[[NEW]], %[[NULL]]) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[SUB:.*]] = llvm.load %[[NEW]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @ncclCommDestroy(%[[SUB]]) : (!llvm.ptr) -> i32
func.func @split_and_destroy(%parent: !spmd.comm, %color: i32, %key: i32) {
  %sub, %e1 = "spmd.commSplit"(%parent, %color, %key) <{usedModel = 3 : i32}> : (!spmd.comm, i32, i32) -> (!spmd.comm, !spmd.error)
  %e2 = "spmd.commDestroy"(%sub) <{usedModel = 3 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// -----

// A communicator the program keeps in an index is as wide as the pointer it
// is.  The check that reports one kept in an int used to ask the index for a
// bit width it does not have.
// CHECK-LABEL: func.func @comm_in_index
// CHECK:         llvm.call @ncclCommUserRank(
func.func @comm_in_index(%handle: index) -> i32 {
  %comm = builtin.unrealized_conversion_cast %handle : index to !spmd.comm
  %rank, %e = "spmd.getRankInComm"(%comm) <{usedModel = 3 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  return %rank : i32
}
