// RUN: spmd-opt %s -lower-spmd | FileCheck %s --implicit-check-not='"spmd.' --implicit-check-not="@MPI_Barrier" --implicit-check-not="@shmem_sync_all" --implicit-check-not="@shmem_team_sync"
//
// The module of driver-mixed-model.mlir under the default table, which now
// lowers OpenSHMEM as well as MPI.  Both process-level parallels go in the
// structure phase, before either model is lowered, and with them the
// self-matching barrier the raiser put in front of each yield -- neither
// becomes a call.  The two models' calls stay where the program made them:
// shmem_init around MPI_Init and MPI_Finalize, shmem_finalize after.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         llvm.call @shmem_init() : () -> ()
// CHECK:         llvm.call @MPI_Init(
// CHECK:         llvm.call @MPI_Finalize()
// CHECK:         llvm.call @shmem_finalize() : () -> ()
// CHECK:         llvm.return
llvm.func @main() -> i32 {
  %shmemWorld = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %code = "spmd.parallel"(%shmemWorld) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 1 : i32}> ({
    %shmemSingle = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 1 : i32}> : () -> !spmd.threadLevel
    %s0 = "spmd.init"(%shmemSingle) <{usedModel = 1 : i32}> : (!spmd.threadLevel) -> !spmd.error
    %mpiWorld = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
    %inner = "spmd.parallel"(%mpiWorld) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 0 : i32, usedModel = 0 : i32}> ({
      %c0 = llvm.mlir.constant(0 : i32) : i32
      %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
      %e0 = "spmd.init"(%single) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
      %e1 = "spmd.finalize"() <{usedModel = 0 : i32}> : () -> !spmd.error
      %e2 = "spmd.barrier"(%mpiWorld) <{isBlocking = true, isUniqueMatching = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    %s1 = "spmd.finalize"() <{usedModel = 1 : i32}> : () -> !spmd.error
    %s2 = "spmd.barrier"(%shmemWorld) <{isBlocking = true, isUniqueMatching = true, usedModel = 1 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%inner) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  llvm.return %code : i32
}
