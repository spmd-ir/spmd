// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s --implicit-check-not="usedModel = 0" --implicit-check-not="llvm.call @MPI_Barrier("
//
// A program using both SHMEM and MPI gets one process-level spmd.parallel per
// model from -convert-apiCalls-to-spmd, each with its own commWorld,
// self-matching barrier and usedModel, and the two nest: the SHMEM parallel
// outside, the MPI parallel inside.  -lower-spmd-to-mpi splices only the MPI
// one, together with its barrier.  The SHMEM parallel, its barrier and its
// yield stay for the SHMEM lowering, and now carry the MPI region's exit code.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         %[[SW:[0-9]+]] = "spmd.commWorld"() <{usedModel = 1 : i32}>
// CHECK:         %[[CODE:[0-9]+]] = "spmd.parallel"(%[[SW]]) <{isProcessLevel = true, {{.*}}usedModel = 1 : i32}> ({
// CHECK:           "spmd.init"({{.*}}) <{usedModel = 1 : i32}>
// CHECK:           llvm.br ^[[BODY:bb[0-9]+]]
// CHECK:         ^[[BODY]]:
// CHECK:           llvm.call @MPI_Init(
// CHECK:           llvm.call @MPI_Finalize()
// CHECK-NEXT:      llvm.br ^[[JOIN:bb[0-9]+]](%{{.*}} : i32)
// CHECK:         ^[[JOIN]](%[[INNER:[0-9]+]]: i32):
// CHECK:           llvm.br ^[[AFTER:bb[0-9]+]]
// CHECK:         ^[[AFTER]]:
// CHECK-NEXT:      "spmd.finalize"() <{usedModel = 1 : i32}>
// CHECK-NEXT:      "spmd.barrier"(%[[SW]]) <{isBlocking = true, isUniqueMatching = true, usedModel = 1 : i32}>
// CHECK-NEXT:      "spmd.parallel_yield"(%[[INNER]]) : (i32) -> ()
// CHECK-NEXT:    }) : (!spmd.comm) -> i32
// CHECK-NEXT:    llvm.return %[[CODE]] : i32
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
