// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s --implicit-check-not=spmd. --implicit-check-not="llvm.call @MPI_Barrier("
//
// -convert-apiCalls-to-spmd moves main's body into a process-level
// spmd.parallel and ends each path out of it with a barrier that matches only
// itself, followed by an spmd.parallel_yield of the exit code.  MPI has no call
// for either: the region is spliced back into main, each yield branches to the
// code after the region, and the barrier in front of it goes.  A barrier the
// program called itself stays.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         llvm.br ^[[BODY:bb[0-9]+]]
// CHECK:       ^[[BODY]]:
// CHECK:         llvm.call @MPI_Init(
// CHECK:         llvm.cond_br %{{.*}}, ^[[EARLY:bb[0-9]+]], ^[[LATE:bb[0-9]+]]
// CHECK:       ^[[EARLY]]:
// CHECK-NEXT:    llvm.br ^[[JOIN:bb[0-9]+]](%{{.*}} : i32)
// CHECK:       ^[[LATE]]:
// CHECK:         llvm.call @MPI_Barrier(
// CHECK:         llvm.call @MPI_Finalize()
// CHECK-NEXT:    llvm.br ^[[JOIN]](%{{.*}} : i32)
// CHECK:       ^[[JOIN]](%[[CODE:.*]]: i32):
// CHECK:         llvm.br ^[[AFTER:bb[0-9]+]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    llvm.return %[[CODE]] : i32
llvm.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %code = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 0 : i32, usedModel = 0 : i32}> ({
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %cond = llvm.mlir.constant(true) : i1
    %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
    %e0 = "spmd.init"(%single) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
    llvm.cond_br %cond, ^early, ^late
  ^early:
    %e1 = "spmd.barrier"(%world) <{isBlocking = true, isUniqueMatching = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c1) : (i32) -> ()
  ^late:
    %e2 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    %e3 = "spmd.finalize"() <{usedModel = 0 : i32}> : () -> !spmd.error
    %e4 = "spmd.barrier"(%world) <{isBlocking = true, isUniqueMatching = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  llvm.return %code : i32
}

// Without an exit code there is nothing to carry across, and the code after
// the region follows directly.
// CHECK-LABEL: llvm.func @no_exit_code()
// CHECK:         llvm.br ^[[BODY:bb[0-9]+]]
// CHECK:       ^[[BODY]]:
// CHECK:         llvm.call @MPI_Finalize()
// CHECK-NEXT:    llvm.br ^[[AFTER:bb[0-9]+]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    llvm.return
llvm.func @no_exit_code() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %e0 = "spmd.finalize"() <{usedModel = 0 : i32}> : () -> !spmd.error
    %e1 = "spmd.barrier"(%world) <{isBlocking = true, isUniqueMatching = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  llvm.return
}
