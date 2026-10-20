// RUN: spmd-opt %s -lower-spmd-to-omp | FileCheck %s --implicit-check-not='"spmd.'
//
// A thread team as -omp-to-spmd raises it: an spmd.commDup of the enclosing
// team, used by a thread-level spmd.parallel, whose queries ask that team.  The
// team is no OpenMP object and goes; the parallel region is omp.parallel again,
// and the queries are the calls they were raised from.

// CHECK-DAG: llvm.func @omp_get_thread_num() -> i32
// CHECK-DAG: llvm.func @omp_get_num_threads() -> i32

// CHECK-LABEL: llvm.func @team
// CHECK-NOT:     llvm.mlir.constant
// CHECK:         omp.parallel num_threads(%[[N:.*]] : i32) {
// CHECK-NEXT:      %[[T:.*]] = llvm.call @omp_get_thread_num() : () -> i32
// CHECK-NEXT:      %[[S:.*]] = llvm.call @omp_get_num_threads() : () -> i32
// CHECK-NEXT:      llvm.call @use(%[[T]], %[[S]])
// CHECK-NEXT:      omp.terminator
// CHECK-NEXT:    }
llvm.func @use(i32, i32)
llvm.func @team(%n: i32) {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %team, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  "spmd.parallel"(%team, %n) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 1, 0, 0>, usedModel = 5 : i32}> ({
    %rank, %e0 = "spmd.getRankInComm"(%team) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
    %size, %e1 = "spmd.getSizeOfComm"(%team) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
    llvm.call @use(%rank, %size) : (i32, i32) -> ()
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm, i32) -> ()
  llvm.return
}

// A nested team duplicates the one it is in, and its queries ask the inner
// team; the outer region's queries, before and after it, still ask the outer.
// A read error code is OpenMP's success.
// CHECK-LABEL: llvm.func @nested
// CHECK:         omp.parallel {
// CHECK-NEXT:      %[[OUTER:.*]] = llvm.call @omp_get_thread_num()
// CHECK-NEXT:      omp.parallel {
// CHECK-NEXT:        %[[INNER:.*]] = llvm.call @omp_get_thread_num()
// CHECK-NEXT:        %[[OK:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:        llvm.call @use(%[[OUTER]], %[[INNER]])
// CHECK-NEXT:        llvm.call @use(%[[OK]], %[[OK]])
// CHECK-NEXT:        omp.terminator
// CHECK-NEXT:      }
// CHECK-NEXT:      omp.terminator
// CHECK-NEXT:    }
llvm.func @nested() {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %outer, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  "spmd.parallel"(%outer) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    %o, %e0 = "spmd.getRankInComm"(%outer) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
    %inner, %e1 = "spmd.commDup"(%outer) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    "spmd.parallel"(%inner) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      %i, %e2 = "spmd.getRankInComm"(%inner) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
      %c = "spmd.castSPMDErrorToI32"(%e2) : (!spmd.error) -> i32
      llvm.call @use(%o, %i) : (i32, i32) -> ()
      llvm.call @use(%c, %c) : (i32, i32) -> ()
      "spmd.parallel_yield"() : () -> ()
    }) : (!spmd.comm) -> ()
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  llvm.return
}

// An orphaned query -- in a function a parallel region calls -- asks
// spmd.getParallelComm, which only folds inside a region.  It is the current
// team all the same, and so is the initial team outside every region.
// CHECK-LABEL: llvm.func @orphaned
// CHECK-NEXT:    %[[T:.*]] = llvm.call @omp_get_thread_num() : () -> i32
// CHECK-NEXT:    %[[S:.*]] = llvm.call @omp_get_num_threads() : () -> i32
// CHECK-NEXT:    llvm.call @use(%[[T]], %[[S]])
llvm.func @orphaned() {
  %team = "spmd.getParallelComm"() : () -> !spmd.comm
  %rank, %e0 = "spmd.getRankInComm"(%team) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %size, %e1 = "spmd.getSizeOfComm"(%world) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
  llvm.call @use(%rank, %size) : (i32, i32) -> ()
  llvm.return
}
