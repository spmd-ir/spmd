// RUN: spmd-opt %s -lower-spmd-to-omp | FileCheck %s --implicit-check-not='"spmd.'
//
// -omp-to-spmd raises a worksharing construct without nowait to the construct
// and an spmd.barrier of its team right after it, and one with nowait to the
// construct alone.  Lowering reads that back: a construct its barrier follows
// has none, one no barrier follows is nowait, and every other barrier of a
// thread team is an omp.barrier.

// CHECK-LABEL: llvm.func @wsloops
// CHECK:         omp.parallel {
//                  The barrier after the loop is its own.
// CHECK-NEXT:      omp.wsloop schedule(static = %[[CHUNK:.*]] : i32) {
// CHECK-NEXT:        omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB:.*]]) to (%[[UB:.*]]) inclusive step (%[[STEP:.*]]) {
// CHECK-NEXT:          llvm.call @body(%[[I]])
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }
// CHECK-NEXT:      }
//                  No barrier follows: nowait.
// CHECK-NEXT:      omp.wsloop nowait schedule(dynamic) {
// CHECK-NEXT:        omp.loop_nest (%{{.*}}) : i32 = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) {
// CHECK:           omp.wsloop nowait schedule(runtime) {
//                  A constant between the loop and the barrier is not in the way.
// CHECK:           omp.wsloop schedule(auto) {
// CHECK:           %[[C7:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK-NEXT:      llvm.call @use(%[[C7]])
// CHECK-NEXT:      omp.terminator
llvm.func @body(i32)
llvm.func @use(i32)
llvm.func @wsloops(%lb: i32, %ub: i32, %step: i32, %chunk: i32) {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %team, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    "spmd.wsloop"(%team, %chunk) <{operandSegmentSizes = array<i32: 1, 1, 0>, schedule_kind = 0 : i32, usedModel = 5 : i32}> ({
      "spmd.loop_nest"(%lb, %ub, %step) <{loop_inclusive}> ({
      ^bb0(%i: i32):
        llvm.call @body(%i) : (i32) -> ()
        "spmd.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
    }) : (!spmd.comm, i32) -> ()
    %b0 = "spmd.barrier"(%team) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.wsloop"(%team) <{operandSegmentSizes = array<i32: 1, 0, 0>, schedule_kind = 1 : i32, usedModel = 5 : i32}> ({
      "spmd.loop_nest"(%lb, %ub, %step) ({
      ^bb0(%i: i32):
        "spmd.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
    }) : (!spmd.comm) -> ()
    "spmd.wsloop"(%team) <{operandSegmentSizes = array<i32: 1, 0, 0>, schedule_kind = 3 : i32, usedModel = 5 : i32}> ({
      "spmd.loop_nest"(%lb, %ub, %step) ({
      ^bb0(%i: i32):
        "spmd.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
    }) : (!spmd.comm) -> ()
    "spmd.wsloop"(%team) <{operandSegmentSizes = array<i32: 1, 0, 0>, schedule_kind = 4 : i32, usedModel = 5 : i32}> ({
      "spmd.loop_nest"(%lb, %ub, %step) ({
      ^bb0(%i: i32):
        "spmd.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
    }) : (!spmd.comm) -> ()
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %b1 = "spmd.barrier"(%team) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}> : (!spmd.comm) -> !spmd.error
    llvm.call @use(%c7) : (i32) -> ()
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  llvm.return
}

// `#pragma omp single` followed by `#pragma omp barrier` raises to two
// barriers; the first is the single's, the second an omp.barrier.  Without a
// barrier after it, a single is nowait.  A barrier whose error the program
// reads is still lowered, with success for the error.
// CHECK-LABEL: llvm.func @singles
// CHECK:         omp.parallel {
// CHECK-NEXT:      omp.single {
// CHECK-NEXT:        llvm.call @use
// CHECK-NEXT:        omp.terminator
// CHECK-NEXT:      }
// CHECK-NEXT:      omp.barrier
// CHECK-NEXT:      %[[OK:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      omp.single nowait {
// CHECK-NEXT:        omp.terminator
// CHECK-NEXT:      }
// CHECK-NEXT:      llvm.call @use(%[[OK]])
// CHECK-NEXT:      omp.terminator
llvm.func @singles(%x: i32) {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %team, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    "spmd.single"(%team) <{usedModel = 5 : i32}> ({
      llvm.call @use(%x) : (i32) -> ()
      "spmd.terminator"() : () -> ()
    }) : (!spmd.comm) -> ()
    %b0 = "spmd.barrier"(%team) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}> : (!spmd.comm) -> !spmd.error
    %b1 = "spmd.barrier"(%team) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}> : (!spmd.comm) -> !spmd.error
    %code = "spmd.castSPMDErrorToI32"(%b1) : (!spmd.error) -> i32
    "spmd.single"(%team) <{usedModel = 5 : i32}> ({
      "spmd.terminator"() : () -> ()
    }) : (!spmd.comm) -> ()
    llvm.call @use(%code) : (i32) -> ()
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  llvm.return
}

// spmd.sections has no terminator; omp.sections ends in one.
// CHECK-LABEL: llvm.func @sections
// CHECK:         omp.parallel {
// CHECK-NEXT:      omp.sections {
// CHECK-NEXT:        omp.section {
// CHECK-NEXT:          llvm.call @use
// CHECK-NEXT:          omp.terminator
// CHECK-NEXT:        }
// CHECK-NEXT:        omp.section {
// CHECK-NEXT:          omp.terminator
// CHECK-NEXT:        }
// CHECK-NEXT:        omp.terminator
// CHECK-NEXT:      }
// CHECK-NEXT:      omp.terminator
llvm.func @sections(%x: i32) {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %team, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    %s = "spmd.sections"(%team) <{operandSegmentSizes = array<i32: 1, 0, 0>, usedModel = 5 : i32}> ({
      "spmd.section"() ({
        llvm.call @use(%x) : (i32) -> ()
        "spmd.terminator"() : () -> ()
      }) : () -> ()
      "spmd.section"() ({
        "spmd.terminator"() : () -> ()
      }) : () -> ()
    }) : (!spmd.comm) -> !spmd.error
    %b = "spmd.barrier"(%team) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  llvm.return
}

// An orphaned worksharing loop -- in a function called from a parallel region
// -- works on the team it is called in, which is what -omp-to-spmd's
// spmd.commWorld stands for there.
// CHECK-LABEL: llvm.func @orphaned
// CHECK-NEXT:    omp.wsloop nowait {
// CHECK:         omp.barrier
llvm.func @orphaned(%lb: i32, %ub: i32, %step: i32) {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  "spmd.wsloop"(%world) <{operandSegmentSizes = array<i32: 1, 0, 0>, usedModel = 5 : i32}> ({
    "spmd.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%i: i32):
      "spmd.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
  }) : (!spmd.comm) -> ()
  llvm.call @use(%lb) : (i32) -> ()
  %b = "spmd.barrier"(%world) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}> : (!spmd.comm) -> !spmd.error
  llvm.return
}
