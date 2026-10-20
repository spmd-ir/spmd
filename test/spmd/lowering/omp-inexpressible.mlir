// RUN: spmd-opt %s -split-input-file -lower-spmd-to-omp -verify-diagnostics
//
// What OpenMP cannot express, reported before anything is rewritten, each
// with its reason.

func.func @other_kinds() {
  // expected-error @+1 {{'spmd.finalize' op has no OpenMP counterpart; OpenMP lowers thread teams, worksharing, tasks, barriers and the thread number and count queries}}
  %e = "spmd.finalize"() <{usedModel = 5 : i32}> : () -> !spmd.error
  return
}

// -----

func.func @process_team() {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.parallel' op is a process team; OpenMP starts threads, not processes}}
  "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  return
}

// -----

func.func @private_vars(%x: !llvm.ptr) {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %team, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  // expected-error @+1 {{'spmd.parallel' op has private variables, but -omp-to-spmd kept none of the omp.private declarations that say how to copy them}}
  "spmd.parallel"(%team, %x) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 1, 0>, usedModel = 5 : i32}> ({
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm, !llvm.ptr) -> ()
  return
}

// -----

// OpenMP only ever asks the current team: a query of the outer team inside an
// inner parallel region has no call.
func.func @outer_team_inside() {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %outer, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  "spmd.parallel"(%outer) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    %inner, %e1 = "spmd.commDup"(%outer) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    // expected-note @+1 {{the current team starts here}}
    "spmd.parallel"(%inner) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      // expected-error @+1 {{'spmd.getRankInComm' op names a team other than the one of the parallel region it is in; OpenMP can only act on the current team}}
      %r, %e2 = "spmd.getRankInComm"(%outer) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
      "spmd.parallel_yield"() : () -> ()
    }) : (!spmd.comm) -> ()
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  return
}

// -----

// A team outlives its parallel region only as a value; OpenMP cannot ask it.
func.func @team_after_region() {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %team, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  // expected-error @+1 {{'spmd.barrier' op names the team of a parallel region it is not in; OpenMP can only act on the current team}}
  %b = "spmd.barrier"(%team) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// -----

// A thread team is no communicator another model's call could take.
func.func @team_used_by_mpi() {
  // expected-note @+1 {{team produced here}}
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.barrier' op uses an OpenMP team, which is no communicator a call of another programming model can take}}
  %b = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// -----

func.func @reductions(%x: !llvm.ptr) {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %team, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  // expected-error @+1 {{'spmd.parallel' op names a reduction that is not declared by an spmd.declare_reduction}}
  "spmd.parallel"(%team, %x) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 1>, reduction_syms = [@nowhere], usedModel = 5 : i32}> ({
  ^bb0(%px: !llvm.ptr):
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm, !llvm.ptr) -> ()
  return
}

// -----

spmd.declare_reduction @add_i32 : i32 init {
^bb0(%arg0: i32):
  %c0 = arith.constant 0 : i32
  "spmd.yield"(%c0) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %s = arith.addi %a, %b : i32
  "spmd.yield"(%s) : (i32) -> ()
}

func.func @reduction_without_argument(%x: !llvm.ptr) {
  // expected-error @+1 {{'spmd.taskgroup' op has 1 reductions, but its region has no entry argument for the private copy of each}}
  %g = "spmd.taskgroup"(%x) <{operandSegmentSizes = array<i32: 1>, task_reduction_syms = [@add_i32], usedModel = 5 : i32}> ({
    "spmd.terminator"() : () -> ()
  }) : (!llvm.ptr) -> !spmd.error
  return
}

// -----

func.func @depend_without_kinds(%p: !llvm.ptr) {
  // expected-error @+1 {{'spmd.taskwait' op has depend variables but no depend_kinds for them}}
  %w = "spmd.taskwait"(%p) <{usedModel = 5 : i32}> : (!llvm.ptr) -> !spmd.error
  return
}

// -----

func.func @nonblocking_barrier() {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.barrier' op is a non-blocking barrier, which OpenMP does not have}}
  %req, %b = "spmd.barrier"(%world) <{isBlocking = false, usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.req, !spmd.error)
  return
}
