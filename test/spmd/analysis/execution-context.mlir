// RUN: spmd-opt %s -split-input-file -test-spmd-execution-context | FileCheck %s
//
// ExecutionContextAnalysis: before every op, for each parallel level from the
// outside in, which ranks (or threads) execute it, and whether all of them do
// so equally often and none may have left. The checkers and the
// optimisations read this; -test-spmd-execution-context prints it:
//
//   ExecutionContext[{mg=<usedModel> proc=<process level?> kind=<ExecKind>
//                     execs=[<ranks>]} ... nonuniform-count early-exit]
//
// The first level, mg=-1, is the implicit one a function is entered at. kind
// is All, Static (exactly the listed ranks), AllBut (all but the listed),
// One (a single thread whose number is not known), or Dynamic (not known).
// nonuniform-count: a condition, loop bound or call on the way here depends on
// the rank, so the ranks that get here may do so different numbers of times.
// early-exit: some rank may have stopped before this point.

// Straight-line code in main: every rank, equally often.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} ]"
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A guard on the rank selects ranks: `rank == 1` is rank 1, its else branch
// every other rank, in either order of the comparison's operands. After the if
// every rank continues, equally often.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=Static execs=[1]} nonuniform-count ]", test.name = "then"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=AllBut execs=[1]} nonuniform-count ]", test.name = "else"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=Static execs=[1]} nonuniform-count ]", test.name = "reversed"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} ]", test.name = "after"
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %is1 = arith.cmpi eq, %rank, %c1 : i32
    scf.if %is1 {
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "then"} : (!spmd.comm) -> !spmd.error
    } else {
      %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "else"} : (!spmd.comm) -> !spmd.error
    }
    %rev = arith.cmpi eq, %c1, %rank : i32
    scf.if %rev {
      %e2 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "reversed"} : (!spmd.comm) -> !spmd.error
    }
    %e3 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "after"} : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// `rank != 2` is every rank but 2, and its else branch rank 2.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=AllBut execs=[2]} nonuniform-count ]", test.name = "then"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=Static execs=[2]} nonuniform-count ]", test.name = "else"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} ]", test.name = "after"
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %not2 = arith.cmpi ne, %rank, %c2 : i32
    scf.if %not2 {
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "then"} : (!spmd.comm) -> !spmd.error
    } else {
      %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "else"} : (!spmd.comm) -> !spmd.error
    }
    %e2 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "after"} : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A rank condition that is not a comparison with one rank selects ranks the
// analysis does not enumerate. What follows the if is not checked here: it
// stays Dynamic (MR 16's description, finding 20).
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=Dynamic} nonuniform-count ]", test.name = "then"
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %low = arith.cmpi slt, %rank, %c2 : i32
    scf.if %low {
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "then"} : (!spmd.comm) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A condition on something every rank agrees on, the size, selects no ranks
// and keeps the count uniform. A loop bound on the rank does not change who
// runs the body, but how often; leaving the loop restores the count.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} ]", test.name = "size"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} nonuniform-count ]", test.name = "loop"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} ]", test.name = "after"
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %size, %es = "spmd.getSizeOfComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %big = arith.cmpi sgt, %size, %c1 : i32
    scf.if %big {
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "size"} : (!spmd.comm) -> !spmd.error
    }
    %n = arith.index_cast %rank : i32 to index
    scf.for %i = %i0 to %n step %i1 {
      %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "loop"} : (!spmd.comm) -> !spmd.error
    }
    %e2 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "after"} : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// exit() under a rank guard: the ranks that reach the next op still all execute
// it, but one may be gone, and it stays gone.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} early-exit ]", test.name = "after"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} early-exit ]", test.name = "later"
llvm.func @exit(i32) attributes {noreturn}
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %is0 = arith.cmpi eq, %rank, %c0 : i32
    scf.if %is0 {
      llvm.call @exit(%c1) : (i32) -> ()
    }
    %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "after"} : (!spmd.comm) -> !spmd.error
    %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "later"} : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A thread-level region adds a level: model 5, proc=0. single and critical run
// their body on one thread, a guard on the thread number on that thread; the
// process level is untouched inside, and after the region the level is gone.
// Inside, the count is not claimed uniform even with a constant number of
// threads: the analysis does not know how often the dialect's own regions
// run (MR 16's description, finding 23).
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} {mg=5 proc=0 kind=All} nonuniform-count ]", test.name = "team"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} {mg=5 proc=0 kind=One} nonuniform-count ]", test.name = "single"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} {mg=5 proc=0 kind=One} nonuniform-count ]", test.name = "critical"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} {mg=5 proc=0 kind=Static execs=[0]} nonuniform-count ]", test.name = "master"
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=All} ]", test.name = "joined"
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    "spmd.parallel"(%team, %c4) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 1, 0, 0>, usedModel = 5 : i32}> ({
      %e0 = "spmd.barrier"(%team) <{isBlocking = true, usedModel = 5 : i32}> {test.name = "team"} : (!spmd.comm) -> !spmd.error
      "spmd.single"(%team) <{usedModel = 5 : i32}> ({
        %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "single"} : (!spmd.comm) -> !spmd.error
        "spmd.terminator"() : () -> ()
      }) : (!spmd.comm) -> ()
      %ec = "spmd.critical"() <{usedModel = 5 : i32}> ({
        %e2 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "critical"} : (!spmd.comm) -> !spmd.error
        "spmd.terminator"() : () -> ()
      }) : () -> !spmd.error
      %tid, %et = "spmd.getRankInComm"(%team) <{usedModel = 5 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
      %master = arith.cmpi eq, %tid, %c0 : i32
      scf.if %master {
        %e3 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "master"} : (!spmd.comm) -> !spmd.error
      }
      "spmd.parallel_yield"() : () -> ()
    }) : (!spmd.comm, i32) -> ()
    %e4 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "joined"} : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A function called under a guard runs on the ranks of the call site. One
// nothing in the module calls may be entered by any subset of ranks, any
// number of times.
// CHECK-LABEL: func.func private @helper
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} {mg=0 proc=1 kind=Static execs=[3]} nonuniform-count ]"
// CHECK-LABEL: func.func @library
// CHECK:         spmd.barrier{{.*}}test.exec = "ExecutionContext[{mg=-1 proc=1 kind=All} nonuniform-count ]"
func.func private @helper(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %is3 = arith.cmpi eq, %rank, %c3 : i32
    scf.if %is3 {
      func.call @helper(%world) : (!spmd.comm) -> ()
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
func.func @library(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
