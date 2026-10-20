// RUN: spmd-opt %s -split-input-file -test-spmd-uniformity | FileCheck %s
//
// isUniformlyExecuted: every rank executes the op, each the same number of
// times. One way for control to diverge per case, and each case is a program of
// its own, since a rank that may have left taints everything after it.
//
// ExecKind says which ranks execute an op, and several of these shapes leave
// that set alone while still diverging: a rank-dependent loop bound, an exit()
// under a rank guard, a function no call in the module shows is entered by
// every rank. roundtrip's ExecutionUniformity missed the last two.

// Straight-line code.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = true
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

// A condition every rank agrees on: the communicator size.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = true
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %size, %es = "spmd.getSizeOfComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %cond = arith.cmpi sgt, %size, %c1 : i32
    scf.if %cond {
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A condition on the rank: not uniform inside, uniform again after the if.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = true
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %isRoot = arith.cmpi eq, %rank, %c0 : i32
    scf.if %isRoot {
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A condition nothing shows every rank agrees on: a value received from
// another rank.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c1_64 = arith.constant 1 : i64
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %buf = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
    %er = "spmd.recv"(%world, %buf, %c1_64, %dt, %c0, %c0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
    %v = llvm.load %buf : !llvm.ptr -> i32
    %cond = arith.cmpi sgt, %v, %c0 : i32
    scf.if %cond {
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A loop whose bound is the rank: every rank enters it, a different number of
// times. Uniform again after the loop.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = true
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %ub = arith.index_cast %rank : i32 to index
    scf.for %i = %i0 to %ub step %i1 {
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// The same loop with a constant bound.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = true
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i8 = arith.constant 8 : index
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    scf.for %i = %i0 to %i8 step %i1 {
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// scf.while with a rank-dependent exit test.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %n = scf.while (%i = %c0) : (i32) -> i32 {
      %cond = arith.cmpi slt, %i, %rank : i32
      scf.condition(%cond) %i : i32
    } do {
    ^bb0(%i: i32):
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      %next = arith.addi %i, %c1 : i32
      scf.yield %next : i32
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A block reached through a rank-dependent cf.cond_br; the block where both
// paths meet again is reached by every rank.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = true
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %isRoot = arith.cmpi eq, %rank, %c0 : i32
    cf.cond_br %isRoot, ^root, ^merge
  ^root:
    %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    cf.br ^merge
  ^merge:
    %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// The shape the raiser gives `if (rank == 1) exit(1);`: a flag out of the if,
// and a cf.switch into an unreachable block. Rank 1 does not get past it.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %isOne = arith.cmpi eq, %rank, %c1 : i32
    %flag = scf.if %isOne -> (i32) {
      scf.yield %c1 : i32
    } else {
      scf.yield %c0 : i32
    }
    cf.switch %flag : i32, [
      default: ^exit,
      0: ^go
    ]
  ^go:
    %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  ^exit:
    llvm.unreachable
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// exit() called under a rank guard, with the if left structured: the ranks
// that went on are fewer than the ranks that reached the if.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %isOne = arith.cmpi eq, %rank, %c1 : i32
    scf.if %isOne {
      llvm.call @exit(%c1) {noreturn} : (i32) -> ()
    }
    %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
llvm.func @exit(i32) attributes {noreturn}

// -----

// exit() under a condition every rank agrees on: either all ranks stop or none
// does.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = true
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %size, %es = "spmd.getSizeOfComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %tooSmall = arith.cmpi slt, %size, %c1 : i32
    scf.if %tooSmall {
      llvm.call @exit(%c1) {noreturn} : (i32) -> ()
    }
    %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
llvm.func @exit(i32) attributes {noreturn}

// -----

// A callee called under a rank guard, and one called by every rank. Their
// bodies look alike; only the call sites differ.
// CHECK-LABEL: func.func private @guarded
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
// CHECK-LABEL: func.func private @plain
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = true
func.func private @guarded(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
func.func private @plain(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %isRoot = arith.cmpi eq, %rank, %c0 : i32
    scf.if %isRoot {
      func.call @guarded(%world) : (!spmd.comm) -> ()
    }
    func.call @plain(%world) : (!spmd.comm) -> ()
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A public function with no caller in the module. roundtrip took it to be an
// entry point every rank runs; only main is.
// CHECK-LABEL: func.func @helper
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
// CHECK-LABEL: func.func @main
func.func @helper(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// Inside a thread team, a process-level op runs once per thread.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    "spmd.parallel"(%world) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      "spmd.parallel_yield"() : () -> ()
    }) : (!spmd.comm) -> ()
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// isUniformValue on the operands of a collective: a constant count, and a
// count derived from the rank.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.allreduce{{.*}}test.uniform_exec = true, test.uniform_operands = true
// CHECK:         spmd.allreduce{{.*}}test.uniform_exec = true, test.uniform_operands = false
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i64
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
    %recv = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
    %e0 = "spmd.allreduce"(%world, %send, %recv, %c8, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %n = arith.extsi %rank : i32 to i64
    %e1 = "spmd.allreduce"(%world, %send, %recv, %n, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A loop built from cf branches whose exit test is the rank. The body
// post-dominates the branch that loops back to it, so the question is not
// whether it runs but how often.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = true
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    cf.br ^body(%c0 : i32)
  ^body(%i: i32):
    %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    %next = arith.addi %i, %c1 : i32
    %again = arith.cmpi slt, %next, %rank : i32
    cf.cond_br %again, ^body(%next : i32), ^done
  ^done:
    %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// An exit() only rank 1 takes, inside spmd.parallel: rank 1 does not reach
// the code after the parallel region either.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
// CHECK:         spmd.barrier{{.*}}test.uniform_exec = false
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %isOne = arith.cmpi eq, %rank, %c1 : i32
    scf.if %isOne {
      llvm.call @exit(%c1) {noreturn} : (i32) -> ()
    }
    %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return %r : i32
}
llvm.func @exit(i32) attributes {noreturn}
