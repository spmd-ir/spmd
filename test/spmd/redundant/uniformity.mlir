// RUN: spmd-opt %s -split-input-file -eliminate-redundant-comm | FileCheck %s
//
// Establishing a fact (docs/optimizations.md, "Available collectives"): a
// collective may only make a later one redundant if *every* rank performed
// it.  These are the cases that separate "available on this rank" from
// "available on all of them" - the ones where getting it wrong is a
// silent-corruption bug rather than a missed optimisation.
//
// Each case is a program of its own, with its body in main's process-level
// spmd.parallel: a fact is only established by an op every rank executes, and
// that is only known there.

// @rank_dependent_if: the condition is rank-dependent, so neither arm
// establishes anything, and the collective after the join has to stay.  Were
// the arms treated as uniform, the removal would be textual - every rank would
// lose the third allreduce - while the ranks that took neither arm never had
// the result at all.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-3: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %rank, %rerr = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %isRoot = arith.cmpi eq, %rank, %zero : i32
    scf.if %isRoot {
      %a = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    } else {
      %b = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    }
    %e = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @uniform_if_both_arms: the same shape on a uniform condition - the
// communicator size, which every rank agrees on.  Both arms perform the
// collective, so it is available after the join and the third one goes.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %size, %serr = "spmd.getSizeOfComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %isSingle = arith.cmpi eq, %size, %one : i32
    scf.if %isSingle {
      %a = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    } else {
      %b = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    }
    %e = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @uniform_if_one_arm: only one arm performs it, so it is not available on the
// other path: the join is an intersection, not a union.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %size, %serr = "spmd.getSizeOfComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %isSingle = arith.cmpi eq, %size, %one : i32
    scf.if %isSingle {
      %a = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    }
    %e = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @fact_survives_rank_dependent_branch: a collective before a rank-dependent
// branch is still available after it, as long as nothing in the branch writes
// the buffers.  Killing needs no uniformity; only establishing does.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i32
    %zero = arith.constant 0 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %rank, %rerr = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %other = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %isRoot = arith.cmpi eq, %rank, %zero : i32
    scf.if %isRoot {
      memref.store %c42, %other[%c0] : memref<1xi32>
    }
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @rank_dependent_write_kills: ... and a write on *one* rank still kills it,
// because a fact only some ranks hold is a fact no rank may use.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i32
    %zero = arith.constant 0 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %rank, %rerr = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %isRoot = arith.cmpi eq, %rank, %zero : i32
    scf.if %isRoot {
      memref.store %c42, %send[%c0] : memref<1xi32>
    }
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @rank_dependent_loop: a rank-dependent loop bound: the body may run a
// different number of times per rank, so nothing established inside it is
// available to every rank.  ExecKind alone says every rank executes the body,
// which is why isUniformlyExecuted also looks at the trip count.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %rank, %rerr = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %ub = arith.index_cast %rank : i32 to index
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    scf.for %i = %c0 to %ub step %c1 {
      %a = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    }
    %e = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @callee: availability does not cross a call: neither into the callee nor
// back out.
// CHECK-LABEL: func.func @callee
// CHECK:         spmd.allreduce
// CHECK-LABEL: func.func @main
// CHECK:         spmd.allreduce
func.func @callee(%comm: !spmd.comm, %send: memref<1xi32>, %recv: memref<1xi32>) {
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %e = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    func.call @callee(%world, %send, %recv) : (!spmd.comm, memref<1xi32>, memref<1xi32>) -> ()
    %e = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
