// RUN: spmd-opt %s -control-flow-sink | FileCheck %s
//
// -control-flow-sink moves an op into a region when all of its uses are there,
// gated only on isMemoryEffectFree. Sinking a communication op into one arm of
// an `scf.if` makes it conditional; with a rank-dependent condition only some
// ranks execute it, which is a deadlock. The effect declarations have to keep
// every communication op where it is.
//
// FileCheck ordering does the work: the op must appear *before* the scf.if.

// CHECK-LABEL: func.func @comm_op_used_in_one_arm
// CHECK:         spmd.commDup
// CHECK:         scf.if
func.func @comm_op_used_in_one_arm(%cond: i1) -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dup, %e0 = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %r = scf.if %cond -> (i32) {
    %rank, %e1 = "spmd.getRankInComm"(%dup) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    scf.yield %rank : i32
  } else {
    %z = arith.constant 0 : i32
    scf.yield %z : i32
  }
  return %r : i32
}

// The rank-dependent form of the same thing, the one that actually hangs.
// CHECK-LABEL: func.func @comm_op_used_in_rank_dependent_arm
// CHECK:         spmd.commSplit
// CHECK:         scf.if
func.func @comm_op_used_in_rank_dependent_arm(%color: i32, %key: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %isRoot = arith.cmpi eq, %rank, %zero : i32
  %new, %e0 = "spmd.commSplit"(%world, %color, %key) <{usedModel = 0 : i32}> : (!spmd.comm, i32, i32) -> (!spmd.comm, !spmd.error)
  %r = scf.if %isRoot -> (i32) {
    %size, %e1 = "spmd.getSizeOfComm"(%new) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    scf.yield %size : i32
  } else {
    scf.yield %zero : i32
  }
  return %r : i32
}

// commWorld is Pure and *does* sink: it names a constant communicator, so making
// it conditional desynchronises nobody.
// CHECK-LABEL: func.func @comm_world_may_sink
// CHECK:         scf.if
// CHECK:           spmd.commWorld
func.func @comm_world_may_sink(%cond: i1) -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = scf.if %cond -> (i32) {
    %rank, %e = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    scf.yield %rank : i32
  } else {
    %z = arith.constant 0 : i32
    scf.yield %z : i32
  }
  return %r : i32
}
