// RUN: spmd-opt %s -cse | FileCheck %s
//
// CSE on communication ops. Two MPI_Comm_dup calls produce two *distinct*
// communicators, so merging them is wrong regardless of block structure; across
// regions it would also be a motion out of a conditional. Communicator creation
// writes the new communicator, which rules CSE out. Constant-like ops and
// read-only local queries remain foldable.

// CHECK-LABEL: func.func @dup_twice_same_block
// CHECK-COUNT-2: spmd.commDup
func.func @dup_twice_same_block() -> (i32, i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a, %e0 = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %b, %e1 = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %r0, %e2 = "spmd.getRankInComm"(%a) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %r1, %e3 = "spmd.getRankInComm"(%b) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  return %r0, %r1 : i32, i32
}

// CHECK-LABEL: func.func @dup_across_regions
// CHECK-COUNT-2: spmd.commDup
func.func @dup_across_regions(%cond: i1) -> i32 {
  %zero = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a, %e0 = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %r = scf.if %cond -> (i32) {
    %b, %e1 = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %rank, %e2 = "spmd.getRankInComm"(%b) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    scf.yield %rank : i32
  } else {
    scf.yield %zero : i32
  }
  %r0, %e3 = "spmd.getRankInComm"(%a) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %sum = arith.addi %r, %r0 : i32
  return %sum : i32
}

// Two identical blocking collectives are two calls, even back to back.
// CHECK-LABEL: func.func @barrier_twice
// CHECK-COUNT-2: spmd.barrier
func.func @barrier_twice() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// commWorld is Pure, so duplicates collapse.
// CHECK-LABEL: func.func @comm_world_cses
// CHECK-COUNT-1: spmd.commWorld
// CHECK-NOT:     spmd.commWorld
func.func @comm_world_cses() -> (i32, i32) {
  %a = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %b = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r0, %e0 = "spmd.getRankInComm"(%a) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %r1, %e1 = "spmd.getRankInComm"(%b) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  return %r0, %r1 : i32, i32
}

// The rank query only reads, so adjacent duplicates fold.
// CHECK-LABEL: func.func @rank_query_cses
// CHECK-COUNT-1: spmd.getRankInComm
// CHECK-NOT:     spmd.getRankInComm
func.func @rank_query_cses() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r0, %e0 = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %r1, %e1 = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %sum = arith.addi %r0, %r1 : i32
  return %sum : i32
}
