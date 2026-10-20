// RUN: spmd-opt %s -split-input-file -convert-arith-select-to-if | FileCheck %s
//
// An if-converted fork, `op(select(c, a, b))`, back into control flow: an
// scf.if on c that runs the op on a in one branch and on b in the other, each
// yielding the op's results. Every user of the select is converted on its own,
// anchored at the user, so values the user needs that are defined after the
// select still dominate it. A terminator cannot move into a branch and keeps
// the select. spmd-verify does not run this pass (see spmd/CLAUDE.md).

// CHECK-LABEL: func.func @fork(
// CHECK-SAME:    %[[C:.*]]: i1, %[[A:.*]]: !spmd.comm, %[[B:.*]]: !spmd.comm)
// CHECK:         %[[SEVEN:.*]] = arith.constant 7 : i32
// CHECK:         %[[SIZE:.*]]:2 = scf.if %[[C]] -> (i32, !spmd.error) {
// CHECK-NEXT:      %[[SA:.*]], %[[EA:.*]] = "spmd.getSizeOfComm"(%[[A]])
// CHECK-NEXT:      scf.yield %[[SA]], %[[EA]] : i32, !spmd.error
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[SB:.*]], %[[EB:.*]] = "spmd.getSizeOfComm"(%[[B]])
// CHECK-NEXT:      scf.yield %[[SB]], %[[EB]] : i32, !spmd.error
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.if %[[C]] -> (!spmd.error) {
// CHECK-NEXT:      "spmd.barrier"(%[[A]])
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    } else {
// CHECK-NEXT:      "spmd.barrier"(%[[B]])
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[SUM:.*]] = arith.addi %[[SIZE]]#0, %[[SEVEN]] : i32
// CHECK-NEXT:    return %[[SUM]] : i32
// CHECK-NOT:     arith.select
func.func @fork(%c: i1, %a: !spmd.comm, %b: !spmd.comm) -> i32 {
  %s = arith.select %c, %a, %b : !spmd.comm
  %n = arith.constant 7 : i32
  %size, %e = "spmd.getSizeOfComm"(%s) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %e2 = "spmd.barrier"(%s) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %sum = arith.addi %size, %n : i32
  return %sum : i32
}

// -----

// CHECK-LABEL: func.func @terminator(
// CHECK:         %[[S:.*]] = arith.select
// CHECK-NEXT:    return %[[S]] : i32
func.func @terminator(%c: i1, %a: i32, %b: i32) -> i32 {
  %s = arith.select %c, %a, %b : i32
  return %s : i32
}
