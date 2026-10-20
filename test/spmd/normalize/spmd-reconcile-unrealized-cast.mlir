// RUN: spmd-opt %s -split-input-file -spmd-reconcile-unrealized-cast | FileCheck %s
//
// A select between two handles the raiser cast from the program's int, whose
// result is only ever cast back, selects between the ints: the casts on both
// sides go. The same pattern runs inside -convert-apiCalls-to-spmd, and
// spmd-verify runs this pass after raising.
//
// Only when the two sides have the same type and every use casts back to it
// is the rewrite valid; neither is checked yet (MR 16's description, finding
// 18), so neither mismatch is tested here.

// CHECK-LABEL: func.func @same(
// CHECK-SAME:    %[[C:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK-NEXT:    %[[S:.*]] = arith.select %[[C]], %[[A]], %[[B]] : i32
// CHECK-NEXT:    return %[[S]] : i32
func.func @same(%c: i1, %a: i32, %b: i32) -> i32 {
  %ca = builtin.unrealized_conversion_cast %a : i32 to !spmd.comm
  %cb = builtin.unrealized_conversion_cast %b : i32 to !spmd.comm
  %s = arith.select %c, %ca, %cb : !spmd.comm
  %back = builtin.unrealized_conversion_cast %s : !spmd.comm to i32
  return %back : i32
}

// -----

// A use of the selected handle itself: nothing to reconcile.
// CHECK-LABEL: func.func @used_as_handle(
// CHECK:         %[[S:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : !spmd.comm
// CHECK:         "spmd.barrier"(%[[S]])
func.func @used_as_handle(%c: i1, %a: i32, %b: i32) {
  %ca = builtin.unrealized_conversion_cast %a : i32 to !spmd.comm
  %cb = builtin.unrealized_conversion_cast %b : i32 to !spmd.comm
  %s = arith.select %c, %ca, %cb : !spmd.comm
  %e = "spmd.barrier"(%s) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// -----

// One side is no cast: left alone.
// CHECK-LABEL: func.func @one_side(
// CHECK:         arith.select %{{.*}}, %{{.*}}, %{{.*}} : !spmd.comm
// CHECK:         builtin.unrealized_conversion_cast
func.func @one_side(%c: i1, %a: i32, %world: !spmd.comm) -> i32 {
  %ca = builtin.unrealized_conversion_cast %a : i32 to !spmd.comm
  %s = arith.select %c, %ca, %world : !spmd.comm
  %back = builtin.unrealized_conversion_cast %s : !spmd.comm to i32
  return %back : i32
}
