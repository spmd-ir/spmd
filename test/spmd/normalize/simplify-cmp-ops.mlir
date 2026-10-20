// RUN: spmd-opt %s -split-input-file -simplify-cmp-ops | FileCheck %s
//
// reactant-clang writes the short circuit `if (x == A || x == B)` as an
// scf.if that yields 0 when the first comparison holds and `x != B`
// otherwise, and then tests that result against 0. -simplify-cmp-ops folds
// this back into `x == A || x == B`, so the rank a guard selects is one
// comparison the analyses can read (spmd-verify runs it after raising).
//
// Only the shape above is rewritten. The first comparison must be `eq` for the
// result to be this disjunction; that is not checked yet, and the rewrite of
// `x != A || x == B` is wrong (MR 16's description, finding 17), so it is not
// tested here.

// CHECK-LABEL: func.func @eq_or_eq(
// CHECK-SAME:    %[[X:.*]]: i32)
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
// CHECK:         %[[A:.*]] = arith.cmpi eq, %[[X]], %[[C0]] : i32
// CHECK:         %[[B:.*]] = arith.cmpi eq, %[[X]], %[[C1]] : i32
// CHECK:         %[[OR:.*]] = arith.ori %[[A]], %[[B]] : i1
// CHECK:         return %[[OR]] : i1
func.func @eq_or_eq(%x: i32) -> i1 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %first = arith.cmpi eq, %x, %c0 : i32
  %r = scf.if %first -> (i32) {
    scf.yield %c0 : i32
  } else {
    %second = arith.cmpi ne, %x, %c1 : i32
    %w = arith.extui %second : i1 to i32
    scf.yield %w : i32
  }
  %taken = arith.cmpi eq, %r, %c0 : i32
  return %taken : i1
}

// -----

// The constant may come first, the else branch may yield the i1 itself, and
// the result may be cast to index before it is tested.
// CHECK-LABEL: func.func @variants(
// CHECK-SAME:    %[[X:.*]]: i32)
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : i32
// CHECK:         %[[A:.*]] = arith.cmpi eq, %[[X]], %[[C2]] : i32
// CHECK:         %[[B:.*]] = arith.cmpi eq, %[[X]], %[[C3]] : i32
// CHECK:         %[[OR:.*]] = arith.ori %[[A]], %[[B]] : i1
// CHECK:         return %[[OR]] : i1
func.func @variants(%x: i32) -> i1 {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %first = arith.cmpi eq, %c2, %x : i32
  %r = scf.if %first -> (i1) {
    scf.yield %false : i1
  } else {
    %second = arith.cmpi ne, %c3, %x : i32
    scf.yield %second : i1
  }
  %wide = arith.index_castui %r : i1 to index
  %taken = arith.cmpi eq, %wide, %c0 : index
  return %taken : i1
}

// -----

// Two different values compared: no rank the guard selects, left alone.
// CHECK-LABEL: func.func @different_values
// CHECK:         scf.if
// CHECK-NOT:     arith.ori
func.func @different_values(%x: i32, %y: i32) -> i1 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %first = arith.cmpi eq, %x, %c0 : i32
  %r = scf.if %first -> (i32) {
    scf.yield %c0 : i32
  } else {
    %second = arith.cmpi ne, %y, %c1 : i32
    %w = arith.extui %second : i1 to i32
    scf.yield %w : i32
  }
  %taken = arith.cmpi eq, %r, %c0 : i32
  return %taken : i1
}

// -----

// The then branch yields 1, so the result is not the disjunction.
// CHECK-LABEL: func.func @then_not_zero
// CHECK:         scf.if
// CHECK-NOT:     arith.ori
func.func @then_not_zero(%x: i32) -> i1 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %first = arith.cmpi eq, %x, %c0 : i32
  %r = scf.if %first -> (i32) {
    scf.yield %c1 : i32
  } else {
    %second = arith.cmpi ne, %x, %c1 : i32
    %w = arith.extui %second : i1 to i32
    scf.yield %w : i32
  }
  %taken = arith.cmpi eq, %r, %c0 : i32
  return %taken : i1
}

// -----

// The else comparison is `eq`: `x == A ? 0 : x == B` is not the disjunction.
// CHECK-LABEL: func.func @else_eq
// CHECK:         scf.if
// CHECK-NOT:     arith.ori
func.func @else_eq(%x: i32) -> i1 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %first = arith.cmpi eq, %x, %c0 : i32
  %r = scf.if %first -> (i32) {
    scf.yield %c0 : i32
  } else {
    %second = arith.cmpi eq, %x, %c1 : i32
    %w = arith.extui %second : i1 to i32
    scf.yield %w : i32
  }
  %taken = arith.cmpi eq, %r, %c0 : i32
  return %taken : i1
}

// -----

// The result is tested against 1, not 0.
// CHECK-LABEL: func.func @tested_against_one
// CHECK:         scf.if
// CHECK-NOT:     arith.ori
func.func @tested_against_one(%x: i32) -> i1 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %first = arith.cmpi eq, %x, %c0 : i32
  %r = scf.if %first -> (i32) {
    scf.yield %c0 : i32
  } else {
    %second = arith.cmpi ne, %x, %c1 : i32
    %w = arith.extui %second : i1 to i32
    scf.yield %w : i32
  }
  %taken = arith.cmpi eq, %r, %c1 : i32
  return %taken : i1
}
