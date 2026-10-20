// RUN: spmd-opt %s -eliminate-redundant-barriers | FileCheck %s
//
// A blocking barrier immediately preceded by an equivalent one synchronizes
// nothing the first did not, so it goes. Every other shape keeps both.

// CHECK-LABEL: func.func @adjacent
// CHECK-COUNT-1: spmd.barrier
// CHECK-NOT:     spmd.barrier
func.func @adjacent() {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// A run of three folds to one.
// CHECK-LABEL: func.func @run_of_three
// CHECK-COUNT-1: spmd.barrier
// CHECK-NOT:     spmd.barrier
func.func @run_of_three() {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e2 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// Anything in between keeps both.
// CHECK-LABEL: func.func @not_adjacent
// CHECK-COUNT-2: spmd.barrier
func.func @not_adjacent() {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<1xi32>
  %e0 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  memref.store %c42, %buf[%c0] : memref<1xi32>
  %e1 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// The second barrier's error code is inspected.
// CHECK-LABEL: func.func @error_observed
// CHECK-COUNT-2: spmd.barrier
func.func @error_observed() -> i32 {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %code = "spmd.castSPMDErrorToI32"(%e1) : (!spmd.error) -> i32
  return %code : i32
}

// Barriers on two communicators synchronize two different groups.
// CHECK-LABEL: func.func @different_comms
// CHECK-COUNT-2: spmd.barrier
func.func @different_comms(%other: !spmd.comm) {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.barrier"(%other) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// Nor are barriers of two programming models the same operation.
// CHECK-LABEL: func.func @different_models
// CHECK-COUNT-2: spmd.barrier
func.func @different_models() {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// A non-blocking barrier completes somewhere else.
// CHECK-LABEL: func.func @non_blocking
// CHECK-COUNT-2: spmd.barrier
func.func @non_blocking() {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %req, %e1 = "spmd.barrier"(%comm) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.req, !spmd.error)
  %w = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}
