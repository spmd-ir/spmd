// RUN: spmd-opt %s -canonicalize | FileCheck %s
//
// What -canonicalize's dead-code elimination may do to communication ops.
//
// An op is trivially dead when it has no uses and every declared effect is
// either a read or an allocation of one of its own results. The SPMD dialect
// marks communicator creation as CommAlloc *and* CommWrite on the new
// communicator, and collectives/lifecycle calls as RuntimeRead + RuntimeWrite,
// so none of them qualify. That matters for verification: a mismatched
// MPI_Comm_dup nobody uses afterwards is still a mismatch, and it must survive
// the canonicalization that runs ahead of the checkers.

// An unused communicator still costs a collective call on every rank.
// CHECK-LABEL: func.func @dup_unused
// CHECK:         spmd.commDup
func.func @dup_unused() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dup, %e0 = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  return
}

// CHECK-LABEL: func.func @split_unused
// CHECK:         spmd.commSplit
func.func @split_unused(%color: i32, %key: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %new, %e0 = "spmd.commSplit"(%world, %color, %key) <{usedModel = 0 : i32}> : (!spmd.comm, i32, i32) -> (!spmd.comm, !spmd.error)
  return
}

// A dup/destroy pair whose error codes nobody inspects: both calls stay.
// CHECK-LABEL: func.func @dup_destroy_pair
// CHECK-COUNT-1: spmd.commDup
// CHECK-COUNT-1: spmd.commDestroy
func.func @dup_destroy_pair() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dup, %e0 = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %e1 = "spmd.commDestroy"(%dup) <{usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// A blocking collective whose error result is unused.
// CHECK-LABEL: func.func @barrier_unused
// CHECK:         spmd.barrier
func.func @barrier_unused() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// A matching probe consumes a message from the queue, so removing it changes
// which messages later receives match, even though nothing uses its results.
// CHECK-LABEL: func.func @mprobe_unused
// CHECK:         spmd.mProbe
func.func @mprobe_unused(%src: i32, %tag: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %msg, %e0 = "spmd.mProbe"(%world, %src, %tag) <{usedModel = 0 : i32}> : (!spmd.comm, i32, i32) -> (!spmd.msg, !spmd.error)
  return
}

// dimsCreate is local, but writes through $dims.
// CHECK-LABEL: func.func @dims_create_unused
// CHECK:         spmd.dimsCreate
func.func @dims_create_unused(%nproc: i32, %ndims: i32) {
  %dims = memref.alloca() : memref<2xi32>
  %e = "spmd.dimsCreate"(%nproc, %ndims, %dims) <{usedModel = 0 : i32}> : (i32, i32, memref<2xi32>) -> !spmd.error
  return
}

// The constant-like ops are Pure and do go away when unused.
// CHECK-LABEL: func.func @constants_unused
// CHECK-NOT:     spmd.commWorld
// CHECK-NOT:     spmd.constDatatype
// CHECK:         return
func.func @constants_unused() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  return
}
