// RUN: spmd-opt %s -split-input-file -check-collectives -verify-diagnostics
//
// -check-collectives on hand-written SPMD IR: the collective-ordering findings
// it reports, and the shapes it must leave alone. Each case is one `main` with
// a process-level spmd.parallel, the form -convert-apiCalls-to-spmd produces.
//
// What the checker compares is a CollectiveOpKey: the operation name, whether
// it is blocking, the model group, and the communicator's unique identity.
// Arguments are deliberately *not* part of it, so a differing root, count or
// datatype is not a mismatch here, and neither is a rank-dependent *number* of
// calls (a loop whose trip count is the rank). Those are the business of the
// benchmark suites, where the corresponding cases are XFAIL; nothing below
// pins them, in either direction.
//
// -verify-diagnostics also absorbs the pass failure the checker signals when
// it has findings, which is what makes the negative cases runnable.

// Every rank reaches both collectives, in the same order: nothing to report.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    %e1 = "spmd.bcast"(%world, %send, %recv, %count, %dt, %c0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A collective under a guard on the rank: only rank 0 calls it, so the other
// ranks never match it. The checker names the collective and, as a remark, the
// conditional it blames.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %is0 = arith.cmpi eq, %rank, %c0 : i32
    // expected-remark @below {{probable causing conditional number 0}}
    scf.if %is0 {
      // expected-warning @below {{[CheckCollectives] barrier operation/call may not be called by all processes in communicator/team (Call Ordering Error)}}
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// The same guard with the collective in *both* arms is legal: every rank
// calls one barrier on the world, whichever arm it takes.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %is0 = arith.cmpi eq, %rank, %c0 : i32
    scf.if %is0 {
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    } else {
      %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A guard on the communicator's *size* holds or fails on every rank alike, so
// the collective under it is still matched by all of them.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %size, %es = "spmd.getSizeOfComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %big = arith.cmpi sgt, %size, %c4 : i32
    scf.if %big {
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// Both arms call both collectives, but in the opposite order: rank 0 is in the
// barrier while everyone else is in the bcast. Two findings, each naming the
// two operations that would meet.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %is0 = arith.cmpi eq, %rank, %c0 : i32
    // expected-remark @below {{probable causing conditional number 0}}
    scf.if %is0 {
      // expected-warning @below {{barrier operation/call may not be called by all processes}}
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      // expected-warning @below {{bcast operation/call may not be called by all processes}}
      %e1 = "spmd.bcast"(%world, %send, %recv, %count, %dt, %c0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
    } else {
      // expected-warning @below {{bcast operation/call may not be called by all processes}}
      %e2 = "spmd.bcast"(%world, %send, %recv, %count, %dt, %c0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
      // expected-warning @below {{barrier operation/call may not be called by all processes}}
      %e3 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A blocking barrier does not match a non-blocking one: whether the operation
// blocks is part of the key the checker compares.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %is0 = arith.cmpi eq, %rank, %c0 : i32
    // expected-remark @below {{probable causing conditional number 0}}
    scf.if %is0 {
      // expected-warning @below {{barrier operation/call may not be called by all processes}}
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    } else {
      // expected-warning @below {{barrier operation/call may not be called by all processes}}
      %req, %e1 = "spmd.barrier"(%world) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.req, !spmd.error)
      %e2 = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// Two barriers of the same kind, but on communicators of different identity:
// rank 0 synchronises the world while everyone else synchronises its duplicate.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %dup, %ed = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %is0 = arith.cmpi eq, %rank, %c0 : i32
    // expected-remark @below {{probable causing conditional number 0}}
    scf.if %is0 {
      // expected-warning @below {{barrier operation/call may not be called by all processes}}
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    } else {
      // expected-warning @below {{barrier operation/call may not be called by all processes}}
      %e1 = "spmd.barrier"(%dup) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// Two models in one program are two independent ordering streams: the
// OpenSHMEM barrier between the two MPI ones does not disturb their matching,
// and is itself matched by every rank.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %sworld = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %o = "spmd.parallel"(%sworld) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 1 : i32}> ({
    %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
      %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      %e1 = "spmd.barrier"(%sworld) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm) -> !spmd.error
      %e2 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%r) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %o : i32
}
