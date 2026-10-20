// RUN: spmd-opt %s -remove-dead-values | FileCheck %s --check-prefix=RDV
// RUN: spmd-opt %s -symbol-dce      | FileCheck %s --check-prefix=SYMDCE
// RUN: spmd-opt %s -sccp            | FileCheck %s --check-prefix=SCCP
// RUN: spmd-opt %s -inline          | FileCheck %s --check-prefix=INLINE
//
// The rest of the count-preservation matrix. These passes do not move
// communication, but they are the ones an LLVM bump is most likely to change
// under us, and the assertion is on the op *count*.

// A collective whose destination is read.
// RDV-LABEL:    func.func @live_collective
// RDV-COUNT-1:    spmd.allreduce
// SYMDCE-LABEL: func.func @live_collective
// SYMDCE-COUNT-1: spmd.allreduce
// SCCP-LABEL:   func.func @live_collective
// SCCP-COUNT-1:   spmd.allreduce
// INLINE-LABEL: func.func @live_collective
// INLINE-COUNT-1: spmd.allreduce
func.func @live_collective() -> i32 {
  %c0 = arith.constant 0 : index
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// A barrier: no data, no observed results.
// RDV-LABEL:    func.func @lone_barrier
// RDV-COUNT-1:    spmd.barrier
// SYMDCE-LABEL: func.func @lone_barrier
// SYMDCE-COUNT-1: spmd.barrier
// SCCP-LABEL:   func.func @lone_barrier
// SCCP-COUNT-1:   spmd.barrier
// INLINE-LABEL: func.func @lone_barrier
// INLINE-COUNT-1: spmd.barrier
func.func @lone_barrier() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// A collective behind a call. The SPMD dialect implements no
// DialectInlinerInterface, so -inline leaves the calls in place. What is pinned
// is that the barrier is neither lost nor duplicated. If the dialect gains an
// inliner interface this starts failing, which is the moment to revisit it:
// inlining duplicates a collective once per call site, which is only
// count-preserving because every rank executed the call.

func.func private @sync(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// INLINE-LABEL: func.func private @sync
// INLINE-COUNT-1: spmd.barrier

// INLINE-LABEL: func.func @two_call_sites
// INLINE-COUNT-2: call @sync
func.func @two_call_sites() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  func.call @sync(%world) : (!spmd.comm) -> ()
  func.call @sync(%world) : (!spmd.comm) -> ()
  return
}
