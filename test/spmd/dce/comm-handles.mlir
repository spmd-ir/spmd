// RUN: spmd-opt %s -eliminate-dead-comm | FileCheck %s
//
// Request/completion matching.  A non-blocking initiation may only be
// removed together with the operations that complete it, and only when that
// matched set is *unique*.  These are the shapes where it is, and the bail-outs
// where it is not (docs/optimizations.md, "Completions").

// One dead initiation, one plain spmd.wait: the pair goes.
// CHECK-LABEL: func.func @wait
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.wait
func.func @wait() {
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// spmd.reqFree completes a request just as well as a wait does.
// CHECK-LABEL: func.func @req_free
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.reqFree
func.func @req_free() {
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %ferr = "spmd.reqFree"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// The wait's own error code is inspected, so the completion is observed and
// cannot silently disappear.
// CHECK-LABEL: func.func @wait_error_observed
// CHECK:         spmd.bcast
// CHECK:         spmd.wait
func.func @wait_error_observed() -> i32 {
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %code = "spmd.castSPMDErrorToI32"(%werr) : (!spmd.error) -> i32
  return %code : i32
}

// spmd.test reports completion as a flag the program branches on: removing the
// request would remove an observable value, not just bookkeeping.
// CHECK-LABEL: func.func @test_flag_observed
// CHECK:         spmd.bcast
// CHECK:         spmd.test
func.func @test_flag_observed() -> i1 {
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %flag, %terr = "spmd.test"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> (i1, !spmd.error)
  return %flag : i1
}

// The request escapes into an opaque call: nothing local decides where it is
// completed.
// CHECK-LABEL: func.func @request_escapes
// CHECK:         spmd.bcast
func.func private @complete(!spmd.req)
func.func @request_escapes() {
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  func.call @complete(%req) : (!spmd.req) -> ()
  return
}

//===----------------------------------------------------------------------===//
// The request-array shape: memref.store into a slot, one spmd.waitAll over it.
//===----------------------------------------------------------------------===//

// Every request in the array belongs to a dead initiation, so the whole set -
// both initiations, both stores, the waitAll - goes together.
// CHECK-LABEL: func.func @waitall_all_dead
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.waitAll
// CHECK-NOT:     memref.store
func.func @waitall_all_dead() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recvA = memref.alloca() : memref<1xi32>
  %recvB = memref.alloca() : memref<1xi32>
  %reqs = memref.alloca() : memref<2x!spmd.req>
  %reqA, %errA = "spmd.bcast"(%comm, %send, %recvA, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %reqA, %reqs[%c0] : memref<2x!spmd.req>
  %reqB, %errB = "spmd.bcast"(%comm, %send, %recvB, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %reqB, %reqs[%c1] : memref<2x!spmd.req>
  %werr = "spmd.waitAll"(%c2_i32, %reqs) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>) -> !spmd.error
  return
}

// The second initiation's destination *is* read, so it stays - and with it the
// waitAll, and therefore the first initiation too: removing that one alone
// would leave the waitAll reading an uninitialised slot.  Both survive.
// CHECK-LABEL: func.func @waitall_one_live
// CHECK-COUNT-2: spmd.bcast
// CHECK:         spmd.waitAll
func.func @waitall_one_live() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recvA = memref.alloca() : memref<1xi32>
  %recvB = memref.alloca() : memref<1xi32>
  %reqs = memref.alloca() : memref<2x!spmd.req>
  %reqA, %errA = "spmd.bcast"(%comm, %send, %recvA, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %reqA, %reqs[%c0] : memref<2x!spmd.req>
  %reqB, %errB = "spmd.bcast"(%comm, %send, %recvB, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %reqB, %reqs[%c1] : memref<2x!spmd.req>
  %werr = "spmd.waitAll"(%c2_i32, %reqs) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>) -> !spmd.error
  %v = memref.load %recvB[%c0] : memref<1xi32>
  return %v : i32
}

// A P2P request shares the array.  P2P is out of scope for this pass, so that
// request can never go, and the array cannot be dismantled around it.
// CHECK-LABEL: func.func @waitall_with_p2p
// CHECK:         spmd.bcast
// CHECK:         spmd.send
// CHECK:         spmd.waitAll
func.func @waitall_with_p2p() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %reqs = memref.alloca() : memref<2x!spmd.req>
  %reqA, %errA = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %reqA, %reqs[%c0] : memref<2x!spmd.req>
  %reqB, %errB = "spmd.send"(%comm, %send, %count, %dt, %dest, %tag) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  memref.store %reqB, %reqs[%c1] : memref<2x!spmd.req>
  %werr = "spmd.waitAll"(%c2_i32, %reqs) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>) -> !spmd.error
  return
}

// spmd.waitAny completes exactly one request and reports which: the matched set
// is not determined statically.
// CHECK-LABEL: func.func @waitany
// CHECK:         spmd.bcast
// CHECK:         spmd.waitAny
func.func @waitany() {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %reqs = memref.alloca() : memref<1x!spmd.req>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %req, %reqs[%c0] : memref<1x!spmd.req>
  %idx, %werr = "spmd.waitAny"(%c1_i32, %reqs) <{usedModel = 0 : i32}> : (i32, memref<1x!spmd.req>) -> (i32, !spmd.error)
  return
}

// Two consumers of the array, both completions that complete nothing but this
// request: all of them go.
// CHECK-LABEL: func.func @two_waitalls
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.waitAll
func.func @two_waitalls() {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %reqs = memref.alloca() : memref<1x!spmd.req>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %req, %reqs[%c0] : memref<1x!spmd.req>
  %werr0 = "spmd.waitAll"(%c1_i32, %reqs) <{usedModel = 0 : i32}> : (i32, memref<1x!spmd.req>) -> !spmd.error
  %werr1 = "spmd.waitAll"(%c1_i32, %reqs) <{usedModel = 0 : i32}> : (i32, memref<1x!spmd.req>) -> !spmd.error
  return
}

// The request is read back out of the array for a wait, which is how the raiser
// spells MPI_Wait(&req). Everything the loaded value reaches is a completion,
// so the ibcast, the store, the load and the wait all go.
// CHECK-LABEL: func.func @request_loaded_back
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.wait
func.func @request_loaded_back() {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %reqs = memref.alloca() : memref<1x!spmd.req>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %req, %reqs[%c0] : memref<1x!spmd.req>
  %loaded = memref.load %reqs[%c0] : memref<1x!spmd.req>
  %werr = "spmd.wait"(%loaded) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// The request array is a function argument: other requests may already sit in
// it, and the waitAll is not ours to remove.
// CHECK-LABEL: func.func @request_array_is_argument
// CHECK:         spmd.bcast
// CHECK:         spmd.waitAll
func.func @request_array_is_argument(%reqs: memref<4x!spmd.req>) {
  %c0 = arith.constant 0 : index
  %c4_i32 = arith.constant 4 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %req, %reqs[%c0] : memref<4x!spmd.req>
  %werr = "spmd.waitAll"(%c4_i32, %reqs) <{usedModel = 0 : i32}> : (i32, memref<4x!spmd.req>) -> !spmd.error
  return
}

//===----------------------------------------------------------------------===//
// The raiser's shapes: MPI_Request variables and arrays of i32.
//===----------------------------------------------------------------------===//

// MPI_Ibcast(..., &req); MPI_Wait(&req): the request is cast to i32, stored
// into the variable, loaded back and cast to a request for the wait. Every
// op on the way goes with the pair.
// CHECK-LABEL: func.func @raised_variable
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.wait
// CHECK-NOT:     memref.store
// CHECK-NOT:     memref.load
func.func @raised_variable() {
  %c1 = arith.constant 1 : i32
  %i0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %slot = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %req, %err = "spmd.bcast"(%comm, %buf, %buf, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %m0 = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<1xi32>
  %s0 = builtin.unrealized_conversion_cast %req : !spmd.req to i32
  memref.store %s0, %m0[%i0] : memref<1xi32>
  %m1 = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<1xi32>
  %l = memref.load %m1[%i0] : memref<1xi32>
  %q = builtin.unrealized_conversion_cast %l : i32 to !spmd.req
  %werr = "spmd.wait"(%q) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// MPI_Request reqs[2]; two dead ibcasts write reqs[0] and reqs[1], MPI_Wait
// loads reqs[1] back, MPI_Waitall completes the array. The load and the
// waitAll complete requests of both, and both go.
// CHECK-LABEL: func.func @raised_array
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.wait
func.func @raised_array() {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %i0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %reqs = llvm.alloca %c1 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
  %ra, %ea = "spmd.bcast"(%comm, %a, %a, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %m0 = "enzymexla.pointer2memref"(%reqs) : (!llvm.ptr) -> memref<1xi32>
  %s0 = builtin.unrealized_conversion_cast %ra : !spmd.req to i32
  memref.store %s0, %m0[%i0] : memref<1xi32>
  %elem = llvm.getelementptr inbounds %reqs[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %rb, %eb = "spmd.bcast"(%comm, %b, %b, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %m1 = "enzymexla.pointer2memref"(%elem) : (!llvm.ptr) -> memref<1xi32>
  %s1 = builtin.unrealized_conversion_cast %rb : !spmd.req to i32
  memref.store %s1, %m1[%i0] : memref<1xi32>
  %m2 = "enzymexla.pointer2memref"(%elem) : (!llvm.ptr) -> memref<1xi32>
  %l = memref.load %m2[%i0] : memref<1xi32>
  %q = builtin.unrealized_conversion_cast %l : i32 to !spmd.req
  %w0 = "spmd.wait"(%q) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %all = builtin.unrealized_conversion_cast %m0 : memref<1xi32> to memref<2x!spmd.req>
  %w1 = "spmd.waitAll"(%c2, %all) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>) -> !spmd.error
  return
}

// The same array, but b's result is read: its ibcast stays, and with it the
// waitAll and the wait that may complete a's request, so a's stays too.
// CHECK-LABEL: func.func @raised_array_one_live
// CHECK-COUNT-2: spmd.bcast
// CHECK:         spmd.wait
// CHECK:         spmd.waitAll
func.func @raised_array_one_live() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %i0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %reqs = llvm.alloca %c1 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
  %ra, %ea = "spmd.bcast"(%comm, %a, %a, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %m0 = "enzymexla.pointer2memref"(%reqs) : (!llvm.ptr) -> memref<1xi32>
  %s0 = builtin.unrealized_conversion_cast %ra : !spmd.req to i32
  memref.store %s0, %m0[%i0] : memref<1xi32>
  %elem = llvm.getelementptr inbounds %reqs[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %rb, %eb = "spmd.bcast"(%comm, %b, %b, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %m1 = "enzymexla.pointer2memref"(%elem) : (!llvm.ptr) -> memref<1xi32>
  %s1 = builtin.unrealized_conversion_cast %rb : !spmd.req to i32
  memref.store %s1, %m1[%i0] : memref<1xi32>
  %m2 = "enzymexla.pointer2memref"(%elem) : (!llvm.ptr) -> memref<1xi32>
  %l = memref.load %m2[%i0] : memref<1xi32>
  %q = builtin.unrealized_conversion_cast %l : i32 to !spmd.req
  %w0 = "spmd.wait"(%q) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %all = builtin.unrealized_conversion_cast %m0 : memref<1xi32> to memref<2x!spmd.req>
  %w1 = "spmd.waitAll"(%c2, %all) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>) -> !spmd.error
  %v = llvm.load %b : !llvm.ptr -> i32
  return %v : i32
}

// The array is initialized with MPI_REQUEST_NULL before the requests go in: a
// store of something no op produced, so the array is not a plain request
// array.
// CHECK-LABEL: func.func @array_initialized
// CHECK:         spmd.bcast
// CHECK:         spmd.waitAll
func.func @array_initialized() {
  %c1 = arith.constant 1 : i32
  %i0 = arith.constant 0 : index
  %null = arith.constant 738197504 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %reqs = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %m0 = "enzymexla.pointer2memref"(%reqs) : (!llvm.ptr) -> memref<1xi32>
  memref.store %null, %m0[%i0] : memref<1xi32>
  %req, %err = "spmd.bcast"(%comm, %buf, %buf, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %s0 = builtin.unrealized_conversion_cast %req : !spmd.req to i32
  memref.store %s0, %m0[%i0] : memref<1xi32>
  %all = builtin.unrealized_conversion_cast %m0 : memref<1xi32> to memref<1x!spmd.req>
  %w = "spmd.waitAll"(%c1, %all) <{usedModel = 0 : i32}> : (i32, memref<1x!spmd.req>) -> !spmd.error
  return
}

// The loaded request is compared against MPI_REQUEST_NULL: an observed value.
// CHECK-LABEL: func.func @loaded_request_compared
// CHECK:         spmd.bcast
func.func @loaded_request_compared() -> i1 {
  %c1 = arith.constant 1 : i32
  %i0 = arith.constant 0 : index
  %null = arith.constant 738197504 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %slot = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %req, %err = "spmd.bcast"(%comm, %buf, %buf, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %m0 = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<1xi32>
  %s0 = builtin.unrealized_conversion_cast %req : !spmd.req to i32
  memref.store %s0, %m0[%i0] : memref<1xi32>
  %l = memref.load %m0[%i0] : memref<1xi32>
  %isNull = arith.cmpi eq, %l, %null : i32
  return %isNull : i1
}

// A select picks one of two requests for the wait. Both go when both
// initiations do.
// CHECK-LABEL: func.func @selected_request
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.wait
func.func @selected_request(%cond: i1) {
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recvA = memref.alloca() : memref<1xi32>
  %recvB = memref.alloca() : memref<1xi32>
  %reqA, %errA = "spmd.bcast"(%comm, %send, %recvA, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %reqB, %errB = "spmd.bcast"(%comm, %send, %recvB, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %req = arith.select %cond, %reqA, %reqB : !spmd.req
  %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// The request comes in as an argument too: the wait may complete a request
// nothing here produces, and has to stay.
// CHECK-LABEL: func.func @selected_with_argument
// CHECK:         spmd.bcast
// CHECK:         spmd.wait
func.func @selected_with_argument(%cond: i1, %other: !spmd.req) {
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %reqA, %errA = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %req = arith.select %cond, %reqA, %other : !spmd.req
  %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}
