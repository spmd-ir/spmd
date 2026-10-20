// RUN: spmd-opt %s -split-input-file -check-dataRace -verify-diagnostics
//
// -check-dataRace on hand-written SPMD IR: a local buffer touched while a
// non-blocking operation on it is still pending. The checker reports the
// access and, as a second warning, the operation that owns the buffer at that
// point; -verify-diagnostics absorbs the pass failure that comes with a
// finding.
//
// The rules the cases below encode are MPI's: between the initiation and the
// completion the buffer belongs to the library. A send buffer may still be
// read (the message content is what it is), but not written; a receive buffer
// may be neither read nor written, because the incoming data may land at any
// moment.

// Writing the buffer of a pending send: the classic finding.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<1xi32>
    // expected-warning @below {{send operation/call}}
    %req, %err = "spmd.send"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    // expected-warning @below {{[CheckDataRace]store operation/call may access the same memory as the previous operation without proper synch. (Data Race)}}
    memref.store %c0, %buf[%idx] : memref<1xi32>
    %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// Reading the buffer of a pending send is allowed and must not be reported.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<1xi32>
    %req, %err = "spmd.send"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    %v = memref.load %buf[%idx] : memref<1xi32>
    %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A pending receive owns its buffer outright: writing it is a finding ...
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<1xi32>
    // expected-warning @below {{recv operation/call}}
    %req, %err = "spmd.recv"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    // expected-warning @below {{store operation/call may access the same memory as the previous operation}}
    memref.store %c0, %buf[%idx] : memref<1xi32>
    %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// ... and so is reading it, unlike the send case above.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<1xi32>
    // expected-warning @below {{recv operation/call}}
    %req, %err = "spmd.recv"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    // expected-warning @below {{load operation/call may access the same memory as the previous operation}}
    %v = memref.load %buf[%idx] : memref<1xi32>
    %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// After the wait the buffer is the program's again: neither access is a
// finding.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<1xi32>
    %req, %err = "spmd.recv"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    %v = memref.load %buf[%idx] : memref<1xi32>
    memref.store %v, %buf[%idx] : memref<1xi32>
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A blocking receive has no window in which the buffer is pending at all.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<1xi32>
    %err = "spmd.recv"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
    %v = memref.load %buf[%idx] : memref<1xi32>
    memref.store %v, %buf[%idx] : memref<1xi32>
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// The access need not name the same value: a second view of the same
// allocation is found through the alias analysis.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<4xi32>
    %alias = memref.cast %buf : memref<4xi32> to memref<?xi32>
    // expected-warning @below {{recv operation/call}}
    %req, %err = "spmd.recv"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    // expected-warning @below {{store operation/call may access the same memory as the previous operation}}
    memref.store %c0, %alias[%idx] : memref<?xi32>
    %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A different allocation while the receive is pending is not a conflict.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<4xi32>
    %other = memref.alloca() : memref<4xi32>
    %req, %err = "spmd.recv"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    memref.store %c0, %other[%idx] : memref<4xi32>
    %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
