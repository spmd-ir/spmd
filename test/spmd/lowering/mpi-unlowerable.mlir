// RUN: spmd-opt %s -split-input-file -lower-spmd-to-mpi -verify-diagnostics
//
// The pass runs as a dialect conversion with every MPI-tagged spmd op marked
// illegal, so an op it has no pattern for is a hard error rather than an op
// that quietly survives into "lowered" output.  spmd.memset is one such op: it
// is reachable from the CUDA and SHMEM front ends, has no MPI spelling, and
// must not be silently dropped when someone tags it for MPI.

func.func @no_mpi_spelling(%dst: !llvm.ptr) {
  %value = arith.constant 0 : i32
  %count = arith.constant 4 : i64
  // expected-error @+1 {{failed to legalize operation 'spmd.memset'}}
  %e = "spmd.memset"(%dst, %value, %count) <{usedModel = 0 : i32}> : (!llvm.ptr, i32, i64) -> !spmd.error
  return
}

// -----

// The same op tagged for a different programming model is left alone -- this
// pass only claims the ops that say they came from MPI -- so it draws no
// diagnostic and -verify-diagnostics still passes.
func.func @other_model(%dst: !llvm.ptr) {
  %value = arith.constant 0 : i32
  %count = arith.constant 4 : i64
  %e = "spmd.memset"(%dst, %value, %count) <{usedModel = 1 : i32}> : (!llvm.ptr, i32, i64) -> !spmd.error
  return
}

// -----

// Only a process-level spmd.parallel is the team MPI starts; a thread-level
// one tagged for MPI has no MPI spelling.
func.func @thread_level_parallel() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{failed to legalize operation 'spmd.parallel'}}
  "spmd.parallel"(%world) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  return
}

// -----

// A type without a predefined MPI datatype, and a name without a predefined
// handle, used to become MPI_INT and MPI_SUM.  Every one is reported, before
// anything is rewritten.
func.func @no_predefined_handle() {
  // expected-error @+1 {{'spmd.constDatatype' op has no predefined MPI datatype for 'i128'}}
  %dt = "spmd.constDatatype"() <{typeAttr = i128, usedModel = 0 : i32}> : () -> !spmd.datatype
  // expected-error @+1 {{'spmd.constOpType' op 'XOR' is not a predefined MPI reduction operation}}
  %op = "spmd.constOpType"() <{stringAttr = "XOR", usedModel = 0 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.constLockTypeOp' op 'READ' is not an MPI lock type}}
  %lock = "spmd.constLockTypeOp"() <{stringAttr = "READ", usedModel = 0 : i32}> : () -> !spmd.lock
  // expected-error @+1 {{'spmd.constThreadLevel' op 'ALL' is not an MPI thread support level}}
  %level = "spmd.constThreadLevel"() <{level = "ALL", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  return
}

// -----

// A datatype of another model is not this pass's to judge.
func.func @other_model_datatype() {
  %dt = "spmd.constDatatype"() <{typeAttr = i128, usedModel = 1 : i32}> : () -> !spmd.datatype
  return
}

// -----

// MAX and MIN compare as the datatype's signedness says, and an integer of
// unknown signedness does not say it: MPI_INT and MPI_UNSIGNED give different
// maxima.  Any other reduction moves the same bits either way, and MPI_CHAR's
// i8 lowers to MPI_CHAR, which leaves the signedness open as the program did.
func.func @unknown_signedness(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %count: i64) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = i32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %char = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 0 : i32}> : () -> !spmd.datatype
  %max = "spmd.constOpType"() <{stringAttr = "MAX", usedModel = 0 : i32}> : () -> !spmd.op
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.allreduce' op reduces with 'MAX' over i32, whose signedness the result depends on and the datatype does not record}}
  %e0 = "spmd.allreduce"(%world, %sbuf, %rbuf, %count, %dt, %max) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e1 = "spmd.allreduce"(%world, %sbuf, %rbuf, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e2 = "spmd.allreduce"(%world, %sbuf, %rbuf, %count, %char, %max) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}
