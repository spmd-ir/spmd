// RUN: spmd-opt %s -split-input-file -verify-diagnostics
//
// spmd.parallel's hand-written verifier. The op's region is a whole program
// region -- everything between init and finalize -- so it is not a single
// block: -convert-llvm-to-func and the lowerings leave cf branches, and a
// path that aborts ends in llvm.unreachable instead of a yield. The rules the
// verifier enforces over that shape:
//
//   * the region is not empty,
//   * every block that does not branch onwards ends in spmd.parallel_yield or
//     llvm.unreachable,
//   * and the yield carries an exit code exactly when the op produces one.
//
// The verifier's fourth message, "block in parallel region has no
// terminator", is not reachable from parsed IR: Block::getTerminator() hands
// back the block's last operation whatever it is, so a block ending in
// something else takes the "must terminate with" arm below instead.
//
// No pass is run: parsing the file is what invokes the verifier.

// The ordinary shape: one block, a yield carrying the program's exit code.
func.func @yield_code() -> i32 {
  %c0 = arith.constant 0 : i32
  %w = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%w) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %e = "spmd.barrier"(%w) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A thread-level region has no exit code, and its yield carries none.
func.func @yield_nothing() {
  %w = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  "spmd.parallel"(%w) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  return
}

// -----

// Several blocks: the branching one is interior and unchecked, the two leaves
// end in a yield and in llvm.unreachable -- the shape an abort path takes.
func.func @multi_block() -> i32 {
  %c0 = arith.constant 0 : i32
  %t = arith.constant true
  %w = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%w) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    cf.cond_br %t, ^yield, ^aborted
  ^yield:
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  ^aborted:
    llvm.unreachable
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// An empty region has no yield to produce the result from.
func.func @empty() -> i32 {
  %w = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @below {{'spmd.parallel' op parallel region must not be empty}}
  %r = "spmd.parallel"(%w) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// spmd.terminator ends the region ops that have no result (single, critical,
// task); it is not a way out of a parallel region.
func.func @wrong_terminator() -> i32 {
  %w = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @below {{'spmd.parallel' op leaf block in parallel region must terminate with spmd.parallel_yield or llvm.unreachable, got: spmd.terminator}}
  %r = "spmd.parallel"(%w) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    "spmd.terminator"() : () -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// A block whose last operation is not a terminator at all reaches the same
// message, naming that operation.
func.func @no_terminator() -> i32 {
  %w = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @below {{'spmd.parallel' op leaf block in parallel region must terminate with spmd.parallel_yield or llvm.unreachable, got: arith.constant}}
  %r = "spmd.parallel"(%w) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %x = arith.constant 1 : i32
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// The op produces an exit code, so the yield has to supply one.
func.func @missing_exit_code() -> i32 {
  %w = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @below {{'spmd.parallel' op parallel_yield must carry an exit code when spmd.parallel has an exit_code result}}
  %r = "spmd.parallel"(%w) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// And the other way round: no result, so the yield must stay empty.
func.func @unexpected_exit_code() {
  %c0 = arith.constant 0 : i32
  %w = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @below {{'spmd.parallel' op parallel_yield must not carry operands when spmd.parallel has no exit_code result}}
  "spmd.parallel"(%w) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> ()
  return
}
