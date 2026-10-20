// RUN: spmd-opt %s -split-input-file -verify-diagnostics
//
// spmd.declare_reduction's hand-written verifier. The $kind attribute is a
// derived annotation -- -omp-to-spmd sets it when it recognises the combiner,
// and the regions stay authoritative -- so the verifier's job is to catch a
// $kind that contradicts the region it annotates, and to stay out of the way
// everywhere else.
//
// It only looks at a combiner that is a single operation plus its yield; a
// richer body (a user-defined reduction, or one still in the llvm dialect, as
// a reduction raised from LLVM IR is) is not something it can judge, and is
// left alone. No pass is run: parsing the file is what invokes the verifier.

// Every kind with the combiner it stands for. One module, no diagnostics.

spmd.declare_reduction @add : i32 attributes {kind = 0 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

spmd.declare_reduction @mul : i32 attributes {kind = 1 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.muli %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

spmd.declare_reduction @min_s : i32 attributes {kind = 2 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.minsi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

spmd.declare_reduction @min_u : i32 attributes {kind = 3 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.minui %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

spmd.declare_reduction @min_f : f32 attributes {kind = 4 : i32} init {
^bb0(%arg0: f32):
  %c = arith.constant 0.0 : f32
  "spmd.yield"(%c) : (f32) -> ()
} combiner {
^bb0(%a: f32, %b: f32):
  %r = arith.minimumf %a, %b : f32
  "spmd.yield"(%r) : (f32) -> ()
}

spmd.declare_reduction @max_s : i32 attributes {kind = 5 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.maxsi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

spmd.declare_reduction @max_u : i32 attributes {kind = 6 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.maxui %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

spmd.declare_reduction @max_f : f32 attributes {kind = 7 : i32} init {
^bb0(%arg0: f32):
  %c = arith.constant 0.0 : f32
  "spmd.yield"(%c) : (f32) -> ()
} combiner {
^bb0(%a: f32, %b: f32):
  %r = arith.maximumf %a, %b : f32
  "spmd.yield"(%r) : (f32) -> ()
}

spmd.declare_reduction @and : i32 attributes {kind = 8 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.andi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

spmd.declare_reduction @or : i32 attributes {kind = 9 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.ori %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

spmd.declare_reduction @xor : i32 attributes {kind = 10 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.xori %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

spmd.declare_reduction @add_f : f32 attributes {kind = 0 : i32} init {
^bb0(%arg0: f32):
  %c = arith.constant 0.0 : f32
  "spmd.yield"(%c) : (f32) -> ()
} combiner {
^bb0(%a: f32, %b: f32):
  %r = arith.addf %a, %b : f32
  "spmd.yield"(%r) : (f32) -> ()
}

spmd.declare_reduction @mul_f : f32 attributes {kind = 1 : i32} init {
^bb0(%arg0: f32):
  %c = arith.constant 0.0 : f32
  "spmd.yield"(%c) : (f32) -> ()
} combiner {
^bb0(%a: f32, %b: f32):
  %r = arith.mulf %a, %b : f32
  "spmd.yield"(%r) : (f32) -> ()
}

// -----

// $kind=add over a arith.muli combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=add inconsistent with combiner: expected arith.addi or arith.addf}}
spmd.declare_reduction @add : i32 attributes {kind = 0 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.muli %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// $kind=mul over a arith.addi combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=mul inconsistent with combiner: expected arith.muli or arith.mulf}}
spmd.declare_reduction @mul : i32 attributes {kind = 1 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// $kind=min_s over a arith.addi combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=min_s inconsistent with combiner: expected arith.minsi}}
spmd.declare_reduction @min_s : i32 attributes {kind = 2 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// $kind=min_u over a arith.addi combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=min_u inconsistent with combiner: expected arith.minui}}
spmd.declare_reduction @min_u : i32 attributes {kind = 3 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// $kind=min_f over a arith.addf combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=min_f inconsistent with combiner: expected arith.minimumf}}
spmd.declare_reduction @min_f : f32 attributes {kind = 4 : i32} init {
^bb0(%arg0: f32):
  %c = arith.constant 0.0 : f32
  "spmd.yield"(%c) : (f32) -> ()
} combiner {
^bb0(%a: f32, %b: f32):
  %r = arith.addf %a, %b : f32
  "spmd.yield"(%r) : (f32) -> ()
}

// -----

// $kind=max_s over a arith.addi combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=max_s inconsistent with combiner: expected arith.maxsi}}
spmd.declare_reduction @max_s : i32 attributes {kind = 5 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// $kind=max_u over a arith.addi combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=max_u inconsistent with combiner: expected arith.maxui}}
spmd.declare_reduction @max_u : i32 attributes {kind = 6 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// $kind=max_f over a arith.addf combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=max_f inconsistent with combiner: expected arith.maximumf}}
spmd.declare_reduction @max_f : f32 attributes {kind = 7 : i32} init {
^bb0(%arg0: f32):
  %c = arith.constant 0.0 : f32
  "spmd.yield"(%c) : (f32) -> ()
} combiner {
^bb0(%a: f32, %b: f32):
  %r = arith.addf %a, %b : f32
  "spmd.yield"(%r) : (f32) -> ()
}

// -----

// $kind=and over a arith.addi combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=and inconsistent with combiner: expected arith.andi}}
spmd.declare_reduction @and : i32 attributes {kind = 8 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// $kind=or over a arith.addi combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=or inconsistent with combiner: expected arith.ori}}
spmd.declare_reduction @or : i32 attributes {kind = 9 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// $kind=xor over a arith.addi combiner.
// expected-error @below {{'spmd.declare_reduction' op $kind=xor inconsistent with combiner: expected arith.xori}}
spmd.declare_reduction @xor : i32 attributes {kind = 10 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// Without $kind there is nothing to contradict: the regions are the only
// source and the verifier says nothing.

spmd.declare_reduction @no_kind : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.muli %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// A combiner of more than one operation is out of the verifier's reach, so a
// $kind on it is taken at face value even when the first operation disagrees.

spmd.declare_reduction @wide : i32 attributes {kind = 0 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %t = arith.muli %a, %b : i32
  %r = arith.addi %t, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}

// -----

// The three logical kinds (land, lor, lxor) have no arm in the verifier: C's
// `&&`, `||` and `!=`-on-booleans have no single arith operation to compare
// against. A combiner is therefore never rejected under them -- a gap, pinned
// here so that giving them a rule is a deliberate change to this file.

spmd.declare_reduction @land : i32 attributes {kind = 11 : i32} init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  "spmd.yield"(%c) : (i32) -> ()
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  "spmd.yield"(%r) : (i32) -> ()
}
