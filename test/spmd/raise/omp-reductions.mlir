// RUN: spmd-opt %s -omp-to-spmd | FileCheck %s --implicit-check-not="omp."
//
// omp.declare_reduction becomes spmd.declare_reduction with its regions, and
// the combiner is recognised when it is a single arith operation: kind is the
// ReductionKind it computes (0 add, 1 mul, 2 min_s, 3 min_u, 4 min_f, 5 max_s,
// 6 max_u, 7 max_f, 8 and, 9 or, 10 xor). Any other combiner is kept without a
// kind -- one computing more than one operation, and one in the llvm dialect,
// which is what a reduction raised from LLVM IR has. The kind is informational:
// no analysis reads it yet.
//
// A parallel region names its reductions by symbol and passes their variables.

// CHECK: spmd.declare_reduction @add : i32 attributes {kind = 0 : i32} init {
// CHECK: %[[R:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK: "spmd.yield"(%[[R]]) : (i32) -> ()
omp.declare_reduction @add : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.addi %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @mul : i32 attributes {kind = 1 : i32} init {
// CHECK: %[[R:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK: "spmd.yield"(%[[R]]) : (i32) -> ()
omp.declare_reduction @mul : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.muli %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @min_s : i32 attributes {kind = 2 : i32} init {
// CHECK: %[[R:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// CHECK: "spmd.yield"(%[[R]]) : (i32) -> ()
omp.declare_reduction @min_s : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.minsi %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @min_u : i32 attributes {kind = 3 : i32} init {
// CHECK: %[[R:.*]] = arith.minui %{{.*}}, %{{.*}} : i32
// CHECK: "spmd.yield"(%[[R]]) : (i32) -> ()
omp.declare_reduction @min_u : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.minui %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @max_s : i32 attributes {kind = 5 : i32} init {
// CHECK: %[[R:.*]] = arith.maxsi %{{.*}}, %{{.*}} : i32
// CHECK: "spmd.yield"(%[[R]]) : (i32) -> ()
omp.declare_reduction @max_s : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.maxsi %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @max_u : i32 attributes {kind = 6 : i32} init {
// CHECK: %[[R:.*]] = arith.maxui %{{.*}}, %{{.*}} : i32
// CHECK: "spmd.yield"(%[[R]]) : (i32) -> ()
omp.declare_reduction @max_u : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.maxui %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @and : i32 attributes {kind = 8 : i32} init {
// CHECK: %[[R:.*]] = arith.andi %{{.*}}, %{{.*}} : i32
// CHECK: "spmd.yield"(%[[R]]) : (i32) -> ()
omp.declare_reduction @and : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.andi %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @or : i32 attributes {kind = 9 : i32} init {
// CHECK: %[[R:.*]] = arith.ori %{{.*}}, %{{.*}} : i32
// CHECK: "spmd.yield"(%[[R]]) : (i32) -> ()
omp.declare_reduction @or : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.ori %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @xor : i32 attributes {kind = 10 : i32} init {
// CHECK: %[[R:.*]] = arith.xori %{{.*}}, %{{.*}} : i32
// CHECK: "spmd.yield"(%[[R]]) : (i32) -> ()
omp.declare_reduction @xor : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = arith.xori %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @add_f : f64 attributes {kind = 0 : i32} init {
// CHECK: %[[R:.*]] = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK: "spmd.yield"(%[[R]]) : (f64) -> ()
omp.declare_reduction @add_f : f64 init {
^bb0(%arg0: f64):
  %c = arith.constant 0.0 : f64
  omp.yield(%c : f64)
} combiner {
^bb0(%a: f64, %b: f64):
  %r = arith.addf %a, %b : f64
  omp.yield(%r : f64)
}

// CHECK: spmd.declare_reduction @mul_f : f64 attributes {kind = 1 : i32} init {
// CHECK: %[[R:.*]] = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK: "spmd.yield"(%[[R]]) : (f64) -> ()
omp.declare_reduction @mul_f : f64 init {
^bb0(%arg0: f64):
  %c = arith.constant 0.0 : f64
  omp.yield(%c : f64)
} combiner {
^bb0(%a: f64, %b: f64):
  %r = arith.mulf %a, %b : f64
  omp.yield(%r : f64)
}

// CHECK: spmd.declare_reduction @min_f : f64 attributes {kind = 4 : i32} init {
// CHECK: %[[R:.*]] = arith.minimumf %{{.*}}, %{{.*}} : f64
// CHECK: "spmd.yield"(%[[R]]) : (f64) -> ()
omp.declare_reduction @min_f : f64 init {
^bb0(%arg0: f64):
  %c = arith.constant 0.0 : f64
  omp.yield(%c : f64)
} combiner {
^bb0(%a: f64, %b: f64):
  %r = arith.minimumf %a, %b : f64
  omp.yield(%r : f64)
}

// CHECK: spmd.declare_reduction @max_f : f64 attributes {kind = 7 : i32} init {
// CHECK: %[[R:.*]] = arith.maximumf %{{.*}}, %{{.*}} : f64
// CHECK: "spmd.yield"(%[[R]]) : (f64) -> ()
omp.declare_reduction @max_f : f64 init {
^bb0(%arg0: f64):
  %c = arith.constant 0.0 : f64
  omp.yield(%c : f64)
} combiner {
^bb0(%a: f64, %b: f64):
  %r = arith.maximumf %a, %b : f64
  omp.yield(%r : f64)
}

// CHECK: spmd.declare_reduction @llvm_add : i32 init {
omp.declare_reduction @llvm_add : i32 init {
^bb0(%arg0: i32):
  %c = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %r = llvm.add %a, %b : i32
  omp.yield(%r : i32)
}

// CHECK: spmd.declare_reduction @twice_sum : i32 init {
omp.declare_reduction @twice_sum : i32 init {
^bb0(%arg0: i32):
  %c = arith.constant 0 : i32
  omp.yield(%c : i32)
} combiner {
^bb0(%a: i32, %b: i32):
  %s = arith.addi %a, %b : i32
  %r = arith.addi %s, %s : i32
  omp.yield(%r : i32)
}

// CHECK-LABEL: llvm.func @reduce(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr)
// CHECK:         "spmd.parallel"(%{{.*}}, %arg0, %arg1) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 2>, reduction_syms = [@add, @max_s], usedModel = 5 : i32}>
llvm.func @reduce(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.parallel reduction(@add %x -> %px, @max_s %y -> %py : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  }
  llvm.return
}
