// RUN: spmd-opt %s -lower-spmd-to-omp | FileCheck %s --implicit-check-not='"spmd.'
//
// spmd.declare_reduction is omp.declare_reduction, regions and all, and a
// construct's reductions refer to it by the same symbol.  -omp-to-spmd does
// not keep reduction_byref; OpenMP passes a reduction by reference when its
// declaration's type is the pointer rather than the value, so that is what it
// is rebuilt from.

// CHECK:       omp.declare_reduction @add_i64 : i64 init {
// CHECK-NEXT:  ^bb0(%{{.*}}: i64):
// CHECK-NEXT:    %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-NEXT:    omp.yield(%[[ZERO]] : i64)
// CHECK-NEXT:  } combiner {
// CHECK-NEXT:  ^bb0(%[[A:.*]]: i64, %[[B:.*]]: i64):
// CHECK-NEXT:    %[[SUM:.*]] = arith.addi %[[A]], %[[B]] : i64
// CHECK-NEXT:    omp.yield(%[[SUM]] : i64)
// CHECK-NEXT:  }
spmd.declare_reduction @add_i64 : i64 attributes {kind = 0 : i32} init {
^bb0(%arg0: i64):
  %zero = arith.constant 0 : i64
  "spmd.yield"(%zero) : (i64) -> ()
} combiner {
^bb0(%a: i64, %b: i64):
  %sum = arith.addi %a, %b : i64
  "spmd.yield"(%sum) : (i64) -> ()
}

// CHECK:       omp.declare_reduction @add_boxed : !llvm.ptr alloc {
spmd.declare_reduction @add_boxed : !llvm.ptr alloc {
  %one = llvm.mlir.constant(1 : i64) : i64
  %box = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
  "spmd.yield"(%box) : (!llvm.ptr) -> ()
} init {
^bb0(%mold: !llvm.ptr, %alloc: !llvm.ptr):
  "spmd.yield"(%alloc) : (!llvm.ptr) -> ()
} combiner {
^bb0(%a: !llvm.ptr, %b: !llvm.ptr):
  "spmd.yield"(%a) : (!llvm.ptr) -> ()
}

// CHECK-LABEL: llvm.func @reductions
// CHECK:         omp.parallel reduction(@add_i64 %[[X:.*]] -> %[[PX:.*]], byref @add_boxed %{{.*}} -> %{{.*}} : !llvm.ptr, !llvm.ptr) {
// CHECK-NEXT:      omp.wsloop nowait reduction(@add_i64 %[[PX]] -> %{{.*}} : !llvm.ptr) {
// CHECK:           omp.sections nowait reduction(@add_i64 %[[PX]] -> %{{.*}} : !llvm.ptr) {
// CHECK:           omp.terminator
// CHECK:         omp.taskgroup task_reduction(@add_i64 %[[X]] -> %{{.*}} : !llvm.ptr) {
llvm.func @reductions(%x: !llvm.ptr, %y: !llvm.ptr, %lb: i32, %ub: i32, %step: i32) {
  %world = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  %team, %e = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  "spmd.parallel"(%team, %x, %y) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 2>, reduction_syms = [@add_i64, @add_boxed], usedModel = 5 : i32}> ({
  ^bb0(%px: !llvm.ptr, %py: !llvm.ptr):
    "spmd.wsloop"(%team, %px) <{operandSegmentSizes = array<i32: 1, 0, 1>, reduction_syms = [@add_i64], usedModel = 5 : i32}> ({
    ^bb0(%lx: !llvm.ptr):
      "spmd.loop_nest"(%lb, %ub, %step) ({
      ^bb0(%i: i32):
        "spmd.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
    }) : (!spmd.comm, !llvm.ptr) -> ()
    %s = "spmd.sections"(%team, %px) <{operandSegmentSizes = array<i32: 1, 0, 1>, reduction_syms = [@add_i64], usedModel = 5 : i32}> ({
    ^bb0(%sx: !llvm.ptr):
      "spmd.section"() ({
      ^bb0(%ssx: !llvm.ptr):
        "spmd.terminator"() : () -> ()
      }) : () -> ()
    }) : (!spmd.comm, !llvm.ptr) -> !spmd.error
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm, !llvm.ptr, !llvm.ptr) -> ()
  %g = "spmd.taskgroup"(%x) <{operandSegmentSizes = array<i32: 1>, task_reduction_syms = [@add_i64], usedModel = 5 : i32}> ({
  ^bb0(%tx: !llvm.ptr):
    "spmd.terminator"() : () -> ()
  }) : (!llvm.ptr) -> !spmd.error
  llvm.return
}
