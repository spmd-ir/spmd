// RUN: spmd-opt %s -omp-to-spmd | FileCheck %s
//
// llvm-to-omp raises a C loop written `i <= n` (and every loop clang
// normalises to one) to an omp.loop_nest with `inclusive`: the upper bound is
// an iteration of its own.  -omp-to-spmd used to drop the attribute, so the
// dialect said the loop stops one iteration early -- and lowering it back to
// OpenMP lost the last one.

// CHECK-LABEL: llvm.func @inclusive
llvm.func @inclusive(%a: !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c999 = llvm.mlir.constant(999 : i32) : i32
  // CHECK: "spmd.parallel"
  omp.parallel {
    // CHECK: "spmd.wsloop"
    omp.wsloop nowait {
      // CHECK: "spmd.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}) <{loop_inclusive}>
      omp.loop_nest (%i) : i32 = (%c0) to (%c999) inclusive step (%c1) {
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @exclusive
llvm.func @exclusive(%a: !llvm.ptr) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c1000 = llvm.mlir.constant(1000 : i32) : i32
  omp.parallel {
    omp.wsloop nowait {
      // CHECK: "spmd.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}) ({
      // CHECK-NOT: loop_inclusive
      omp.loop_nest (%i) : i32 = (%c0) to (%c1000) step (%c1) {
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}
