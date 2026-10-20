// RUN: spmd-opt %s -omp-to-spmd -canonicalize -convert-apiCalls-to-spmd -cse -lower-spmd | FileCheck %s --implicit-check-not='"spmd.' --implicit-check-not=spmd.declare_reduction
//
// The OpenMP half of the round trip `spmd-verify compile` makes, on what
// llvm-to-omp raises from C: raising and lowering again gives back the omp
// dialect operations and the runtime queries the program started with.

// CHECK-LABEL: llvm.func @main
// CHECK:         omp.parallel num_threads(%{{.*}} : i32) {
// CHECK-NEXT:      %[[T:.*]] = llvm.call @omp_get_thread_num() : () -> i32
// CHECK:           omp.wsloop schedule(static = %{{.*}} : i32) {
// CHECK-NEXT:        omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
// CHECK:               omp.yield
// CHECK:           omp.wsloop nowait schedule(dynamic = %{{.*}} : i32) {
// CHECK:           omp.single {
// CHECK:             omp.terminator
// CHECK-NEXT:      }
// CHECK-NEXT:      omp.barrier
// CHECK-NEXT:      omp.critical {
// CHECK:           omp.terminator
// CHECK:         }
// CHECK:         llvm.return
llvm.func @omp_get_thread_num() -> i32
llvm.func @body(i32, i32)
llvm.func @main() -> i32 {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c4 = llvm.mlir.constant(4 : i32) : i32
  %c7 = llvm.mlir.constant(7 : i32) : i32
  %c99 = llvm.mlir.constant(99 : i32) : i32
  omp.parallel num_threads(%c4 : i32) {
    %t = llvm.call @omp_get_thread_num() : () -> i32
    omp.wsloop nowait schedule(static = %c1 : i32) {
      omp.loop_nest (%i) : i32 = (%c0) to (%c99) inclusive step (%c1) {
        llvm.call @body(%t, %i) : (i32, i32) -> ()
        omp.yield
      }
    }
    omp.barrier
    omp.wsloop nowait schedule(dynamic = %c7 : i32) {
      omp.loop_nest (%i) : i32 = (%c0) to (%c99) step (%c1) {
        llvm.call @body(%i, %t) : (i32, i32) -> ()
        omp.yield
      }
    }
    omp.single {
      llvm.call @body(%t, %t) : (i32, i32) -> ()
      omp.terminator
    }
    omp.barrier
    omp.critical {
      llvm.call @body(%t, %t) : (i32, i32) -> ()
      omp.terminator
    }
    omp.terminator
  }
  llvm.return %c0 : i32
}

// A raised omp.sections keeps its terminator, which is lowered with the rest
// of its region; the lowering used to add a second one, and the module failed
// to verify.  (C programs do not show it: clang makes sections a worksharing
// loop.)
// CHECK-LABEL: llvm.func @sections
// CHECK:         omp.parallel {
// CHECK-NEXT:      omp.sections {
// CHECK-NEXT:        omp.section {
// CHECK-NEXT:          llvm.call @body
// CHECK-NEXT:          omp.terminator
// CHECK-NEXT:        }
// CHECK-NEXT:        omp.section {
// CHECK-NEXT:          llvm.call @body
// CHECK-NEXT:          omp.terminator
// CHECK-NEXT:        }
// CHECK-NEXT:        omp.terminator
// CHECK-NEXT:      }
// CHECK-NEXT:      omp.terminator
// CHECK-NEXT:    }
llvm.func @sections(%x: i32) {
  omp.parallel {
    omp.sections {
      omp.section {
        llvm.call @body(%x, %x) : (i32, i32) -> ()
        omp.terminator
      }
      omp.section {
        llvm.call @body(%x, %x) : (i32, i32) -> ()
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
