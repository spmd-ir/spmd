// RUN: spmd-opt %s -lower-spmd-to-omp | FileCheck %s --implicit-check-not='"spmd.'
//
// The directives that name no team: critical sections and tasks.  What
// -omp-to-spmd keeps of them as discardable attributes -- the depend kinds, the
// taskloop's modifiers and mergeable -- goes back into their clauses.

// A named critical section refers to an omp.critical.declare, which
// -omp-to-spmd does not keep; one declaration serves every section of that
// name.
// CHECK-DAG:   omp.critical.declare @lock
// CHECK-NOT:   omp.critical.declare
// CHECK-LABEL: llvm.func @critical
// CHECK:         omp.critical(@lock) {
// CHECK-NEXT:      llvm.call @use
// CHECK-NEXT:      omp.terminator
// CHECK-NEXT:    }
// CHECK-NEXT:    omp.critical(@lock) {
// CHECK-NEXT:      omp.terminator
// CHECK-NEXT:    }
// CHECK-NEXT:    omp.critical {
// CHECK-NEXT:      omp.terminator
// CHECK-NEXT:    }
llvm.func @use(i32)
llvm.func @critical(%x: i32) {
  %e0 = "spmd.critical"() <{name = "lock", usedModel = 5 : i32}> ({
    llvm.call @use(%x) : (i32) -> ()
    "spmd.terminator"() : () -> ()
  }) : () -> !spmd.error
  %e1 = "spmd.critical"() <{name = "lock", usedModel = 5 : i32}> ({
    "spmd.terminator"() : () -> ()
  }) : () -> !spmd.error
  %e2 = "spmd.critical"() <{usedModel = 5 : i32}> ({
    "spmd.terminator"() : () -> ()
  }) : () -> !spmd.error
  llvm.return
}

// CHECK-LABEL: llvm.func @tasks
// CHECK:         omp.taskgroup {
// CHECK-NEXT:      omp.task depend(taskdependout -> %[[P:.*]] : !llvm.ptr) final(%[[FINAL:.*]]) if(%[[IF:.*]]) mergeable untied {
// CHECK-NEXT:        llvm.call @use
// CHECK-NEXT:        omp.terminator
// CHECK-NEXT:      }
// CHECK-NEXT:      omp.task {
// CHECK-NEXT:        omp.terminator
// CHECK-NEXT:      }
// CHECK-NEXT:      omp.taskwait depend(taskdependin -> %[[P]] : !llvm.ptr)
// CHECK-NEXT:      omp.taskwait
// CHECK-NEXT:      omp.terminator
// CHECK-NEXT:    }
llvm.func @tasks(%x: i32, %p: !llvm.ptr, %if: i1, %final: i1) {
  %g = "spmd.taskgroup"() <{operandSegmentSizes = array<i32: 0>, usedModel = 5 : i32}> ({
    "spmd.task"(%p, %if, %final) <{mergeable = true, operandSegmentSizes = array<i32: 1, 1, 1>, untied = true, usedModel = 5 : i32}> ({
      llvm.call @use(%x) : (i32) -> ()
      "spmd.terminator"() : () -> ()
    }) {depend_kinds = [#omp<clause_task_depend(taskdependout)>]} : (!llvm.ptr, i1, i1) -> ()
    "spmd.task"() <{mergeable = false, operandSegmentSizes = array<i32: 0, 0, 0>, untied = false, usedModel = 5 : i32}> ({
      "spmd.terminator"() : () -> ()
    }) : () -> ()
    %w0 = "spmd.taskwait"(%p) <{usedModel = 5 : i32}> {depend_kinds = [#omp<clause_task_depend(taskdependin)>]} : (!llvm.ptr) -> !spmd.error
    %w1 = "spmd.taskwait"() <{usedModel = 5 : i32}> : () -> !spmd.error
    "spmd.terminator"() : () -> ()
  }) : () -> !spmd.error
  llvm.return
}

// CHECK-LABEL: llvm.func @taskloop
// CHECK:         omp.taskloop grainsize(strict, %{{.*}}: i32) mergeable nogroup untied {
// CHECK-NEXT:      omp.loop_nest
// CHECK:         omp.taskloop num_tasks(%{{.*}}: i32) {
llvm.func @taskloop(%lb: i32, %ub: i32, %step: i32, %grain: i32, %n: i32) {
  "spmd.taskloop"(%grain) <{nogroup = true, operandSegmentSizes = array<i32: 1, 0>, untied = true, usedModel = 5 : i32}> ({
    "spmd.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%i: i32):
      "spmd.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
  }) {grainsize_modifier = #omp<grainsizetype strict>, mergeable} : (i32) -> ()
  "spmd.taskloop"(%n) <{nogroup = false, operandSegmentSizes = array<i32: 0, 1>, untied = false, usedModel = 5 : i32}> ({
    "spmd.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%i: i32):
      "spmd.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()
  }) : (i32) -> ()
  llvm.return
}
