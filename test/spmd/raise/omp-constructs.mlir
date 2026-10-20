// RUN: spmd-opt %s -omp-to-spmd -canonicalize | FileCheck %s --implicit-check-not="omp."
//
// The OpenMP directives -omp-to-spmd raises, in the order spmd-verify runs it:
// followed by -canonicalize, which resolves every spmd.getParallelComm to the
// team of the enclosing region.
//
// A parallel region is a thread-level spmd.parallel over a team of its own, a
// duplicate of the world, with num_threads as given. The directives that
// synchronise the team end in a barrier that matches exactly once, unless they
// are nowait: a barrier directive itself, single, and a worksharing loop.
// master and masked do not synchronise; they run their body on the thread
// whose rank in the team is 0, or the filter. A critical section, tasks,
// taskwait and taskgroup keep their structure; taskyield is only a scheduling
// hint and goes.
//
// Not here: sections and a loop's inclusive bound, which MR 12 changes, and
// a named critical section, which does not raise (MR 16's description,
// finding 15).

// CHECK-LABEL: llvm.func @main
// CHECK:         %[[WORLD:.*]] = "spmd.commWorld"() <{usedModel = 5 : i32}>
// CHECK:         %[[TEAM:.*]], %{{.*}} = "spmd.commDup"(%[[WORLD]]) <{usedModel = 5 : i32}>
// CHECK:         "spmd.parallel"(%[[TEAM]], %c4_i32) <{isProcessLevel = false, {{.*}}usedModel = 5 : i32}> ({
// CHECK-NEXT:      llvm.call @work(%c0_i32)
// CHECK-NEXT:      "spmd.barrier"(%[[TEAM]]) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}>
// CHECK-NEXT:      "spmd.single"(%[[TEAM]]) <{usedModel = 5 : i32}> ({
// CHECK-NEXT:        llvm.call @work(%c1_i32)
// CHECK-NEXT:        "spmd.terminator"()
// CHECK-NEXT:      })
// CHECK-NEXT:      "spmd.barrier"(%[[TEAM]]) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}>
// CHECK-NEXT:      "spmd.single"(%[[TEAM]]) <{usedModel = 5 : i32}> ({
// CHECK-NEXT:        llvm.call @work(%c2_i32)
// CHECK-NEXT:        "spmd.terminator"()
// CHECK-NEXT:      })
// CHECK-NEXT:      "spmd.critical"() <{usedModel = 5 : i32}> ({
// CHECK-NEXT:        llvm.call @work(%c3_i32)
// CHECK-NEXT:        "spmd.terminator"()
// CHECK-NEXT:      }) : () -> !spmd.error
// CHECK-NEXT:      %[[R0:.*]], %{{.*}} = "spmd.getRankInComm"(%[[TEAM]]) <{usedModel = 5 : i32}>
// CHECK-NEXT:      %[[IS0:.*]] = arith.cmpi eq, %[[R0]], %c0_i32 : i32
// CHECK-NEXT:      scf.if %[[IS0]] {
// CHECK-NEXT:        llvm.call @work(%c4_i32)
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[R2:.*]], %{{.*}} = "spmd.getRankInComm"(%[[TEAM]]) <{usedModel = 5 : i32}>
// CHECK-NEXT:      %[[IS2:.*]] = arith.cmpi eq, %[[R2]], %c2_i32 : i32
// CHECK-NEXT:      scf.if %[[IS2]] {
// CHECK-NEXT:        llvm.call @work(%c5_i32)
// CHECK-NEXT:      }
// CHECK-NEXT:      "spmd.task"() <{mergeable = false, {{.*}}untied = true, usedModel = 5 : i32}> ({
// CHECK-NEXT:        llvm.call @work(%c6_i32)
// CHECK-NEXT:        "spmd.terminator"()
// CHECK-NEXT:      })
// CHECK-NEXT:      "spmd.taskwait"() <{usedModel = 5 : i32}>
// CHECK-NEXT:      "spmd.taskgroup"() <{usedModel = 5 : i32}> ({
// CHECK-NEXT:        llvm.call @work(%c7_i32)
// CHECK-NEXT:        "spmd.terminator"()
// CHECK-NEXT:      })
// Schedule kinds: 0 static, 1 dynamic.
// CHECK-NEXT:      "spmd.wsloop"(%[[TEAM]], %c2_i32) <{{{.*}}schedule_kind = 1 : i32, usedModel = 5 : i32}> ({
// CHECK-NEXT:        "spmd.loop_nest"(%c0_i32, %c10_i32, %c1_i32) ({
// CHECK-NEXT:        ^bb0(%[[I:.*]]: i32):
// CHECK-NEXT:          llvm.call @work(%[[I]])
// CHECK-NEXT:          "spmd.yield"()
// CHECK-NEXT:        })
// CHECK-NEXT:      })
// CHECK-NEXT:      "spmd.wsloop"(%[[TEAM]]) <{{{.*}}schedule_kind = 0 : i32, usedModel = 5 : i32}> ({
// CHECK:             }) : (i32, i32, i32) -> ()
// CHECK-NEXT:      }) : (!spmd.comm) -> ()
// CHECK-NEXT:      "spmd.barrier"(%[[TEAM]]) <{isBlocking = true, isUniqueMatching = true, usedModel = 5 : i32}>
// CHECK-NEXT:      "spmd.parallel_yield"() : () -> ()
// CHECK-NEXT:    })
llvm.func @work(i32)
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c5 = arith.constant 5 : i32
  %c6 = arith.constant 6 : i32
  %c7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : i32
  %c10 = arith.constant 10 : i32
  omp.parallel num_threads(%c4 : i32) {
    llvm.call @work(%c0) : (i32) -> ()
    omp.barrier
    omp.single {
      llvm.call @work(%c1) : (i32) -> ()
      omp.terminator
    }
    omp.single nowait {
      llvm.call @work(%c2) : (i32) -> ()
      omp.terminator
    }
    omp.critical {
      llvm.call @work(%c3) : (i32) -> ()
      omp.terminator
    }
    omp.master {
      llvm.call @work(%c4) : (i32) -> ()
      omp.terminator
    }
    omp.masked filter(%c2 : i32) {
      llvm.call @work(%c5) : (i32) -> ()
      omp.terminator
    }
    omp.task untied {
      llvm.call @work(%c6) : (i32) -> ()
      omp.terminator
    }
    omp.taskyield
    omp.taskwait
    omp.taskgroup {
      llvm.call @work(%c7) : (i32) -> ()
      omp.terminator
    }
    omp.wsloop schedule(dynamic = %c2 : i32) nowait {
      omp.loop_nest (%i) : i32 = (%c0) to (%c10) step (%c1) {
        llvm.call @work(%i) : (i32) -> ()
        omp.yield
      }
    }
    omp.wsloop schedule(static) {
      omp.loop_nest (%i) : i32 = (%c0) to (%c10) step (%c1) {
        llvm.call @work(%i) : (i32) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return %c0 : i32
}
