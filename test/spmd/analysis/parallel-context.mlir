// RUN: spmd-opt %s -test-spmd-parallel-context | FileCheck %s
//
// ParallelContextAnalysis: the spmd.parallel regions an op runs in, outermost
// first, as "mg<usedModel>.<proc|thread>". The uniformity analyses key their
// per-level facts by these regions, so a wrong stack puts a fact on the wrong
// level. A function takes the stack of its call site. Not tested: a function
// called from inside a region and from outside it (MR 16's description,
// finding 22).

// CHECK-DAG:   test.name = "in_team", test.parallel = "mg0.proc mg5.thread"
// CHECK-DAG:   test.name = "uncalled", test.parallel = ""
// CHECK-DAG:   test.name = "outside", test.parallel = ""
// CHECK-DAG:   test.name = "process", test.parallel = "mg0.proc"
// CHECK-DAG:   test.name = "nested", test.parallel = "mg0.proc mg5.thread mg5.thread"
// CHECK-DAG:   test.name = "after", test.parallel = "mg0.proc"
func.func private @in_team(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "in_team"} : (!spmd.comm) -> !spmd.error
  return
}
func.func @uncalled(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "uncalled"} : (!spmd.comm) -> !spmd.error
  return
}
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> {test.name = "outside"} : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "process"} : (!spmd.comm) -> !spmd.error
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      %inner, %ei = "spmd.commDup"(%team) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
      "spmd.parallel"(%inner) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
        %e1 = "spmd.barrier"(%inner) <{isBlocking = true, usedModel = 5 : i32}> {test.name = "nested"} : (!spmd.comm) -> !spmd.error
        "spmd.parallel_yield"() : () -> ()
      }) : (!spmd.comm) -> ()
      func.call @in_team(%world) : (!spmd.comm) -> ()
      "spmd.parallel_yield"() : () -> ()
    }) : (!spmd.comm) -> ()
    %e2 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> {test.name = "after"} : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
