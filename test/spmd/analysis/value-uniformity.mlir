// RUN: spmd-opt %s -test-spmd-value-uniformity | FileCheck %s
//
// UniformityValueAnalysis and UniformityMemoryAnalysis: whether a value is the
// same on every rank (process level) or every thread (thread level) at each
// level it is used in, printed per result as "mg<usedModel>.<level>=<kind>".
// This is what decides whether a guard or loop bound makes ranks diverge.
//
// A rank is not uniform, a size is, and so is anything computed from uniform
// values only. Memory is tracked per byte of an allocation: loading the slot
// the size was stored in is uniform although the rank sits next to it. A
// thread number is not uniform at the thread level, nor at the process level
// the region is nested in.
//
// Not tested: that a collective makes its receive buffer uniform; stores are
// weak updates, so it does not yet (MR 16's description, finding 21).

// CHECK-LABEL: func.func @main
// CHECK:         %[[RANK:.*]], %{{.*}} = "spmd.getRankInComm"{{.*}}test.uniformity = ["mg0.proc=non-uniform", "mg0.proc=non-uniform"]
// CHECK:         %[[SIZE:.*]], %{{.*}} = "spmd.getSizeOfComm"{{.*}}test.uniformity = ["mg0.proc=uniform", "mg0.proc=uniform"]
// CHECK:         arith.muli %[[SIZE]], %[[SIZE]] {{{.*}}test.uniformity = ["mg0.proc=uniform"]
// CHECK:         arith.addi %[[RANK]], %[[SIZE]] {{{.*}}test.uniformity = ["mg0.proc=non-uniform"]
// CHECK:         memref.load {{.*}}test.name = "load_rank", test.uniformity = ["mg0.proc=non-uniform"]
// CHECK:         memref.load {{.*}}test.name = "load_size", test.uniformity = ["mg0.proc=uniform"]
// CHECK:         "spmd.getRankInComm"{{.*}}test.uniformity = ["mg0.proc=non-uniform mg5.thread=non-uniform", "mg0.proc=non-uniform mg5.thread=non-uniform"]
// CHECK:         "spmd.getSizeOfComm"{{.*}}test.uniformity = ["mg0.proc=uniform mg5.thread=uniform", "mg0.proc=uniform mg5.thread=uniform"]
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %size, %es = "spmd.getSizeOfComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %square = arith.muli %size, %size : i32
    %mixed = arith.addi %rank, %size : i32
    %pair = memref.alloca() : memref<2xi32>
    memref.store %rank, %pair[%i0] : memref<2xi32>
    memref.store %size, %pair[%i1] : memref<2xi32>
    %lrank = memref.load %pair[%i0] {test.name = "load_rank"} : memref<2xi32>
    %lsize = memref.load %pair[%i1] {test.name = "load_size"} : memref<2xi32>
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    "spmd.parallel"(%team, %c4) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 1, 0, 0>, usedModel = 5 : i32}> ({
      %tid, %et = "spmd.getRankInComm"(%team) <{usedModel = 5 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
      %tn, %etn = "spmd.getSizeOfComm"(%team) <{usedModel = 5 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
      "spmd.parallel_yield"() : () -> ()
    }) : (!spmd.comm, i32) -> ()
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
