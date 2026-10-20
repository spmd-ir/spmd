// RUN: spmd-opt %s -coalesce-comm=allow-staging=false | FileCheck %s
//
// Which communicator a message uses decides which census its tags go into.
// The communicator is resolved by CommOriginsAnalysis, not read off its
// defining op, so it survives function arguments and memory; one that stays
// unknown could be any communicator, and stops merging everywhere.

//===----------------------------------------------------------------------===//
// Resolved through a call
//===----------------------------------------------------------------------===//

// The communicator arrives as an argument of a private function whose only
// caller passes MPI_COMM_WORLD.  Every use of tags 0 and 1 is in this group,
// so it merges.
// CHECK-LABEL: func.func private @halo_on_argument
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func private @halo_on_argument(%comm : !spmd.comm, %peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 0 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @caller
func.func @caller(%peer : i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  func.call @halo_on_argument(%world, %peer) : (!spmd.comm, i32) -> ()
  return
}
