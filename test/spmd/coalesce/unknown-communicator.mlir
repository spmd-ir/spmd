// RUN: spmd-opt %s --split-input-file -coalesce-comm=allow-staging=false | FileCheck %s
//
// A communicator CommOriginsAnalysis cannot resolve could be any communicator,
// so a single message on one stops merging on every communicator - whether the
// module mentions that communicator before or after it.  Each scenario is a
// module of its own, since the poison reaches everything in its module.

// The argument of a public function: nothing shows which communicator a caller
// outside the module passes.  It comes before any message on MPI_COMM_WORLD.
// CHECK-LABEL: func.func @unknown_communicator_first
// CHECK:         spmd.send
func.func @unknown_communicator_first(%comm : !spmd.comm, %peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 0 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %buf = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %e = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The message above may be on MPI_COMM_WORLD with tag 0, expecting 4 elements,
// so merging tags 0 and 1 here could hand it 8.
// CHECK-LABEL: func.func @world_group_after
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @world_group_after(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 0 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// -----

// The same two functions the other way round.
// CHECK-LABEL: func.func @world_group_before
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @world_group_before(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 0 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @unknown_communicator_last
// CHECK:         spmd.send
func.func @unknown_communicator_last(%comm : !spmd.comm, %peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 0 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %buf = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %e = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}
