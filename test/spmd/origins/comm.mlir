// RUN: spmd-opt %s -test-spmd-comm-origins | FileCheck %s
//
// resolveHandleFacts over CommOriginsAnalysis and HandleSlotAnalysis: which
// communicator a !spmd.comm operand denotes, also after it went through
// memory - stored into a variable and loaded back, the way the raiser hands on
// a communicator a call wrote through a pointer.

// CHECK-LABEL: func.func @loaded
// CHECK:         %[[DUP:[^,]*]], %{{.*}} = "spmd.commDup"
// CHECK:         spmd.barrier{{.*}}test.origins = ["%[[DUP]]"]
// CHECK:         spmd.barrier{{.*}}test.origins = ["%[[DUP]]"]
func.func @loaded() {
  %c1 = arith.constant 1 : i32
  %i0 = arith.constant 0 : index
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dup, %e = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %e0 = "spmd.barrier"(%dup) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %boxed = builtin.unrealized_conversion_cast %dup : !spmd.comm to !llvm.ptr
  %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  %store = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<1x!llvm.ptr>
  memref.store %boxed, %store[%i0] : memref<1x!llvm.ptr>
  %load = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<1x!llvm.ptr>
  %ptr = memref.load %load[%i0] : memref<1x!llvm.ptr>
  %comm = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !spmd.comm
  %e1 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// A communicator passed into a function that anyone may call.
// CHECK-LABEL: func.func @argument
// CHECK:         spmd.barrier{{.*}}test.origins = ["unknown"]
func.func @argument(%comm: !spmd.comm) {
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
