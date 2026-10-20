// RUN: spmd-opt %s -test-spmd-request-origins | FileCheck %s
//
// resolveHandleFacts over HandleOriginsAnalysis and HandleSlotAnalysis: which
// request a !spmd.req operand denotes. The shape is the raiser's for
// MPI_Request reqs[2]: MPI_Isend and MPI_Irecv write their requests into
// reqs[0] and reqs[1] as i32, and MPI_Wait(&reqs[1]) loads the second one back.
// The loaded request is the receive's, and only that.

// CHECK-LABEL: func.func @loaded
// CHECK:         %[[SEND:[^,]*]], %{{.*}} = "spmd.send"
// CHECK:         %[[RECV:[^,]*]], %{{.*}} = "spmd.recv"
// CHECK:         spmd.wait{{.*}}test.origins = ["%[[SEND]]"]
// CHECK:         spmd.wait{{.*}}test.origins = ["%[[RECV]]"]
func.func @loaded() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c1_64 = arith.constant 1 : i64
  %i0 = arith.constant 0 : index
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %buf = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %reqs = llvm.alloca %c1 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
  %sreq, %se = "spmd.send"(%world, %buf, %c1_64, %dt, %c0, %c0) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %m0 = "enzymexla.pointer2memref"(%reqs) : (!llvm.ptr) -> memref<1xi32>
  %s0 = builtin.unrealized_conversion_cast %sreq : !spmd.req to i32
  memref.store %s0, %m0[%i0] : memref<1xi32>
  %elem = llvm.getelementptr inbounds %reqs[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i32>
  %rreq, %re = "spmd.recv"(%world, %buf, %c1_64, %dt, %c0, %c0) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %m1 = "enzymexla.pointer2memref"(%elem) : (!llvm.ptr) -> memref<1xi32>
  %s1 = builtin.unrealized_conversion_cast %rreq : !spmd.req to i32
  memref.store %s1, %m1[%i0] : memref<1xi32>
  %w0 = "spmd.wait"(%sreq) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %m2 = "enzymexla.pointer2memref"(%elem) : (!llvm.ptr) -> memref<1xi32>
  %l = memref.load %m2[%i0] : memref<1xi32>
  %q = builtin.unrealized_conversion_cast %l : i32 to !spmd.req
  %w1 = "spmd.wait"(%q) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}
