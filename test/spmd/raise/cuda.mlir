// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @cudaMem"
//
// The two CUDA runtime calls the raiser maps (cudaSpmdEntries.cpp), tagged
// usedModel = 4. The byte count and the copy direction pass through unchanged;
// a buffer becomes a memref of what the pointer points to. Whether a call
// blocks is what USIS says about it: cudaMemcpy returns once the copy is done,
// cudaMemset is ordered on the stream.

// CHECK-LABEL: llvm.func @copy
// CHECK:         %[[HOST:.*]] = llvm.alloca %c16_i32 x i32
// CHECK:         %[[DEV:.*]] = llvm.alloca %c8_i32 x i32
llvm.func @copy() -> i32 {
  %c7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %c32 = arith.constant 32 : i64
  // cudaMemcpyHostToDevice
  %h2d = arith.constant 1 : i32
  %host = llvm.alloca %c16 x i32 : (i32) -> !llvm.ptr
  %dev = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr

  // cudaMemcpy(dst, src, count, kind)
  // CHECK: %[[DST:.*]] = "enzymexla.pointer2memref"(%[[DEV]]) : (!llvm.ptr) -> memref<8xi32>
  // CHECK: %[[SRC:.*]] = "enzymexla.pointer2memref"(%[[HOST]]) : (!llvm.ptr) -> memref<16xi32>
  // CHECK: "spmd.memcpy"(%[[DST]], %[[SRC]], %c32_i64, %c1_i32) <{usedModel = 4 : i32}> {isBlocking = true}
  // CHECK-SAME: : (memref<8xi32>, memref<16xi32>, i64, i32) -> !spmd.error
  %0 = llvm.call @cudaMemcpy(%dev, %host, %c32, %h2d) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32

  // cudaMemset(devPtr, value, count)
  // CHECK: %[[SET:.*]] = "enzymexla.pointer2memref"(%[[DEV]]) : (!llvm.ptr) -> memref<8xi32>
  // CHECK: "spmd.memset"(%[[SET]], %c7_i32, %c32_i64) <{usedModel = 4 : i32}> {isBlocking = false}
  // CHECK-SAME: : (memref<8xi32>, i32, i64) -> !spmd.error
  %1 = llvm.call @cudaMemset(%dev, %c7, %c32) : (!llvm.ptr, i32, i64) -> i32

  llvm.return %c7 : i32
}
llvm.func @cudaMemcpy(!llvm.ptr, !llvm.ptr, i64, i32) -> i32
llvm.func @cudaMemset(!llvm.ptr, i32, i64) -> i32
