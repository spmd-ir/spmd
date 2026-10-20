// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @nccl"
//
// NCCL's datatypes and reduction operations are read from
// include/spmd/TargetABI/NCCL.h, which //test/abi:nccl_abi_test checks against
// the vendored nccl.h.  The raiser used to know ncclInt and ncclSum only and
// stopped the tool on anything else.
//
// An integer datatype raises to the signed or unsigned type of its width, as
// MPI's do: ncclUint8 is ui8, so that ncclMax over it lowers back to ncclUint8
// rather than to the ncclInt8 that compares differently.

// CHECK-LABEL: llvm.func @reductions
// CHECK-DAG:     %[[F32:.*]] = "spmd.constDatatype"() <{typeAttr = f32, usedModel = 3 : i32}>
// CHECK-DAG:     %[[MAX:.*]] = "spmd.constOpType"() <{stringAttr = "MAX", usedModel = 3 : i32}>
// CHECK-DAG:     %[[U8:.*]] = "spmd.constDatatype"() <{typeAttr = ui8, usedModel = 3 : i32}>
// CHECK-DAG:     %[[AVG:.*]] = "spmd.constOpType"() <{stringAttr = "AVG", usedModel = 3 : i32}>
// CHECK-DAG:     %[[FP8:.*]] = "spmd.constDatatype"() <{typeAttr = f8E4M3FN, usedModel = 3 : i32}>
// CHECK-DAG:     %[[PROD:.*]] = "spmd.constOpType"() <{stringAttr = "PROD", usedModel = 3 : i32}>
// CHECK:         "spmd.allreduce"({{.*}}, %[[F32]], %[[MAX]], {{.*}})
// CHECK:         "spmd.allreduce"({{.*}}, %[[U8]], %[[AVG]], {{.*}})
// CHECK:         "spmd.allreduce"({{.*}}, %[[FP8]], %[[PROD]], {{.*}})
llvm.func @reductions(%comm: !llvm.ptr, %stream: !llvm.ptr, %sbuf: !llvm.ptr, %rbuf: !llvm.ptr) {
  %n = arith.constant 4 : i64
  %float = arith.constant 7 : i32
  %max = arith.constant 2 : i32
  %uint8 = arith.constant 1 : i32
  %avg = arith.constant 4 : i32
  %float8e4m3 = arith.constant 10 : i32
  %prod = arith.constant 1 : i32
  %r0 = llvm.call @ncclAllReduce(%sbuf, %rbuf, %n, %float, %max, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  %r1 = llvm.call @ncclAllReduce(%sbuf, %rbuf, %n, %uint8, %avg, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  %r2 = llvm.call @ncclAllReduce(%sbuf, %rbuf, %n, %float8e4m3, %prod, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.return
}
llvm.func @ncclAllReduce(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
