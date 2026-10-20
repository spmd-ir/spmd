// RUN: spmd-opt %s -lower-spmd-to-nccl | FileCheck %s --implicit-check-not='"spmd.'
//
// The NCCL collectives.  Each becomes the nccl* call of its shape, with the
// arguments in the order of its prototype: the buffers, the count, the
// datatype and the reduction as NCCL's enumerators (ncclInt32 = 2, ncclFloat64
// = 8, ncclSum = 0, ncclMax = 2, ncclAvg = 4), the communicator, and the CUDA
// stream the program passed -- spmd.castStream is that cudaStream_t.
//
// roundtrip's lowering passed a null stream to every call, and a datatype or
// reduction it did not recognise as int and sum.

// CHECK-LABEL: func.func @allreduce
// CHECK-SAME:    (%[[COMM:.*]]: !spmd.comm, %[[STREAM:.*]]: !llvm.ptr, %[[SEND:.*]]: !llvm.ptr, %[[RECV:.*]]: !llvm.ptr, %[[N:.*]]: i64)
// CHECK:         %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !spmd.comm to !llvm.ptr
// CHECK:         %[[F64:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:         %[[MAX:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[E:.*]] = llvm.call @ncclAllReduce(%[[SEND]], %[[RECV]], %[[N]], %[[F64]], %[[MAX]], %[[C]], %[[STREAM]]) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         return %[[E]] : i32
func.func @allreduce(%comm: !spmd.comm, %stream: !llvm.ptr, %send: !llvm.ptr, %recv: !llvm.ptr, %n: i64) -> i32 {
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 3 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "MAX", usedModel = 3 : i32}> : () -> !spmd.op
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %e = "spmd.allreduce"(%comm, %send, %recv, %n, %dt, %op, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, !spmd.stream) -> !spmd.error
  %ret = "spmd.castSPMDErrorToI32"(%e) : (!spmd.error) -> i32
  return %ret : i32
}

// An unsigned maximum stays unsigned: ncclUint32 = 3 and ncclUint64 = 5, not
// the ncclInt32 and ncclInt64 of the same width, which compare differently.
// CHECK-LABEL: func.func @unsigned_max
// CHECK-DAG:     %[[U32:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG:     %[[U64:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-DAG:     %[[MAX:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-DAG:     %[[MIN:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         llvm.call @ncclAllReduce(%arg2, %arg3, %arg4, %[[U32]], %[[MAX]], %{{.*}}, %arg1)
// CHECK:         llvm.call @ncclAllReduce(%arg2, %arg3, %arg4, %[[U64]], %[[MIN]], %{{.*}}, %arg1)
func.func @unsigned_max(%comm: !spmd.comm, %stream: !llvm.ptr, %send: !llvm.ptr, %recv: !llvm.ptr, %n: i64) {
  %u32 = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %u64 = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 3 : i32}> : () -> !spmd.datatype
  %max = "spmd.constOpType"() <{stringAttr = "MAX", usedModel = 3 : i32}> : () -> !spmd.op
  %min = "spmd.constOpType"() <{stringAttr = "MIN", usedModel = 3 : i32}> : () -> !spmd.op
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %e0 = "spmd.allreduce"(%comm, %send, %recv, %n, %u32, %max, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, !spmd.stream) -> !spmd.error
  %e1 = "spmd.allreduce"(%comm, %send, %recv, %n, %u64, %min, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, !spmd.stream) -> !spmd.error
  return
}

// An in-place broadcast is ncclBroadcast with the buffer twice, not the
// deprecated ncclBcast.  The root is passed as it is.
// CHECK-LABEL: func.func @bcast
// CHECK:         %[[INT:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         llvm.call @ncclBroadcast(%arg2, %arg2, %arg3, %[[INT]], %arg4, %{{.*}}, %arg1)
func.func @bcast(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64, %root: i32) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %e = "spmd.bcast"(%comm, %buf, %buf, %n, %dt, %root, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32, !spmd.stream) -> !spmd.error
  return
}

// A rooted reduction is ncclReduce, which roundtrip handed only the datatype
// and reduction it knew.
// CHECK-LABEL: func.func @reduce
// CHECK:         %[[F32:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[AVG:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:         llvm.call @ncclReduce(%arg2, %arg3, %arg4, %[[F32]], %[[AVG]], %arg5, %{{.*}}, %arg1) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
func.func @reduce(%comm: !spmd.comm, %stream: !llvm.ptr, %send: !llvm.ptr, %recv: !llvm.ptr, %n: i64, %root: i32) {
  %dt = "spmd.constDatatype"() <{typeAttr = f32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "AVG", usedModel = 3 : i32}> : () -> !spmd.op
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %e = "spmd.reduce"(%comm, %send, %recv, %n, %dt, %op, %root, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, i32, !spmd.stream) -> !spmd.error
  return
}

// The collectives that move the same count to and from every rank take it
// once, and the datatype once.
// CHECK-LABEL: func.func @gather_family
// CHECK:         %[[I64:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         llvm.call @ncclAllGather(%arg2, %arg3, %arg4, %[[I64]], %[[C:.*]], %arg1) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.call @ncclAlltoAll(%arg2, %arg3, %arg4, %[[I64]], %[[C]], %arg1) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.call @ncclReduceScatter(%arg2, %arg3, %arg4, %[[I64]], %[[SUM]], %[[C]], %arg1) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.call @ncclGather(%arg2, %arg3, %arg4, %[[I64]], %arg5, %[[C]], %arg1) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.call @ncclScatter(%arg2, %arg3, %arg4, %[[I64]], %arg5, %[[C]], %arg1) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
func.func @gather_family(%comm: !spmd.comm, %stream: !llvm.ptr, %send: !llvm.ptr, %recv: !llvm.ptr, %n: i64, %root: i32) {
  %dt = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 3 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 3 : i32}> : () -> !spmd.op
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %e0 = "spmd.allgather"(%comm, %send, %n, %dt, %recv, %n, %dt, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, !spmd.stream) -> !spmd.error
  %e1 = "spmd.alltoall"(%comm, %send, %n, %dt, %recv, %n, %dt, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, !spmd.stream) -> !spmd.error
  %e2 = "spmd.reduceScatter"(%comm, %send, %recv, %n, %dt, %op, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, !spmd.stream) -> !spmd.error
  %e3 = "spmd.gather"(%comm, %send, %n, %dt, %recv, %n, %dt, %root, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, i32, !spmd.stream) -> !spmd.error
  %e4 = "spmd.scatter"(%comm, %send, %n, %dt, %recv, %n, %dt, %root, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, i32, !spmd.stream) -> !spmd.error
  return
}
