// RUN: spmd-opt %s -lower-spmd-to-nvshmem | FileCheck %s --implicit-check-not='"spmd.'
//
// The collectives, as shmem-collectives.mlir has them for OpenSHMEM, on the
// same operations: NVSHMEM spells the element type and the reduction
// operation in the function name as OpenSHMEM does, so the calls are the same
// with an nvshmem_ prefix.  The team is NVSHMEM_TEAM_WORLD, the constant 0.

// The destination comes first.  The bitwise reductions exist for the unsigned
// and the fixed-width names only, so an and over si64 is
// nvshmem_int64_and_reduce, and an xor over i8, of unknown signedness,
// nvshmem_int8_xor_reduce.  A program translated from MPI would spell the
// operations SUM, BXOR and PROD; they are the same ones.
// CHECK-LABEL: func.func @allreduce
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[STATUS:.*]] = llvm.call @nvshmem_int_sum_reduce(%[[WORLD]], %arg1, %arg0, %arg2) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @nvshmem_double_max_reduce(%[[WORLD]], %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_int64_and_reduce(%[[WORLD]], %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_int_sum_reduce(%[[WORLD]], %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_int8_xor_reduce(%[[WORLD]], %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_double_prod_reduce(%[[WORLD]], %arg1, %arg0, %arg2)
func.func @allreduce(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %i32 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %f64 = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 2 : i32}> : () -> !spmd.datatype
  %i64 = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 2 : i32}> : () -> !spmd.datatype
  %i8 = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 2 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 2 : i32}> : () -> !spmd.op
  %max = "spmd.constOpType"() <{stringAttr = "max", usedModel = 2 : i32}> : () -> !spmd.op
  %and = "spmd.constOpType"() <{stringAttr = "and", usedModel = 2 : i32}> : () -> !spmd.op
  %mpiSum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 2 : i32}> : () -> !spmd.op
  %mpiBxor = "spmd.constOpType"() <{stringAttr = "BXOR", usedModel = 2 : i32}> : () -> !spmd.op
  %mpiProd = "spmd.constOpType"() <{stringAttr = "PROD", usedModel = 2 : i32}> : () -> !spmd.op
  %e0 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %i32, %sum) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e1 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %f64, %max) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e2 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %i64, %and) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e3 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %i32, %mpiSum) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e4 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %i8, %mpiBxor) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e5 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %f64, %mpiProd) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %code = "spmd.castSPMDErrorToI32"(%e0) : (!spmd.error) -> i32
  return
}

// Signedness picks the call where it changes the result.
// CHECK-LABEL: func.func @signedness
// CHECK:         llvm.call @nvshmem_uint_max_reduce(%{{.*}}, %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_int_max_reduce(%{{.*}}, %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_ulong_min_reduce(%{{.*}}, %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_uint_and_reduce(%{{.*}}, %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_uint_sum_reduce(%{{.*}}, %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_int_sum_reduce(%{{.*}}, %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_char_max_reduce(%{{.*}}, %arg1, %arg0, %arg2)
func.func @signedness(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %ui32 = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %si32 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %ui64 = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 2 : i32}> : () -> !spmd.datatype
  %i32 = "spmd.constDatatype"() <{typeAttr = i32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %i8 = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 2 : i32}> : () -> !spmd.datatype
  %max = "spmd.constOpType"() <{stringAttr = "max", usedModel = 2 : i32}> : () -> !spmd.op
  %mpiMin = "spmd.constOpType"() <{stringAttr = "MIN", usedModel = 2 : i32}> : () -> !spmd.op
  %and = "spmd.constOpType"() <{stringAttr = "and", usedModel = 2 : i32}> : () -> !spmd.op
  %sum = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 2 : i32}> : () -> !spmd.op
  %e0 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %ui32, %max) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e1 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %si32, %max) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e2 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %ui64, %mpiMin) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e3 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %ui32, %and) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e4 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %ui32, %sum) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e5 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %i32, %sum) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e6 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %i8, %max) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}

// An allgather is nvshmem_<T>_fcollect.  A blocking collective the
// optimisations gave a request to hands back one that needs no completing,
// and waiting on it is nvshmem_quiet().
// CHECK-LABEL: func.func @bcast_gather
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         llvm.call @nvshmem_float_broadcast(%[[WORLD]], %arg1, %arg0, %arg2, %arg3) : (i32, !llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK:         llvm.call @nvshmem_short_fcollect(%[[WORLD]], %arg1, %arg0, %arg2) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @nvshmem_short_alltoall(%[[WORLD]], %arg1, %arg0, %arg2) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @nvshmem_float_broadcast(%[[WORLD]], %arg1, %arg0, %arg2, %arg3)
// CHECK:         llvm.call @nvshmem_quiet() : () -> ()
func.func @bcast_gather(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %f32 = "spmd.constDatatype"() <{typeAttr = f32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %i16 = "spmd.constDatatype"() <{typeAttr = si16, usedModel = 2 : i32}> : () -> !spmd.datatype
  %e0 = "spmd.bcast"(%world, %sbuf, %rbuf, %n, %f32, %root) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.allgather"(%world, %sbuf, %n, %i16, %rbuf, %n, %i16) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype) -> !spmd.error
  %e2 = "spmd.alltoall"(%world, %sbuf, %n, %i16, %rbuf, %n, %i16) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype) -> !spmd.error
  %req, %e3 = "spmd.bcast"(%world, %sbuf, %rbuf, %n, %f32, %root) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %e4 = "spmd.wait"(%req) <{usedModel = 2 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// NVSHMEM reduces and collects half and bfloat16, which OpenSHMEM has no
// names for.
// CHECK-LABEL: func.func @half_types
// CHECK:         llvm.call @nvshmem_half_sum_reduce(%{{.*}}, %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_bfloat16_max_reduce(%{{.*}}, %arg1, %arg0, %arg2)
// CHECK:         llvm.call @nvshmem_bfloat16_fcollect(%{{.*}}, %arg1, %arg0, %arg2)
func.func @half_types(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %f16 = "spmd.constDatatype"() <{typeAttr = f16, usedModel = 2 : i32}> : () -> !spmd.datatype
  %bf16 = "spmd.constDatatype"() <{typeAttr = bf16, usedModel = 2 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 2 : i32}> : () -> !spmd.op
  %max = "spmd.constOpType"() <{stringAttr = "max", usedModel = 2 : i32}> : () -> !spmd.op
  %e0 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %f16, %sum) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e1 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %bf16, %max) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e2 = "spmd.allgather"(%world, %sbuf, %n, %bf16, %rbuf, %n, %bf16) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype) -> !spmd.error
  return
}
