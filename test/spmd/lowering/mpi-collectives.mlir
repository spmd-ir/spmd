// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// Every collective, blocking and non-blocking.  The arguments follow each C
// prototype - MPI_Bcast has one buffer where spmd.bcast has two, and a rooted
// collective takes the root before the communicator - and the non-blocking
// form appends the address its request is written to and read back from.
// Element counts are i64 in the dialect and int at MPI.

// CHECK-LABEL: func.func @barrier
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         llvm.call @MPI_Barrier(%[[WORLD]]) : (i32) -> i32
// CHECK:         llvm.call @MPI_Ibarrier(%[[WORLD]], %[[REQ:[0-9]+]]) : (i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @barrier() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %req, %e1 = "spmd.barrier"(%world) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @bcast
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Bcast(%arg1, %{{.*}}, %[[DOUBLE]], %arg3, %[[WORLD]]) : (!llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Ibcast(%arg1, %{{.*}}, %[[DOUBLE]], %arg3, %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @bcast(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.bcast"(%world, %rbuf, %rbuf, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> !spmd.error
  %req, %e1 = "spmd.bcast"(%world, %rbuf, %rbuf, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @reduce
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Reduce(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %arg3, %[[WORLD]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Ireduce(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %arg3, %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @reduce(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.reduce"(%world, %sbuf, %rbuf, %count, %dt, %op, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  %req, %e1 = "spmd.reduce"(%world, %sbuf, %rbuf, %count, %dt, %op, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, i32) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @allreduce
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Allreduce(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %[[WORLD]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Iallreduce(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @allreduce(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.allreduce"(%world, %sbuf, %rbuf, %count, %dt, %op) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %req, %e1 = "spmd.allreduce"(%world, %sbuf, %rbuf, %count, %dt, %op) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @gather
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Gather(%arg0, %{{.*}}, %[[DOUBLE]], %arg1, %{{.*}}, %[[DOUBLE]], %arg3, %[[WORLD]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Igather(%arg0, %{{.*}}, %[[DOUBLE]], %arg1, %{{.*}}, %[[DOUBLE]], %arg3, %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @gather(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.gather"(%world, %sbuf, %count, %dt, %rbuf, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, i32) -> !spmd.error
  %req, %e1 = "spmd.gather"(%world, %sbuf, %count, %dt, %rbuf, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @scatter
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Scatter(%arg0, %{{.*}}, %[[DOUBLE]], %arg1, %{{.*}}, %[[DOUBLE]], %arg3, %[[WORLD]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Iscatter(%arg0, %{{.*}}, %[[DOUBLE]], %arg1, %{{.*}}, %[[DOUBLE]], %arg3, %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @scatter(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.scatter"(%world, %sbuf, %count, %dt, %rbuf, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, i32) -> !spmd.error
  %req, %e1 = "spmd.scatter"(%world, %sbuf, %count, %dt, %rbuf, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @allgather
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Allgather(%arg0, %{{.*}}, %[[DOUBLE]], %arg1, %{{.*}}, %[[DOUBLE]], %[[WORLD]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Iallgather(%arg0, %{{.*}}, %[[DOUBLE]], %arg1, %{{.*}}, %[[DOUBLE]], %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @allgather(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.allgather"(%world, %sbuf, %count, %dt, %rbuf, %count, %dt) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype) -> !spmd.error
  %req, %e1 = "spmd.allgather"(%world, %sbuf, %count, %dt, %rbuf, %count, %dt) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @alltoall
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Alltoall(%arg0, %{{.*}}, %[[DOUBLE]], %arg1, %{{.*}}, %[[DOUBLE]], %[[WORLD]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Ialltoall(%arg0, %{{.*}}, %[[DOUBLE]], %arg1, %{{.*}}, %[[DOUBLE]], %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @alltoall(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.alltoall"(%world, %sbuf, %count, %dt, %rbuf, %count, %dt) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype) -> !spmd.error
  %req, %e1 = "spmd.alltoall"(%world, %sbuf, %count, %dt, %rbuf, %count, %dt) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @scan
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Scan(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %[[WORLD]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Iscan(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @scan(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.scan"(%world, %sbuf, %rbuf, %count, %dt, %op) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %req, %e1 = "spmd.scan"(%world, %sbuf, %rbuf, %count, %dt, %op) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @exscan
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Exscan(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %[[WORLD]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Iexscan(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @exscan(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.exscan"(%world, %sbuf, %rbuf, %count, %dt, %op) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %req, %e1 = "spmd.exscan"(%world, %sbuf, %rbuf, %count, %dt, %op) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  return
}

// CHECK-LABEL: func.func @reduce_scatter
// CHECK:         %[[WORLD:.*]] = llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         %[[DOUBLE:.*]] = llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         %[[SUM:.*]] = llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.call @MPI_Reduce_scatter_block(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %[[WORLD]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
// CHECK:         llvm.call @MPI_Ireduce_scatter_block(%arg0, %arg1, %{{.*}}, %[[DOUBLE]], %[[SUM]], %[[WORLD]], %[[REQ:[0-9]+]]) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.load %[[REQ]] : !llvm.ptr -> i32
func.func @reduce_scatter(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i32, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %count = arith.extui %n : i32 to i64
  %e0 = "spmd.reduceScatter"(%world, %sbuf, %rbuf, %count, %dt, %op) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %req, %e1 = "spmd.reduceScatter"(%world, %sbuf, %rbuf, %count, %dt, %op) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  return
}
