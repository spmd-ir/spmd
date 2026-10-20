// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @shmem_"
//
// OpenSHMEM one-sided transfers. OpenSHMEM has no windows: the remote side of
// a transfer is an address in symmetric memory. The raiser gives every
// transfer a window of its own over that address (spmd.winCreate on the
// world, displacement 0), so a put or get has the same shape as MPI's. The
// element type comes from the call's name, and nelems is both counts.
//
// Here the remote address is `dest` (8 ints) for a put and `source` (16 ints)
// for a get; nelems is 5 and the PE 3. A stride is 1 unless the call has one.
//
// Not here, because they do not raise to valid operations yet (MR 16's
// description, finding 6): shmem_<T>_p, _atomic_set, _atomic_swap and
// _atomic_compare_swap. The strides of shmem_<T>_iget are not checked
// (finding 7).

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK-DAG:     %[[NOORIGIN:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:     %[[DEST:.*]] = llvm.alloca %c8_i32 x i32
// CHECK-DAG:     %[[SRC:.*]] = llvm.alloca %c16_i32 x i32
// CHECK-DAG:     %[[SIG:.*]] = llvm.alloca %c3_i32 x i64
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %c2_i64 = arith.constant 2 : i64
  %c5_i64 = arith.constant 5 : i64
  %c6_i64 = arith.constant 6 : i64
  %c42_i64 = arith.constant 42 : i64
  // SHMEM_SIGNAL_SET
  %set = arith.constant 0 : i32
  llvm.call @shmem_init() : () -> ()
  %dest = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr
  %src = llvm.alloca %c16 x i32 : (i32) -> !llvm.ptr
  %sigaddr = llvm.alloca %c3 x i64 : (i32) -> !llvm.ptr

  // shmem_int_put(dest, source, nelems, pe)
  // CHECK: %[[O0:.*]] = "enzymexla.pointer2memref"(%[[SRC]]) : (!llvm.ptr) -> memref<16xi32>
  // CHECK: %[[T0:.*]] = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}>
  // CHECK: %[[D0:.*]] = "enzymexla.pointer2memref"(%[[DEST]]) : (!llvm.ptr) -> memref<8xi32>
  // CHECK: %[[C0:.*]] = "spmd.commWorld"() <{usedModel = 1 : i32}>
  // CHECK: %[[W0:.*]], %{{.*}} = "spmd.winCreate"(%[[C0]], %[[D0]], %c0_i32) <{usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%[[O0]], %c5_i64, %[[T0]], %c1_i64, %c3_i32, %[[W0]], %c0_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}>
  llvm.call @shmem_int_put(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  // shmem_int_get(dest, source, nelems, pe): dest is local, source remote.
  // CHECK: %[[O1:.*]] = "enzymexla.pointer2memref"(%[[DEST]]) : (!llvm.ptr) -> memref<8xi32>
  // CHECK: %[[D1:.*]] = "enzymexla.pointer2memref"(%[[SRC]]) : (!llvm.ptr) -> memref<16xi32>
  // CHECK: %[[W1:.*]], %{{.*}} = "spmd.winCreate"(%{{.*}}, %[[D1]], %c0_i32)
  // CHECK: "spmd.get"(%[[O1]], %c5_i64, %{{.*}}, %c1_i64, %c3_i32, %[[W1]], %c0_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}>
  llvm.call @shmem_int_get(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // shmem_int_iput(dest, source, dst = 2, sst = 6, nelems, pe): the origin
  // (source) steps by sst, the target (dest) by dst.
  // CHECK: %[[O2:.*]] = "enzymexla.pointer2memref"(%[[SRC]]) : (!llvm.ptr) -> memref<16xi32>
  // CHECK: %[[W2:.*]], %{{.*}} = "spmd.winCreate"(%{{.*}}, %{{.*}}, %c0_i32) {{.*}} -> (!spmd.win, !spmd.error)
  // CHECK: "spmd.put"(%[[O2]], %c5_i64, %{{.*}}, %c6_i64, %c3_i32, %[[W2]], %c0_i64, %c5_i64, %{{.*}}, %c2_i64) <{isAtomic = false, isBlocking = true
  llvm.call @shmem_int_iput(%dest, %src, %c2_i64, %c6_i64, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32) -> ()
  // CHECK: %[[O3:.*]] = "enzymexla.pointer2memref"(%[[DEST]]) : (!llvm.ptr) -> memref<8xi32>
  // CHECK: %[[W3:.*]], %{{.*}} = "spmd.winCreate"(%{{.*}}, %{{.*}}, %c0_i32) {{.*}} -> (!spmd.win, !spmd.error)
  // CHECK: "spmd.get"(%[[O3]], %c5_i64, %{{.*}}, %{{.*}}, %c3_i32, %[[W3]], %c0_i64, %c5_i64, %{{.*}}, %{{.*}}) <{isAtomic = false, isBlocking = true
  llvm.call @shmem_int_iget(%dest, %src, %c2_i64, %c6_i64, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32) -> ()

  // The _nbi forms return nothing to wait on: the op only says it does not
  // block.
  // CHECK: "spmd.put"(%{{.*}}, %c5_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = false, usedModel = 1 : i32}>
  // CHECK-SAME: -> !spmd.error
  llvm.call @shmem_int_put_nbi(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  // CHECK: "spmd.get"(%{{.*}}, %c5_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = false, usedModel = 1 : i32}>
  // CHECK-SAME: -> !spmd.error
  llvm.call @shmem_int_get_nbi(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // shmem_int_g(source, pe) returns the element: it is fetched into temporary
  // storage and loaded from there.
  // CHECK: %[[TMP5:.*]] = memref.alloca() : memref<1xi32>
  // CHECK: "spmd.get"(%[[TMP5]], %c1_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c1_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = true
  // CHECK: %[[G:.*]] = memref.load %[[TMP5]][%c0] : memref<1xi32>
  %g = llvm.call @shmem_int_g(%src, %c3) : (!llvm.ptr, i32) -> i32

  // shmem_int_put_signal(dest, source, nelems, sig_addr, signal = 42,
  // sig_op, pe): a put, plus the signal written to a window over sig_addr.
  // CHECK: %[[SW:.*]] = "enzymexla.pointer2memref"(%[[SIG]]) : (!llvm.ptr) -> memref<3xi64>
  // CHECK: %[[SWIN:.*]], %{{.*}} = "spmd.winCreate"(%{{.*}}, %[[SW]], %c0_i32)
  // CHECK: "spmd.putSignal"(%{{.*}}, %c5_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c5_i64, %{{.*}}, %c1_i64, %[[SWIN]], %c42_i64, %[[SETOP:.*]]) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (memref<16xi32>, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64, !spmd.win, i64, !spmd.signalOp) -> !spmd.error
  llvm.call @shmem_int_put_signal(%dest, %src, %c5_i64, %sigaddr, %c42_i64, %set, %c3) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32) -> ()
  // CHECK: "spmd.putSignal"({{.*}}) <{isAtomic = false, isBlocking = false, usedModel = 1 : i32}>
  llvm.call @shmem_int_put_signal_nbi(%dest, %src, %c5_i64, %sigaddr, %c42_i64, %set, %c3) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32) -> ()

  // Atomic operations on one element.
  // CHECK: %[[TMP8:.*]] = memref.alloca() : memref<1xi32>
  // CHECK: "spmd.get"(%[[TMP8]], %c1_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c1_i64, %{{.*}}, %c1_i64) <{isAtomic = true, isBlocking = true
  // CHECK: %[[F:.*]] = memref.load %[[TMP8]][%c0] : memref<1xi32>
  %f = llvm.call @shmem_int_atomic_fetch(%src, %c3) : (!llvm.ptr, i32) -> i32
  // Fetch and increment: a get-accumulate with the "inc" operation and no
  // origin, the old value fetched into temporary storage.
  // CHECK: %[[TMP9:.*]] = memref.alloca() : memref<1xi32>
  // CHECK: %[[INC:.*]] = "spmd.constOpType"() <{stringAttr = "inc", usedModel = 1 : i32}>
  // CHECK: "spmd.getAccumulate"(%[[NOORIGIN]], %c1_i64, %{{.*}}, %c1_i64, %[[TMP9]], %c1_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c1_i64, %{{.*}}, %c1_i64, %[[INC]]) <{isAtomic = true, isBlocking = true
  // CHECK: %[[FI:.*]] = memref.load %[[TMP9]][%c0] : memref<1xi32>
  %fi = llvm.call @shmem_int_atomic_fetch_inc(%dest, %c3) : (!llvm.ptr, i32) -> i32
  // shmem_int_atomic_fetch_nbi(fetch, source, pe): the program provides the
  // storage.
  // CHECK: %[[FETCH:.*]] = "enzymexla.pointer2memref"(%[[DEST]]) : (!llvm.ptr) -> memref<8xi32>
  // CHECK: "spmd.get"(%[[FETCH]], %c1_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c1_i64, %{{.*}}, %c1_i64) <{isAtomic = true, isBlocking = false
  llvm.call @shmem_int_atomic_fetch_nbi(%dest, %src, %c3) : (!llvm.ptr, !llvm.ptr, i32) -> ()

  // The sized calls move untyped words: signless integers.
  // CHECK: %[[T32:.*]] = "spmd.constDatatype"() <{typeAttr = i32, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c5_i64, %[[T32]], %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = true
  llvm.call @shmem_put32(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  // CHECK: %[[T64:.*]] = "spmd.constDatatype"() <{typeAttr = i64, usedModel = 1 : i32}>
  // CHECK: "spmd.get"(%{{.*}}, %c5_i64, %[[T64]], %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = false
  llvm.call @shmem_get64_nbi(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  // CHECK: %[[T128:.*]] = "spmd.constDatatype"() <{typeAttr = i128, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c5_i64, %[[T128]], %c6_i64, %c3_i32, %{{.*}}, %c0_i64, %c5_i64, %{{.*}}, %c2_i64) <{isAtomic = false, isBlocking = true
  llvm.call @shmem_iput128(%dest, %src, %c2_i64, %c6_i64, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32) -> ()

  llvm.call @shmem_finalize() : () -> ()
  // CHECK: %[[S:.*]] = arith.addi %[[G]], %[[F]] : i32
  // CHECK: arith.addi %[[S]], %[[FI]] : i32
  %s = arith.addi %g, %f : i32
  %t = arith.addi %s, %fi : i32
  llvm.return %t : i32
}
llvm.func @shmem_init()
llvm.func @shmem_finalize()
llvm.func @shmem_int_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int_get(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int_iput(!llvm.ptr, !llvm.ptr, i64, i64, i64, i32)
llvm.func @shmem_int_iget(!llvm.ptr, !llvm.ptr, i64, i64, i64, i32)
llvm.func @shmem_int_put_nbi(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int_get_nbi(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int_g(!llvm.ptr, i32) -> i32
llvm.func @shmem_int_put_signal(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32)
llvm.func @shmem_int_put_signal_nbi(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32)
llvm.func @shmem_int_atomic_fetch(!llvm.ptr, i32) -> i32
llvm.func @shmem_int_atomic_fetch_inc(!llvm.ptr, i32) -> i32
llvm.func @shmem_int_atomic_fetch_nbi(!llvm.ptr, !llvm.ptr, i32)
llvm.func @shmem_put32(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_get64_nbi(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_iput128(!llvm.ptr, !llvm.ptr, i64, i64, i64, i32)
