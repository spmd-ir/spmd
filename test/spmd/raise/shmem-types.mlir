// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @shmem_"
//
// The element type of a typed OpenSHMEM call comes from its name: one put per
// type of OpenSHMEM 1.6 Table 5, and the sized puts. Signed and unsigned
// where C says so; char is signless, since C leaves its sign to the
// platform, and so are the sized calls, which move words of no type. size_t
// and ptrdiff_t are 64 bits here, as on every platform spmd-verify targets.
// long double is x86's 80-bit extended type.
//
// Each call has its own count, so every CHECK pairs a name with its call.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c64 = arith.constant 64 : i32
  llvm.call @shmem_init() : () -> ()
  %dest = llvm.alloca %c64 x i8 : (i32) -> !llvm.ptr
  %src = llvm.alloca %c64 x i8 : (i32) -> !llvm.ptr

  // CHECK: %[[T1:.*]] = "spmd.constDatatype"() <{typeAttr = f32, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c1_i64, %[[T1]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c1_i64, %{{.*}}, %c1_i64)
  %n1 = arith.constant 1 : i64
  llvm.call @shmem_float_put(%dest, %src, %n1, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T2:.*]] = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c2_i64, %[[T2]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c2_i64, %{{.*}}, %c1_i64)
  %n2 = arith.constant 2 : i64
  llvm.call @shmem_double_put(%dest, %src, %n2, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T3:.*]] = "spmd.constDatatype"() <{typeAttr = f80, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c3_i64, %[[T3]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c3_i64, %{{.*}}, %c1_i64)
  %n3 = arith.constant 3 : i64
  llvm.call @shmem_longdouble_put(%dest, %src, %n3, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T4:.*]] = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c4_i64, %[[T4]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c4_i64, %{{.*}}, %c1_i64)
  %n4 = arith.constant 4 : i64
  llvm.call @shmem_char_put(%dest, %src, %n4, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T5:.*]] = "spmd.constDatatype"() <{typeAttr = si8, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c5_i64, %[[T5]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c5_i64, %{{.*}}, %c1_i64)
  %n5 = arith.constant 5 : i64
  llvm.call @shmem_schar_put(%dest, %src, %n5, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T6:.*]] = "spmd.constDatatype"() <{typeAttr = si16, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c6_i64, %[[T6]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c6_i64, %{{.*}}, %c1_i64)
  %n6 = arith.constant 6 : i64
  llvm.call @shmem_short_put(%dest, %src, %n6, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T7:.*]] = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c7_i64, %[[T7]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c7_i64, %{{.*}}, %c1_i64)
  %n7 = arith.constant 7 : i64
  llvm.call @shmem_int_put(%dest, %src, %n7, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T8:.*]] = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c8_i64, %[[T8]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c8_i64, %{{.*}}, %c1_i64)
  %n8 = arith.constant 8 : i64
  llvm.call @shmem_long_put(%dest, %src, %n8, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T9:.*]] = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c9_i64, %[[T9]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c9_i64, %{{.*}}, %c1_i64)
  %n9 = arith.constant 9 : i64
  llvm.call @shmem_longlong_put(%dest, %src, %n9, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T10:.*]] = "spmd.constDatatype"() <{typeAttr = ui8, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c10_i64, %[[T10]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c10_i64, %{{.*}}, %c1_i64)
  %n10 = arith.constant 10 : i64
  llvm.call @shmem_uchar_put(%dest, %src, %n10, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T11:.*]] = "spmd.constDatatype"() <{typeAttr = ui16, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c11_i64, %[[T11]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c11_i64, %{{.*}}, %c1_i64)
  %n11 = arith.constant 11 : i64
  llvm.call @shmem_ushort_put(%dest, %src, %n11, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T12:.*]] = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c12_i64, %[[T12]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c12_i64, %{{.*}}, %c1_i64)
  %n12 = arith.constant 12 : i64
  llvm.call @shmem_uint_put(%dest, %src, %n12, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T13:.*]] = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c13_i64, %[[T13]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c13_i64, %{{.*}}, %c1_i64)
  %n13 = arith.constant 13 : i64
  llvm.call @shmem_ulong_put(%dest, %src, %n13, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T14:.*]] = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c14_i64, %[[T14]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c14_i64, %{{.*}}, %c1_i64)
  %n14 = arith.constant 14 : i64
  llvm.call @shmem_ulonglong_put(%dest, %src, %n14, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T15:.*]] = "spmd.constDatatype"() <{typeAttr = si8, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c15_i64, %[[T15]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c15_i64, %{{.*}}, %c1_i64)
  %n15 = arith.constant 15 : i64
  llvm.call @shmem_int8_put(%dest, %src, %n15, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T16:.*]] = "spmd.constDatatype"() <{typeAttr = si16, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c16_i64, %[[T16]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c16_i64, %{{.*}}, %c1_i64)
  %n16 = arith.constant 16 : i64
  llvm.call @shmem_int16_put(%dest, %src, %n16, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T17:.*]] = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c17_i64, %[[T17]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c17_i64, %{{.*}}, %c1_i64)
  %n17 = arith.constant 17 : i64
  llvm.call @shmem_int32_put(%dest, %src, %n17, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T18:.*]] = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c18_i64, %[[T18]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c18_i64, %{{.*}}, %c1_i64)
  %n18 = arith.constant 18 : i64
  llvm.call @shmem_int64_put(%dest, %src, %n18, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T19:.*]] = "spmd.constDatatype"() <{typeAttr = ui8, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c19_i64, %[[T19]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c19_i64, %{{.*}}, %c1_i64)
  %n19 = arith.constant 19 : i64
  llvm.call @shmem_uint8_put(%dest, %src, %n19, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T20:.*]] = "spmd.constDatatype"() <{typeAttr = ui16, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c20_i64, %[[T20]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c20_i64, %{{.*}}, %c1_i64)
  %n20 = arith.constant 20 : i64
  llvm.call @shmem_uint16_put(%dest, %src, %n20, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T21:.*]] = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c21_i64, %[[T21]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c21_i64, %{{.*}}, %c1_i64)
  %n21 = arith.constant 21 : i64
  llvm.call @shmem_uint32_put(%dest, %src, %n21, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T22:.*]] = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c22_i64, %[[T22]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c22_i64, %{{.*}}, %c1_i64)
  %n22 = arith.constant 22 : i64
  llvm.call @shmem_uint64_put(%dest, %src, %n22, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T23:.*]] = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c23_i64, %[[T23]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c23_i64, %{{.*}}, %c1_i64)
  %n23 = arith.constant 23 : i64
  llvm.call @shmem_size_put(%dest, %src, %n23, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T24:.*]] = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c24_i64, %[[T24]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c24_i64, %{{.*}}, %c1_i64)
  %n24 = arith.constant 24 : i64
  llvm.call @shmem_ptrdiff_put(%dest, %src, %n24, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T25:.*]] = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c25_i64, %[[T25]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c25_i64, %{{.*}}, %c1_i64)
  %n25 = arith.constant 25 : i64
  llvm.call @shmem_put8(%dest, %src, %n25, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T26:.*]] = "spmd.constDatatype"() <{typeAttr = i16, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c26_i64, %[[T26]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c26_i64, %{{.*}}, %c1_i64)
  %n26 = arith.constant 26 : i64
  llvm.call @shmem_put16(%dest, %src, %n26, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T27:.*]] = "spmd.constDatatype"() <{typeAttr = i32, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c27_i64, %[[T27]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c27_i64, %{{.*}}, %c1_i64)
  %n27 = arith.constant 27 : i64
  llvm.call @shmem_put32(%dest, %src, %n27, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T28:.*]] = "spmd.constDatatype"() <{typeAttr = i64, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c28_i64, %[[T28]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c28_i64, %{{.*}}, %c1_i64)
  %n28 = arith.constant 28 : i64
  llvm.call @shmem_put64(%dest, %src, %n28, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[T29:.*]] = "spmd.constDatatype"() <{typeAttr = i128, usedModel = 1 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c29_i64, %[[T29]], %c1_i64, %c1_i32, %{{.*}}, %c0_i64, %c29_i64, %{{.*}}, %c1_i64)
  %n29 = arith.constant 29 : i64
  llvm.call @shmem_put128(%dest, %src, %n29, %c1) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  llvm.call @shmem_finalize() : () -> ()
  llvm.return %c0 : i32
}
llvm.func @shmem_init()
llvm.func @shmem_finalize()
llvm.func @shmem_float_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_double_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_longdouble_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_char_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_schar_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_short_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_long_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_longlong_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_uchar_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_ushort_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_uint_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_ulong_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_ulonglong_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int8_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int16_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int32_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int64_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_uint8_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_uint16_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_uint32_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_uint64_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_size_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_ptrdiff_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_put8(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_put16(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_put32(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_put64(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_put128(!llvm.ptr, !llvm.ptr, i64, i32)
