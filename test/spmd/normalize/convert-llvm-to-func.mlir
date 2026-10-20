// RUN: spmd-opt %s -convert-llvm-to-func | FileCheck %s
//
// llvm.func and llvm.call become func.func and func.call, private, with a
// single-block body's llvm.return turned into return. The func dialect has no
// variadic functions: those stay in the llvm dialect, and so do calls to them.
// spmd-verify does not run this pass. Not tested: an indirect call, which
// crashes the pass, and a multi-block function, which keeps its llvm.return
// (MR 16's description, finding 16).

// CHECK-LABEL: func.func private @callee(
// CHECK-SAME:    %[[X:.*]]: i32) -> i32 {
// CHECK-NEXT:    return %[[X]] : i32
// CHECK:       func.func private @decl(i32) -> i32
// CHECK:       llvm.func @printf(!llvm.ptr, ...) -> i32
// CHECK-LABEL: func.func private @main(
// CHECK-SAME:    %[[P:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[A:.*]] = call @callee(%{{.*}}) : (i32) -> i32
// CHECK-NEXT:    %[[B:.*]] = call @decl(%[[A]]) : (i32) -> i32
// CHECK-NEXT:    llvm.call @printf(%[[P]]) vararg(!llvm.func<i32 (ptr, ...)>)
// CHECK-NEXT:    return %[[B]] : i32
llvm.func @callee(%x: i32) -> i32 {
  llvm.return %x : i32
}
llvm.func @decl(i32) -> i32
llvm.func @printf(!llvm.ptr, ...) -> i32
llvm.func @main(%p: !llvm.ptr) -> i32 {
  %c1 = arith.constant 1 : i32
  %a = llvm.call @callee(%c1) : (i32) -> i32
  %b = llvm.call @decl(%a) : (i32) -> i32
  %c = llvm.call @printf(%p) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
  llvm.return %b : i32
}
