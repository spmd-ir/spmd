// RUN: spmd-opt %s -privatize-symbols="exclude=kept" | FileCheck %s
//
// Every function defined in the module becomes private, so passes that drop
// unused symbols may drop it; a declaration has to stay visible to be a
// declaration, and a name in `exclude` keeps its visibility. Globals are not
// touched.

// CHECK:       func.func private @defined()
// CHECK:       func.func private @declared()
// CHECK:       func.func @kept()
// CHECK:       llvm.func @llvm_defined() attributes {sym_visibility = "private"}
// CHECK:       llvm.func @llvm_declared()
// CHECK-NOT:   sym_visibility
// CHECK:       llvm.mlir.global external @g
func.func @defined() {
  return
}
func.func private @declared()
func.func @kept() {
  return
}
llvm.func @llvm_defined() {
  llvm.return
}
llvm.func @llvm_declared()
llvm.mlir.global external @g(0 : i32) : i32
