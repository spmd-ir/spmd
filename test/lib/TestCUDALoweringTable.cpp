// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// -test-cuda-lowering-table: prints the CUDA lowering's call descriptions,
// which are derived from the CACTS map and the USIS API description rather
// than written out (lib/spmd/Conversion/CUDALoweringTable.cpp).
//
// The derivation is the part of -lower-spmd-to-cuda the lowering tests cannot
// see: they show the calls the lowering emits for the operations they contain,
// not what it would emit for the rest of the map.  Printing every entry pins
// all of them, so that a change to the raiser's map -- a new call, a moved
// argument, a new alias -- shows up here rather than in whichever program
// happens to use it.  For CUDA the map is two calls, so this dump is also
// where the MR's scope decision is visible.

#include "TestPasses.h"

#include "CUDALoweringTable.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::spmd::test {
namespace {

struct TestCUDALoweringTablePass
    : public PassWrapper<TestCUDALoweringTablePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCUDALoweringTablePass)

  StringRef getArgument() const final { return "test-cuda-lowering-table"; }
  StringRef getDescription() const final {
    return "Print the CUDA calls -lower-spmd-to-cuda derives from the CACTS "
           "map and the API description";
  }

  void runOnOperation() override {
    OpBuilder builder(&getContext());
    if (getCUDALoweringTable().empty()) {
      getOperation().emitError()
          << "no CUDA API description; set CUDA_JSON_PATH";
      return signalPassFailure();
    }
    printCUDALoweringTable(llvm::outs(), builder);
    // mlir-opt prints the module through its own stream on the same file
    // descriptor, so flush before it does.
    llvm::outs().flush();
  }
};

} // namespace

void registerTestCUDALoweringTablePass() {
  PassRegistration<TestCUDALoweringTablePass>();
}

} // namespace mlir::spmd::test
