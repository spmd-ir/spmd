// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// -test-nccl-lowering-table: prints the NCCL lowering's call descriptions, which
// are derived from the CACTS map and the USIS API description rather than
// written out (lib/spmd/Conversion/NCCLLoweringTable.cpp).
//
// The derivation is the part of -lower-spmd-to-nccl the lowering tests cannot
// see: they show the calls the lowering emits for the operations they contain,
// not what it would emit for the rest of the map.  Printing every entry pins
// all of them, so that a change to the raiser's map -- a new call, a moved
// argument, a new alias -- shows up here rather than in whichever program
// happens to use it.

#include "TestPasses.h"

#include "NCCLLoweringTable.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::spmd::test {
namespace {

struct TestNCCLLoweringTablePass
    : public PassWrapper<TestNCCLLoweringTablePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestNCCLLoweringTablePass)

  StringRef getArgument() const final { return "test-nccl-lowering-table"; }
  StringRef getDescription() const final {
    return "Print the NCCL calls -lower-spmd-to-nccl derives from the CACTS map "
           "and the API description";
  }

  void runOnOperation() override {
    OpBuilder builder(&getContext());
    if (getNCCLLoweringTable().empty()) {
      getOperation().emitError()
          << "no NCCL API description; set NCCL_JSON_PATH";
      return signalPassFailure();
    }
    printNCCLLoweringTable(llvm::outs(), builder);
    // mlir-opt prints the module through its own stream on the same file
    // descriptor, so flush before it does.
    llvm::outs().flush();
  }
};

} // namespace

void registerTestNCCLLoweringTablePass() {
  PassRegistration<TestNCCLLoweringTablePass>();
}

} // namespace mlir::spmd::test
