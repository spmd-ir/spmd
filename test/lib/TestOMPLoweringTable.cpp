// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// -test-omp-lowering-table: prints the OpenMP lowering's call descriptions, which
// are derived from the CACTS map and the USIS API description rather than
// written out (lib/spmd/Conversion/OMPLoweringTable.cpp).
//
// The derivation is the part of -lower-spmd-to-omp the lowering tests cannot
// see: they show the calls the lowering emits for the operations they contain,
// not what it would emit for the rest of the map.  Printing every entry pins
// all of them, so that a change to the raiser's map -- a new call, a moved
// argument, a new alias -- shows up here rather than in whichever program
// happens to use it.

#include "TestPasses.h"

#include "OMPLoweringTable.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::spmd::test {
namespace {

struct TestOMPLoweringTablePass
    : public PassWrapper<TestOMPLoweringTablePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOMPLoweringTablePass)

  StringRef getArgument() const final { return "test-omp-lowering-table"; }
  StringRef getDescription() const final {
    return "Print the OpenMP calls -lower-spmd-to-omp derives from the CACTS map "
           "and the API description";
  }

  void runOnOperation() override {
    OpBuilder builder(&getContext());
    if (getOMPLoweringTable().empty()) {
      getOperation().emitError()
          << "no OpenMP API description; set OMP_JSON_PATH";
      return signalPassFailure();
    }
    printOMPLoweringTable(llvm::outs(), builder);
    // mlir-opt prints the module through its own stream on the same file
    // descriptor, so flush before it does.
    llvm::outs().flush();
  }
};

} // namespace

void registerTestOMPLoweringTablePass() {
  PassRegistration<TestOMPLoweringTablePass>();
}

} // namespace mlir::spmd::test
