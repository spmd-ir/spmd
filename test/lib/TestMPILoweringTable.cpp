// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// -test-mpi-lowering-table: prints the MPI lowering's call descriptions, which
// are derived from the CACTS map and the USIS API description rather than
// written out (lib/spmd/Conversion/MPILoweringTable.cpp).
//
// The derivation is the part of -lower-spmd-to-mpi the lowering tests cannot
// see: they show the calls the lowering emits for the operations they contain,
// not what it would emit for the rest of the map.  Printing every entry pins
// all of them, so that a change to the raiser's map -- a new call, a moved
// argument, a new alias -- shows up here rather than in whichever program
// happens to use it.

#include "TestPasses.h"

#include "MPILoweringTable.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::spmd::test {
namespace {

struct TestMPILoweringTablePass
    : public PassWrapper<TestMPILoweringTablePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMPILoweringTablePass)

  StringRef getArgument() const final { return "test-mpi-lowering-table"; }
  StringRef getDescription() const final {
    return "Print the MPI calls -lower-spmd-to-mpi derives from the CACTS map "
           "and the API description";
  }

  void runOnOperation() override {
    OpBuilder builder(&getContext());
    if (getMPILoweringTable().empty()) {
      getOperation().emitError()
          << "no MPI API description; set MPI_JSON_PATH";
      return signalPassFailure();
    }
    printMPILoweringTable(llvm::outs(), builder);
    // mlir-opt prints the module through its own stream on the same file
    // descriptor, so flush before it does.
    llvm::outs().flush();
  }
};

} // namespace

void registerTestMPILoweringTablePass() {
  PassRegistration<TestMPILoweringTablePass>();
}

} // namespace mlir::spmd::test
