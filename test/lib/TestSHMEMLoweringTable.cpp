// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// -test-shmem-lowering-table and -test-nvshmem-lowering-table: print the
// OpenSHMEM and the NVSHMEM lowering's call descriptions, which are derived
// from the CACTS map and the USIS API description rather than written out
// (lib/spmd/Conversion/SHMEMLoweringTable.cpp).
//
// The derivation is the part of -lower-spmd-to-shmem the lowering tests cannot
// see: they show the calls the lowering emits for the operations they contain,
// not what it would emit for the rest of the map.  It matters more here than
// for MPI, because both models put the element type and the reduction
// operation in the function name, so which of the ~1300 names the lowering
// picks is the lowering.

#include "TestPasses.h"

#include "SHMEMLoweringTable.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::spmd::test {
namespace {

/// The names that tell the two passes apart.
struct TableNames {
  StringLiteral argument;
  StringLiteral description;
  StringLiteral jsonVariable;
  StringLiteral displayName;
};

constexpr TableNames kOpenSHMEMNames = {
    "test-shmem-lowering-table",
    "Print the OpenSHMEM calls -lower-spmd-to-shmem derives from the CACTS "
    "map and the API description",
    "SHMEM_JSON_PATH", "OpenSHMEM"};
constexpr TableNames kNVSHMEMNames = {
    "test-nvshmem-lowering-table",
    "Print the NVSHMEM calls -lower-spmd-to-nvshmem derives from the CACTS "
    "map and the API description",
    "NVSHMEM_JSON_PATH", "NVSHMEM"};

template <SHMEMFlavour Flavour, const TableNames &Names>
struct TestSHMEMLoweringTablePass
    : public PassWrapper<TestSHMEMLoweringTablePass<Flavour, Names>,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSHMEMLoweringTablePass)

  TestSHMEMLoweringTablePass() = default;
  // An Option member is not copyable, so a pass that has one spells out how it
  // is cloned.
  TestSHMEMLoweringTablePass(const TestSHMEMLoweringTablePass &other)
      : PassWrapper<TestSHMEMLoweringTablePass<Flavour, Names>,
                    OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const final { return Names.argument; }
  StringRef getDescription() const final { return Names.description; }
  /// Only the calls the lowering would pick, which is what a test wants to
  /// read: the map has one entry per spelling of every element type, and the
  /// aliases are thousands of lines of the same call.
  Pass::Option<bool> canonicalOnly{
      *this, "canonical-only",
      llvm::cl::desc("Leave out the calls the lowering never picks"),
      llvm::cl::init(false)};
  /// The table has one entry per spelling of every element type; a test reads
  /// the handful of callees it is about.
  Pass::Option<std::string> filter{
      *this, "filter",
      llvm::cl::desc("Only the calls whose callee contains this"),
      llvm::cl::init("")};

  void runOnOperation() override {
    OpBuilder builder(&this->getContext());
    if (getSHMEMLoweringTable(Flavour).empty()) {
      this->getOperation().emitError()
          << "no " << Names.displayName << " API description; set "
          << Names.jsonVariable;
      return this->signalPassFailure();
    }
    printSHMEMLoweringTable(Flavour, llvm::outs(), builder, canonicalOnly,
                            filter);
    // mlir-opt prints the module through its own stream on the same file
    // descriptor, so flush before it does.
    llvm::outs().flush();
  }
};

} // namespace

void registerTestSHMEMLoweringTablePass() {
  PassRegistration<
      TestSHMEMLoweringTablePass<SHMEMFlavour::OpenSHMEM, kOpenSHMEMNames>>();
}

void registerTestNVSHMEMLoweringTablePass() {
  PassRegistration<
      TestSHMEMLoweringTablePass<SHMEMFlavour::NVSHMEM, kNVSHMEMNames>>();
}

} // namespace mlir::spmd::test
