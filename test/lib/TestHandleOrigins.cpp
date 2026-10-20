// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// -test-spmd-comm-origins, -test-spmd-request-origins: solve
// CommOriginsAnalysis or HandleOriginsAnalysis, each with a HandleSlotAnalysis
// for the same kind of handle, then write on every op of the spmd dialect what
// resolveHandleFacts answers for each of its communicator or request operands:
//
//   test.origins = ["<atoms>", ...]
//       <atoms>   the values the handle may originate from, sorted, or
//                 "unknown", or "none"

#include "TestPasses.h"

#include "spmd/Analysis/CommOriginsAnalysis.h"
#include "spmd/Analysis/HandleOriginsAnalysis.h"
#include "spmd/Analysis/HandleSlotAnalysis.h"
#include "spmd/SPMDTypes.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

#include <string>

namespace mlir::spmd::test {
namespace {

std::string describeFacts(const HandleFactsValue &facts) {
  if (facts.unknown)
    return "unknown";
  if (facts.atoms.empty())
    return "none";
  SmallVector<std::string> names;
  for (Value atom : facts.atoms)
    names.push_back(valAsString(atom));
  llvm::sort(names);
  return llvm::join(names, ", ");
}

/// Solves `OriginsAnalysis` with a HandleSlotAnalysis for `kind` over `module`
/// and annotates the operands of type `HandleT`.
template <typename OriginsAnalysis, typename HandleT>
LogicalResult annotateOrigins(ModuleOp module, HandleKind kind) {
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<OriginsAnalysis>();
  solver.load<HandleSlotAnalysis>(kind);
  if (failed(solver.initializeAndRun(module)))
    return failure();

  Builder builder(module.getContext());
  module.walk([&](Operation *op) {
    if (!isSpmdOp(op))
      return;
    SmallVector<Attribute> origins;
    for (Value operand : op->getOperands())
      if (isa<HandleT>(operand.getType()))
        origins.push_back(builder.getStringAttr(
            describeFacts(resolveHandleFacts(solver, operand, op, kind))));
    if (!origins.empty())
      op->setAttr("test.origins", builder.getArrayAttr(origins));
  });
  return success();
}

struct TestCommOriginsPass
    : public PassWrapper<TestCommOriginsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCommOriginsPass)

  StringRef getArgument() const final { return "test-spmd-comm-origins"; }
  StringRef getDescription() const final {
    return "Annotate communicator operands with the communicators they may "
           "denote";
  }

  void runOnOperation() override {
    if (failed(annotateOrigins<CommOriginsAnalysis, CommType>(
            getOperation(), HandleKind::Communicator)))
      signalPassFailure();
  }
};

struct TestRequestOriginsPass
    : public PassWrapper<TestRequestOriginsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRequestOriginsPass)

  StringRef getArgument() const final { return "test-spmd-request-origins"; }
  StringRef getDescription() const final {
    return "Annotate request operands with the requests they may denote";
  }

  void runOnOperation() override {
    if (failed(annotateOrigins<HandleOriginsAnalysis, ReqType>(
            getOperation(), HandleKind::Request)))
      signalPassFailure();
  }
};

} // namespace

void registerTestHandleOriginsPasses() {
  PassRegistration<TestCommOriginsPass>();
  PassRegistration<TestRequestOriginsPass>();
}

} // namespace mlir::spmd::test
