// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// -test-spmd-uniformity: solves the analyses -check-collectives solves, then
// writes the two uniformity queries an optimization over communication needs.
//
//   test.uniform_exec      on every op of the spmd dialect:
//                          isUniformlyExecuted
//   test.uniform_operands  on every CommBufferOpInterface op: isUniformValue
//                          holds for its communicator, counts and root

#include "TestPasses.h"

#include "spmd/Analysis/ExecutionContextAnalysis.h"
#include "spmd/Analysis/ParallelContextAnalysis.h"
#include "spmd/Analysis/UniformityAnalysis.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::spmd::test {
namespace {

bool operandsAreUniform(CommBufferOpInterface op, DataFlowSolver &solver) {
  CommBufferOperands operands = op.getCommBufferOperands();
  for (OpOperand *operand : {operands.sendCount, operands.recvCount,
                             operands.root})
    if (operand && !isUniformValue(operand->get(), op, solver))
      return false;
  for (Value operand : op->getOperands())
    if (isa<CommType>(operand.getType()) &&
        !isUniformValue(operand, op, solver))
      return false;
  return true;
}

struct TestUniformityPass
    : public PassWrapper<TestUniformityPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestUniformityPass)

  StringRef getArgument() const final { return "test-spmd-uniformity"; }
  StringRef getDescription() const final {
    return "Annotate spmd ops with whether every rank executes them the same "
           "number of times";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    DataFlowSolver aliasSolver;
    aliasSolver.load<enzyme::AliasAnalysis>(&getContext());
    aliasSolver.load<enzyme::PointsToPointerAnalysis>();
    aliasSolver.load<dataflow::DeadCodeAnalysis>();
    aliasSolver.load<dataflow::SparseConstantPropagation>();
    if (failed(aliasSolver.initializeAndRun(module)))
      return signalPassFailure();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<ParallelContextAnalysis>();
    solver.load<UniformityValueAnalysis>(aliasSolver);
    solver.load<UniformityMemoryAnalysis>(aliasSolver);
    solver.load<ExecutionContextAnalysis>();
    if (failed(solver.initializeAndRun(module)))
      return signalPassFailure();

    Builder builder(module.getContext());
    module.walk([&](Operation *op) {
      if (!isSpmdOp(op))
        return;
      op->setAttr("test.uniform_exec",
                  builder.getBoolAttr(isUniformlyExecuted(op, solver)));
      if (auto commOp = dyn_cast<CommBufferOpInterface>(op))
        op->setAttr("test.uniform_operands",
                    builder.getBoolAttr(operandsAreUniform(commOp, solver)));
    });
  }
};

} // namespace

void registerTestUniformityPass() { PassRegistration<TestUniformityPass>(); }

} // namespace mlir::spmd::test
