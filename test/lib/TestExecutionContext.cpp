// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Solve the analyses -check-collectives and -check-thread-safety solve, then
// write what they computed into test.* attributes:
//
// -test-spmd-parallel-context   on every op of the spmd dialect:
//   test.parallel = "<levels>"
//       the ParallelContextState before the op, outermost level first, each
//       "mg<usedModel>.<proc|thread>", or "<uninitialized>"
//
// -test-spmd-execution-context  on every op of the spmd dialect:
//   test.exec = "<ExecutionContext>"
//       the ExecutionContextLattice before the op, as ExecutionContext::print
//       writes it, or "<uninitialized>"
//
// -test-spmd-value-uniformity   on every op with results inside an
//                               spmd.parallel:
//   test.uniformity = ["<levels>", ...]
//       per result, its uniformity at every level of the parallel context
//       before the op, "mg<usedModel>.<proc|thread>=<uniformity>"

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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::spmd::test {
namespace {

/// The solvers of -check-collectives: a frozen alias solver, and the
/// uniformity solver with ParallelContextAnalysis and ExecutionContextAnalysis
/// on top of it.
struct Solvers {
  DataFlowSolver alias;
  DataFlowSolver solver;

  LogicalResult run(ModuleOp module) {
    alias.load<enzyme::AliasAnalysis>(module.getContext());
    alias.load<enzyme::PointsToPointerAnalysis>();
    alias.load<dataflow::DeadCodeAnalysis>();
    alias.load<dataflow::SparseConstantPropagation>();
    if (failed(alias.initializeAndRun(module)))
      return failure();

    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<ParallelContextAnalysis>();
    solver.load<UniformityValueAnalysis>(alias);
    solver.load<UniformityMemoryAnalysis>(alias);
    solver.load<ExecutionContextAnalysis>();
    return solver.initializeAndRun(module);
  }
};

std::string describeLevel(Operation *parallelOp) {
  auto parallel = cast<ParallelOp>(parallelOp);
  std::string rendered;
  llvm::raw_string_ostream os(rendered);
  os << "mg" << parallel.getUsedModel() << "."
     << (parallel.getIsProcessLevel() ? "proc" : "thread");
  return rendered;
}

const ParallelContextState *getParallelContext(Operation *op,
                                               DataFlowSolver &solver) {
  const auto *state = solver.lookupState<ParallelContextState>(
      solver.getProgramPointBefore(op));
  return state && state->isInitialized() ? state : nullptr;
}

struct TestParallelContextPass
    : public PassWrapper<TestParallelContextPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestParallelContextPass)

  StringRef getArgument() const final { return "test-spmd-parallel-context"; }
  StringRef getDescription() const final {
    return "Annotate spmd ops with the stack of parallel regions they run in";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Solvers solvers;
    if (failed(solvers.run(module)))
      return signalPassFailure();

    Builder builder(module.getContext());
    module.walk([&](Operation *op) {
      if (!isSpmdOp(op))
        return;
      std::string rendered = "<uninitialized>";
      if (const ParallelContextState *state =
              getParallelContext(op, solvers.solver)) {
        SmallVector<std::string> levels;
        for (Operation *level : state->getStack())
          levels.push_back(describeLevel(level));
        rendered = llvm::join(levels, " ");
      }
      op->setAttr("test.parallel", builder.getStringAttr(rendered));
    });
  }
};

struct TestExecutionContextPass
    : public PassWrapper<TestExecutionContextPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestExecutionContextPass)

  StringRef getArgument() const final { return "test-spmd-execution-context"; }
  StringRef getDescription() const final {
    return "Annotate spmd ops with the execution context before them";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Solvers solvers;
    if (failed(solvers.run(module)))
      return signalPassFailure();

    Builder builder(module.getContext());
    module.walk([&](Operation *op) {
      if (!isSpmdOp(op))
        return;
      std::string rendered;
      llvm::raw_string_ostream os(rendered);
      const auto *lattice = solvers.solver.lookupState<ExecutionContextLattice>(
          solvers.solver.getProgramPointBefore(op));
      if (lattice)
        lattice->print(os);
      else
        os << "<uninitialized>";
      op->setAttr("test.exec", builder.getStringAttr(rendered));
    });
  }
};

struct TestValueUniformityPass
    : public PassWrapper<TestValueUniformityPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestValueUniformityPass)

  StringRef getArgument() const final { return "test-spmd-value-uniformity"; }
  StringRef getDescription() const final {
    return "Annotate the results of ops in parallel regions with their "
           "uniformity at every parallel level";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Solvers solvers;
    if (failed(solvers.run(module)))
      return signalPassFailure();

    Builder builder(module.getContext());
    module.walk([&](Operation *op) {
      if (op->getNumResults() == 0 || !op->getParentOfType<ParallelOp>())
        return;
      const ParallelContextState *state =
          getParallelContext(op, solvers.solver);
      if (!state)
        return;
      SmallVector<Attribute> perResult;
      for (Value result : op->getResults()) {
        SmallVector<std::string> levels;
        for (Operation *level : state->getStack()) {
          std::string rendered;
          llvm::raw_string_ostream os(rendered);
          os << describeLevel(level) << "="
             << getUniformityAtLevel(result, level, solvers.solver);
          levels.push_back(rendered);
        }
        perResult.push_back(builder.getStringAttr(llvm::join(levels, " ")));
      }
      op->setAttr("test.uniformity", builder.getArrayAttr(perResult));
    });
  }
};

} // namespace

void registerTestExecutionContextPasses() {
  PassRegistration<TestParallelContextPass>();
  PassRegistration<TestExecutionContextPass>();
  PassRegistration<TestValueUniformityPass>();
}

} // namespace mlir::spmd::test
