// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Pass/Pass.h"
#include "spmd/SPMDPasses.h"

#include "spmd/Analysis/CollectiveMismatchAnalysis.h"
#include "spmd/Analysis/CommOriginsAnalysis.h"
#include "spmd/Analysis/ExecutionContextAnalysis.h"
#include "spmd/Analysis/UniformityAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#include "CACTS/utils.h"
#include "common.h"
#include "spmd/Utils/PassUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <sstream>
#include <string>

#define DEBUG_TYPE "check-collectives"
#define PATTERN "main"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

namespace mlir::spmd {
#define GEN_PASS_DEF_CHECKCOLLECTIVES
#include "spmd/SPMDPasses.h.inc"

namespace {

static constexpr llvm::StringLiteral kSeparator =
    "----------------------------------------------------------------------"
    "------------------------";

static std::string getDefaultJsonReportPath() {
  return (std::filesystem::current_path() / "collective_mismatch_report.json")
      .string();
}

static void
dumpErrorsToJson(const SmallVector<CollectiveMismatchErrorInfo> &errorInfos,
                 llvm::StringRef outputFilePath) {
  llvm::json::Array messages;

  for (const auto &err : errorInfos) {
    llvm::json::Array references;
    for (Operation *op : err.conflictOprs)
      references.push_back(getLocationJson(findLocToShowWrapper(op->getLoc())));

    std::ostringstream oss;
    oss << "Detected conflict involving " << err.conflictOprs.size()
        << " operation(s) and " << err.causingOprs.size()
        << " causing conditional(s).";

    messages.push_back(llvm::json::Object{
        {"error_id", ""},
        {"type", "Collective Mismatch"},
        {"text", oss.str()},
        {"references", std::move(references)},
    });
  }

  std::string actualPath = outputFilePath.empty() ? getDefaultJsonReportPath()
                                                  : outputFilePath.str();

  std::error_code ec;
  llvm::raw_fd_ostream outFile(actualPath, ec);
  if (ec) {
    llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
    return;
  }

  llvm::json::Object full{{"messages", std::move(messages)}};
  outFile << llvm::formatv("{0:2}", llvm::json::Value(std::move(full))) << "\n";
}

static void
printErrorInfos(const SmallVector<CollectiveMismatchErrorInfo> &errorInfos) {
  fprintf(stderr, "\n\n%s\n", kSeparator.data());
  fprintf(stderr, "Static Verification of Collective Communication:");
  fprintf(stderr, "\n%s\n\n", kSeparator.data());

  if (errorInfos.empty()) {
    fprintf(stderr, "No Collective Communication Error Found");
    fprintf(stderr, "\n\n%s\n\n", kSeparator.data());
    return;
  }

  for (const auto &errorInfo : errorInfos) {
    for (size_t i = 0; i < errorInfo.conflictOprs.size(); ++i) {
      errorInfo.conflictOprs[i]->emitWarning(
          "[CheckCollectives] " +
          errorInfo.conflictOprs[i]->getName().stripDialect().str() +
          " operation/call may not be called by all processes in "
          "communicator/team (Call Ordering Error)");

      if (i + 1 < errorInfo.conflictOprs.size())
        fprintf(stderr, "\n\nand\n\n");
    }

    fprintf(stderr, "\n----------------\n");
    fprintf(stderr, "\nProbably due to the conditionals:");
    fprintf(stderr, "\n--------\n");

    for (size_t i = 0; i < errorInfo.causingOprs.size(); ++i) {
      errorInfo.causingOprs[i]->emitRemark(
          llvm::Twine("probable causing conditional number ") + llvm::Twine(i));

      if (i + 1 < errorInfo.causingOprs.size())
        fprintf(stderr, "\n\nand\n\n");
    }

    fprintf(stderr, "\n%s\n\n", kSeparator.data());
  }

  fprintf(stderr, "\n\n");
}

struct CheckCollectives : impl::CheckCollectivesBase<CheckCollectives> {
  using CheckCollectivesBase::CheckCollectivesBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    Operation *mainOpr = moduleOp.lookupSymbol("main");
    if (!mainOpr) {
      LLVM_DEBUG(DBGS() << "Pass failed since no main func exists\n");
      signalPassFailure();
      return;
    }

    // // Phase 1: alias analysis — must run to fixpoint before
    // ReachingDefinition
    DataFlowSolver aliasSolver;
    aliasSolver.load<enzyme::AliasAnalysis>(&getContext());
    aliasSolver.load<enzyme::PointsToPointerAnalysis>();
    aliasSolver.load<dataflow::DeadCodeAnalysis>();
    aliasSolver.load<dataflow::SparseConstantPropagation>();
    if (failed(aliasSolver.initializeAndRun(moduleOp))) {
      LLVM_DEBUG(DBGS() << "Alias analysis failed\n");
      signalPassFailure();
      return;
    }

    // Phase 2: remaining analyses with frozen alias results
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<ParallelContextAnalysis>();
    solver.load<UniformityValueAnalysis>(aliasSolver);
    solver.load<UniformityMemoryAnalysis>(aliasSolver);
    solver.load<ExecutionContextAnalysis>();
    solver.load<CommOriginsAnalysis>();
    solver.load<HandleSlotAnalysis>(HandleKind::Communicator);
    if (failed(solver.initializeAndRun(moduleOp))) {
      LLVM_DEBUG(DBGS() << "Dataflow initialization/run failed\n");
      signalPassFailure();
      return;
    }

    CollectiveMismatchAnalysis collectiveAnalysis(
        solver, static_cast<uint64_t>(numRanks));

    if (failed(collectiveAnalysis.run(moduleOp))) {
      LLVM_DEBUG(DBGS() << "Collective mismatch analysis failed\n");
      signalPassFailure();
      return;
    }

    const auto &errorInfos = collectiveAnalysis.getErrors();

    if (!errorInfos.empty())
      signalPassFailure();

    if (writeJsonReport) {
      LLVM_DEBUG(
          DBGS() << "Writing collective mismatch report in JSON format.\n");
      dumpErrorsToJson(errorInfos, "");
    } else {
      LLVM_DEBUG(DBGS() << "Writing collective mismatch report to stderr.\n");
      printErrorInfos(errorInfos);
    }
  }
};

} // namespace
} // namespace mlir::spmd