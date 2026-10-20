// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Pass/Pass.h"
#include "spmd/SPMDPasses.h"

#include "spmd/Analysis/ExecutionContextAnalysis.h"
#include "spmd/Analysis/HandleOriginsAnalysis.h"
#include "spmd/Analysis/PendingAccessAnalysis.h"
#include "spmd/Analysis/UniformityAnalysis.h"

#include "Enzyme/MLIR/Analysis/DataFlowAliasAnalysis.h"

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

#define DEBUG_TYPE "check-dataRace"
#define PATTERN "main"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

namespace mlir::spmd {
#define GEN_PASS_DEF_CHECKDATARACE
#include "spmd/SPMDPasses.h.inc"

namespace {

static constexpr int kMaxErrorCount = 100;
static constexpr llvm::StringLiteral kSeparator =
    "----------------------------------------------------------------------"
    "----------------------------------------------";

static std::string getDefaultJsonReportPath() {
  return (std::filesystem::current_path() / "local_data_race_report.json")
      .string();
}

static llvm::json::Object buildJsonMessage(const ErrorInfo &err) {
  Location prevLoc = findLocToShowWrapper(err.prevOpr->getLoc());
  Location causingLoc = findLocToShowWrapper(err.causingOpr->getLoc());

  llvm::json::Array references;
  references.push_back(getLocationJson(prevLoc));
  references.push_back(getLocationJson(causingLoc));

  std::ostringstream text;
  text << "Found " << err.prevOpr->getName().stripDialect().str()
       << " operation/call at " << locToString(prevLoc)
       << " which is in conflict with "
       << err.causingOpr->getName().stripDialect().str()
       << " operation/call at " << locToString(causingLoc) << " before release";

  return llvm::json::Object{
      {"error_id", ""},
      {"type", "Local Data Race"},
      {"text", text.str()},
      {"references", std::move(references)},
  };
}

static void dumpErrorsToJson(const ErrorSet &errors,
                             llvm::StringRef outputFilePath) {
  llvm::json::Array messages;
  for (const auto &err : errors)
    messages.push_back(buildJsonMessage(err));

  std::string actualPath = outputFilePath.empty() ? getDefaultJsonReportPath()
                                                  : outputFilePath.str();

  std::error_code ec;
  llvm::raw_fd_ostream outFile(actualPath, ec);
  if (ec) {
    llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
    return;
  }

  llvm::json::Object fullReport{{"messages", std::move(messages)}};
  outFile << llvm::formatv("{0:2}", llvm::json::Value(std::move(fullReport)))
          << "\n";
}

static void emitSingleError(const ErrorInfo &errorInfo) {
  errorInfo.causingOpr->emitWarning(
      "[CheckDataRace]" + errorInfo.causingOpr->getName().stripDialect().str() +
      " operation/call may access the same memory as the previous "
      "operation without proper synch. (Data Race)");

  fprintf(stderr, "\n\nPrevious operation:\n\n");
  errorInfo.prevOpr->emitWarning(
      errorInfo.prevOpr->getName().stripDialect().str() + " operation/call");

  fprintf(stderr, "\n\nWith the alias values:\n");
  fprintf(stderr, "Causing Access on: %s\n",
          valAsString(errorInfo.aliasCausing).c_str());
  fprintf(stderr, "Previous Access on: %s\n",
          valAsString(errorInfo.aliasPrev).c_str());

  fprintf(stderr, "\n%s\n\n", kSeparator.data());
}

static void printErrorInfos(const ErrorSet &errors) {
  fprintf(stderr, "\n\n%s\n", kSeparator.data());
  fprintf(stderr, "Static Verification of Data Races:\n");
  fprintf(stderr, "%s\n\n", kSeparator.data());

  if (errors.empty()) {
    fprintf(stderr, "No Data Race Found\n");
    fprintf(stderr, "\n%s\n\n\n", kSeparator.data());
    return;
  }

  size_t emitted = 0;
  for (const auto &errorInfo : errors) {
    if (emitted >= kMaxErrorCount)
      break;
    emitSingleError(errorInfo);
    ++emitted;
  }

  fprintf(stderr, "\n\n");
  if (emitted >= kMaxErrorCount) {
    fprintf(stderr, "\n%s\n", kSeparator.data());
    fprintf(stderr, "Too many errors/warnings to emit! Aborted.");
    fprintf(stderr, "\n%s\n\n\n", kSeparator.data());
  }
}

struct CheckDataRace : impl::CheckDataRaceBase<CheckDataRace> {
  using CheckDataRaceBase::CheckDataRaceBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    Operation *mainOpr = moduleOp.lookupSymbol("main");
    if (!mainOpr) {
      LLVM_DEBUG(DBGS() << "No main function found\n");
      signalPassFailure();
      return;
    }

    // Phase 1
    DataFlowSolver aliasSolver;
    aliasSolver.load<enzyme::AliasAnalysis>(&getContext());
    aliasSolver.load<enzyme::PointsToPointerAnalysis>();
    aliasSolver.load<dataflow::DeadCodeAnalysis>();
    aliasSolver.load<dataflow::SparseConstantPropagation>();

    if (failed(aliasSolver.initializeAndRun(moduleOp))) {
      signalPassFailure();
      return;
    }

    DataFlowSolver uniformitySolver;
    uniformitySolver.load<dataflow::DeadCodeAnalysis>();
    uniformitySolver.load<dataflow::SparseConstantPropagation>();
    uniformitySolver.load<ParallelContextAnalysis>();
    uniformitySolver.load<UniformityValueAnalysis>(aliasSolver);
    uniformitySolver.load<UniformityMemoryAnalysis>(aliasSolver);
    uniformitySolver.load<ExecutionContextAnalysis>();
    uniformitySolver.load<HandleOriginsAnalysis>();
    auto *pending = uniformitySolver.load<PendingAccessAnalysis>(aliasSolver);

    if (failed(uniformitySolver.initializeAndRun(moduleOp))) {
      signalPassFailure();
      return;
    }
    const ErrorSet &errors = pending->getErrors();
    if (!errors.empty())
      signalPassFailure();

    if (writeJsonReport)
      dumpErrorsToJson(errors, "");
    else
      printErrorInfos(errors);
  }
};

} // namespace
} // namespace mlir::spmd