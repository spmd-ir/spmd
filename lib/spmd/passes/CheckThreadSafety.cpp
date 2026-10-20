// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Pass/Pass.h"
#include "spmd/SPMDAttributes.h"
#include "spmd/SPMDOps.h"
#include "spmd/SPMDPasses.h"

#include "spmd/Analysis/ExecutionContextAnalysis.h"
#include "spmd/Analysis/UniformityAnalysis.h"

#include "Enzyme/MLIR/Analysis/DataFlowAliasAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#include "common.h"
#include "spmd/Utils/PassUtils.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <sstream>

#define DEBUG_TYPE "check-thread-safety"
#define DBGS() ::llvm::dbgs() << "[check-thread-safety] "

namespace mlir::spmd {
#define GEN_PASS_DEF_CHECKTHREADSAFETY
#include "spmd/SPMDPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Violation record
//===----------------------------------------------------------------------===//

struct ThreadSafetyViolation {
  Operation *commOp;
  spmd::ParallelOp processParallel; // enclosing process-level parallel
  spmd::ParallelOp threadParallel;  // innermost thread-level parallel
  ThreadLevel level;                // the constraint that was violated
};

//===----------------------------------------------------------------------===//
// Conservative ExecutionContext fallback
//
// Used when the solver has no lattice state for an op. Injects a maximally
// unsafe thread-level entry so the violation predicate fires conservatively.
//===----------------------------------------------------------------------===//

static ExecutionContext makeConservativeFallback() {
  ExecutionContext ctx = ExecutionContext::getEntry();
  LevelInfo threadLevel;
  threadLevel.modelGroup = -1;
  threadLevel.isProcessLevel = false;
  threadLevel.execKind = ExecKind::Dynamic;
  ctx.levels.push_back(threadLevel);
  return ctx;
}

/// Return the ECA execution context for \p op, or a conservative fallback when
/// the solver carries no initialised state for it (should not happen in
/// well-formed IR, but defended against).
static ExecutionContext getContextOrFallback(Operation *op,
                                             DataFlowSolver &solver) {
  auto *lat = solver.lookupState<ExecutionContextLattice>(
      solver.getProgramPointBefore(op));
  if (lat && lat->isInitialized())
    return lat->getValue();

  LLVM_DEBUG(
      DBGS() << "  *** no lattice state — using conservative fallback\n");
  return makeConservativeFallback();
}

//===----------------------------------------------------------------------===//
// Core violation predicate
//===----------------------------------------------------------------------===//

/// Returns true when a communication op violates `level` given the stacked
/// execution context `ctx`. Purely lattice-based: the ECA models
/// spmd.single/critical as ExecKind::One and scf.if rank==X guards as
/// ExecKind::Static, so this function needs no knowledge of op types.
static bool isViolation(ThreadLevel level, const ExecutionContext &ctx) {
  switch (level) {

  case ThreadLevel::Single:
    // Any thread-level parallel violates the contract. Checked structurally
    // before this predicate is ever called, so this arm is defensive.
    return true;

  case ThreadLevel::Multiple:
    return false;

  case ThreadLevel::Funneled:
    // Contract: only the master thread (OMP thread 0) calls MPI.
    //
    // Safe iff the ECA shows ExecKind::Static with executors=={0} at the
    // innermost thread level (set by scf.if rank==0 guards).
    //
    // NOT safe: ExecKind::One — spmd.single/spmd.critical pick an arbitrary
    //   thread, not necessarily the master.
    // NOT safe: ExecKind::All/AllBut/Dynamic — all or most threads run.
    return !ctx.isMasterOnly();

  case ThreadLevel::Serialized:
    // Contract: MPI calls must not be concurrent (any thread may call MPI, but
    // only one at a time).
    //
    // Safe iff the ECA shows a single executor at the innermost thread level:
    //   ExecKind::One   — spmd.single / spmd.critical body
    //   ExecKind::Static with exactly one executor — scf.if rank==X guard
    return ctx.isThreadLevelConcurrent();
  }
  llvm_unreachable("unhandled ThreadLevel");
}

//===----------------------------------------------------------------------===//
// shouldCheckForThreadSafety
//===----------------------------------------------------------------------===//

static bool shouldCheckForThreadSafety(Operation *op, int modelGroup) {
  bool hasRelevantTrait = false;

  if (op->hasTrait<spmd::CommManagement>() ||
      op->hasTrait<spmd::ReqManagement>())
    hasRelevantTrait = !mlir::isPure(op);
  else
    hasRelevantTrait = op->hasTrait<spmd::Communication>() ||
                       op->hasTrait<spmd::ProcessManagement>() ||
                       op->hasTrait<spmd::ReqManagement>() ||
                       op->hasTrait<spmd::DataManagement>() ||
                       op->hasTrait<spmd::NonBlockingSync>() ||
                       op->hasTrait<spmd::P2P>() || op->hasTrait<spmd::RMA>();

  if (!hasRelevantTrait)
    return false;
  if (modelGroup == -1)
    return true; // no filter requested

  auto mg = op->getAttrOfType<IntegerAttr>("usedModel");
  if (!mg)
    return true; // no model attr → assume match
  return (int)mg.getInt() == modelGroup;
}

//===----------------------------------------------------------------------===//
// getFollowableCallee
//
// Resolve a call-like op to a callee body we can descend into, or null when it
// cannot be followed (indirect call, external symbol, empty body). The caller
// owns the `visited` decision, since that differs between use sites.
//===----------------------------------------------------------------------===//

static FunctionOpInterface getFollowableCallee(Operation *op) {
  auto call = dyn_cast<CallOpInterface>(op);
  if (!call)
    return {};

  auto sym = call.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return {};

  auto func = dyn_cast_or_null<FunctionOpInterface>(
      SymbolTable::lookupNearestSymbolFrom(op, sym));
  if (!func || func.isExternal() || func.getFunctionBody().empty())
    return {};

  return func;
}

//===----------------------------------------------------------------------===//
// containsCandidateRecursive  (used by the GlobalConcurrency / SERIALIZED path)
//
// Returns true if `root` or any function it transitively calls contains a
// shouldCheckForThreadSafety op belonging to `processModelGroup`.
//
// Stops at spmd.critical: a global mutex serialises all threads, so on the
// SERIALIZED path (the only caller) its body is safe.
//===----------------------------------------------------------------------===//

static bool containsCandidateRecursive(Operation *root,
                                       llvm::DenseSet<Operation *> &visited,
                                       int processModelGroup) {

  bool found = false;

  // Pre-order is required: the skip on spmd.critical must fire BEFORE its
  // children are visited. A post-order walk visits children first, making the
  // skip useless for parent-guarding.
  root->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (found)
      return WalkResult::interrupt();

    // Critical is a global mutex — safe under SERIALIZED.
    if (isa<spmd::CriticalOp>(op))
      return WalkResult::skip();

    if (shouldCheckForThreadSafety(op, processModelGroup)) {
      found = true;
      return WalkResult::interrupt();
    }

    // Follow call edges into callees.
    if (auto func = getFollowableCallee(op)) {
      if (!visited.insert(func.getOperation()).second)
        return WalkResult::advance(); // recursion guard

      if (containsCandidateRecursive(func.getOperation(), visited,
                                     processModelGroup))
        found = true;
    }

    return found ? WalkResult::interrupt() : WalkResult::advance();
  });

  return found;
}

//===----------------------------------------------------------------------===//
// findParallelsRecursive
//
// Walk `root` following region nesting and call edges, collecting every
// reachable spmd.parallel whose level matches `wantProcessLevel`.
//
//   wantProcessLevel == true  — collect process-level parallels; their bodies
//     are separate units of analysis, so they are not descended.
//   wantProcessLevel == false — collect thread-level parallels; their bodies
//     ARE descended so nested thread-level parallels are also found.
//
// A nested parallel of the opposite kind delimits a foreign scope: a thread
// search stops at a nested process-level parallel (its own contract); a process
// search descends through a thread-level parallel (defensive — a process-level
// parallel is not expected below a thread-level one).
//===----------------------------------------------------------------------===//

static void findParallelsRecursive(Operation *root, bool wantProcessLevel,
                                   DenseSet<Operation *> &visited,
                                   SmallVectorImpl<spmd::ParallelOp> &result) {

  root->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    // The starting op is transparent: we search *within* it, never collect or
    // skip it. Operation::walk visits the root itself, so without this guard a
    // thread search rooted at a process-level parallel would hit the
    // opposite-kind branch and skip the entire body.
    if (op == root)
      return WalkResult::advance();

    if (auto parallelOp = dyn_cast<spmd::ParallelOp>(op)) {
      if (parallelOp.getIsProcessLevel() == wantProcessLevel) {
        result.push_back(parallelOp);
        return wantProcessLevel ? WalkResult::skip()     // body = own unit
                                : WalkResult::advance(); // descend for nested
      }
      // Opposite kind.
      return wantProcessLevel ? WalkResult::advance() // defensive descend
                              : WalkResult::skip();   // foreign process scope
    }

    // Follow call edges (shared by both search modes).
    if (auto func = getFollowableCallee(op)) {
      if (!visited.insert(func.getOperation()).second)
        return WalkResult::advance(); // recursion guard

      findParallelsRecursive(func.getOperation(), wantProcessLevel, visited,
                             result);
    }

    return WalkResult::advance();
  });
}

//===----------------------------------------------------------------------===//
// reportViolationsInCallee
//
// ECA-free structural walk of `root`, recording every MPI op not sheltered by
// a serialising construct. Called only after the call-site ECA has already
// established a violating threading context, so no further context queries are
// needed.
//
// Serialisation harbours (level-dependent — must mirror isViolation):
//   spmd.critical — global mutex.   spmd.single — one thread at a time.
//   Both satisfy SERIALIZED (ExecKind::One is non-concurrent), so their bodies
//   are skipped. NEITHER is safe for FUNNELED: each picks an arbitrary thread,
//   not the master. Under FUNNELED we therefore descend and report the calls
//   inside. We cannot recognise a rank==0 guard structurally here, so FUNNELED
//   callee reporting is conservative (it may over-report a genuinely
//   master-guarded call).
//
//   spmd.parallel — a new parallelism scope discovered independently by the
//   Level-1 finder; skip to avoid double-processing.
//
// `visited` is shared with the lattice walk for the enclosing threadOp,
// preventing both double-reporting (same callee reached from two violating
// sites) and infinite recursion through cyclic call graphs.
//===----------------------------------------------------------------------===//

static void
reportViolationsInCallee(Operation *root, spmd::ParallelOp processOp,
                         spmd::ParallelOp threadOp, ThreadLevel level,
                         int processModel, DenseSet<Operation *> &visited,
                         SmallVectorImpl<ThreadSafetyViolation> &out) {

  root->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    // Serialisation harbours — safe for SERIALIZED only.
    if (level == ThreadLevel::Serialized &&
        (isa<spmd::CriticalOp>(op) || isa<spmd::SingleOp>(op)))
      return WalkResult::skip();

    // New parallelism scope — handled separately by the Level-1 finder.
    if (isa<spmd::ParallelOp>(op))
      return WalkResult::skip();

    // MPI op: the call site that brought us here is already violating, so every
    // comm op reached structurally is a violation.
    if (shouldCheckForThreadSafety(op, processModel)) {
      out.push_back({op, processOp, threadOp, level});
      return WalkResult::advance();
    }

    // Follow call edges.
    if (auto func = getFollowableCallee(op)) {
      if (!visited.insert(func.getOperation()).second)
        return WalkResult::advance();

      reportViolationsInCallee(func.getOperation(), processOp, threadOp, level,
                               processModel, visited, out);
    }

    return WalkResult::advance();
  });
}

//===----------------------------------------------------------------------===//
// collectViolations
//===----------------------------------------------------------------------===//

static void collectViolations(spmd::ParallelOp processOp, ThreadLevel level,
                              DataFlowSolver &mvSolver,
                              SmallVectorImpl<ThreadSafetyViolation> &out) {

  const int processModel = (int)processOp.getUsedModel();

  // ── Level 1: all thread-level parallels reachable from the process op ──────
  // (region nesting + call edges).
  SmallVector<spmd::ParallelOp> threadOps;
  {
    DenseSet<Operation *> l1Visited;
    l1Visited.insert(processOp.getOperation());
    findParallelsRecursive(processOp.getOperation(), /*wantProcessLevel=*/false,
                           l1Visited, threadOps);
  }

  for (spmd::ParallelOp threadOp : threadOps) {

    // ── MPI_THREAD_SINGLE ────────────────────────────────────────────────────
    // The contract forbids all thread-level parallelism; the region itself is
    // the violation regardless of its contents.
    if (level == ThreadLevel::Single) {
      out.push_back({threadOp.getOperation(), processOp, threadOp, level});
      continue;
    }

    Region *boundary = &threadOp.getRegion();

    // Shared across the lattice walk and every reportViolationsInCallee call
    // for this threadOp. A callee is inserted only when a violating call site
    // actually dispatches into it, so a callee first reached from a safe site
    // stays eligible for analysis from a later violating one. Also prevents
    // double-reporting and infinite recursion on cyclic call graphs.
    DenseSet<Operation *> l2Visited;

    boundary->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      // ── SERIALIZED: GlobalConcurrency construct ────────────────────────────
      // Constructs the ECA models as ExecKind::One locally but which are
      // globally concurrent (spmd.task / wsloop / sections / taskloop). Report
      // at construct granularity and skip the subtree. This single skip
      // subsumes the former per-op parent climb AND the separate construct-
      // level loop, and reports only the outermost of nested GlobalConcurrency
      // constructs (no double-report). FUNNELED is intentionally not handled
      // here — its inner ops are checked individually by the lattice walk.
      if (level == ThreadLevel::Serialized &&
          op->hasTrait<spmd::GlobalConcurrency>()) {
        llvm::DenseSet<Operation *> gcVisited;
        gcVisited.insert(op);
        if (containsCandidateRecursive(op, gcVisited, processModel))
          out.push_back({op, processOp, threadOp, level});
        return WalkResult::skip();
      }

      // ── Call site (interprocedural extension) ──────────────────────────────
      // The ECA lattice at the call op is valid: the op is structurally inside
      // the thread region where the ECA ran correctly. If the call site is in a
      // violating context, descend into the callee with an ECA-free structural
      // walk (the ECA is intraprocedural and uninformed about the threading
      // context inside callees).
      if (auto func = getFollowableCallee(op)) {
        ExecutionContext ctx = getContextOrFallback(op, mvSolver);
        if (isViolation(level, ctx)) {
          // Insert only when dispatching, to avoid re-analysing a callee
          // already covered from a prior violating call site.
          if (l2Visited.insert(func.getOperation()).second)
            reportViolationsInCallee(func.getOperation(), processOp, threadOp,
                                     level, processModel, l2Visited, out);
        }
        return WalkResult::advance();
      }

      // ── Direct MPI op (original lattice logic, unchanged) ──────────────────
      // Model-group filtering lives inside shouldCheckForThreadSafety; model
      // ids do not overlap between process- and thread-level parallels, so no
      // separate ownership guard is needed.
      if (!shouldCheckForThreadSafety(op, processModel))
        return WalkResult::advance();

      ExecutionContext ctx = getContextOrFallback(op, mvSolver);
      if (isViolation(level, ctx))
        out.push_back({op, processOp, threadOp, level});

      return WalkResult::advance();
    });
  }
}

//===----------------------------------------------------------------------===//
// Reporting
//===----------------------------------------------------------------------===//

static void emitViolation(const ThreadSafetyViolation &v) {
  std::string levelStr =
      llvm::StringRef(mlir::spmd::stringifyThreadLevel(v.level)).upper();

  std::string remedy;
  switch (v.level) {
  case ThreadLevel::Single:
    // The contract forbids thread-level parallelism entirely during
    // communication — the region itself is the violation, not the individual
    // calls inside it.
    remedy = "the declared thread-support level requires the process to remain "
             "fully single-threaded; remove all thread-parallel regions or "
             "upgrade the thread-support level";
    break;
  case ThreadLevel::Funneled:
    remedy = "only the master thread may issue communication calls; "
             "guard the call with a master-thread-only conditional";
    break;
  case ThreadLevel::Serialized:
    remedy = "communication calls must not be issued concurrently; "
             "place the call inside a mutual-exclusion or "
             "single-execution construct";
    break;
  default:
    break;
  }

  llvm::StringRef opName = (v.level == ThreadLevel::Single)
                               ? llvm::StringRef("thread-parallel region")
                               : v.commOp->getName().stripDialect();

  v.commOp->emitWarning() << "[CheckThreadSafety] " << opName
                          << " may violate thread-support level " << levelStr
                          << " — " << remedy;
}

static std::string getDefaultJsonPath() {
  return (std::filesystem::current_path() / "thread_safety_report.json")
      .string();
}

static void dumpViolationsToJson(ArrayRef<ThreadSafetyViolation> violations) {
  llvm::json::Array messages;

  for (const auto &v : violations) {
    Location loc = findLocToShowWrapper(v.commOp->getLoc());

    std::ostringstream text;
    text << v.commOp->getName().stripDialect().str() << " at "
         << locToString(loc) << " may violate MPI_THREAD_"
         << llvm::StringRef(mlir::spmd::stringifyThreadLevel(v.level)).upper();

    messages.push_back(llvm::json::Object{
        {"type", "Thread Safety Violation"},
        {"text", text.str()},
        {"references", llvm::json::Array{getLocationJson(loc)}},
    });
  }

  std::string actualPath = getDefaultJsonPath();

  std::error_code ec;
  llvm::raw_fd_ostream outFile(actualPath, ec);
  if (ec) {
    llvm::errs() << "[CheckThreadSafety] failed to open '" << actualPath
                 << "': " << ec.message() << "\n";
    return;
  }

  outFile << llvm::formatv("{0:2}", llvm::json::Value(llvm::json::Object{
                                        {"messages", std::move(messages)}}))
          << "\n";
}

static void printViolations(ArrayRef<ThreadSafetyViolation> violations) {
  constexpr llvm::StringLiteral kSep =
      "--------------------------------------------------------------------"
      "--------------------";

  fprintf(stderr, "\n\n%s\n", kSep.data());
  fprintf(stderr, "Static Verification of Thread-Support Level Compliance:\n");
  fprintf(stderr, "%s\n\n", kSep.data());

  if (violations.empty()) {
    fprintf(stderr, "No thread-safety violations found\n");
    fprintf(stderr, "\n%s\n\n", kSep.data());
    return;
  }

  for (const auto &v : violations) {
    emitViolation(v);
    fprintf(stderr, "%s\n\n", kSep.data());
  }
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CheckThreadSafety : impl::CheckThreadSafetyBase<CheckThreadSafety> {
  using CheckThreadSafetyBase::CheckThreadSafetyBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // ------------------------------------------------------------------ //
    // Phase 1: Assert each modelGroup appears in at most one process-level
    //          spmd.parallel. A module walk is appropriate: this is a
    //          structural invariant over the whole IR, independent of
    //          reachability from main. Thread-level parallels are
    //          unconstrained.
    // ------------------------------------------------------------------ //
    DenseMap<int, spmd::ParallelOp> seenModelGroups;
    bool precondFailed = false;

    module.walk([&](spmd::ParallelOp op) {
      if (!op.getIsProcessLevel())
        return;
      int mg = (int)op.getUsedModel();
      auto [it, inserted] = seenModelGroups.try_emplace(mg, op);
      if (!inserted) {
        op.emitError(
            "[CheckThreadSafety] multiple process-level spmd.parallel ops "
            "for model group " +
            std::to_string(mg) + " — IR invariant violated");
        precondFailed = true;
      }
    });

    if (precondFailed) {
      signalPassFailure();
      return;
    }

    // ------------------------------------------------------------------ //
    // Phase 2: Run the solver chain.
    //          Mirrors CheckDataRace and CheckCollectives exactly.
    // ------------------------------------------------------------------ //

    // Alias analysis must run to fixpoint first.
    DataFlowSolver aliasSolver;
    aliasSolver.load<enzyme::AliasAnalysis>(&getContext());
    aliasSolver.load<enzyme::PointsToPointerAnalysis>();
    aliasSolver.load<dataflow::DeadCodeAnalysis>();
    aliasSolver.load<dataflow::SparseConstantPropagation>();

    if (failed(aliasSolver.initializeAndRun(module))) {
      LLVM_DEBUG(DBGS() << "alias analysis failed\n");
      signalPassFailure();
      return;
    }

    // Uniformity and ExecutionContextAnalysis with frozen alias results.
    DataFlowSolver mvSolver;
    mvSolver.load<dataflow::DeadCodeAnalysis>();
    mvSolver.load<dataflow::SparseConstantPropagation>();
    mvSolver.load<ParallelContextAnalysis>();
    mvSolver.load<UniformityValueAnalysis>(aliasSolver);
    mvSolver.load<UniformityMemoryAnalysis>(aliasSolver);
    mvSolver.load<ExecutionContextAnalysis>();

    if (failed(mvSolver.initializeAndRun(module))) {
      LLVM_DEBUG(DBGS() << "ExecutionContextAnalysis failed\n");
      signalPassFailure();
      return;
    }

    // ------------------------------------------------------------------ //
    // Phase 3: Discover process-level parallels, then collect violations.
    //
    // Primary mode is main-based: start from `main` and follow call edges to
    // every reachable process-level parallel. If there is no `main` (library
    // translation unit, alternate entry), fall back to a whole-module walk so
    // the pass still analyses something.
    // ------------------------------------------------------------------ //
    FunctionOpInterface mainFunc;
    module.walk([&](FunctionOpInterface func) -> WalkResult {
      if (func.getName() == "main") {
        mainFunc = func;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    SmallVector<spmd::ParallelOp> processParallels;
    if (mainFunc) {
      DenseSet<Operation *> visited;
      visited.insert(mainFunc.getOperation());
      findParallelsRecursive(mainFunc.getOperation(), /*wantProcessLevel=*/true,
                             visited, processParallels);
    } else {
      module.emitWarning(
          "[CheckThreadSafety] no 'main' function found; falling back to "
          "whole-module analysis");
      module.walk([&](spmd::ParallelOp op) {
        if (op.getIsProcessLevel())
          processParallels.push_back(op);
      });
    }

    SmallVector<ThreadSafetyViolation> violations;

    for (spmd::ParallelOp processOp : processParallels) {
      // Absent threadLevel → assume Single (MPI standard §12.4.3:
      // MPI_Init is equivalent to MPI_Init_Thread with SINGLE).
      auto attrOpt = processOp.getThreadLevel();
      ThreadLevel level = attrOpt ? *attrOpt : ThreadLevel::Single;

      // Multiple imposes no threading restrictions — skip entirely.
      if (level == ThreadLevel::Multiple)
        continue;

      collectViolations(processOp, level, mvSolver, violations);
    }

    // ------------------------------------------------------------------ //
    // Phase 4: Report.
    // ------------------------------------------------------------------ //
    if (writeJsonReport)
      dumpViolationsToJson(violations);
    else
      printViolations(violations);

    if (!violations.empty())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spmd