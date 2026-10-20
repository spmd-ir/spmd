// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMD.cpp - Lower every programming model of a module ----------===//
//
// -lower-spmd lowers a whole module under a table that says what each model is
// lowered to: "mpi=mpi" lowers MPI to MPI calls, "mpi=shmem" translates an
// MPI-raised program to OpenSHMEM, "omp=keep" leaves a model alone.  Its first
// phase is also -spmd-retarget-model on its own.
//
// The phases run in this order so that a diagnostic never arrives after half
// the module has been rewritten: retarget, check every target model, undo the
// structure the raiser wrapped around the program, and lower the models one
// after the other.  See the pass descriptions in include/spmd/SPMDPasses.td.
//
//===----------------------------------------------------------------------===//

#include "LowerSPMDCommon.h"
#include "LowerSPMDToCUDA.h"
#include "LowerSPMDToMPI.h"
#include "LowerSPMDToNCCL.h"
#include "LowerSPMDToNVSHMEM.h"
#include "LowerSPMDToOMP.h"
#include "LowerSPMDToSHMEM.h"

#include "spmd/SPMDOps.h"
#include "spmd/SPMDPasses.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstddef>
#include <optional>
#include <string>

namespace mlir::spmd {

#define GEN_PASS_DEF_LOWERSPMD
#define GEN_PASS_DEF_SPMDRETARGETMODEL
#include "spmd/SPMDPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// The models a table can name
//===----------------------------------------------------------------------===//

/// One programming model: how the table spells it, the usedModel it stands
/// for, and the entry points that check and lower it.
struct ModelEntry {
  StringRef name;
  ProgrammingModel model;
  /// Reports the operations the model cannot express, before any rewrite.
  LogicalResult (*check)(ModuleOp);
  LogicalResult (*lower)(ModuleOp, StructureLowering);
  /// Why the model's operations cannot be translated to or from another
  /// model's, or nothing when they can.
  StringLiteral noTranslation = "";
};

constexpr size_t kNumModels = 6;

/// Every model -convert-apiCalls-to-spmd tags for.
constexpr ModelEntry kModels[kNumModels] = {
    {"mpi", ProgrammingModel::MPI, checkSPMDToMPI, lowerSPMDToMPI},
    {"shmem", ProgrammingModel::SHMEM, checkSPMDToSHMEM, lowerSPMDToSHMEM},
    // NVSHMEM's API is OpenSHMEM's, but not its memory: the symmetric heap is
    // on the GPU, and every buffer of an NVSHMEM program is device memory the
    // host copies to and from.  A program written for host memory -- MPI's or
    // OpenSHMEM's -- would touch device memory from the host, and one written
    // for device memory would hand MPI or OpenSHMEM buffers the host cannot
    // read.  NCCL's buffers are on the GPU too, but nothing translates between
    // the two.
    {"nvshmem", ProgrammingModel::NVSHMEM, checkSPMDToNVSHMEM,
     lowerSPMDToNVSHMEM,
     "its buffers are GPU memory, where MPI's and OpenSHMEM's are host "
     "memory; NCCL's are GPU memory too, but no lowering translates between "
     "NVSHMEM and NCCL"},
    // NCCL's buffers are GPU memory as well, and every communication call
    // takes a CUDA stream.  The checks of each operation catch a stream that
    // is missing or left over, but not an operation that has none either way:
    // an spmd.malloc translated from MPI would hand the host the device memory
    // of ncclMemAlloc.
    {"nccl", ProgrammingModel::NCCL, checkSPMDToNCCL, lowerSPMDToNCCL,
     "its buffers are GPU memory, where MPI's and OpenSHMEM's are host "
     "memory, and every communication call takes a CUDA stream, which no "
     "MPI, OpenSHMEM or NVSHMEM program passes"},
    {"cuda", ProgrammingModel::CUDA, checkSPMDToCUDA, lowerSPMDToCUDA},
    {"omp", ProgrammingModel::OMP, checkSPMDToOMP, lowerSPMDToOMP},
};

constexpr bool modelsAreInUsedModelOrder() {
  for (size_t i = 0; i < kNumModels; ++i)
    if (static_cast<size_t>(kModels[i].model) != i)
      return false;
  return true;
}
static_assert(modelsAreInUsedModelOrder(),
              "kModels is indexed by the usedModel it stands for");

constexpr bool everyModelHasALowering() {
  for (const ModelEntry &entry : kModels)
    if (!entry.check || !entry.lower)
      return false;
  return true;
}
// The default table lowers every model to itself, and a models= table may
// name any model as a target.  A model added without a lowering would need
// both to learn about it.
static_assert(everyModelHasALowering(),
              "every model -convert-apiCalls-to-spmd tags for has a lowering");

const ModelEntry *findModel(StringRef name) {
  for (const ModelEntry &entry : kModels)
    if (entry.name == name)
      return &entry;
  return nullptr;
}

StringRef nameOfModel(int usedModel) {
  if (usedModel < 0 || static_cast<size_t>(usedModel) >= kNumModels)
    return "?";
  return kModels[usedModel].name;
}

std::string knownModels() {
  std::string list;
  for (const ModelEntry &entry : kModels) {
    if (!list.empty())
      list += ", ";
    list += entry.name.str();
  }
  return list;
}

/// What each model is lowered to, indexed by usedModel.  A null entry is
/// "keep": the model's operations stay as they are.
using ModelTable = std::array<const ModelEntry *, kNumModels>;

/// Every model, to itself.
ModelTable defaultTable() {
  ModelTable table{};
  for (const ModelEntry &entry : kModels)
    table[static_cast<size_t>(entry.model)] = &entry;
  return table;
}

/// What the table says is wrong with the pass options rather than with the
/// module, so it has no place in the program to point at.
InFlightDiagnostic optionError(MLIRContext *ctx) {
  return emitError(UnknownLoc::get(ctx));
}

/// Reports a translation from \p from to \p to that one of the two models
/// cannot take part in; \p what says who asked for it.
LogicalResult checkTranslatable(const ModelEntry &from, const ModelEntry &to,
                                MLIRContext *ctx, StringRef what) {
  if (&from == &to)
    return success();
  for (const ModelEntry *model : {&from, &to})
    if (!model->noTranslation.empty())
      return optionError(ctx)
             << what << " '" << from.name << "' to '" << to.name << "', but '"
             << model->name
             << "' cannot be translated to or from: " << model->noTranslation;
  return success();
}

/// Parses models="<source>=<target>,...", each pair overriding the default for
/// its source.
FailureOr<ModelTable> parseModelTable(StringRef spec, MLIRContext *ctx) {
  ModelTable table = defaultTable();
  std::array<bool, kNumModels> given{};

  SmallVector<StringRef> entries;
  spec.split(entries, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef entry : entries) {
    entry = entry.trim();
    if (entry.empty())
      continue;

    auto [sourceSpec, targetSpec] = entry.split('=');
    StringRef source = sourceSpec.trim();
    StringRef target = targetSpec.trim();
    if (target.empty())
      return optionError(ctx)
             << "'" << entry << "' in models= is not a <source>=<target> pair";

    // A target for one class of operations needs a communicator of the target
    // model for every communicator of the source, and NCCL a stream the source
    // does not carry; the NCCL lowering bootstraps only its own world
    // communicator.  Reading it as the whole model would silently lower more
    // than was asked for.
    if (source.contains('.'))
      return optionError(ctx)
             << "'" << source
             << "' in models= names a class of operations; only whole models "
                "can be retargeted so far, as in models=\"mpi=shmem\"";

    const ModelEntry *from = findModel(source);
    if (!from)
      return optionError(ctx)
             << "'" << source << "' in models= is not a programming model; "
             << "known models are " << knownModels();

    const ModelEntry *to = nullptr;
    if (target != "keep") {
      to = findModel(target);
      if (!to)
        return optionError(ctx)
               << "'" << target << "' in models= is not a programming model; "
               << "known models are " << knownModels() << ", or 'keep'";
      if (failed(checkTranslatable(*from, *to, ctx, "models= translates")))
        return failure();
    }

    size_t index = static_cast<size_t>(from->model);
    if (given[index])
      return optionError(ctx) << "models= names '" << source << "' twice";
    given[index] = true;
    table[index] = to;
  }

  // A model that operations are retargeted onto must be lowered to itself.
  // Once they carry its usedModel nothing tells them from the operations that
  // were already tagged for it, so neither "keep" nor a second translation
  // could be honoured for either set.
  for (const ModelEntry &entry : kModels) {
    const ModelEntry *to = table[static_cast<size_t>(entry.model)];
    if (!to || to == &entry)
      continue;
    const ModelEntry *onward = table[static_cast<size_t>(to->model)];
    if (onward == to)
      continue;
    std::string fate = onward ? ("retargeted to '" + onward->name.str() + "'")
                              : std::string("kept");
    return optionError(ctx)
           << "models= retargets '" << entry.name << "' to '" << to->name
           << "', but '" << to->name << "' is itself " << fate
           << "; the operations would be indistinguishable once they carry "
              "its usedModel";
  }
  return table;
}

//===----------------------------------------------------------------------===//
// Retarget
//===----------------------------------------------------------------------===//

/// Whether \p type stands for a runtime object of one model, which that
/// model's lowering turns into its own handle.  An !spmd.error is a return
/// code, and the cast that reads it follows the operation it comes from.
bool isModelHandleType(Type type) {
  if (type.getDialect().getNamespace() != SPMDDialect::getDialectNamespace())
    return false;
  return !isa<ErrorType>(type);
}

/// The operations \p value can come from, followed back through the block
/// arguments a branch passes it as, so that a handle reaching its user through
/// a cf.br is seen as well as one used where it is defined.  A value entering a
/// region as an entry argument is not followed: it comes from the operation
/// around the region rather than from a branch.
void collectHandleSources(Value value, SmallVectorImpl<Operation *> &sources,
                          DenseSet<Value> &seen) {
  if (!seen.insert(value).second)
    return;
  if (Operation *def = value.getDefiningOp()) {
    sources.push_back(def);
    return;
  }
  auto arg = cast<BlockArgument>(value);
  Block *block = arg.getOwner();
  if (block->isEntryBlock())
    return;
  for (Block *pred : block->getPredecessors()) {
    auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!branch)
      continue;
    for (unsigned i = 0, e = pred->getNumSuccessors(); i < e; ++i) {
      if (pred->getSuccessor(i) != block)
        continue;
      SuccessorOperands operands = branch.getSuccessorOperands(i);
      if (Value forwarded = operands[arg.getArgNumber()])
        collectHandleSources(forwarded, sources, seen);
    }
  }
}

void setUsedModel(Operation *op, int model) {
  op->setAttr("usedModel",
              IntegerAttr::get(IntegerType::get(op->getContext(), 32), model));
}

/// Re-tags every operation the table moves and reports a handle left crossing
/// two models.  usedModel is a required attribute of every spmd operation, the
/// handle producers included, so spmd.commWorld and the spmd.const* operations
/// move with the calls that use them rather than having to be closed over.
/// The module is not rewritten: only usedModel changes.
LogicalResult retargetModels(ModuleOp module, const ModelTable &table) {
  DenseSet<Operation *> moved;
  module.walk([&](Operation *op) {
    if (!isSpmdOp(op))
      return;
    std::optional<int> model = getUsedModel(op);
    if (!model || *model < 0 || static_cast<size_t>(*model) >= kNumModels)
      return;
    const ModelEntry *target = table[static_cast<size_t>(*model)];
    if (!target || target->model == static_cast<ProgrammingModel>(*model))
      return;
    setUsedModel(op, static_cast<int>(target->model));
    moved.insert(op);
  });

  if (moved.empty())
    return success();

  // A handle the retarget moved one end of.  A module that mixed models
  // before is none of this pass's business: the single-model passes lower
  // such a module one model at a time.
  bool anyError = false;
  module.walk([&](Operation *user) {
    if (!isSpmdOp(user))
      return;
    std::optional<int> userModel = getUsedModel(user);
    if (!userModel)
      return;
    for (OpOperand &operand : user->getOpOperands()) {
      if (!isModelHandleType(operand.get().getType()))
        continue;
      SmallVector<Operation *> sources;
      DenseSet<Value> seen;
      collectHandleSources(operand.get(), sources, seen);
      for (Operation *def : sources) {
        if (!isSpmdOp(def))
          continue;
        std::optional<int> defModel = getUsedModel(def);
        if (!defModel || *defModel == *userModel)
          continue;
        if (!moved.contains(user) && !moved.contains(def))
          continue;
        InFlightDiagnostic diag =
            user->emitError()
            << "operand #" << operand.getOperandNumber() << " is a '"
            << nameOfModel(*defModel)
            << "' handle, but the operation is tagged for '"
            << nameOfModel(*userModel) << "': the value crosses programming "
            << "models";
        diag.attachNote(def->getLoc()) << "handle produced here";
        anyError = true;
      }
    }
  });

  return failure(anyError);
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

struct LowerSPMDPass : public impl::LowerSPMDBase<LowerSPMDPass> {
  using LowerSPMDBase::LowerSPMDBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(checkUsedModels(module, "lower-spmd")))
      return signalPassFailure();

    FailureOr<ModelTable> table = parseModelTable(models, &getContext());
    if (failed(table))
      return signalPassFailure();

    // 1. Retarget.
    if (failed(retargetModels(module, *table)))
      return signalPassFailure();

    // The models the table lowers to, each once, in usedModel order.
    SmallVector<const ModelEntry *> targets;
    for (const ModelEntry &entry : kModels)
      if (llvm::is_contained(*table, &entry))
        targets.push_back(&entry);

    // 2. Check all of them before rewriting anything.
    bool inexpressible = false;
    for (const ModelEntry *target : targets)
      if (failed(target->check(module)))
        inexpressible = true;
    if (inexpressible)
      return signalPassFailure();

    // 3. The process team belongs to the launcher rather than to a model, so
    // the process-level parallels of every model about to be lowered go in one
    // step, before any of the calls: a parallel still standing during a
    // conversion would need its communicator materialised back into an
    // !spmd.comm, and that cast would outlive it.
    SmallVector<ProgrammingModel> lowered;
    for (const ModelEntry *target : targets)
      lowered.push_back(target->model);
    if (failed(lowerProcessParallels(module, lowered)))
      return signalPassFailure();

    // 4. Lower the models one after the other.  Their handle types differ and
    // a type carries no model, so unrealized casts bridge what one lowering
    // leaves for the next.
    for (const ModelEntry *target : targets)
      if (failed(target->lower(module, StructureLowering::Deferred)))
        return signalPassFailure();
  }
};

struct SPMDRetargetModelPass
    : public impl::SPMDRetargetModelBase<SPMDRetargetModelPass> {
  using SPMDRetargetModelBase::SPMDRetargetModelBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    if (from.empty() || to.empty()) {
      optionError(ctx) << "-spmd-retarget-model needs from= and to=, each "
                          "naming a programming model ("
                       << knownModels() << ")";
      return signalPassFailure();
    }

    const ModelEntry *source = findModel(from);
    if (!source) {
      optionError(ctx) << "'" << from
                       << "' in from= is not a programming model; known "
                          "models are "
                       << knownModels();
      return signalPassFailure();
    }
    const ModelEntry *target = findModel(to);
    if (!target) {
      optionError(ctx) << "'" << to
                       << "' in to= is not a programming model; known "
                          "models are "
                       << knownModels();
      return signalPassFailure();
    }

    if (failed(checkTranslatable(*source, *target, ctx,
                                 "-spmd-retarget-model moves")))
      return signalPassFailure();

    if (failed(checkUsedModels(module, "spmd-retarget-model")))
      return signalPassFailure();

    ModelTable table{};
    table[static_cast<size_t>(source->model)] = target;
    if (failed(retargetModels(module, table)))
      return signalPassFailure();

    // What the target cannot express is reported here, so that no lowering
    // starts on operations it will have to give up on.
    if (failed(target->check(module)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace mlir::spmd
