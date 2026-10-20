// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/ParallelContextAnalysis.h"
#include "spmd/SPMDOps.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "parallel-context-analysis"

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// ParallelContextState
//===----------------------------------------------------------------------===//

ChangeResult ParallelContextState::join(const AbstractDenseLattice &other_) {
  const auto &other = static_cast<const ParallelContextState &>(other_);
  if (!other.initialized)
    return ChangeResult::NoChange;
  if (!initialized)
    return initializeTo(other.parallelStack);

  if (llvm::equal(parallelStack, other.parallelStack))
    return ChangeResult::NoChange;

  // The dense framework seeds a callable's entry block with the empty entry
  // stack (setToEntryState) whenever the callsite predecessors are not yet
  // known, and that seeding is never retracted. When the real caller context
  // arrives afterwards over the call edge it has to win: a function invoked
  // from inside a parallel region executes with that region's stack. Keeping
  // the empty stack left every value in the callee — notably the result of
  // spmd.getRankInComm — unmarked at the enclosing parallel level, so it
  // looked uniform and no rank guard inside a called function (a CUDA kernel,
  // say) could refine the executing ranks.
  if (parallelStack.empty())
    return initializeTo(other.parallelStack);

  // Otherwise the stacks must be identical in well-formed SPMD IR: a parallel
  // region cannot be entered on one CFG path and skipped on another, so all
  // paths reaching a merge point carry the same stack.
  assert(other.parallelStack.empty() &&
         "ParallelContextState mismatch at a CFG join — IR has mismatched "
         "parallel nesting across control-flow paths");
  return ChangeResult::NoChange;
}

ChangeResult ParallelContextState::initializeTo(ArrayRef<Operation *> stack) {
  const bool same = initialized && llvm::equal(parallelStack, stack);
  if (same)
    return ChangeResult::NoChange;
  initialized = true;
  parallelStack.assign(stack.begin(), stack.end());
  return ChangeResult::Change;
}

void ParallelContextState::print(raw_ostream &os) const {
  if (!initialized) {
    os << "<uninitialized>";
    return;
  }
  os << "ParallelContext[";
  for (Operation *op : parallelStack)
    os << op << " ";
  os << "]";
}

//===----------------------------------------------------------------------===//
// ParallelContextAnalysis
//===----------------------------------------------------------------------===//

void ParallelContextAnalysis::setToEntryState(ParallelContextState *lattice) {
  // Function / region entry point: no enclosing parallel ops yet.
  if (auto *pp = dyn_cast<ProgramPoint *>(lattice->getAnchor())) {
    if (Block *block = pp->getBlock()) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(block->getParentOp())) {
        propagateIfChanged(lattice, lattice->initializeTo({}));
        return;
      }
    }
  }
  // All other program points remain uninitialized; propagation fills them.
}

LogicalResult
ParallelContextAnalysis::visitOperation(Operation *op,
                                        const ParallelContextState &before,
                                        ParallelContextState *after) {
  if (!before.isInitialized())
    return success(); // not yet reachable — nothing to propagate

  if (isa<spmd::ParallelOp>(op)) {
    // Propagate stack ++ [op] into every region of the parallel op.
    for (Region &region : op->getRegions()) {
      if (region.empty())
        continue;
      SmallVector<Operation *> bodyStack(before.getStack());
      bodyStack.push_back(op);
      auto *bodyLat = getOrCreate<ParallelContextState>(
          getProgramPointBefore(&region.front()));
      propagateIfChanged(bodyLat, bodyLat->initializeTo(bodyStack));
    }
    // Past the op itself the stack is unchanged — executors reconverge.
  } else if (op->getNumRegions() > 0 && !isa<RegionBranchOpInterface>(op) &&
             !isa<FunctionOpInterface>(op) && !isa<ModuleOp>(op)) {
    // Region-holding ops the dense framework never seeds: not a
    // RegionBranchOpInterface (whose region edges it drives itself) and not a
    // callable. `gpu.launch` is the case that matters — it wraps the
    // `llvm.call` to a CUDA kernel's device stub. Leaving its body
    // uninitialized left UniformityValueAnalysis with no parallel stack for
    // any op in the kernel, so `spmd.getRankInComm` there was never marked
    // non-uniform and rank guards inside kernels could not be refined. The
    // body runs under the same enclosing parallels as the launch site.
    for (Region &region : op->getRegions()) {
      if (region.empty())
        continue;
      auto *bodyLat = getOrCreate<ParallelContextState>(
          getProgramPointBefore(&region.front()));
      propagateIfChanged(bodyLat, bodyLat->initializeTo(before.getStack()));
    }
  }

  // Identity for all ops (including ParallelOp) at the post-op point.
  propagateIfChanged(after, after->join(before));
  return success();
}

void ParallelContextAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const ParallelContextState &before, ParallelContextState *after) {
  // Propagate the caller's stack into the callee and back on all call
  // control-flow edges.  This ensures functions invoked from within a
  // parallel body correctly inherit that body's parallel context —
  // something a structural parent-op walk cannot do.
  propagateIfChanged(after, after->join(before));
}

} // namespace mlir::spmd