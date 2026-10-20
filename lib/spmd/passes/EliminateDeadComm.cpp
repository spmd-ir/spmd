// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- EliminateDeadComm.cpp - SPMD Pass ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dead communication elimination for the SPMD dialect.  Three independent
// removals share one buffer-reasoning substrate:
//
//   dead-dest         a data-moving collective whose destination buffer is
//                     never read is removed (together with its completion).
//   overwritten-dest  a data-moving collective whose destination buffer is
//                     completely overwritten by a later collective before
//                     anything reads it is removed.
//   dead-resource     an RMA window that no data-movement operation ever
//                     targets is removed together with its whole lifecycle
//                     (fences, locks, free); likewise an spmd.malloc whose
//                     address is only ever passed to spmd.free.
//
// Two failure modes are kept apart (docs/optimizations.md, "Two ways to get it
// wrong"):
//
//   silent corruption - deleting communication whose result some rank
//      genuinely needed.  This is what the buffer analysis below guards:
//      liveness is computed over the textual CFG, so a buffer that is "read on
//      some path" is read on every rank that takes that path.
//   mismatch - ranks disagreeing on how many operations they
//      execute.  Every removal here is *textual and whole-op*: all ranks lose
//      exactly the same operation, so per-rank counts stay aligned by
//      construction.  No uniformity precondition is therefore needed.  That
//      argument stops holding the moment an operation is *moved* or removed on
//      one path only, which is why this pass never does either.
//
// Deliberately out of scope: P2P (send/recv/sendrecv) and RMA data movement.
// Their observable effect is on a *peer's* memory, about which local liveness
// says nothing, and removing one half of a matched pair is a textbook mismatch.
//
//===----------------------------------------------------------------------===//

#include "spmd/Analysis/CommBuffers.h"
#include "spmd/Analysis/CommCompletions.h"
#include "spmd/SPMDOps.h"
#include "spmd/SPMDPasses.h"
#include "spmd/Utils/MemoryOffsets.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <optional>

namespace mlir::spmd {

#define GEN_PASS_DEF_ELIMINATEDEADCOMM
#include "spmd/SPMDPasses.h.inc"

namespace {

/// A collective whose bookkeeping results are all dead and which is therefore a
/// candidate for removal, together with the buffers it writes.
///
/// Whether it can *actually* go is a separate question answered later by
/// getCommCompletions, because it depends on which other collectives are going
/// away at the same time.
struct Candidate {
  Operation *op;
  SmallVector<Value> written;
};

static std::optional<Candidate> getCandidate(Operation *op) {
  if (!isDataMovingCollective(op) || hasStreamOperand(op))
    return std::nullopt;

  // Everything except the request (which the completion consumes) must be dead;
  // an observed error code or status is a reason to keep the op.
  for (Value result : op->getResults())
    if (!isa<ReqType>(result.getType()) && !result.use_empty())
      return std::nullopt;

  Candidate candidate{op, {}};
  if (failed(collectWrittenAddresses(op, candidate.written)))
    return std::nullopt;
  if (candidate.written.empty())
    return std::nullopt;
  return candidate;
}

/// The buffer denotes the base address of the value it resolves to.
static bool preservesBase(const BufferSlice &slice) {
  return slice.offset && *slice.offset == ByteOffset();
}

/// The communicator `op` communicates over, when it has exactly one.
static Value getCommOperand(Operation *op) {
  Value comm;
  for (Value operand : op->getOperands()) {
    if (!isa<CommType>(operand.getType()))
      continue;
    if (comm)
      return {};
    comm = operand;
  }
  return comm;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct EliminateDeadComm : impl::EliminateDeadCommBase<EliminateDeadComm> {
  using EliminateDeadCommBase::EliminateDeadCommBase;

  void runOnOperation() override {
    DataLayout dl(getOperation());
    for (unsigned iteration = 0; iteration < maxIterations; ++iteration) {
      bool changed = false;
      if (deadDest)
        changed |= eliminateDeadDestinations(dl);
      if (overwrittenDest)
        changed |= eliminateOverwrittenDestinations(dl);
      if (deadResource)
        changed |= eliminateDeadResources();
      if (!changed)
        break;
    }
  }

  /// Narrows `ops` to those whose completion set is unique *given that all of
  /// `ops` goes away*, and erases each one together with that set.  Returns how
  /// many communication operations were removed.
  ///
  /// Selection and erasure cannot be separated.  Whether a non-blocking
  /// initiation can go depends on the other initiations sharing its request
  /// array, so dropping one strands the rest; the set is therefore shrunk to a
  /// fixpoint before anything is touched.  Nothing here ever erases a
  /// communication operation on its own - see collectRemovals.
  unsigned eraseWithCompletions(SmallVector<Operation *> ops,
                                const DataLayout &dl) {
    // One completion can serve several initiations (a single spmd.waitAll over
    // a whole request array), so collect before erasing.
    SetVector<Operation *> removals = collectRemovals(ops, dl);
    IRRewriter rewriter(&getContext());
    eraseOps(rewriter, std::move(removals));
    return ops.size();
  }

  //===--------------------------------------------------------------------===//
  // dead-dest: nothing ever reads what the collective produced.
  //===--------------------------------------------------------------------===//

  bool eliminateDeadDestinations(const DataLayout &dl) {
    SmallVector<Operation *> dead;

    getOperation().walk([&](Operation *op) {
      std::optional<Candidate> candidate = getCandidate(op);
      if (!candidate)
        return;

      for (Value buffer : candidate->written) {
        BufferFacts facts;
        if (failed(computeBufferFacts(buffer, dl, facts)))
          return;
        // The op's own read of an in-place buffer does not keep it alive.  A
        // completion never reads the data buffer - it only consumes the
        // request - so it cannot show up here.
        for (Operation *reader : facts.readers)
          if (reader != op)
            return;
      }
      dead.push_back(op);
    });

    unsigned removed = eraseWithCompletions(std::move(dead), dl);
    numDeadDest += removed;
    return removed != 0;
  }

  //===--------------------------------------------------------------------===//
  // overwritten-dest: a later collective rewrites the whole destination before
  // anything can observe this one.
  //===--------------------------------------------------------------------===//

  /// True when `later` writes at least the memory `earlier` writes, on every
  /// rank on which `earlier` writes it.
  static bool fullyOverwrites(Operation *later, Operation *earlier,
                              const DataLayout &dl) {
    if (!isDataMovingCollective(later) || hasStreamOperand(later))
      return false;
    // A non-blocking write only lands at its completion; do not reason about
    // where that is.
    if (isNonBlocking(later))
      return false;

    auto laterOp = dyn_cast<CommBufferOpInterface>(later);
    auto earlierOp = dyn_cast<CommBufferOpInterface>(earlier);
    if (!laterOp || !earlierOp)
      return false;
    CommBufferOperands lhs = laterOp.getCommBufferOperands();
    CommBufferOperands rhs = earlierOp.getCommBufferOperands();
    if (!lhs.recvBuffer || !rhs.recvBuffer)
      return false;

    // Over the same communicator in the same programming model, which a
    // per-rank count and a root are relative to ...
    Value comm = getCommOperand(later);
    if (!comm || comm != getCommOperand(earlier) ||
        getUsedModel(later) != getUsedModel(earlier))
      return false;

    // ... the same element count, counted the same way, and element type ...
    if (lhs.recvCount->get() != rhs.recvCount->get() ||
        lhs.recvCountIsPerRank != rhs.recvCountIsPerRank ||
        lhs.recvDatatype->get() != rhs.recvDatatype->get())
      return false;

    // ... covering the same extent of the same allocation, from its base ...
    BufferSlice lhsSlice = resolveBufferSlice(lhs.recvBuffer->get(), dl);
    BufferSlice rhsSlice = resolveBufferSlice(rhs.recvBuffer->get(), dl);
    if (lhsSlice.base != rhsSlice.base || !preservesBase(lhsSlice) ||
        !preservesBase(rhsSlice))
      return false;

    // ... on every rank `earlier` writes it: `later` writes on all ranks, or on
    // the same ones, being the same op with the same root. A root is what
    // decides which ranks write: reduce(root=0) and reduce(root=1) into one
    // buffer leave rank 0 with the first result.
    if (lhs.writesRecvBufferOnAllRanks)
      return true;
    return later->getName() == earlier->getName() &&
           CommBufferOperands::valueOf(lhs.root) ==
               CommBufferOperands::valueOf(rhs.root);
  }

  bool eliminateOverwrittenDestinations(const DataLayout &dl) {
    SmallVector<Operation *> dead;

    getOperation().walk([&](Operation *op) {
      std::optional<Candidate> candidate = getCandidate(op);
      if (!candidate || candidate->written.size() != 1)
        return;
      // A non-blocking write only lands at the completion, which sits somewhere
      // else entirely; the forward scan below says nothing about that point.
      // Only handle the blocking case here.
      if (isNonBlocking(op))
        return;

      BufferFacts facts;
      if (failed(computeBufferFacts(candidate->written.front(), dl, facts)))
        return;
      // The scan below reads the order of accesses off the order of ops in the
      // block. A pending non-blocking op or a window accesses the buffer
      // somewhere else in time, and other threads of a team run concurrently.
      if (isAccessedAwayFromOps(facts) || isInThreadTeam(op))
        return;

      // Scan forward inside the enclosing block.  Staying in one block is what
      // makes the overwrite unconditional: control cannot leave a block before
      // its terminator, so every operation between `op` and the overwrite runs.
      // Because the buffer provably does not escape, an operation that does not
      // name one of its aliases cannot touch it.
      for (Operation *cursor = op->getNextNode(); cursor;
           cursor = cursor->getNextNode()) {
        if (readsBuffer(cursor, facts))
          return;
        if (fullyOverwrites(cursor, op, dl)) {
          dead.push_back(op);
          return;
        }
      }
    });

    unsigned removed = eraseWithCompletions(std::move(dead), dl);
    numOverwrittenDest += removed;
    return removed != 0;
  }

  //===--------------------------------------------------------------------===//
  // dead-resource: windows and symmetric allocations nothing ever uses.
  //===--------------------------------------------------------------------===//

  /// RMA operations that only synchronise a window.  A window touched by these
  /// alone moves no data, so the window and all of them can go: fences are
  /// collective over the window's communicator and locks are passive-target, so
  /// in both cases removing the operation on every rank keeps counts aligned.
  ///
  /// spmd.winStart / spmd.winComplete are deliberately *not* in this list. They
  /// are the origin half of a PSCW epoch whose target half (MPI_Win_post /
  /// MPI_Win_wait) has no operation in this dialect, so nothing here can check
  /// that the other side goes away with them.
  static bool isWindowSyncOnly(Operation *op) {
    return isa<FenceOp, FenceOrderingOp, FlushOp, FlushAllOp, LockOp, LockAllOp,
               UnlockOp, UnlockAllOp>(op);
  }

  /// Collects the users of `resource` when they are all disposable, i.e. all of
  /// `allowed` kind and with no observed results themselves.
  static bool
  collectDisposableUsers(Value resource,
                         llvm::function_ref<bool(Operation *)> allowed,
                         SmallVectorImpl<Operation *> &users) {
    for (Operation *user : resource.getUsers()) {
      if (!allowed(user))
        return false;
      if (!llvm::all_of(user->getResults(),
                        [](Value result) { return result.use_empty(); }))
        return false;
      if (!llvm::is_contained(users, user))
        users.push_back(user);
    }
    return true;
  }

  bool eliminateDeadResources() {
    SmallVector<Operation *> dead;

    getOperation().walk([&](Operation *op) {
      Value window;
      if (auto create = dyn_cast<WinCreateOp>(op)) {
        window = create.getWin();
        if (!create.getError().use_empty())
          return;
      } else if (auto alloc = dyn_cast<WinAllocOp>(op)) {
        // The allocated memory itself must be unused too - it is symmetric
        // memory, so any local use may be mirrored by a remote access.
        if (!alloc.getBaseAddress().use_empty() ||
            !alloc.getError().use_empty())
          return;
        window = alloc.getWin();
      } else if (auto malloc = dyn_cast<MallocOp>(op)) {
        if (!malloc.getError().use_empty())
          return;
        SmallVector<Operation *> frees;
        if (!collectDisposableUsers(
                malloc.getAddress(),
                [](Operation *user) { return isa<FreeOp>(user); }, frees))
          return;
        llvm::append_range(dead, frees);
        dead.push_back(op);
        return;
      } else {
        return;
      }

      SmallVector<Operation *> users;
      if (!collectDisposableUsers(
              window,
              [](Operation *user) {
                return isa<WinFreeOp>(user) || isWindowSyncOnly(user);
              },
              users))
        return;
      llvm::append_range(dead, users);
      dead.push_back(op);
    });

    // Users before producers: everything in `dead` is either a consumer of a
    // resource also in `dead` or the producer itself, and walk() visited the
    // producer first, so erasing in reverse order is not enough - erase all
    // consumers first.
    IRRewriter rewriter(&getContext());
    for (Operation *op : dead)
      if (!isa<WinCreateOp, WinAllocOp, MallocOp>(op))
        rewriter.eraseOp(op);
    unsigned removedProducers = 0;
    for (Operation *op : dead) {
      if (!isa<WinCreateOp, WinAllocOp, MallocOp>(op))
        continue;
      rewriter.eraseOp(op);
      ++removedProducers;
    }

    numDeadResource += removedProducers;
    return !dead.empty();
  }
};

} // namespace

} // namespace mlir::spmd
