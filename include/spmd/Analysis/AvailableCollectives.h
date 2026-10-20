/*
 * SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- AvailableCollectives.h - Available-collective dataflow ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The must-analysis behind -eliminate-redundant-comm: at each program point,
// which collectives have already been performed and whose result therefore
// still sits in the destination buffer.
//
// Soundness rests on two rules (docs/optimizations.md, "Available
// collectives"):
//
//   KILLs are unconditional - any write that may reach a buffer in the key
//   removes the fact, whether or not the writer is uniformly executed, whether
//   or not it can be attributed to an operand.
//
//   GENs happen only under uniform control - a fact may only be established by
//   an operation *every* rank executes, on operands every rank agrees on.
//   Otherwise one rank would carry a fact another does not, and the later
//   removal - which is textual, so it happens on all ranks - would drop a
//   collective some rank still needed (silent corruption).
//
// Facts are also only established over private allocations that nothing
// accesses away from where its op sits (docs/optimizations.md, "Accesses away
// from the op").
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_ANALYSIS_AVAILABLECOLLECTIVES_H
#define SPMD_ANALYSIS_AVAILABLECOLLECTIVES_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// CommKey
//===----------------------------------------------------------------------===//

/// Everything that has to match for one collective to make another one
/// redundant.
///
/// Buffers are identified by their *allocation*, not by the SSA value the
/// operation names, so that two collectives reached through different casts of
/// the same `memref.alloca` still compare equal.  Only base-preserving chains
/// qualify: a pointer into the middle of an allocation describes a different
/// extent, and CommKey has no notion of offset.
///
/// The remaining operands - count, datatype, reduction operator, root - are
/// compared by SSA identity, which is why the pass wants `-canonicalize -cse`
/// ahead of it.  Two `arith.constant 1` operations that CSE has not merged
/// simply produce two different keys and nothing is removed.
struct CommKey {
  /// The operation's name, held as the uniqued identifier so that the struct
  /// stays default-constructible (OperationName is not).
  StringAttr kind;
  Value comm;
  Value sendAlloc;
  Value recvAlloc;
  /// count / datatype / reduction op / root, in operand order.
  SmallVector<Value, 4> shape;
  int32_t usedModel;

  bool operator==(const CommKey &other) const {
    return kind == other.kind && comm == other.comm &&
           sendAlloc == other.sendAlloc && recvAlloc == other.recvAlloc &&
           shape == other.shape && usedModel == other.usedModel;
  }
  bool operator!=(const CommKey &other) const { return !(*this == other); }

  void print(raw_ostream &os) const;
};

/// The key of `op`, when it is a collective this analysis can reason about at
/// all: data-moving, blocking, not stream-ordered, with both buffers resolving
/// to the base of an allocation.
///
/// Returns nullopt for the in-place shape (`sendAlloc == recvAlloc`, i.e.
/// MPI_IN_PLACE).  An in-place collective is *not* idempotent - a second
/// allreduce over the already-reduced value produces a different answer, which
/// at two ranks may even look right - so it must never enter the available set.
std::optional<CommKey> getCommKey(Operation *op, const DataLayout &dl);

//===----------------------------------------------------------------------===//
// The lattice
//===----------------------------------------------------------------------===//

/// The set of collectives available at a program point.  Join is intersection:
/// a fact survives a merge only if it holds on every incoming path.
///
/// `bottom` distinguishes "unreached, so no information yet" from "reached, and
/// nothing is available".  Without it the first join at a merge point would
/// intersect against an empty set and wipe everything.
class AvailableCollectives : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &lattice) override;

  /// Drops every fact and leaves the state reachable-but-empty.
  ChangeResult killAll();

  /// Drops the facts `predicate` selects.
  ChangeResult kill(llvm::function_ref<bool(const CommKey &)> predicate);

  /// Records `key` as available.
  ChangeResult gen(const CommKey &key);

  bool contains(const CommKey &key) const;

  /// True while nothing has reached this point yet.  A lookup answering true
  /// carries no information and must not be used to remove anything.
  bool isBottom() const { return bottom; }

  void print(raw_ostream &os) const override;

private:
  bool bottom = true;
  SmallVector<CommKey> available;
};

//===----------------------------------------------------------------------===//
// The analysis
//===----------------------------------------------------------------------===//

/// Load it together with dataflow::DeadCodeAnalysis and
/// dataflow::SparseConstantPropagation. Without DeadCodeAnalysis the framework
/// never marks anything live and every lattice lookup comes back uninitialised.
/// DeadCodeAnalysis in turn reads `ConstantValue` to resolve the successors of
/// a RegionBranchOpInterface operation; with no constant propagation loaded
/// those lattices stay uninitialised, an `scf.if` never gets known
/// predecessors, and the state after it stays bottom - which looks exactly
/// like "nothing is available" and silently disables the pass across every
/// conditional.
class AvailableCollectivesAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<AvailableCollectives> {
public:
  /// `uniformity` is a solver on which ExecutionContextAnalysis and the
  /// uniformity analyses have already run to a fixpoint.
  AvailableCollectivesAnalysis(DataFlowSolver &solver,
                               AliasAnalysis &aliasAnalysis,
                               DataFlowSolver &uniformity, const DataLayout &dl)
      : DenseForwardDataFlowAnalysis(solver), aliasAnalysis(aliasAnalysis),
        uniformity(uniformity), dl(dl) {}

  LogicalResult visitOperation(Operation *op,
                               const AvailableCollectives &before,
                               AvailableCollectives *after) override;

  void setToEntryState(AvailableCollectives *lattice) override;

  /// Availability is intraprocedural in this version: a call neither carries
  /// facts into a callee nor brings any back.  Making it interprocedural means
  /// deciding what an MPI communicator means across a call boundary, which is a
  /// separate question from the one this pass answers.
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const AvailableCollectives &before,
                                    AvailableCollectives *after) override;

private:
  /// Removes every fact whose buffers `written` may alias.  A null `written`
  /// means "an effect we could not attribute", which kills everything.
  ChangeResult killWrites(Value written, AvailableCollectives *after);

  /// `allocation` is private, so that every access to it is an op in this
  /// function, and none of those accesses it away from where it sits: no
  /// pending non-blocking op writes it behind the analysis' back, no window
  /// lets another rank do so. Cached per allocation.
  bool isLocalBuffer(Value allocation);

  AliasAnalysis &aliasAnalysis;
  DataFlowSolver &uniformity;
  const DataLayout &dl;
  llvm::DenseMap<Value, bool> localBuffers;
};

} // namespace mlir::spmd

#endif // SPMD_ANALYSIS_AVAILABLECOLLECTIVES_H
