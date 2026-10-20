/*
 * SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- CommCompletions.h - Completions of communication ops -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The operations that have to go together with a communication op: a
// non-blocking op cannot be erased without whatever completes its request, and
// a completion cannot be erased while it still completes a request of an op
// that stays. See "Completions" in docs/optimizations.md.
//
// Requests are followed through their SSA uses, not through
// HandleOriginsAnalysis: the question is where a request goes, and a load of a
// request array through a different address expression than the store's gets
// no facts at all from the origins analyses, where it has to count as a use.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_ANALYSIS_COMMCOMPLETIONS_H
#define SPMD_ANALYSIS_COMMCOMPLETIONS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::spmd {

/// Everything that goes together with one communication op.
struct CommCompletions {
  /// The ops whose requests meet the op's own at a completion or in a request
  /// array, the op included. All of them go or none does.
  llvm::SetVector<Operation *> initiations;
  /// The spmd.wait, spmd.reqFree and spmd.waitAll ops completing them.
  llvm::SetVector<Operation *> completions;
  /// What only carries a request to a completion: casts and selects, stores
  /// into a request array and loads out of it.
  llvm::SetVector<Operation *> carriers;
};

/// What goes together with `op`. A blocking op goes alone.
///
/// A request may reach a completion directly, through a cast or select, or
/// through a request array: a private allocation that only stores of requests
/// write and only loads and spmd.waitAll read. A load of such an array may
/// yield any request stored into it. Fails when a request, or a result or
/// operand of a completion, reaches anything else - spmd.test, waitAny,
/// waitSome, a comparison, a call, a function argument - or when a completion's
/// result is used.
FailureOr<CommCompletions> getCommCompletions(Operation *op,
                                              const DataLayout &dl);

/// Narrows `ops` to those that can go given that all of the remaining ones go,
/// and returns every op to erase for them: the ops, their completions and
/// carriers. Shrinks to a fixpoint, since dropping one op may leave another
/// with a completion that has to stay.
llvm::SetVector<Operation *> collectRemovals(SmallVectorImpl<Operation *> &ops,
                                             const DataLayout &dl);

/// Erases `ops`, each once its results are unused. Every use of a result of an
/// op in `ops` must be in `ops` too.
void eraseOps(RewriterBase &rewriter, llvm::SetVector<Operation *> ops);

} // namespace mlir::spmd

#endif // SPMD_ANALYSIS_COMMCOMPLETIONS_H
