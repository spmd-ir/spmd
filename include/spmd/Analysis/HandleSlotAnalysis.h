/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- HandleSlotAnalysis.h - Handles stored in memory ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Which handles a memory slot may hold, at each program point.
//
// CommOriginsAnalysis and HandleOriginsAnalysis follow a communicator or a
// request through SSA values, but the raiser just as often hands one over
// through memory: MPI_Comm_dup writes the new communicator into a variable a
// later call reads, MPI_Isend writes its request into an array element that
// MPI_Wait loads back. HandleSlotAnalysis records the facts of every value
// stored into a slot, and resolveHandleFacts reads them back for a value
// loaded from one.
//
// A slot is keyed by its address with the enzymexla memref/pointer bridges
// and unrealized casts stripped - not by alias class or byte offset - so a
// load finds what was stored through the same address expression. A store of
// unknown facts goes into unknownSlotContents, which every load sees.
//
// Both origins analyses keep their facts in HandleFactsLattice, so one solver
// runs one of them, together with a HandleSlotAnalysis for the same kind of
// handle.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_ANALYSIS_HANDLESLOTANALYSIS_H
#define SPMD_ANALYSIS_HANDLESLOTANALYSIS_H

#include "spmd/Analysis/HandleFacts.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir::spmd {

/// The kind of handle a HandleSlotAnalysis tracks, which decides the types that
/// can carry one.
enum class HandleKind {
  /// Communicators, tracked by CommOriginsAnalysis.
  Communicator,
  /// Requests and windows, tracked by HandleOriginsAnalysis.
  Request,
};

/// A type that can carry a handle of `kind` through a store or a cast: the
/// handle type itself, the pointers and memrefs it is stored through, and for
/// requests the i32 the raiser casts them to before storing them.
bool isHandleCarrierType(Type type, HandleKind kind);

//===----------------------------------------------------------------------===//
// HandleSlotValue
//===----------------------------------------------------------------------===//

struct HandleSlotValue {
  llvm::DenseMap<Value, HandleFactsValue> slotContents;
  HandleFactsValue unknownSlotContents;

  bool operator==(const HandleSlotValue &other) const;
  ChangeResult joinInPlace(const HandleSlotValue &other);
};

//===----------------------------------------------------------------------===//
// HandleSlotLattice
//===----------------------------------------------------------------------===//

class HandleSlotLattice : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &other) override {
    return value.joinInPlace(
        static_cast<const HandleSlotLattice &>(other).value);
  }

  void print(raw_ostream &os) const override {
    os << "HandleSlotLattice(" << value.slotContents.size() << " slots)";
  }

  HandleSlotValue &getValue() { return value; }
  const HandleSlotValue &getValue() const { return value; }

private:
  HandleSlotValue value;
};

//===----------------------------------------------------------------------===//
// HandleSlotAnalysis
//===----------------------------------------------------------------------===//

class HandleSlotAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<HandleSlotLattice> {
public:
  using Base = dataflow::DenseForwardDataFlowAnalysis<HandleSlotLattice>;

  HandleSlotAnalysis(DataFlowSolver &solver, HandleKind kind)
      : Base(solver), kind(kind) {}

  LogicalResult visitOperation(Operation *op, const HandleSlotLattice &before,
                               HandleSlotLattice *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const HandleSlotLattice &before,
                                    HandleSlotLattice *after) override;

  void setToEntryState(HandleSlotLattice *lattice) override {}

  HandleFactsValue resolveValueFacts(Operation *op, Value val,
                                     const HandleSlotValue &slotState);

private:
  HandleKind kind;
};

/// The facts about `value` once the solver has run: its own
/// HandleFactsLattice, through casts, bridges and selects, and for a loaded
/// value the contents of the slot it was loaded from as the slot state before
/// `contextOp` holds them.
HandleFactsValue resolveHandleFacts(DataFlowSolver &solver, Value value,
                                    Operation *contextOp, HandleKind kind);

} // namespace mlir::spmd

#endif // SPMD_ANALYSIS_HANDLESLOTANALYSIS_H
