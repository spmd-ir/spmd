// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- OMPLoweringTable.cpp - The OpenMP queries, read off the raiser -----===//

#include "OMPLoweringTable.h"

#include "ompSpmdEntries.h"

#include "spmd/SPMDOps.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::spmd {

namespace {

// The routines the raiser maps take no parameter, so there is no argument to
// classify; a parameter showing up here is a routine the map gained that this
// lowering does not know how to pass, and it is left out of the table.
bool classifyOMPArg(ArgSlot &, CallDescription &) { return false; }

bool isCanonicalOMPCallee(StringRef, const APICallEntry &) { return true; }

// The raiser injects spmd.getParallelComm, the team the call runs in, and
// canonicalisation folds it into the communicator of the enclosing
// spmd.parallel -- of a thread team, or of the process team when the query is
// not in one.  OpenMP has no way to ask another team, so any communicator is
// the injected one here; that it really is the current team is what
// checkSPMDToOMP makes sure of.
bool isTheCurrentTeam(Value actual, Operation *expected) {
  return isa<GetParallelCommOp>(expected) && actual &&
         isa<CommType>(actual.getType());
}

StringRef ompKindName(int) { return "none"; }

} // namespace

const TableSpec &getOMPTableSpec() {
  static const TableSpec spec = {
      /*model=*/"omp",
      /*displayName=*/"OpenMP",
      /*entries=*/&ompToSpmdOperands,
      /*classify=*/classifyOMPArg,
      /*isCanonical=*/isCanonicalOMPCallee,
      // The queries are blocking in USIS, and the raiser copies that into
      // $isBlocking; no OpenMP routine hands out a request.
      /*blockingForm=*/BlockingForm::Execution,
      /*allowsRequestNull=*/false,
      /*alsoSameInjected=*/isTheCurrentTeam,
      /*contiguousOnlyNote=*/"",
      /*extraOperandNotes=*/{},
      /*missingOperandNotes=*/{},
  };
  return spec;
}

ArrayRef<CallDescription> getOMPLoweringTable() {
  return getLoweringTable(getOMPTableSpec());
}

const CallDescription *selectOMPCall(Operation *op, OpBuilder &builder,
                                     SmallVectorImpl<std::string> &reasons) {
  return selectCall(getOMPTableSpec(), op, builder, reasons);
}

void printOMPLoweringTable(raw_ostream &os, OpBuilder &builder) {
  printLoweringTable(getOMPTableSpec(), os, builder, ompKindName);
}

} // namespace mlir::spmd
