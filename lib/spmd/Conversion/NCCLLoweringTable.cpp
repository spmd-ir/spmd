// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- NCCLLoweringTable.cpp - The NCCL lowering, read off the raiser -----===//

#include "NCCLLoweringTable.h"

#include "ncclSpmdEntries.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::spmd {

namespace {

//===----------------------------------------------------------------------===//
// What the two sources cannot say
//===----------------------------------------------------------------------===//

// ncclBcast is the in-place ncclBroadcast, which nccl.h marks deprecated, and
// both raise to spmd.bcast.  An in-place bcast fits both, so one of them has to
// go: ncclBroadcast with the same buffer twice is the same transfer, and it is
// the one NCCL keeps.
constexpr StringLiteral kNonCanonicalCallees[] = {
    "ncclBcast",
};

bool isCanonicalNCCLCallee(StringRef callee, const APICallEntry &) {
  return !llvm::is_contained(kNonCanonicalCallees, callee);
}

// What the diagnostic calls an operand NCCL has no parameter for, where its
// type does not say.
constexpr OperandNote kExtraOperandNotes[] = {
    {"spmd.malloc",
     "asks for an alignment, which ncclMemAlloc has no parameter for"},
};

// A program translated from a model without streams has none to pass, and the
// stream is the last operand of every communication call.  The table sees only
// that the operation is one operand short; which one is missing is said here.
constexpr StringLiteral kNoStream =
    "has no CUDA stream; every NCCL communication call takes one";

constexpr OperandNote kMissingOperandNotes[] = {
    {"spmd.send", kNoStream},          {"spmd.recv", kNoStream},
    {"spmd.bcast", kNoStream},         {"spmd.reduce", kNoStream},
    {"spmd.allreduce", kNoStream},     {"spmd.allgather", kNoStream},
    {"spmd.reduceScatter", kNoStream}, {"spmd.alltoall", kNoStream},
    {"spmd.gather", kNoStream},        {"spmd.scatter", kNoStream},
};

//===----------------------------------------------------------------------===//
// The NCCL ABI of one argument
//===----------------------------------------------------------------------===//

bool isPointerDeclarator(StringRef decl) { return decl.contains('*'); }

bool classifyNCCLArg(ArgSlot &slot, CallDescription &) {
  auto set = [&](NCCLArgKind kind) {
    slot.abiKind = static_cast<int>(kind);
    return true;
  };

  switch (slot.source) {
  case ArgSource::Result:
    // The out-parameters NCCL's map has: a new communicator, an address
    // ncclMemAlloc allocates, and an int.
    if (slot.kind == "COMMUNICATOR")
      return set(NCCLArgKind::OutComm);
    if (slot.kind == "BUFFER")
      return set(NCCLArgKind::OutPtr);
    if (slot.kind == "RANK" || slot.kind == "DEVICE" ||
        slot.kind == "TEAM_SIZE")
      return set(NCCLArgKind::OutInt);
    return false;

  case ArgSource::Operand:
    return set(NCCLArgKind::In);

  case ArgSource::Unused:
    // ncclCommSplit's optional configuration, which spmd.commSplit carries as
    // an optional operand the map does not read.  Nothing else may be left
    // out: ncclCommInitRank's size, unique id and rank are what the raiser
    // drops when it makes the call an spmd.commWorld, and passing zeros for
    // them would start a communicator of no one.  That call is dropped from
    // the table instead, which is what returning false does; the lowering
    // bootstraps the world communicator in spmd.init.
    if (slot.name == "config" && isPointerDeclarator(slot.cType))
      return set(NCCLArgKind::NullPointer);
    return false;
  }
  llvm_unreachable("unknown ArgSource");
}

StringRef ncclKindName(int abiKind) {
  switch (static_cast<NCCLArgKind>(abiKind)) {
  case NCCLArgKind::In:
    return "in";
  case NCCLArgKind::OutComm:
    return "outComm";
  case NCCLArgKind::OutInt:
    return "outInt";
  case NCCLArgKind::OutPtr:
    return "outPtr";
  case NCCLArgKind::NullPointer:
    return "NULL";
  }
  llvm_unreachable("unknown NCCLArgKind");
}

} // namespace

const TableSpec &getNCCLTableSpec() {
  static const TableSpec spec = {
      /*model=*/"nccl",
      /*displayName=*/"NCCL",
      /*entries=*/&ncclToSpmdOperands,
      /*classify=*/classifyNCCLArg,
      /*isCanonical=*/isCanonicalNCCLCallee,
      // NCCL hands out no request, and USIS calls every communication call
      // non-blocking because it only enqueues onto the stream.  The raiser
      // copies that into $isBlocking, so it is what inverts.
      /*blockingForm=*/BlockingForm::Execution,
      // A request result is what the raiser gives the operations of a group;
      // the lowering opens the group before the call and fills the request
      // with a constant, and the spmd.wait that completes it closes the group.
      /*allowsRequestNull=*/true,
      /*alsoSameInjected=*/nullptr,
      /*contiguousOnlyNote=*/"",
      /*extraOperandNotes=*/kExtraOperandNotes,
      /*missingOperandNotes=*/kMissingOperandNotes,
  };
  return spec;
}

ArrayRef<CallDescription> getNCCLLoweringTable() {
  return getLoweringTable(getNCCLTableSpec());
}

const CallDescription *selectNCCLCall(Operation *op, OpBuilder &builder,
                                      SmallVectorImpl<std::string> &reasons) {
  return selectCall(getNCCLTableSpec(), op, builder, reasons);
}

void printNCCLLoweringTable(raw_ostream &os, OpBuilder &builder) {
  printLoweringTable(getNCCLTableSpec(), os, builder, ncclKindName);
}

} // namespace mlir::spmd
