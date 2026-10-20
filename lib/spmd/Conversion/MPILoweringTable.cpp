// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- MPILoweringTable.cpp - The MPI lowering, read off the raiser -------===//

#include "MPILoweringTable.h"

#include "mpiSpmdEntries.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::spmd {

namespace {

//===----------------------------------------------------------------------===//
// What the two sources cannot say
//===----------------------------------------------------------------------===//

// Several MPI calls raise to the same spmd operation, because the dialect does
// not record what tells them apart: synchronous mode is not modelled, and
// nothing in the *shape* of spmd.sendrecv says that one buffer was written
// twice.  selectCall has to pick one of them, and it picks the one that is not
// listed here.  A new alias in the CACTS map that is not listed makes the
// choice ambiguous, which selectCall reports rather than resolving at random.
//
// Sendrecv_replace and Isendrecv_replace are listed for that reason -- as far
// as selectCall can tell, every spmd.sendrecv could be either -- but they are
// not actually unrecoverable the way Ssend's synchronous mode is: the two
// buffers being one *value*, not merely one shape, is exactly what raising
// MPI_Sendrecv_replace produces, and LowerSPMDToMPI.cpp's pickSendrecvCall
// checks that directly (sameOperandValue) and overrides this table's answer
// when it holds.  MPI_Sendrecv/MPI_Isendrecv do not tolerate that aliasing.
constexpr StringLiteral kNonCanonicalCallees[] = {
    "MPI_Ssend",
    "MPI_Issend",
    "MPI_Ssend_init",
    "MPI_Sendrecv_replace",
    "MPI_Isendrecv_replace",
};

bool isCanonicalMPICallee(StringRef callee, const APICallEntry &) {
  return !llvm::is_contained(kNonCanonicalCallees, callee);
}

// What the diagnostic calls an operand MPI has no parameter for, where its
// type does not say.  The table knows that the operand is there and that no
// MPI parameter takes it; what to call it is the dialect's business.
constexpr OperandNote kExtraOperandNotes[] = {
    {"spmd.malloc",
     "asks for an alignment, which MPI_Alloc_mem has no parameter for"},
};

// The same for an operand MPI needs and the operation does not have.  Which of
// the operands is missing is not something the table can tell -- it sees only
// that there are too few -- so the note names it.
constexpr OperandNote kMissingOperandNotes[] = {
    {"spmd.winCreate", "has no size; MPI_Win_create needs one"},
    {"spmd.fence", "has no window; MPI needs one"},
    {"spmd.flushAll", "has no window; MPI needs one"},
};

//===----------------------------------------------------------------------===//
// The MPICH ABI of one argument
//===----------------------------------------------------------------------===//

/// Whether the C binding spells the parameter as a plain int.  An element
/// count the dialect keeps as an MPI_Count then has to be narrowed, or the
/// call replaced by its large-count form; one MPI already takes as an
/// MPI_Count or an MPI_Aint goes straight in.  The call has to have a
/// large-count form to be renamed to it, which is why the caller also asks for
/// the "c_large" binding: MPI_Waitall counts requests, not elements, and has
/// no MPI_Waitall_c.
bool isPlainInt(StringRef decl) {
  decl = decl.trim();
  decl.consume_front("const ");
  return decl.starts_with("int ") && !decl.contains('*') && !decl.contains('[');
}

/// Whether the kind names one of the handles MPI passes as an int, and writes
/// back through a pointer when it invalidates it.
bool isHandleKind(StringRef kind) {
  return kind == "REQUEST" || kind == "COMMUNICATOR" || kind == "WINDOW" ||
         kind == "MESSAGE";
}

bool classifyMPIArg(ArgSlot &slot, CallDescription &call) {
  auto set = [&](MPIArgKind kind) { slot.abiKind = static_cast<int>(kind); };

  switch (slot.source) {
  case ArgSource::Result:
    // An out-parameter the raiser turned into a result of the operation.
    if (slot.kind == "BOOL") {
      set(MPIArgKind::OutFlag);
    } else if (slot.kind == "STATUS") {
      set(MPIArgKind::OutStatus);
      slot.optional = OptionalResult::Status;
    } else if (slot.kind == "BUFFER") {
      set(MPIArgKind::OutPtr);
    } else {
      set(MPIArgKind::OutInt);
      if (slot.kind == "THREAD_LEVEL")
        slot.optional = OptionalResult::ThreadLevel;
    }
    call.producesRequest |= slot.kind == "REQUEST";
    return true;

  case ArgSource::Operand:
    if (slot.kind == "STATUS" && slot.isArray) {
      set(MPIArgKind::StatusArray);
      slot.optionalOperand = true;
    } else if (slot.direction == "inout" && !slot.isArray &&
               isHandleKind(slot.kind)) {
      set(MPIArgKind::ByAddress);
    } else if (slot.kind == "COUNT" && isPlainInt(slot.cType) &&
               slot.hasLargeCountBinding) {
      set(MPIArgKind::Count);
    } else {
      set(MPIArgKind::In);
    }
    return true;

  case ArgSource::Unused:
    // A parameter no operand and no result of the operation feeds.  MPI still
    // needs somewhere to write an out-parameter, so the slot becomes one the
    // lowering passes storage for, whether or not anything reads it back.
    if (slot.kind == "STATUS" && slot.isArray) {
      set(MPIArgKind::StatusArray);
      slot.optionalOperand = true;
    } else if (slot.kind == "STATUS") {
      set(MPIArgKind::OutStatus);
      slot.optional = OptionalResult::Status;
      slot.source = ArgSource::Result;
    } else if (slot.kind == "REQUEST" && slot.direction == "out") {
      set(MPIArgKind::OutInt);
      slot.source = ArgSource::Result;
      call.producesRequest = true;
    } else if (slot.kind == "INFO") {
      set(MPIArgKind::InfoNull);
    } else {
      set(MPIArgKind::NullPointer);
    }
    return true;
  }
  llvm_unreachable("unknown ArgSource");
}

StringRef mpiKindName(int abiKind) {
  switch (static_cast<MPIArgKind>(abiKind)) {
  case MPIArgKind::In:
    return "in";
  case MPIArgKind::Count:
    return "count";
  case MPIArgKind::ByAddress:
    return "byAddress";
  case MPIArgKind::StatusArray:
    return "statuses";
  case MPIArgKind::OutInt:
    return "outInt";
  case MPIArgKind::OutPtr:
    return "outPtr";
  case MPIArgKind::OutFlag:
    return "flag";
  case MPIArgKind::OutStatus:
    return "status";
  case MPIArgKind::InfoNull:
    return "MPI_INFO_NULL";
  case MPIArgKind::NullPointer:
    return "NULL";
  }
  llvm_unreachable("unknown MPIArgKind");
}

} // namespace

const TableSpec &getMPITableSpec() {
  static const TableSpec spec = {
      /*model=*/"mpi",
      /*displayName=*/"MPI",
      /*entries=*/&mpiToSpmdOperands,
      /*classify=*/classifyMPIArg,
      /*isCanonical=*/isCanonicalMPICallee,
      /*blockingForm=*/BlockingForm::Request,
      /*allowsRequestNull=*/true,
      /*alsoSameInjected=*/nullptr,
      /*contiguousOnlyNote=*/"; MPI transfers contiguous elements only",
      /*extraOperandNotes=*/kExtraOperandNotes,
      /*missingOperandNotes=*/kMissingOperandNotes,
  };
  return spec;
}

ArrayRef<CallDescription> getMPILoweringTable() {
  return getLoweringTable(getMPITableSpec());
}

const CallDescription *selectMPICall(Operation *op, OpBuilder &builder,
                                     SmallVectorImpl<std::string> &reasons) {
  return selectCall(getMPITableSpec(), op, builder, reasons);
}

void printMPILoweringTable(raw_ostream &os, OpBuilder &builder) {
  printLoweringTable(getMPITableSpec(), os, builder, mpiKindName);
}

} // namespace mlir::spmd
