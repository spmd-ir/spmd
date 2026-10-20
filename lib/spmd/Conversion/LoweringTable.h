// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LoweringTable.h - A lowering read off the raiser ---------*- C++ -*-===//
//
// Every lowering of the dialect to one programming model's calls describes the
// same thing: which call each spmd operation becomes, and which C argument
// each of its operands fills.  Both are already written down, once, for the
// raiser:
//
//   * the model's CACTS map (lib/spmd/passes/CACTS/<model>SpmdEntries.cpp),
//     which says which spmd operation a call raises to, which C argument each
//     of the operation's operands came from, which of its results was an
//     out-parameter or the return value, which operands the raiser injected
//     because the call had no argument for them, and which attributes it
//     recorded; and
//   * the USIS API description (includesProgramModels/<model>/api.json),
//     which says what the C prototype looks like -- the order of the
//     parameters, their direction, their kind and their C type.
//
// Inverting the two gives, per callee, one slot per C argument saying where it
// comes from.  That inversion is the same for every model, and it is what this
// file does; what differs is the ABI -- how a handle is passed, what an
// out-parameter is stored in, which callee to pick when several raise to the
// same operation -- and each lowering supplies that as a TableSpec.
//
// An argument order written only in the raiser cannot drift from one written
// only in the lowering, because there is only one of them.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_LOWERINGTABLE_H
#define SPMD_CONVERSION_LOWERINGTABLE_H

#include "common.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>

namespace mlir::spmd {

struct CallDescription;

/// Where one C argument comes from, before the model's ABI has a say.
enum class ArgSource {
  /// The operation's operand at ArgSlot::operand.
  Operand,
  /// An out-parameter the raiser turned into one of the operation's results.
  /// ArgSlot::result is its position among them, or -1 when the raiser drops
  /// the value and the ABI still has to pass storage for it.
  Result,
  /// A parameter no operand and no result of the operation feeds.
  Unused,
};

/// Whether an out-parameter's result is one the operation may leave out, and
/// what says whether it did.  Only two of the dialect's results are optional
/// in that way; every other out-parameter is a result the operation always
/// has.
enum class OptionalResult { None, Status, ThreadLevel };

/// One C argument of one call: where it comes from, what USIS says about it,
/// and what the model's ABI makes of it.
struct ArgSlot {
  ArgSource source = ArgSource::Unused;
  /// The operand of the spmd operation this argument comes from, in the order
  /// the operands are stored, or -1 for a slot that reads none.  The CACTS
  /// entry lists its operands in exactly that order -- the raiser fills
  /// `newOperands[i]` and drops the absent ones -- so the entry's position is
  /// the operand's position.
  int operand = -1;
  /// The result of the spmd operation this out-parameter fills, in the same
  /// sense, or -1.
  int result = -1;
  /// What the ABI makes of the argument: the value of the lowering's own kind
  /// enum, which only that lowering's emitter and dump interpret.  TableSpec
  /// classify sets it.
  int abiKind = 0;
  OptionalResult optional = OptionalResult::None;
  /// Whether the operation may leave this operand out.  Only a trailing one
  /// can be left out without making the operation shorter than the call needs.
  bool optionalOperand = false;

  /// What USIS says, for the classification and the diagnostics.
  std::string name;
  std::string kind;
  std::string direction;
  /// The parameter's C declarator, such as "int count" or
  /// "MPI_Request *request".
  std::string cType;
  bool isArray = false;
  /// Whether the C binding passes the parameter by address.  A parameter USIS
  /// calls a SCALAR and does not pass by address carries the datum itself, so
  /// an operand the dialect holds as an address cannot fill it.
  bool isPointer = false;
  /// Whether the parameter has the MPI 4 large-count binding, so that the
  /// call has a form taking every count as an MPI_Count.
  bool hasLargeCountBinding = false;
};

/// An operand the raiser injected because the call had no argument for it.
/// Inverted, it is a condition on the operation: the operand must be the value
/// the raiser would have produced, or this call cannot express the operation.
struct DefaultCheck {
  int operand;
  /// The operand's name in the CACTS entry, which is what the diagnostic calls
  /// it.
  StringRef name;
  std::function<Value(OpBuilder &, Location, int)> value;
};

/// Two operands the entry reads from the same C argument, so the call has one
/// where the operation has two.  Inverted: the two must be the same value.
struct AliasCheck {
  int first;
  int second;
  StringRef firstName;
  StringRef secondName;
};

/// One call, and how to build it from the spmd operation it raises from.
struct CallDescription {
  StringRef callee;
  StringRef spmdOpName;
  /// One slot per C argument, in prototype order.  No inline storage: a slot
  /// carries what USIS says about its parameter, so an inline one would make
  /// every CallDescription several hundred bytes wide.
  SmallVector<ArgSlot, 0> args;
  SmallVector<DefaultCheck, 0> defaults;
  SmallVector<AliasCheck, 0> aliases;
  /// The attributes the raiser records for this callee.  Only those that tell
  /// this call apart from the operation's other candidates are conditions;
  /// see matchesAttributes.
  ArrayRef<std::function<NamedAttribute(OpBuilder &)>> attributes;
  /// How many operands the entry reads.  An operation with more carries
  /// something the model has no parameter for, one with fewer is missing a
  /// parameter the model needs.
  unsigned numOperands = 0;
  /// The operation's result the call's return value becomes, or -1: OpenSHMEM
  /// returns a rank, a size and a symmetric address rather than writing them
  /// through a pointer.
  int returnResult = -1;
  /// The operand whose storage the call's return value is written to, or -1:
  /// the raiser of shmem_<T>_g gives the operation a buffer and stores the
  /// returned element in it, so the lowering has to store it back.
  int returnOperand = -1;
  /// What USIS says the call returns: its kind ("ERROR_CODE", "value",
  /// "RESULT", "bool", "void") and its C type.
  std::string returnKind;
  std::string returnCType;
  /// Whether one of the slots is the request of a non-blocking call.  For MPI
  /// this, not USIS's execution.blocking, is what $isBlocking inverts to: MPI
  /// calls the one-sided transfers non-blocking whether or not they hand out a
  /// request.
  bool producesRequest = false;
  /// USIS's execution.blocking, which is what $isBlocking inverts to for a
  /// model whose non-blocking calls hand out no request at all.
  bool blocking = true;
  /// Whether the entry's last operand is one the operation may leave out --
  /// MPI's trailing array of statuses, which then becomes MPI_STATUSES_IGNORE.
  /// It is the one operand whose absence does not make the operation shorter
  /// than the call needs.
  bool trailingOptionalOperand = false;
  /// The parameter the ABI has no way to fill, if any.  The call stays in the
  /// table so that a diagnostic can name it, but no operation lowers to it:
  /// shmem_<T>_atomic_fetch_add takes the summand by value and the raiser
  /// keeps no operand for it, so inverting the entry would add zero.
  std::string unsupportedParameter;
  /// Whether this is a call the lowering picks for its operation.  Several
  /// calls raise to the same one, because the dialect does not record what
  /// tells them apart; the others are in the table, but not offered.
  bool canonical = true;
};

/// What the diagnostic calls an operand the model has no parameter for, or one
/// it needs and the operation has not, where the operand's type does not say.
/// The table knows that the operand is there and that no parameter takes it;
/// what to call it is the dialect's business.
struct OperandNote {
  StringLiteral spmdOpName;
  StringLiteral text;
};

/// How a model's $isBlocking inverts.
enum class BlockingForm {
  /// To whether the call hands out a request (MPI).
  Request,
  /// To USIS's execution.blocking (OpenSHMEM, whose _nbi calls hand out no
  /// request).
  Execution,
};

/// Everything one model's lowering has to say that the CACTS map and USIS
/// cannot.  One static instance per lowering; the table is built on first use
/// and cached against its address.
struct TableSpec {
  /// The model as USIS and the <MODEL>_JSON_PATH environment variable spell
  /// it.
  StringRef model;
  /// What a diagnostic calls it: "MPI", "OpenSHMEM".
  StringRef displayName;
  /// The CACTS map the raiser reads for it.
  const ApiCallToSpmdMap *entries;
  /// What the ABI makes of one C argument: sets ArgSlot::abiKind, and may
  /// refine the slot (an out-parameter the raiser drops is still a Result the
  /// ABI passes storage for) and the call (whether it hands out a request).
  /// False when the ABI has no rule for the argument, which drops the call
  /// from the table: a parameter the lowering cannot fill is a call it cannot
  /// emit, and passing a zero for it would quietly change what the call does.
  bool (*classify)(ArgSlot &slot, CallDescription &call);
  /// Whether the lowering ever picks \p callee.  Several calls raise to the
  /// same operation; the others stay in the table but are not offered.
  bool (*isCanonical)(StringRef callee, const APICallEntry &entry);
  BlockingForm blockingForm;
  /// Whether an operation may keep a request result the call does not write,
  /// which the lowering then fills with the model's null request.  A model
  /// with no non-blocking form of the operation at all sets this false, so
  /// that the operation is reported rather than quietly made blocking.
  bool allowsRequestNull;
  /// Whether \p actual is the injected value \p expected, when the default
  /// comparison -- same operation, same attributes -- says no.  A model whose
  /// raiser spells a handle differently from the model the operation was
  /// raised from needs this to accept a translated program; null for a model
  /// that does not.
  bool (*alsoSameInjected)(Value actual, Operation *expected);
  /// What a diagnostic about a stride adds, for a model that transfers
  /// contiguous elements only; empty for one with strided calls of its own.
  StringRef contiguousOnlyNote;
  ArrayRef<OperandNote> extraOperandNotes;
  ArrayRef<OperandNote> missingOperandNotes;
};

/// Every call of \p spec's table, in a stable order.
ArrayRef<CallDescription> getLoweringTable(const TableSpec &spec);

/// The canonical calls that raise to \p spmdOpName, in the table's order.
ArrayRef<const CallDescription *> getCallsFor(const TableSpec &spec,
                                              StringRef spmdOpName);

/// Whether \p op carries the attribute values \p call records, for those
/// attributes that tell \p call apart from the operation's other candidates.
bool matchesAttributes(const TableSpec &spec, Operation *op,
                       const CallDescription &call, OpBuilder &builder);

/// The call \p op lowers to, or nullptr.
///
/// When no call fits and the reason is one the table can name -- an operand
/// the model has no parameter for, one it needs and the operation has not, an
/// injected operand the operation spells differently, or two operands the
/// model shares one parameter for -- \p reasons gets one entry per such
/// reason, each the tail of a diagnostic on \p op.  An operation the table
/// does not describe at all, and one whose results no call can produce, leave
/// \p reasons empty: those are for the conversion to report as a failure to
/// legalize.
const CallDescription *selectCall(const TableSpec &spec, Operation *op,
                                  OpBuilder &builder,
                                  SmallVectorImpl<std::string> &reasons);

/// Whether \p lhs and \p rhs are the same value, looking through a widening,
/// narrowing or pointer-to-memref cast both apply to the same value, and
/// comparing constants by value.  This is what an AliasCheck asks of two
/// operands the entry reads from one C argument; a lowering that has to ask
/// the same question of two of its own operation's operands outside an
/// AliasCheck uses it directly -- see LowerSPMDToMPI.cpp's handling of
/// spmd.sendrecv.
bool sameOperandValue(Value lhs, Value rhs);

/// Whether \p op still has the operand \p slot reads.
bool hasSlotOperand(Operation *op, const ArgSlot &slot);

/// Whether the operation has a result for the out-parameter \p slot, which it
/// may leave out only for an optional status or thread level.
bool producesSlotResult(Operation *op, const ArgSlot &slot);

/// Whether \p op keeps a request result \p call does not write, which the
/// lowering fills with the model's null request: a blocking call of an
/// operation the optimisations gave a request to.
bool needsRequestNull(const TableSpec &spec, Operation *op,
                      const CallDescription &call);

/// Prints the derived table, one line per call, for the -test-*-lowering-table
/// passes.  \p kindName spells the lowering's own ArgSlot::abiKind.
/// \p canonicalOnly leaves out the calls the lowering never picks, and
/// \p filter the calls whose callee does not contain it: OpenSHMEM's table has
/// one entry per spelling of every element type, which no test wants in full.
void printLoweringTable(const TableSpec &spec, raw_ostream &os,
                        OpBuilder &builder, StringRef (*kindName)(int),
                        bool canonicalOnly = false, StringRef filter = "");

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_LOWERINGTABLE_H
