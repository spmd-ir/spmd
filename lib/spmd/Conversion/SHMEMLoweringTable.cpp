// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SHMEMLoweringTable.cpp - The OpenSHMEM family, off the raiser ------===//

#include "SHMEMLoweringTable.h"

#include "spmd/SPMDOps.h"

#include "nvshmemSpmdEntries.h"
#include "shmemSpmdEntries.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::spmd {

namespace {

//===----------------------------------------------------------------------===//
// Which of the many spellings of a call to pick
//===----------------------------------------------------------------------===//

// OpenSHMEM spells one element type several ways -- shmem_int_put and
// shmem_int32_put both move four signed bytes -- and the dialect keeps only
// the MLIR type, so the raiser maps every spelling of a group to the same
// spmd.constDatatype and the lowering has to pick one back.  It picks the first
// spelling of the group the CACTS map has a call of that name for, which is
// why the groups are ordered: shmem_schar_and_reduce does not exist, so a
// bitwise and over si8 comes out as shmem_int8_and_reduce.  NVSHMEM spells its
// types the same way.
//
// A group is one signedness: shmem_int_put and shmem_uint_put move the same
// bytes, but shmem_int_max_reduce and shmem_uint_max_reduce do not compute the
// same maximum.  char is a group of its own, because C does not say whether it
// is signed.
//
// The complex types have no group of their own because the dialect has no
// complex type: kNeverCanonical drops them.
struct TypeNameGroup {
  StringLiteral names[4];
};

constexpr TypeNameGroup kIntegerTypeNameGroups[] = {
    {{"char", "", "", ""}},
    {{"schar", "int8", "", ""}},
    {{"uchar", "uint8", "", ""}},
    {{"short", "int16", "", ""}},
    {{"ushort", "uint16", "", ""}},
    {{"int", "int32", "", ""}},
    {{"uint", "uint32", "", ""}},
    {{"long", "longlong", "int64", "ptrdiff"}},
    {{"ulong", "ulonglong", "uint64", "size"}},
};

constexpr TypeNameGroup kOpenSHMEMTypeNameGroups[] = {
    kIntegerTypeNameGroups[0], kIntegerTypeNameGroups[1],
    kIntegerTypeNameGroups[2], kIntegerTypeNameGroups[3],
    kIntegerTypeNameGroups[4], kIntegerTypeNameGroups[5],
    kIntegerTypeNameGroups[6], kIntegerTypeNameGroups[7],
    kIntegerTypeNameGroups[8], {{"float", "", "", ""}},
    {{"double", "", "", ""}},  {{"longdouble", "", "", ""}},
};

// NVSHMEM has no long double, and has the two 16-bit floating-point types
// OpenSHMEM does not.
constexpr TypeNameGroup kNVSHMEMTypeNameGroups[] = {
    kIntegerTypeNameGroups[0],  kIntegerTypeNameGroups[1],
    kIntegerTypeNameGroups[2],  kIntegerTypeNameGroups[3],
    kIntegerTypeNameGroups[4],  kIntegerTypeNameGroups[5],
    kIntegerTypeNameGroups[6],  kIntegerTypeNameGroups[7],
    kIntegerTypeNameGroups[8],  {{"float", "", "", ""}},
    {{"double", "", "", ""}},   {{"half", "", "", ""}},
    {{"bfloat16", "", "", ""}},
};

// Type names the dialect has no type for at all, so the datatype the raiser
// injects for them cannot even be built.  NVSHMEM has none of them.
constexpr StringLiteral kOpenSHMEMUntypedNames[] = {"complexf", "complexd"};

/// What tells the models of the family apart in their tables: the prefix of
/// every callee, the map the raiser reads, and the spellings they have.
struct Names {
  StringLiteral prefix;
  const ApiCallToSpmdMap *entries;
  ArrayRef<TypeNameGroup> typeNameGroups;
  ArrayRef<StringLiteral> untypedNames;
};

const Names &namesOf(SHMEMFlavour flavour) {
  static const Names openshmem = {"shmem_", &shmemToSpmdOperands,
                                  kOpenSHMEMTypeNameGroups,
                                  kOpenSHMEMUntypedNames};
  static const Names nvshmem = {
      "nvshmem_", &nvshmemToSpmdOperands, kNVSHMEMTypeNameGroups, {}};
  switch (flavour) {
  case SHMEMFlavour::OpenSHMEM:
    return openshmem;
  case SHMEMFlavour::NVSHMEM:
    return nvshmem;
  }
  llvm_unreachable("unknown SHMEMFlavour");
}

/// The spellings of \p token's type, in the order the lowering prefers them,
/// or nothing when \p token is not a type name at all ("team", "global").
ArrayRef<StringLiteral> typeNameGroupOf(const Names &names, StringRef token) {
  for (const TypeNameGroup &group : names.typeNameGroups)
    for (StringLiteral name : group.names)
      if (!name.empty() && name == token)
        return ArrayRef<StringLiteral>(group.names);
  return {};
}

/// The type name in \p callee, which both models always put in the second
/// underscore-separated position (shmem_int_atomic_fetch_add), and the callee
/// around it.
bool splitTypeName(const Names &names, StringRef callee, StringRef &before,
                   StringRef &token, StringRef &after) {
  StringRef rest = callee;
  if (!rest.consume_front(names.prefix))
    return false;
  size_t end = rest.find('_');
  if (end == StringRef::npos)
    return false;
  token = rest.take_front(end);
  if (typeNameGroupOf(names, token).empty())
    return false;
  before = names.prefix;
  after = rest.substr(end);
  return true;
}

// Calls the lowering never picks, whatever their type name, and why.
struct RejectedCall {
  /// What the callee ends with, after its type name.
  StringLiteral suffix;
  StringLiteral reason;
};

constexpr RejectedCall kNeverCanonicalSuffixes[] = {
    // shmem_<T>_p(dest, value, pe) and shmem_<T>_atomic_set(dest, value, pe)
    // take the element by value.  The raiser reads that argument into the
    // operation's origin address because there is nowhere else to put it --
    // which does not verify -- so inverting it would pass an address where
    // the call wants a value.
    {"_p", "takes the element by value"},
    {"_atomic_set", "takes the element by value"},
    // The same transfer as shmem_<T>_get of one element.  Keeping the general
    // form means a raised get lowers back to the call it came from.
    {"_g", "is shmem_<T>_get of a single element"},
    // shmem_<T>_collect and shmem_<T>_fcollect raise to the same operation;
    // an allgather moves the same number of elements from every PE, which is
    // what fcollect is.
    {"_collect", "is the variable-length form of shmem_<T>_fcollect"},
    // spmd.swap records its form in $isNonBlocking rather than $isBlocking,
    // and the two calls are otherwise identical, so the blocking one is taken.
    {"_atomic_swap_nbi", "is the non-blocking form of a call the dialect "
                         "cannot tell apart from it"},
};

// shmem_put8 moves the bytes shmem_char_put does, with no more to say about
// their signedness, so the lowering prefers the type name.  The wider ones
// have no type name that leaves signedness open: shmem_put32 is the call for
// an i32.
constexpr StringLiteral kDuplicateSizes[] = {"8"};

bool isDuplicateSizeCallee(StringRef callee) {
  StringRef name = callee;
  name.consume_back("_nbi");
  size_t digits = name.size();
  while (digits > 0 && llvm::isDigit(name[digits - 1]))
    --digits;
  return llvm::is_contained(kDuplicateSizes, name.substr(digits));
}

bool isCanonicalCallee(const Names &names, StringRef callee) {
  // The routines the lowering emits from patterns of its own; leaving them out
  // of the table keeps two patterns from claiming the same operation.
  StringRef unprefixed = callee;
  if (unprefixed.consume_front(names.prefix) &&
      llvm::is_contained({kBarrierAll, kQuiet, kFence}, unprefixed))
    return false;

  // Checked on the name itself: an untyped name is in no group, so
  // splitTypeName does not see it as a type name at all.
  for (StringLiteral untyped : names.untypedNames)
    if (callee.starts_with((names.prefix + untyped + "_").str()))
      return false;

  StringRef before, token, after;
  if (!splitTypeName(names, callee, before, token, after))
    return !isDuplicateSizeCallee(callee);
  for (const RejectedCall &rejected : kNeverCanonicalSuffixes)
    if (after == rejected.suffix)
      return false;

  // A spelling the lowering prefers, that the map has a call of.
  for (StringLiteral preferred : typeNameGroupOf(names, token)) {
    if (preferred.empty())
      continue;
    if (preferred == token)
      return true;
    if (names.entries->count((before + preferred + after).str()))
      return false;
  }
  return true;
}

// TableSpec takes plain function pointers, so each model gets one.
bool isCanonicalOpenSHMEMCallee(StringRef callee, const APICallEntry &) {
  return isCanonicalCallee(namesOf(SHMEMFlavour::OpenSHMEM), callee);
}
bool isCanonicalNVSHMEMCallee(StringRef callee, const APICallEntry &) {
  return isCanonicalCallee(namesOf(SHMEMFlavour::NVSHMEM), callee);
}

//===----------------------------------------------------------------------===//
// One reduction operation, spelled by two raisers
//===----------------------------------------------------------------------===//

struct ReductionName {
  StringLiteral shmem;
  StringLiteral mpi;
};

// The dialect keeps a reduction operation under the name the model it was
// raised from gives it, so a program translated from MPI arrives spelling
// MPI_SUM "SUM" where OpenSHMEM's own raiser would have injected "sum".  The
// two mean the same operation for every reduction OpenSHMEM has.  NVSHMEM
// spells its reductions as OpenSHMEM does.
//
// MPI's logical LAND, LOR and LXOR are deliberately absent: OpenSHMEM's and,
// or and xor are bitwise, and a bitwise reduction of the values 1 and 2 is not
// the logical one.
constexpr ReductionName kReductionNames[] = {
    {"sum", "SUM"},  {"prod", "PROD"}, {"max", "MAX"},  {"min", "MIN"},
    {"and", "BAND"}, {"or", "BOR"},    {"xor", "BXOR"},
};

/// Whether \p actual is the reduction operation \p expected is, under the
/// other raiser's spelling.
bool sameReductionOperation(Value actual, Operation *expected) {
  auto wanted = dyn_cast<ConstOpTypeOp>(expected);
  auto given = actual ? actual.getDefiningOp<ConstOpTypeOp>() : nullptr;
  if (!wanted || !given)
    return false;
  for (const ReductionName &name : kReductionNames)
    if (name.shmem == wanted.getStringAttr())
      return given.getStringAttr() == name.mpi;
  return false;
}

//===----------------------------------------------------------------------===//
// What the diagnostics call an operand OpenSHMEM has no parameter for
//===----------------------------------------------------------------------===//

constexpr OperandNote kExtraOperandNotes[] = {
    {"spmd.commSplitStrided",
     "carries a team configuration, which the lowering does not pass"},
};

constexpr OperandNote kOpenSHMEMMissingOperandNotes[] = {
    {"spmd.barrier", "has no team; OpenSHMEM needs one"},
};
constexpr OperandNote kNVSHMEMMissingOperandNotes[] = {
    {"spmd.barrier", "has no team; NVSHMEM needs one"},
};

//===----------------------------------------------------------------------===//
// The OpenSHMEM ABI of one argument
//===----------------------------------------------------------------------===//

bool isPointerDeclarator(StringRef decl) { return decl.contains('*'); }

// A parameter the operation feeds nothing into, and what the lowering passes
// for it.  Nothing else may be left out: shmem_<T>_atomic_fetch_add's addend
// is a parameter the CACTS map reads nothing into -- the dialect has nowhere
// to keep it -- and passing a zero for it would add nothing where the program
// added something.  Such a call is dropped from the table instead, which is
// what returning false below does.
struct OmittableParameter {
  StringLiteral name;
  SHMEMArgKind kind;
};

constexpr OmittableParameter kOmittableParameters[] = {
    // shmem_team_split_strided's optional shmem_team_config_t and the mask
    // saying which of its fields to read.  spmd.commSplitStrided carries the
    // configuration as an optional operand the lowering does not pass on.
    {"config", SHMEMArgKind::NullPointer},
    {"config_mask", SHMEMArgKind::ZeroI64},
};

bool classifySHMEMArg(ArgSlot &slot, CallDescription &) {
  auto set = [&](SHMEMArgKind kind) {
    slot.abiKind = static_cast<int>(kind);
    return true;
  };

  switch (slot.source) {
  case ArgSource::Result:
    // The two out-parameters the maps have: a team, which is a pointer in
    // OpenSHMEM and an int32_t in NVSHMEM, and the granted thread level,
    // which is an int.
    if (slot.kind == "TEAM")
      return set(SHMEMArgKind::OutTeam);
    if (slot.kind == "THREAD_LEVEL") {
      slot.optional = OptionalResult::ThreadLevel;
      return set(SHMEMArgKind::OutInt);
    }
    return false;

  case ArgSource::Operand:
    // Every handle of this ABI is a pointer or an integer the dialect already
    // holds in the right width, so the converted operand goes straight in.
    return set(SHMEMArgKind::In);

  case ArgSource::Unused:
    for (const OmittableParameter &omittable : kOmittableParameters)
      if (omittable.name == slot.name &&
          isPointerDeclarator(slot.cType) ==
              (omittable.kind == SHMEMArgKind::NullPointer))
        return set(omittable.kind);
    return false;
  }
  llvm_unreachable("unknown ArgSource");
}

StringRef shmemKindName(int abiKind) {
  switch (static_cast<SHMEMArgKind>(abiKind)) {
  case SHMEMArgKind::In:
    return "in";
  case SHMEMArgKind::OutTeam:
    return "outTeam";
  case SHMEMArgKind::OutInt:
    return "outInt";
  case SHMEMArgKind::NullPointer:
    return "NULL";
  case SHMEMArgKind::ZeroI32:
    return "0";
  case SHMEMArgKind::ZeroI64:
    return "0L";
  }
  llvm_unreachable("unknown SHMEMArgKind");
}

} // namespace

namespace {

TableSpec makeTableSpec(StringRef model, StringRef displayName,
                        const ApiCallToSpmdMap *entries,
                        bool (*isCanonical)(StringRef, const APICallEntry &),
                        ArrayRef<OperandNote> missingOperandNotes) {
  return {
      model,
      displayName,
      entries,
      /*classify=*/classifySHMEMArg,
      isCanonical,
      // Neither model's non-blocking calls hand out a request at all, so
      // nothing but USIS's execution.blocking tells the two forms apart.
      /*blockingForm=*/BlockingForm::Execution,
      // A request result of a call that writes none becomes the lowering's
      // null request, which spmd.wait turns into the model's quiet.  What
      // neither has a non-blocking form of at all -- the collectives -- is
      // reported by the lowering's check instead.
      /*allowsRequestNull=*/true,
      /*alsoSameInjected=*/sameReductionOperation,
      /*contiguousOnlyNote=*/"",
      /*extraOperandNotes=*/kExtraOperandNotes,
      missingOperandNotes,
  };
}

} // namespace

const TableSpec &getSHMEMTableSpec(SHMEMFlavour flavour) {
  static const TableSpec openshmem =
      makeTableSpec("shmem", "OpenSHMEM", &shmemToSpmdOperands,
                    isCanonicalOpenSHMEMCallee, kOpenSHMEMMissingOperandNotes);
  static const TableSpec nvshmem =
      makeTableSpec("nvshmem", "NVSHMEM", &nvshmemToSpmdOperands,
                    isCanonicalNVSHMEMCallee, kNVSHMEMMissingOperandNotes);
  switch (flavour) {
  case SHMEMFlavour::OpenSHMEM:
    return openshmem;
  case SHMEMFlavour::NVSHMEM:
    return nvshmem;
  }
  llvm_unreachable("unknown SHMEMFlavour");
}

ArrayRef<CallDescription> getSHMEMLoweringTable(SHMEMFlavour flavour) {
  return getLoweringTable(getSHMEMTableSpec(flavour));
}

const CallDescription *selectSHMEMCall(SHMEMFlavour flavour, Operation *op,
                                       OpBuilder &builder,
                                       SmallVectorImpl<std::string> &reasons) {
  return selectCall(getSHMEMTableSpec(flavour), op, builder, reasons);
}

void printSHMEMLoweringTable(SHMEMFlavour flavour, raw_ostream &os,
                             OpBuilder &builder, bool canonicalOnly,
                             StringRef filter) {
  printLoweringTable(getSHMEMTableSpec(flavour), os, builder, shmemKindName,
                     canonicalOnly, filter);
}

} // namespace mlir::spmd
