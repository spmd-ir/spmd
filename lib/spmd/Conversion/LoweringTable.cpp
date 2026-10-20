// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LoweringTable.cpp - A lowering read off the raiser -----------------===//

#include "LoweringTable.h"
#include "LowerSPMDCommon.h"

#include "spmd/SPMDOps.h"
#include "spmd/SPMDTypes.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>

namespace mlir::spmd {

namespace {

//===----------------------------------------------------------------------===//
// Reading one USIS entry
//===----------------------------------------------------------------------===//

std::string paramString(const llvm::json::Object *param, StringRef key) {
  return getParamString(param, key).value_or("");
}

/// Whether the parameter has the language binding \p name -- "c" for the
/// classic C prototype, "c_large" for the MPI 4 one that takes every count as
/// an MPI_Count.
bool hasParameterBinding(const llvm::json::Object *param, StringRef name) {
  const llvm::json::Array *bindings = param->getArray("parameter_bindings");
  if (!bindings)
    return false;
  for (const llvm::json::Value &binding : *bindings)
    if (binding.getAsString() == name)
      return true;
  return false;
}

/// The parameters of \p entry that the C binding has, in prototype order.
/// USIS lists the Fortran-only ierror with the others, and a call's C argument
/// index -- what the CACTS map counts in -- skips it.
SmallVector<const llvm::json::Object *>
getCParameters(const llvm::json::Object *entry) {
  SmallVector<const llvm::json::Object *> params;
  const llvm::json::Array *all = entry->getArray("parameters");
  if (!all)
    return params;
  for (const llvm::json::Value &value : *all) {
    const llvm::json::Object *param = value.getAsObject();
    if (param && hasParameterBinding(param, "c"))
      params.push_back(param);
  }
  return params;
}

/// The declarators of the C prototype's parameters, in order.  USIS gives MPI
/// each parameter's C type in binding_type.c and OpenSHMEM none of them, so
/// the C binding's whole signature is the second place to read it from.
SmallVector<std::string> getSignatureDeclarators(StringRef signature) {
  SmallVector<std::string> declarators;
  size_t open = signature.find('(');
  size_t close = signature.rfind(')');
  if (open == StringRef::npos || close == StringRef::npos || close < open)
    return declarators;
  StringRef list = signature.slice(open + 1, close).trim();
  if (list.empty() || list == "void")
    return declarators;

  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i <= list.size(); ++i) {
    char c = i < list.size() ? list[i] : ',';
    if (c == '(' || c == '[')
      ++depth;
    else if (c == ')' || c == ']')
      --depth;
    else if (c == ',' && depth == 0) {
      declarators.push_back(list.slice(start, i).trim().str());
      start = i + 1;
    }
  }
  return declarators;
}

std::string getReturnString(const llvm::json::Object *entry, StringRef key) {
  const llvm::json::Object *ret = entry->getObject("return");
  if (!ret)
    return "";
  if (key == "c") {
    const llvm::json::Object *bindings = ret->getObject("binding_type");
    if (!bindings)
      return "";
    std::optional<StringRef> c = bindings->getString("c");
    return c ? c->str() : std::string();
  }
  std::optional<StringRef> value = ret->getString(key);
  return value ? value->str() : std::string();
}

bool isExecutionBlocking(const llvm::json::Object *entry) {
  const llvm::json::Object *execution = entry->getObject("execution");
  if (!execution)
    return true;
  return execution->getBoolean("blocking").value_or(true);
}

//===----------------------------------------------------------------------===//
// Inverting one CACTS entry
//===----------------------------------------------------------------------===//

/// The description of \p callee, or nothing when the entry uses a form of the
/// CACTS map this inversion does not cover.  A new one shows up as a call the
/// lowering does not offer, not as a wrong call.
std::optional<CallDescription> deriveCall(const TableSpec &spec,
                                          StringRef callee,
                                          const APICallEntry &entry,
                                          const llvm::json::Object *usis) {
  SmallVector<const llvm::json::Object *> params = getCParameters(usis);
  SmallVector<std::string> declarators;
  {
    const llvm::json::Object *bindings = usis->getObject("bindings");
    const llvm::json::Object *c = bindings ? bindings->getObject("c") : nullptr;
    std::optional<StringRef> signature =
        c ? c->getString("signature") : std::nullopt;
    if (signature)
      declarators = getSignatureDeclarators(*signature);
  }

  CallDescription call;
  call.callee = callee;
  call.spmdOpName = entry.spmdOpName;
  call.attributes = entry.newAttributes;
  call.numOperands = entry.operands.size();
  call.canonical = spec.isCanonical(callee, entry);
  call.blocking = isExecutionBlocking(usis);
  call.returnKind = getReturnString(usis, "kind");
  call.returnCType = getReturnString(usis, "c");

  // C argument index -> the operands that read it, and the result written to
  // it.  Both are positions in the entry's own lists, which are the positions
  // of the operation's operands and results: the raiser fills newOperands[i]
  // from entry.operands[i] and drops the absent ones.
  SmallVector<SmallVector<int>> operandsOf(params.size());
  SmallVector<int> resultOf(params.size(), -1);
  for (auto [index, operand] : llvm::enumerate(entry.operands)) {
    if (operand.apiCallResIdx) {
      // The raiser gave the operation storage holding the call's return value.
      // Only one return value exists, so only index 0 inverts.
      if (*operand.apiCallResIdx != 0 || call.returnOperand >= 0)
        return std::nullopt;
      call.returnOperand = static_cast<int>(index);
      continue;
    }
    if (!operand.apiCallArgIdx) {
      if (!operand.functionForDefaultValue)
        return std::nullopt;
      call.defaults.push_back({static_cast<int>(index), operand.name,
                               operand.functionForDefaultValue});
      continue;
    }
    if (*operand.apiCallArgIdx >= params.size())
      return std::nullopt;
    operandsOf[*operand.apiCallArgIdx].push_back(static_cast<int>(index));
  }
  for (auto [index, result] : llvm::enumerate(entry.results)) {
    if (result.apiCallResultIdx) {
      if (*result.apiCallResultIdx != 0 || call.returnResult >= 0)
        return std::nullopt;
      call.returnResult = static_cast<int>(index);
      continue;
    }
    if (!result.apiCallArgIdx || *result.apiCallArgIdx >= params.size())
      return std::nullopt;
    resultOf[*result.apiCallArgIdx] = static_cast<int>(index);
  }

  // Two operands on one C argument: the call has one where the operation has
  // two, so the two have to be the same value.
  for (const SmallVector<int> &sharing : operandsOf)
    for (size_t i = 1; i < sharing.size(); ++i)
      call.aliases.push_back({sharing[0], sharing[i],
                              entry.operands[sharing[0]].name,
                              entry.operands[sharing[i]].name});

  for (auto [index, param] : llvm::enumerate(params)) {
    ArgSlot slot;
    slot.name = paramString(param, "name");
    slot.kind = getSpmdParamKind(param).value_or("");
    slot.direction = paramString(param, "direction");
    slot.isArray = getParamString(param, "length").has_value();
    slot.isPointer = param->getBoolean("pointer").value_or(false);
    slot.hasLargeCountBinding = hasParameterBinding(param, "c_large");
    const llvm::json::Object *bindings = param->getObject("binding_type");
    std::optional<StringRef> cType =
        bindings ? bindings->getString("c") : std::nullopt;
    if (cType)
      slot.cType = cType->str();
    else if (index < declarators.size())
      slot.cType = declarators[index];

    if (resultOf[index] >= 0) {
      slot.source = ArgSource::Result;
      slot.result = resultOf[index];
    } else if (!operandsOf[index].empty()) {
      slot.source = ArgSource::Operand;
      slot.operand = operandsOf[index].front();
    }
    if (!spec.classify(slot, call))
      return std::nullopt;
    call.args.push_back(std::move(slot));
  }

  call.trailingOptionalOperand =
      call.numOperands > 0 && llvm::any_of(call.args, [&](const ArgSlot &slot) {
        return slot.optionalOperand &&
               slot.operand == static_cast<int>(call.numOperands) - 1;
      });
  return call;
}

//===----------------------------------------------------------------------===//
// The table
//===----------------------------------------------------------------------===//

struct Table {
  SmallVector<CallDescription, 0> calls;
  llvm::StringMap<SmallVector<const CallDescription *>> byOp;
};

/// The table of \p spec, built on first use.  byOp points into calls, so the
/// two are filled in place rather than built and moved: a move would leave the
/// pointers behind.  The CACTS map is a StringMap, whose order is its hash
/// order, so the callees are sorted first and the table reads the same
/// whatever the run.
const Table &getTable(const TableSpec &spec) {
  static llvm::DenseMap<const TableSpec *, std::unique_ptr<Table>> tables;
  std::unique_ptr<Table> &slot = tables[&spec];
  if (slot)
    return *slot;
  slot = std::make_unique<Table>();
  Table &table = *slot;

  const ApiEntryIndex *index = getJsonIndex(spec.model);
  if (!index)
    return table;

  SmallVector<StringRef> callees;
  for (const auto &entry : *spec.entries)
    callees.push_back(entry.getKey());
  llvm::sort(callees);

  for (StringRef callee : callees) {
    auto usis = index->find(toLowerStr(callee));
    if (usis == index->end())
      continue;
    std::optional<CallDescription> call = deriveCall(
        spec, callee, spec.entries->find(callee)->second, usis->second);
    if (call)
      table.calls.push_back(std::move(*call));
  }
  for (const CallDescription &call : table.calls)
    if (call.canonical)
      table.byOp[call.spmdOpName].push_back(&call);
  return table;
}

} // namespace

ArrayRef<CallDescription> getLoweringTable(const TableSpec &spec) {
  return getTable(spec).calls;
}

ArrayRef<const CallDescription *> getCallsFor(const TableSpec &spec,
                                              StringRef spmdOpName) {
  const Table &table = getTable(spec);
  auto it = table.byOp.find(spmdOpName);
  if (it == table.byOp.end())
    return {};
  return it->second;
}

bool hasSlotOperand(Operation *op, const ArgSlot &slot) {
  return slot.operand >= 0 &&
         slot.operand < static_cast<int>(op->getNumOperands());
}

bool producesSlotResult(Operation *op, const ArgSlot &slot) {
  if (slot.source != ArgSource::Result)
    return false;
  switch (slot.optional) {
  case OptionalResult::None:
    return true;
  case OptionalResult::Status:
    return llvm::any_of(op->getResultTypes(),
                        [](Type type) { return isa<StatusType>(type); });
  case OptionalResult::ThreadLevel:
    return llvm::any_of(op->getResultTypes(),
                        [](Type type) { return isa<ThreadLevelType>(type); });
  }
  llvm_unreachable("unknown OptionalResult");
}

namespace {

/// How many of the operation's results \p call fills.
unsigned producedResults(Operation *op, const CallDescription &call) {
  unsigned produced = call.returnResult >= 0 ? 1 : 0;
  for (const ArgSlot &slot : call.args)
    produced += producesSlotResult(op, slot);
  return produced;
}

} // namespace

bool needsRequestNull(const TableSpec &spec, Operation *op,
                      const CallDescription &call) {
  if (!spec.allowsRequestNull || call.producesRequest ||
      op->getNumResults() < 2)
    return false;
  return producedResults(op, call) + 2 == op->getNumResults() &&
         isa<ReqType>(op->getResult(op->getNumResults() - 2).getType());
}

bool matchesAttributes(const TableSpec &spec, Operation *op,
                       const CallDescription &call, OpBuilder &builder) {
  // Only the attributes that tell this call apart from the operation's other
  // candidates are conditions.  spmd.put carries $isAtomic whichever call it
  // came from, so requiring it would reject an operation of another model's
  // shape for no reason; $isBuffered separates MPI_Bsend from MPI_Send, so it
  // is one.
  ArrayRef<const CallDescription *> others = getCallsFor(spec, call.spmdOpName);
  for (const auto &attribute : call.attributes) {
    NamedAttribute required = attribute(builder);
    bool discriminates = false;
    for (const CallDescription *other : others) {
      if (other == &call)
        continue;
      for (const auto &otherAttribute : other->attributes) {
        NamedAttribute candidate = otherAttribute(builder);
        if (candidate.getName() == required.getName() &&
            candidate.getValue() != required.getValue())
          discriminates = true;
      }
    }
    if (!discriminates)
      continue;
    if (op->getAttr(required.getName()) != required.getValue())
      return false;
  }
  return true;
}

namespace {

StringRef lookupNote(ArrayRef<OperandNote> notes, StringRef spmdOpName) {
  for (const OperandNote &note : notes)
    if (note.spmdOpName == spmdOpName)
      return note.text;
  return StringRef();
}

/// The value the raiser would have injected for \p check, built into a block
/// of its own so that nothing of it reaches the module.
class ScratchValue {
public:
  ScratchValue(MLIRContext *ctx, const DefaultCheck &check, int model)
      : block(std::make_unique<Block>()) {
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(block.get());
    value = check.value(builder, UnknownLoc::get(ctx), model);
  }
  Operation *getDefiningOp() const {
    return value ? value.getDefiningOp() : nullptr;
  }
  ~ScratchValue() { block->clear(); }

private:
  std::unique_ptr<Block> block;
  Value value;
};

/// Whether \p actual is a datatype of unknown signedness -- a signless
/// integer -- and \p expected the datatype of the same width that records one.
bool isUnknownSignednessOf(Operation *actual, Operation *expected) {
  auto actualDatatype = dyn_cast_or_null<ConstDatatypeOp>(actual);
  auto expectedDatatype = dyn_cast<ConstDatatypeOp>(expected);
  if (!actualDatatype || !expectedDatatype)
    return false;
  auto actualType = dyn_cast<IntegerType>(actualDatatype.getTypeAttr());
  auto expectedType = dyn_cast<IntegerType>(expectedDatatype.getTypeAttr());
  return actualType && expectedType && actualType.isSignless() &&
         actualType.getWidth() == expectedType.getWidth();
}

/// Whether \p actual is the value \p expected is.  An injected constant is
/// compared by its value, not by the operation holding it: a stride of 1 is a
/// stride of 1 whether the program spells it arith.constant, llvm.mlir.constant
/// or anything else constant-like.  An injected handle is compared by the
/// operation that produces it, ignoring the model tag every spmd operation
/// carries.  With \p anySignedness, an integer datatype of unknown signedness
/// is also the signed and the unsigned datatype of its width.
///
/// A value with no defining operation -- a block argument -- is not the
/// injected one, because nothing here can show that it is.
bool sameInjectedValue(const TableSpec &spec, Value actual, Operation *expected,
                       bool anySignedness) {
  if (!expected)
    return false;

  APInt injected;
  if (matchPattern(expected, m_ConstantInt(&injected))) {
    APInt value;
    return actual && matchPattern(actual, m_ConstantInt(&value)) &&
           APInt::isSameValue(value, injected);
  }

  Operation *actualOp = actual ? actual.getDefiningOp() : nullptr;
  if (anySignedness && isUnknownSignednessOf(actualOp, expected))
    return true;
  if (actualOp && actualOp->getName() == expected->getName()) {
    bool same = true;
    for (NamedAttribute attribute : expected->getAttrs()) {
      if (attribute.getName() == "usedModel")
        continue;
      if (actualOp->getAttr(attribute.getName()) != attribute.getValue())
        same = false;
    }
    if (same && actualOp->getNumOperands() == 0 &&
        expected->getNumOperands() == 0)
      return true;
  }
  return spec.alsoSameInjected && spec.alsoSameInjected(actual, expected);
}

/// "recvStride" -> "receive stride": the operand's name in the CACTS entry,
/// spelled the way a diagnostic reads.
std::string describeOperand(StringRef name) {
  SmallVector<std::string> words;
  std::string word;
  for (char c : name) {
    if (llvm::isUpper(c) && !word.empty()) {
      words.push_back(word);
      word.clear();
    }
    word.push_back(llvm::toLower(c));
  }
  if (!word.empty())
    words.push_back(word);
  for (std::string &w : words) {
    if (w == "recv")
      w = "receive";
    else if (w == "buf")
      w = "buffer";
    else if (w == "comm")
      w = "communicator";
  }
  return llvm::join(words, " ");
}

StringRef articleFor(StringRef phrase) {
  return !phrase.empty() && StringRef("aeiou").contains(phrase.front()) ? "an"
                                                                        : "a";
}

/// How the diagnostic names the value the raiser would have injected.
std::string describeInjectedValue(Operation *expected) {
  if (!expected)
    return "other than the one the raiser injects";
  APInt constant;
  if (matchPattern(expected, m_ConstantInt(&constant)))
    return ("other than " + llvm::Twine(constant.getSExtValue())).str();
  if (isa<CommWorldOp>(expected))
    return "other than the world communicator";
  if (auto level = dyn_cast<ConstThreadLevelOp>(expected))
    return ("other than " + level.getLevel()).str();
  if (auto datatype = dyn_cast<ConstDatatypeOp>(expected)) {
    std::string type;
    llvm::raw_string_ostream os(type);
    os << datatype.getTypeAttr();
    return "other than " + type;
  }
  if (auto reduceOp = dyn_cast<ConstOpTypeOp>(expected))
    return ("other than " + reduceOp.getStringAttr()).str();
  return ("other than the " + expected->getName().getStringRef() +
          " the raiser injects")
      .str();
}

/// Whether the dialect holds a value of \p type as an address rather than as
/// the datum itself.
bool isAddressLikeType(Type type) {
  return isa<MemRefType>(type) || isa<LLVM::LLVMPointerType>(type) ||
         isa<WinType>(type);
}

/// Whether every argument the call takes by value is one the operation has the
/// datum for.  OpenSHMEM's shmem_<T>_p and its atomic stores take the element
/// itself, and the raiser puts that argument in an operand the dialect
/// declares as an address, so inverting it would pass the address as the
/// value.
bool passesValuesByValue(Operation *op, const CallDescription &call,
                         SmallVectorImpl<std::string> *reasons) {
  bool ok = true;
  for (const ArgSlot &slot : call.args) {
    if (slot.source != ArgSource::Operand || slot.kind != "SCALAR" ||
        slot.isPointer || !hasSlotOperand(op, slot))
      continue;
    if (!isAddressLikeType(op->getOperand(slot.operand).getType()))
      continue;
    ok = false;
    if (reasons)
      reasons->push_back((llvm::Twine("lowers to ") + call.callee +
                          ", which takes its " + slot.name +
                          " by value, and the operation carries only its "
                          "address")
                             .str());
  }
  return ok;
}

/// How the diagnostic names the value the operation has, when no call of the
/// table takes it, or an empty string when there is nothing short to say.
std::string describeActualValue(Value actual) {
  Operation *def = actual ? actual.getDefiningOp() : nullptr;
  if (!def)
    return "";
  APInt constant;
  if (matchPattern(actual, m_ConstantInt(&constant)))
    return llvm::Twine(constant.getSExtValue()).str();
  if (auto datatype = dyn_cast<ConstDatatypeOp>(def)) {
    std::string type;
    llvm::raw_string_ostream os(type);
    os << datatype.getTypeAttr();
    return type;
  }
  if (auto reduceOp = dyn_cast<ConstOpTypeOp>(def))
    return ("'" + reduceOp.getStringAttr() + "'").str();
  return "";
}

/// Whether \p op has the operands \p call reads, and the results its
/// out-parameters fill.  \p requestNull says the operation keeps a request
/// result the call does not write.
bool fitsShape(const TableSpec &spec, Operation *op,
               const CallDescription &call, bool &requestNull) {
  if (op->getNumResults() == 0)
    return false;
  unsigned operands = op->getNumOperands();
  if (operands != call.numOperands &&
      !(call.trailingOptionalOperand && operands + 1 == call.numOperands))
    return false;

  unsigned produced = producedResults(op, call);

  // Every operation of the table ends in its error code.
  unsigned wanted = op->getNumResults() - 1;
  requestNull = false;
  if (produced == wanted)
    return true;
  if (spec.allowsRequestNull && produced + 1 == wanted &&
      !call.producesRequest && wanted >= 1 &&
      isa<ReqType>(op->getResult(wanted - 1).getType())) {
    requestNull = true;
    return true;
  }
  return false;
}

/// The injected operands of \p call that \p op does not have, and how the
/// diagnostic describes the value each should have been.
SmallVector<std::pair<const DefaultCheck *, std::string>>
missedDefaults(const TableSpec &spec, Operation *op,
               const CallDescription &call, int model, bool anySignedness) {
  SmallVector<std::pair<const DefaultCheck *, std::string>> missed;
  for (const DefaultCheck &check : call.defaults) {
    if (check.operand >= static_cast<int>(op->getNumOperands()))
      continue;
    ScratchValue expected(op->getContext(), check, model);
    if (!sameInjectedValue(spec, op->getOperand(check.operand),
                           expected.getDefiningOp(), anySignedness))
      missed.push_back(
          {&check, describeInjectedValue(expected.getDefiningOp())});
  }
  return missed;
}

bool satisfiesDefaults(const TableSpec &spec, Operation *op,
                       const CallDescription &call, int model,
                       bool anySignedness,
                       SmallVectorImpl<std::string> *reasons) {
  bool ok = true;
  for (const DefaultCheck &check : call.defaults) {
    if (check.operand >= static_cast<int>(op->getNumOperands()))
      return false;
    ScratchValue expected(op->getContext(), check, model);
    if (sameInjectedValue(spec, op->getOperand(check.operand),
                          expected.getDefiningOp(), anySignedness))
      continue;
    ok = false;
    if (!reasons)
      continue;
    std::string phrase = describeOperand(check.name);
    std::string reason =
        (llvm::Twine("has ") + articleFor(phrase) + " " + phrase + " " +
         describeInjectedValue(expected.getDefiningOp()))
            .str();
    // A model with no strided transfer at all says so once.
    if (check.name.ends_with("Stride") && !spec.contiguousOnlyNote.empty())
      reason += spec.contiguousOnlyNote;
    reasons->push_back(std::move(reason));
  }
  return ok;
}

} // namespace

/// Whether \p lhs and \p rhs are the same value, looking through a widening or
/// narrowing both apply to the same value.  The raiser converts every operand
/// it reads on its own, so a program passing one count as both MPI_Put's origin
/// and target count raises to two arith.extui of it; the call that takes the
/// count once has to see them as one.  Constants compare by value.  The same
/// goes for buffers: MPI_Bcast's one buffer raises to a send and a receive
/// buffer, each its own enzymexla.pointer2memref view of the one address.
///
/// Exposed (LoweringTable.h) for a lowering that has to ask the same question
/// of two of its own operation's operands outside an AliasCheck -- see
/// LowerSPMDToMPI.cpp's handling of spmd.sendrecv.
bool sameOperandValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;
  APInt lhsConstant, rhsConstant;
  if (matchPattern(lhs, m_ConstantInt(&lhsConstant)) &&
      matchPattern(rhs, m_ConstantInt(&rhsConstant)))
    return APInt::isSameValue(lhsConstant, rhsConstant);
  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef || lhsDef->getName() != rhsDef->getName() ||
      !isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::IndexCastOp,
           LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp,
           enzymexla::Pointer2MemrefOp>(lhsDef))
    return false;
  return sameOperandValue(lhsDef->getOperand(0), rhsDef->getOperand(0));
}

namespace {

bool satisfiesAliases(Operation *op, const CallDescription &call,
                      SmallVectorImpl<std::string> *reasons) {
  bool ok = true;
  for (const AliasCheck &alias : call.aliases) {
    if (alias.first >= static_cast<int>(op->getNumOperands()) ||
        alias.second >= static_cast<int>(op->getNumOperands()))
      return false;
    if (sameOperandValue(op->getOperand(alias.first),
                         op->getOperand(alias.second)))
      continue;
    ok = false;
    if (!reasons)
      continue;
    std::string first = describeOperand(alias.firstName);
    std::string second = describeOperand(alias.secondName);
    SmallVector<StringRef> firstWords, secondWords;
    StringRef(first).split(firstWords, ' ');
    StringRef(second).split(secondWords, ' ');
    reasons->push_back((llvm::Twine("has distinct ") + firstWords.front() +
                        " and " + secondWords.front() + " " +
                        secondWords.back() + "s; " + call.callee + " has one")
                           .str());
  }
  return ok;
}

/// The calls whose attributes \p op carries.
SmallVector<const CallDescription *>
matchingCalls(const TableSpec &spec, Operation *op, OpBuilder &builder) {
  SmallVector<const CallDescription *> matching;
  for (const CallDescription *call :
       getCallsFor(spec, op->getName().getStringRef()))
    if (matchesAttributes(spec, op, *call, builder))
      matching.push_back(call);
  return matching;
}

/// Whether the operation says it is the blocking form, however the dialect
/// spells it.  Most operations carry $isBlocking; spmd.swap carries
/// $isNonBlocking instead.
std::optional<bool> isBlockingForm(Operation *op) {
  if (auto blocking = op->getAttrOfType<BoolAttr>("isBlocking"))
    return blocking.getValue();
  if (auto nonBlocking = op->getAttrOfType<BoolAttr>("isNonBlocking"))
    return !nonBlocking.getValue();
  return std::nullopt;
}

/// Those of \p calls in the form the operation asks for.
SmallVector<const CallDescription *>
callsInBlockingForm(const TableSpec &spec, Operation *op,
                    ArrayRef<const CallDescription *> calls) {
  std::optional<bool> blocking = isBlockingForm(op);
  if (!blocking)
    return SmallVector<const CallDescription *>(calls);
  SmallVector<const CallDescription *> inForm;
  for (const CallDescription *call : calls) {
    bool isBlockingCall = spec.blockingForm == BlockingForm::Request
                              ? !call->producesRequest
                              : call->blocking;
    if (isBlockingCall == *blocking)
      inForm.push_back(call);
  }
  return inForm.empty() ? SmallVector<const CallDescription *>(calls) : inForm;
}

/// Whether one of the datatypes \p call injects is an unsigned integer.
bool injectsUnsignedDatatype(MLIRContext *ctx, const CallDescription &call,
                             int model) {
  for (const DefaultCheck &check : call.defaults) {
    ScratchValue injected(ctx, check, model);
    auto datatype = dyn_cast_or_null<ConstDatatypeOp>(injected.getDefiningOp());
    if (datatype && datatype.getTypeAttr().isUnsignedInteger())
      return true;
  }
  return false;
}

} // namespace

const CallDescription *selectCall(const TableSpec &spec, Operation *op,
                                  OpBuilder &builder,
                                  SmallVectorImpl<std::string> &reasons) {
  std::optional<int> model = getUsedModel(op);
  SmallVector<const CallDescription *> matching =
      matchingCalls(spec, op, builder);
  SmallVector<const CallDescription *> intended =
      callsInBlockingForm(spec, op, matching);

  auto fittingCalls = [&](bool anySignedness) {
    auto fits = [&](const CallDescription *call) {
      bool requestNull = false;
      return call->unsupportedParameter.empty() &&
             fitsShape(spec, op, *call, requestNull) &&
             satisfiesDefaults(spec, op, *call, model.value_or(0),
                               anySignedness, nullptr) &&
             satisfiesAliases(op, *call, nullptr) &&
             passesValuesByValue(op, *call, nullptr);
    };
    SmallVector<const CallDescription *> fitting;
    llvm::copy_if(intended, std::back_inserter(fitting), fits);
    // $isBlocking and the results disagree.  The results decide: a call that
    // hands out a request the operation has no result for would leave it
    // uncompleted, and one that hands out none cannot fill a request result.
    // -convert-apiCalls-to-spmd reads $isBlocking off USIS's
    // execution.blocking, which calls MPI_Put non-blocking although it hands
    // out no request, so a raised MPI_Put lands here.
    if (fitting.empty() && intended.size() != matching.size())
      llvm::copy_if(matching, std::back_inserter(fitting), fits);
    return fitting;
  };

  SmallVector<const CallDescription *> fitting = fittingCalls(false);

  // A datatype of unknown signedness -- MPI_CHAR's i8 -- where the calls
  // inject a signed or an unsigned one: shmem_char_and_reduce does not exist.
  // Either moves the same bits and gives the same sum, so the signed one is
  // taken, as the MPI and NCCL datatype tables do.  Where signedness changes
  // the result, taking either would be a guess, and nothing is taken.
  bool anySignedness = !reducesBySignedness(op);
  if (fitting.empty()) {
    SmallVector<const CallDescription *> relaxed = fittingCalls(true);
    std::string signedness;
    if (!anySignedness && !relaxed.empty() &&
        reducesOverUnknownSignedness(op, &signedness)) {
      reasons.push_back(std::move(signedness));
      return nullptr;
    }
    if (anySignedness)
      fitting = std::move(relaxed);
    auto injectsUnsigned = [&](const CallDescription *call) {
      return injectsUnsignedDatatype(op->getContext(), *call,
                                     model.value_or(0));
    };
    if (!llvm::all_of(fitting, injectsUnsigned))
      llvm::erase_if(fitting, injectsUnsigned);
  }

  // The call that fixes more of the operation's operands describes fewer
  // operations, so it is the one meant: MPI_Init is MPI_Init_thread with the
  // level fixed to SINGLE, and both fit an operation that asks for SINGLE.
  if (fitting.size() > 1) {
    size_t mostDefaults = 0;
    for (const CallDescription *call : fitting)
      mostDefaults = std::max(mostDefaults, call->defaults.size());
    llvm::erase_if(fitting, [&](const CallDescription *call) {
      return call->defaults.size() < mostDefaults;
    });
  }

  if (fitting.size() == 1)
    return fitting.front();
  if (fitting.size() > 1) {
    // Two calls the operation cannot tell apart: the CACTS map grew an alias
    // the lowering's canonical-callee rule does not know about.  It is a fault
    // in the table rather than in the module, but it is still reported on the
    // operation, so that it comes with a location and the tool does not abort
    // in the middle of the check phase.
    std::string names;
    for (const CallDescription *call : fitting)
      names += (names.empty() ? "" : ", ") + call->callee.str();
    reasons.push_back((llvm::Twine("lowers to more than one ") +
                       spec.displayName + " call (" + names +
                       "); mark all but one of them non-canonical")
                          .str());
    return nullptr;
  }

  // Nothing fits.  The call the operation was meant for says why, when the
  // table can name the reason; otherwise the conversion reports it.  That is
  // the candidate the operation misses by the fewest injected operands: of
  // shmem_char_put and shmem_int_put for an i32 transfer with a displacement,
  // the one to blame is the displacement, not the element type.
  if (intended.empty())
    return nullptr;
  const CallDescription *closest = intended.front();
  size_t fewestMisses = SIZE_MAX;
  SmallVector<const CallDescription *> tied;
  for (const CallDescription *candidate : intended) {
    bool requestNull = false;
    if (!fitsShape(spec, op, *candidate, requestNull))
      continue;
    SmallVector<std::string> misses;
    satisfiesDefaults(spec, op, *candidate, model.value_or(0), anySignedness,
                      &misses);
    satisfiesAliases(op, *candidate, &misses);
    if (misses.size() < fewestMisses) {
      fewestMisses = misses.size();
      closest = candidate;
      tied.clear();
    } else if (misses.size() == fewestMisses &&
               candidate->defaults.size() > closest->defaults.size()) {
      // As when several fit: the call that fixes more operands is the one
      // meant, shmem_int_put rather than the strided shmem_int_iput.
      closest = candidate;
    }
    if (misses.size() == fewestMisses)
      tied.push_back(candidate);
  }

  // Several candidates miss by as little, each expecting a different value of
  // the same operand: shmem_int_max_reduce wants max, shmem_int_sum_reduce
  // sum.  Naming one of them would say the operation should have been that
  // one; what is true is that no call takes the value the operation has.
  if (tied.size() > 1) {
    DenseMap<int, std::string> firstExpectation;
    DenseMap<int, unsigned> missedBy;
    DenseSet<int> differs;
    for (const CallDescription *candidate : tied)
      for (const auto &[check, expectation] : missedDefaults(
               spec, op, *candidate, model.value_or(0), anySignedness)) {
        ++missedBy[check->operand];
        auto [it, inserted] =
            firstExpectation.try_emplace(check->operand, expectation);
        if (!inserted && it->second != expectation)
          differs.insert(check->operand);
      }
    bool reported = false;
    for (const DefaultCheck &check : tied.front()->defaults) {
      if (!differs.contains(check.operand) ||
          missedBy[check.operand] != tied.size())
        continue;
      std::string phrase = describeOperand(check.name);
      std::string actual = describeActualValue(op->getOperand(check.operand));
      reasons.push_back((llvm::Twine("has ") + articleFor(phrase) + " " +
                         phrase + (actual.empty() ? "" : " " + actual) +
                         " that no " + spec.displayName +
                         " call of this "
                         "shape takes")
                            .str());
      reported = true;
    }
    if (reported)
      return nullptr;
  }
  const CallDescription &call = *closest;
  unsigned operands = op->getNumOperands();
  if (operands > call.numOperands) {
    Value extra = op->getOperand(call.numOperands);
    if (isa<StreamType>(extra.getType()))
      reasons.push_back((llvm::Twine("carries a stream, which ") +
                         spec.displayName + " has no parameter for")
                            .str());
    else if (isa<CommConfigType>(extra.getType()))
      reasons.push_back((llvm::Twine("carries a communicator config, which ") +
                         call.callee + " has no parameter for")
                            .str());
    else if (StringRef note =
                 lookupNote(spec.extraOperandNotes, call.spmdOpName);
             !note.empty())
      reasons.push_back(note.str());
    return nullptr;
  }
  if (operands < call.numOperands && !call.trailingOptionalOperand) {
    if (StringRef note = lookupNote(spec.missingOperandNotes, call.spmdOpName);
        !note.empty())
      reasons.push_back(note.str());
    return nullptr;
  }
  if (!call.unsupportedParameter.empty()) {
    reasons.push_back((llvm::Twine("lowers to ") + call.callee +
                       ", which takes a " + call.unsupportedParameter +
                       " the operation does not carry")
                          .str());
    return nullptr;
  }
  satisfiesDefaults(spec, op, call, model.value_or(0), anySignedness, &reasons);
  satisfiesAliases(op, call, &reasons);
  passesValuesByValue(op, call, &reasons);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// The dump
//===----------------------------------------------------------------------===//

void printLoweringTable(const TableSpec &spec, raw_ostream &os,
                        OpBuilder &builder, StringRef (*kindName)(int),
                        bool canonicalOnly, StringRef filter) {
  for (const CallDescription &call : getLoweringTable(spec)) {
    if (canonicalOnly && !call.canonical)
      continue;
    if (!filter.empty() && !call.callee.contains(filter))
      continue;
    os << call.callee << " <- " << call.spmdOpName;
    if (!call.canonical)
      os << " [alias]";
    if (call.producesRequest)
      os << " [request]";
    for (const auto &attribute : call.attributes) {
      NamedAttribute named = attribute(builder);
      os << " [" << named.getName().getValue() << "=" << named.getValue()
         << "]";
    }
    os << "\n";
    for (const ArgSlot &slot : call.args) {
      os << "    " << slot.name << ": " << kindName(slot.abiKind);
      if (slot.operand >= 0)
        os << " $" << slot.operand;
      if (slot.optional != OptionalResult::None)
        os << " (optional)";
      os << "\n";
    }
    if (call.returnResult >= 0)
      os << "    return: result #" << call.returnResult << "\n";
    if (call.returnOperand >= 0)
      os << "    return: stored to $" << call.returnOperand << "\n";
    for (const DefaultCheck &check : call.defaults)
      os << "    $" << check.operand << " " << check.name << ": injected\n";
    for (const AliasCheck &alias : call.aliases)
      os << "    $" << alias.first << " " << alias.firstName << " == $"
         << alias.second << " " << alias.secondName << "\n";
  }
}

} // namespace mlir::spmd
