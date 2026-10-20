// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToMPI.cpp - Lower SPMD operations to MPI calls ------------===//
//
// Every operation tagged for MPI becomes a call to the MPI function it models,
// under the MPICH ABI of includesProgramModels/mpi/include: handles are ints,
// an MPI_Status is five ints passed by address, and the predefined handles are
// the ones in include/spmd/TargetABI/MPICH.h.
//
// No call is described here.  MPILoweringTable derives every one of them from
// the CACTS map the raiser reads and the USIS API description, so that the
// argument order exists once rather than once per direction; this file turns
// one of its descriptions into the call (MPICallArgs) and keeps what neither
// source knows: the MPICH handle values and the write-back.
//
//===----------------------------------------------------------------------===//

#include "LowerSPMDToMPI.h"
#include "LowerSPMDCommon.h"
#include "MPILoweringTable.h"

#include "spmd/SPMDPasses.h"
#include "spmd/TargetABI/MPICH.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

namespace mlir::spmd {

#define GEN_PASS_DEF_LOWERSPMDTOMPI
#include "spmd/SPMDPasses.h.inc"

namespace {

namespace mpich = abi::mpich;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// MPICH passes every handle as an int, so all scalar handle types become i32.
// An MPI_Status, and every array or buffer, is a bare pointer at the ABI.
class MPITypeConverter : public SPMDLoweringTypeConverter {
public:
  explicit MPITypeConverter(MLIRContext *ctx) : SPMDLoweringTypeConverter(ctx) {
    auto i32 = IntegerType::get(ctx, 32);
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    addConversion([i32](CommType) -> Type { return i32; });
    addConversion([i32](DatatypeType) -> Type { return i32; });
    addConversion([i32](OpType) -> Type { return i32; });
    addConversion([i32](ReqType) -> Type { return i32; });
    addConversion([i32](WinType) -> Type { return i32; });
    addConversion([i32](ErrorType) -> Type { return i32; });
    addConversion([i32](LockType) -> Type { return i32; });
    addConversion([i32](MsgType) -> Type { return i32; });
    addConversion([i32](ThreadLevelType) -> Type { return i32; });
    addConversion([ptr](StatusType) -> Type { return ptr; });
    addConversion([ptr](MemRefType) -> Type { return ptr; });
  }
};

//===----------------------------------------------------------------------===//
// Predefined handles
//===----------------------------------------------------------------------===//

std::optional<int32_t> handleOf(CommWorldOp) { return mpich::kCommWorld; }
std::optional<int32_t> handleOf(RequestNullOp) { return mpich::kRequestNull; }
std::optional<int32_t> handleOf(ConstDatatypeOp op) {
  return mpich::getDatatypeHandle(op.getTypeAttr());
}
std::optional<int32_t> handleOf(ConstOpTypeOp op) {
  return mpich::lookupHandle(mpich::kReduceOps, op.getStringAttr());
}
std::optional<int32_t> handleOf(ConstLockTypeOp op) {
  return mpich::lookupHandle(mpich::kLockTypes, op.getStringAttr());
}
std::optional<int32_t> handleOf(ConstThreadLevelOp op) {
  return mpich::lookupHandle(mpich::kThreadLevels, op.getLevel());
}

// A constant this pass would have no MPI handle for.  Checked for the whole
// module before the conversion starts, so each one is reported once and with
// its own reason, instead of as the conversion's "failed to legalize".
LogicalResult checkPredefinedHandle(Operation *op) {
  if (auto datatype = dyn_cast<ConstDatatypeOp>(op)) {
    if (!handleOf(datatype))
      return datatype.emitOpError()
             << "has no predefined MPI datatype for " << datatype.getTypeAttr();
  } else if (auto reduceOp = dyn_cast<ConstOpTypeOp>(op)) {
    if (!handleOf(reduceOp))
      return reduceOp.emitOpError()
             << "'" << reduceOp.getStringAttr()
             << "' is not a predefined MPI reduction operation";
  } else if (auto lockType = dyn_cast<ConstLockTypeOp>(op)) {
    if (!handleOf(lockType))
      return lockType.emitOpError()
             << "'" << lockType.getStringAttr() << "' is not an MPI lock type";
  } else if (auto level = dyn_cast<ConstThreadLevelOp>(op)) {
    if (!handleOf(level))
      return level.emitOpError() << "'" << level.getLevel()
                                 << "' is not an MPI thread support level";
  }
  return success();
}

// An operand the dialect has and MPI has no parameter for, one MPI needs and
// the operation has not, and an operand whose value the call fixes.  All three
// are what the CACTS entry says read backwards, so the table finds them:
// roundtrip's lowering dropped each of them, and the call then did something
// other than the operation said.  Before them, a MAX or MIN over an integer
// whose signedness is unknown: MPI_INT and MPI_UNSIGNED compare differently,
// and the datatype does not say which one the program meant.  MPI_CHAR's i8
// is the exception, because it lowers to MPI_CHAR, which does not say either.
LogicalResult checkExpressible(Operation *op) {
  OpBuilder builder(op->getContext());
  SmallVector<std::string> reasons;
  std::string signedness;
  if (reducesOverUnknownSignedness(op, &signedness, [](Type type) {
        return mpich::getExactDatatypeHandle(type).has_value();
      }))
    reasons.push_back(std::move(signedness));
  selectMPICall(op, builder, reasons);
  for (const std::string &reason : reasons)
    op->emitOpError() << reason;
  return failure(!reasons.empty());
}

bool hasOperationTaggedForMPI(ModuleOp module) {
  return module
      .walk([](Operation *op) {
        return isTaggedFor(op, ProgrammingModel::MPI) ? WalkResult::interrupt()
                                                      : WalkResult::advance();
      })
      .wasInterrupted();
}

LogicalResult checkPredefinedHandles(ModuleOp module) {
  bool missing = false;
  module.walk([&](Operation *op) {
    if (!isTaggedFor(op, ProgrammingModel::MPI))
      return;
    // Both checks run, so an op wrong on two counts reports both.
    bool handle = succeeded(checkPredefinedHandle(op));
    bool expressible = succeeded(checkExpressible(op));
    if (!handle || !expressible)
      missing = true;
  });
  return failure(missing);
}

template <typename OpT>
struct LowerPredefinedHandle : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<int32_t> handle = handleOf(op);
    if (!handle)
      return rewriter.notifyMatchFailure(op, "no predefined MPI handle");
    rewriter.replaceOp(op, createI32Constant(rewriter, op.getLoc(), *handle));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Calls
//===----------------------------------------------------------------------===//

/// The arguments of one MPI call, added in the order of its C prototype, and
/// the results of the operation it replaces, which follow from them: the
/// operation's results are its out-parameters in parameter order, then the
/// call's return code.
class MPICallArgs {
public:
  MPICallArgs(ConversionPatternRewriter &rewriter, Operation *op)
      : rewriter(rewriter), op(op), loc(op->getLoc()),
        i32(rewriter.getI32Type()),
        ptr(LLVM::LLVMPointerType::get(rewriter.getContext())) {}

  /// A parameter passed as its operand lowered.
  MPICallArgs &in(Value value) {
    args.push_back(value);
    return *this;
  }

  /// An element count: an MPI_Count (i64) in the dialect, an int in MPI's
  /// classic calls.  A count that provably fits an int is passed as one; if any
  /// count of the call may not, the call becomes its MPI 4 large-count form
  /// (MPI_Send_c, ...), which takes every count as an MPI_Count.  Truncating
  /// would silently send fewer elements.
  MPICallArgs &count(Value count) {
    counts.push_back({args.size(), count});
    args.push_back(count);
    if (!narrowIntOf(count))
      largeCount = true;
    return *this;
  }

  /// An int constant, such as MPI_INFO_NULL.
  MPICallArgs &constant(int32_t value) {
    args.push_back(createI32Constant(rewriter, loc, value));
    return *this;
  }

  /// A null pointer, such as the argc and argv of MPI_Init.
  MPICallArgs &null() {
    if (!nullPtr)
      nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptr);
    args.push_back(nullPtr);
    return *this;
  }

  /// A handle MPI takes by address, and may write back: MPI_Wait and
  /// MPI_Request_free set a request to MPI_REQUEST_NULL, MPI_Comm_free and
  /// MPI_Win_free their handle to the null handle, MPI_Mrecv a message to
  /// MPI_MESSAGE_NULL.  When \p original, the operand before lowering, was read
  /// from a variable of the program right before the operation, MPI gets that
  /// variable's address, so what MPI writes is what the program reads next.
  /// Otherwise the handle has no variable to write back to, and a stack slot
  /// takes the write.
  MPICallArgs &byAddress(Value handle, Value original) {
    if (Value address = addressReadFrom(original)) {
      args.push_back(address);
      return *this;
    }
    Value slot = createStackSlot(rewriter, loc, i32);
    LLVM::StoreOp::create(rewriter, loc, handle, slot);
    args.push_back(slot);
    return *this;
  }

  /// An out-parameter of \p type, read back as the operation's next result.
  MPICallArgs &out(Type type) {
    Value slot = createStackSlot(rewriter, loc, type);
    args.push_back(slot);
    results.push_back({ResultKind::Load, slot, type, 0});
    return *this;
  }

  /// An int out-parameter, such as a rank or a new handle.
  MPICallArgs &outInt() { return out(i32); }

  /// A pointer out-parameter, such as the base address MPI allocates.  The
  /// dialect may spell the address as a memref; MPI writes a void *.
  MPICallArgs &outPtr() { return out(ptr); }

  /// An int out-parameter nobody reads back.
  MPICallArgs &unreadInt() {
    args.push_back(createStackSlot(rewriter, loc, i32));
    return *this;
  }

  /// An int flag, read back as the operation's next result, compared with 0.
  MPICallArgs &flag() {
    Value slot = createStackSlot(rewriter, loc, i32);
    args.push_back(slot);
    results.push_back({ResultKind::Flag, slot, i32, 0});
    return *this;
  }

  /// An MPI_Status: storage that becomes the operation's next result when the
  /// operation has a status result, MPI_STATUS_IGNORE otherwise.
  MPICallArgs &status(bool wanted) {
    if (!wanted)
      return ignoreStatus();
    Value slot = createStackSlot(
        rewriter, loc, LLVM::LLVMArrayType::get(i32, mpich::kStatusNumFields));
    args.push_back(slot);
    results.push_back({ResultKind::Pointer, slot, ptr, 0});
    return *this;
  }

  /// An array of statuses, or MPI_STATUSES_IGNORE when there is none.
  MPICallArgs &statuses(Value array) {
    if (!array)
      return ignoreStatus();
    return in(array);
  }

  /// The request of a call with a blocking and a non-blocking form.  The
  /// non-blocking form writes it through its last parameter; the blocking one
  /// has no request, so a request result becomes MPI_REQUEST_NULL.
  MPICallArgs &request(bool isBlocking, bool hasResult) {
    if (!isBlocking)
      return out(i32);
    if (hasResult)
      results.push_back(
          {ResultKind::Constant, Value(), i32, mpich::kRequestNull});
    return *this;
  }

  /// Marks the operation as one MPI cannot express.
  MPICallArgs &fail(StringRef reason) {
    failure = reason;
    return *this;
  }

  /// Emits the call to \p callee and replaces the operation.
  LogicalResult replace(StringRef callee) {
    if (failure)
      return rewriter.notifyMatchFailure(op, *failure);

    std::string largeCountCallee;
    if (largeCount) {
      largeCountCallee = (callee + "_c").str();
      callee = largeCountCallee;
    } else {
      for (auto [index, count] : counts)
        args[index] = asInt(count);
    }
    if (results.size() + 1 != op->getNumResults())
      return rewriter.notifyMatchFailure(op, "unexpected number of results");

    SmallVector<Type> types;
    for (Value arg : args)
      types.push_back(arg.getType());
    FailureOr<LLVM::LLVMFuncOp> func = getOrDeclareRuntimeFunction(
        rewriter, op, callee, LLVM::LLVMFunctionType::get(i32, types));
    if (failed(func))
      return LogicalResult::failure();
    auto call = LLVM::CallOp::create(rewriter, loc, *func, args);

    SmallVector<Value> values;
    for (const Result &result : results) {
      switch (result.kind) {
      case ResultKind::Load:
        values.push_back(
            LLVM::LoadOp::create(rewriter, loc, result.type, result.slot));
        break;
      case ResultKind::Flag: {
        Value flag = LLVM::LoadOp::create(rewriter, loc, i32, result.slot);
        Value zero = createI32Constant(rewriter, loc, 0);
        values.push_back(LLVM::ICmpOp::create(
            rewriter, loc, LLVM::ICmpPredicate::ne, flag, zero));
        break;
      }
      case ResultKind::Pointer:
        values.push_back(result.slot);
        break;
      case ResultKind::Constant:
        values.push_back(createI32Constant(rewriter, loc, result.value));
        break;
      }
    }
    values.push_back(call.getResult());
    rewriter.replaceOp(op, values);
    return success();
  }

private:
  /// Whether \p count provably fits an int, and if so the narrowest value it
  /// is known from: \p count itself when it already is an i32, the operand of
  /// an extension of a value of at most 32 bits, or \p count when it is a
  /// constant in int range.  A zero extension of a 32-bit value is accepted
  /// too: that is how the raiser widens an int count, and a count MPI accepted
  /// was never negative.  Creates nothing.
  std::optional<Value> narrowIntOf(Value count) {
    if (count.getType().isInteger(32))
      return count;
    Value narrow;
    if (auto ext = count.getDefiningOp<arith::ExtSIOp>())
      narrow = ext.getIn();
    else if (auto ext = count.getDefiningOp<arith::ExtUIOp>())
      narrow = ext.getIn();
    else if (auto ext = count.getDefiningOp<LLVM::SExtOp>())
      narrow = ext.getArg();
    else if (auto ext = count.getDefiningOp<LLVM::ZExtOp>())
      narrow = ext.getArg();
    if (narrow) {
      auto type = dyn_cast<IntegerType>(narrow.getType());
      if (!type || type.getWidth() > 32)
        return std::nullopt;
      return narrow;
    }
    APInt value;
    if (matchPattern(count, m_ConstantInt(&value)) && value.isSignedIntN(32))
      return count;
    return std::nullopt;
  }

  /// \p count, which narrowIntOf accepted, as an i32.
  Value asInt(Value count) {
    Value narrow = *narrowIntOf(count);
    if (narrow.getType().isInteger(32))
      return narrow;
    return LLVM::TruncOp::create(rewriter, loc, i32, count);
  }

  MPICallArgs &ignoreStatus() {
    Value one = createI64Constant(rewriter, loc, mpich::kStatusIgnore);
    args.push_back(LLVM::IntToPtrOp::create(rewriter, loc, ptr, one));
    return *this;
  }

  /// The address of the program's variable holding \p value.  The raiser
  /// spells a handle MPI takes by address in two ways: as the variable's
  /// address itself, cast to the handle type (a communicator, for
  /// MPI_Comm_free), or as a memref.load of a one-element view at that address,
  /// cast to the handle type (requests, windows, messages).  Hand-written IR
  /// may use llvm.load.  A load counts only in the operation's block with
  /// nothing that can touch memory between the two.
  Value addressReadFrom(Value value) {
    while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() != 1)
        return Value();
      value = cast->getOperand(0);
      if (isa<LLVM::LLVMPointerType>(value.getType()))
        return value;
    }
    if (!value.getType().isInteger(32))
      return Value();
    Operation *load = value.getDefiningOp();
    if (!load || load->getBlock() != op->getBlock() ||
        !load->isBeforeInBlock(op))
      return Value();

    Value address;
    if (auto llvmLoad = dyn_cast<LLVM::LoadOp>(load)) {
      address = llvmLoad.getAddr();
    } else if (auto memLoad = dyn_cast<memref::LoadOp>(load)) {
      for (Value index : memLoad.getIndices())
        if (!matchPattern(index, m_Zero()))
          return Value();
      auto view =
          memLoad.getMemref().getDefiningOp<enzymexla::Pointer2MemrefOp>();
      if (!view)
        return Value();
      address = view.getSource();
    } else {
      return Value();
    }

    for (Operation *between = load->getNextNode(); between != op;
         between = between->getNextNode())
      if (!isMemoryEffectFree(between) && !isa<LLVM::AllocaOp>(between))
        return Value();
    return address;
  }

  enum class ResultKind { Load, Flag, Pointer, Constant };
  struct Result {
    ResultKind kind;
    Value slot;
    Type type;
    int32_t value;
  };

  ConversionPatternRewriter &rewriter;
  Operation *op;
  Location loc;
  Type i32;
  Type ptr;
  SmallVector<Value> args;
  SmallVector<std::pair<size_t, Value>> counts;
  bool largeCount = false;
  SmallVector<Result> results;
  Value nullPtr;
  std::optional<StringRef> failure;
};

/// MPI_Sendrecv_replace and MPI_Isendrecv_replace raise to spmd.sendrecv with
/// sendBuf and recvBuf the same value (MPILoweringTable.cpp's
/// kNonCanonicalCallees), so selectMPICall always answers MPI_Sendrecv or
/// MPI_Isendrecv for it -- the dialect has no attribute saying the source
/// called the _replace form. It does not need one: unlike the _replace forms,
/// MPI_Sendrecv and MPI_Isendrecv do not promise anything for a send and
/// receive buffer that still are one value (no internal temporary), and
/// calling them that way is observed to lose the exchanged value once the two
/// ranks do not share memory (test/exec/mpi/test-sendrecv.c, across nodes).
/// So when \p op's own operands still name the same buffer, answer the
/// _replace form instead of whichever \p canonical selectMPICall picked.
const CallDescription *pickSendrecvCall(Operation *op,
                                        const CallDescription *canonical) {
  auto sendrecv = dyn_cast<SendrecvOp>(op);
  if (!sendrecv ||
      !sameOperandValue(sendrecv.getSendBuf(), sendrecv.getRecvBuf()))
    return canonical;
  StringRef replaceCallee = sendrecv.getIsBlocking() ? "MPI_Sendrecv_replace"
                                                     : "MPI_Isendrecv_replace";
  for (const CallDescription &candidate : getMPILoweringTable())
    if (candidate.callee == replaceCallee)
      return &candidate;
  return canonical;
}

/// One operation lowered to the MPI call MPILoweringTable picks for it: the
/// slots of that call say, argument by argument, which builder call to make.
struct MPITableCallPattern : public ConversionPattern {
  MPITableCallPattern(const TypeConverter &converter, StringRef rootName,
                      MLIRContext *ctx)
      : ConversionPattern(converter, rootName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<std::string> reasons;
    const CallDescription *call = selectMPICall(op, rewriter, reasons);
    if (!call)
      return rewriter.notifyMatchFailure(
          op, reasons.empty() ? "no MPI call of this shape" : reasons.front());
    call = pickSendrecvCall(op, call);

    MPICallArgs args(rewriter, op);
    for (const ArgSlot &slot : call->args) {
      switch (static_cast<MPIArgKind>(slot.abiKind)) {
      case MPIArgKind::In:
        args.in(operands[slot.operand]);
        break;
      case MPIArgKind::Count:
        args.count(operands[slot.operand]);
        break;
      case MPIArgKind::ByAddress:
        args.byAddress(operands[slot.operand], op->getOperand(slot.operand));
        break;
      case MPIArgKind::StatusArray:
        args.statuses(hasSlotOperand(op, slot) ? operands[slot.operand]
                                               : Value());
        break;
      case MPIArgKind::OutInt:
        if (producesSlotResult(op, slot))
          args.outInt();
        else
          args.unreadInt();
        break;
      case MPIArgKind::OutPtr:
        args.outPtr();
        break;
      case MPIArgKind::OutFlag:
        args.flag();
        break;
      case MPIArgKind::OutStatus:
        args.status(producesSlotResult(op, slot));
        break;
      case MPIArgKind::InfoNull:
        args.constant(mpich::kInfoNull);
        break;
      case MPIArgKind::NullPointer:
        args.null();
        break;
      }
    }
    // A blocking call of an operation the optimisations gave a request to: the
    // request it hands back is the one MPI never started.
    if (needsRequestNull(getMPITableSpec(), op, *call))
      args.request(/*isBlocking=*/true, /*hasResult=*/true);
    return args.replace(call->callee);
  }
};

void populateMPICallPatterns(const TypeConverter &c, RewritePatternSet &p) {
  SmallVector<StringRef> lowered;
  for (const CallDescription &call : getMPILoweringTable())
    if (call.canonical && !llvm::is_contained(lowered, call.spmdOpName))
      lowered.push_back(call.spmdOpName);
  for (StringRef spmdOpName : lowered)
    p.add<MPITableCallPattern>(c, spmdOpName, p.getContext());
}

} // namespace

LogicalResult checkSPMDToMPI(ModuleOp module) {
  if (failed(checkUsedModels(module, "lower-spmd-to-mpi")))
    return failure();
  // The calls are derived from the same API description the raiser reads, so
  // the lowering needs it too.  Without it there would be no patterns at all,
  // and every operation would fail to legalize for no stated reason.  Only a
  // module that has something to lower needs it, though: -lower-spmd runs this
  // check for every model of its table, whether or not the module holds any
  // operation of that model.
  if (getMPILoweringTable().empty() && hasOperationTaggedForMPI(module))
    // Not module.emitError(): the module is not what is wrong with the run, and
    // printing it back as the offending operation buries the one line that is.
    return emitError(module.getLoc())
           << "lower-spmd-to-mpi: no MPI API description; set MPI_JSON_PATH to "
              "includesProgramModels/mpi/api.json";
  return checkPredefinedHandles(module);
}

LogicalResult lowerSPMDToMPI(ModuleOp module, StructureLowering structure) {
  MLIRContext *ctx = module.getContext();

  if (failed(checkSPMDToMPI(module)))
    return failure();

  MPITypeConverter converter(ctx);
  RewritePatternSet patterns(ctx);
  patterns.add<LowerPredefinedHandle<CommWorldOp>,
               LowerPredefinedHandle<RequestNullOp>,
               LowerPredefinedHandle<ConstDatatypeOp>,
               LowerPredefinedHandle<ConstOpTypeOp>,
               LowerPredefinedHandle<ConstLockTypeOp>,
               LowerPredefinedHandle<ConstThreadLevelOp>>(converter, ctx);
  populateMPICallPatterns(converter, patterns);
  populateSPMDStructureLoweringPatterns(converter, patterns,
                                        ProgrammingModel::MPI, structure);

  return applySPMDLowering(module, ProgrammingModel::MPI, converter,
                           std::move(patterns));
}

namespace {

struct LowerSPMDtoMPIPass
    : public impl::LowerSPMDtoMPIBase<LowerSPMDtoMPIPass> {
  using LowerSPMDtoMPIBase::LowerSPMDtoMPIBase;

  void runOnOperation() override {
    if (failed(lowerSPMDToMPI(getOperation())))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::spmd
