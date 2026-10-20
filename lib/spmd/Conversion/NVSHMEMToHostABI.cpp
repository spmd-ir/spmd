// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- NVSHMEMToHostABI.cpp - Map stand-in NVSHMEM to NVIDIA's library ----===//
//
// The NVSHMEM headers the frontend compiles against
// (includesProgramModels/mynvshmem/include) are a stand-in for NVIDIA's.  They
// declare every host routine as an ordinary function, which is what lets the
// raiser see nvshmem_init as a call; and they declare a few hundred routines
// NVIDIA's host library does not export.  A program compiled against them,
// raised and lowered, calls exactly what it called before -- which is right
// for the verification and wrong for a binary: it would not link against
// NVIDIA's libnvshmem_host.
//
// -nvshmem-to-host-abi closes that gap, for the routines the lowering emitted
// and for those the raiser never raised alike, since both are calls of the
// same declarations.  It runs after the lowering, in `spmd-verify compile`
// only.  Three kinds of routine need it:
//
// * Every routine the stand-in declares for the host and defines inline for
//   the device (`__device__ inline int nvshmem_my_pe() { return 0; }`).  A C++
//   host compile sees both as one function, so the module carries the stub
//   body as a linkonce_odr definition, and a call the raiser did not raise --
//   nvshmem_barrier, nvshmem_team_translate_pe -- would link against the stub
//   and do nothing.  The pass drops those bodies, so the calls reach the
//   library.
//
// * nvshmem_init, nvshmem_init_thread and nvshmem_finalize are static inline
//   functions of NVIDIA's header (host/nvshmem_api.h).  They call
//   nvshmemi_init_thread and nvshmemi_finalize in libnvshmem_device.a, whose
//   device code has to be linked by nvcc -rdc.  The pass defines the three
//   in the module instead, as the header does, over the host library's own
//   entry points: nvshmemx_hostlib_init_attr and nvshmemx_hostlib_finalize.
//   That is what nvshmemi_init_thread calls too; what it adds is the state of
//   the device library, which only the program's own GPU kernels read, and no
//   kernel is compiled here.  The collectives a host calls launch kernels of
//   the host library itself.  The thread level loses nothing either: NVSHMEM
//   provides NVSHMEM_THREAD_SERIALIZED whatever was requested, which is what
//   nvshmem_query_thread reports.
//
// * The rest are not in the host library at all: the routines kHostless
//   names.  A call of one is reported, with the reason, rather than left for
//   the linker to report as an undefined symbol.  A declaration nothing calls
//   is left alone.
//
// Both lists were checked against the 586 host routines the stand-in headers
// declare and the symbols NVSHMEM 3.8's libnvshmem_host.so.3 exports: the
// three inline routines and kHostless together are exactly the 270 it does
// not export.
//
//===----------------------------------------------------------------------===//

#include "LowerSPMDCommon.h"

#include "spmd/SPMDPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::spmd {

#define GEN_PASS_DEF_NVSHMEMTOHOSTABI
#include "spmd/SPMDPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// The routines NVIDIA's host library does not have
//===----------------------------------------------------------------------===//

/// A routine of the stand-in headers that the host library does not export:
/// nvshmem_<type><suffix> for every type, or for one of \p types only, or
/// the routine \p exact.
struct Hostless {
  StringLiteral exact;
  StringLiteral suffix;
  ArrayRef<StringLiteral> types;
  StringLiteral reason;
};

// The types NVSHMEM has no bitwise reduction for; its and, or and xor reduce
// the unsigned and the fixed-width integers only.
constexpr StringLiteral kNoBitwiseTypes[] = {
    "char", "schar", "short", "int", "long", "longlong", "float", "double"};

constexpr StringLiteral kDeviceOnly =
    "it is a device routine in NVSHMEM, which only a GPU kernel can call";
constexpr StringLiteral kSignalOnStream =
    "the host signals only through nvshmemx_<T>_put_signal_on_stream and "
    "nvshmemx_<T>_put_signal_nbi_on_stream, which take a CUDA stream";
constexpr StringLiteral kNoBitwise =
    "NVSHMEM reduces bitwise over the unsigned and the fixed-width integer "
    "types only";

constexpr Hostless kHostless[] = {
    {"nvshmem_realloc", "", {}, "NVSHMEM has no realloc"},
    {"", "_put_signal", {}, kSignalOnStream},
    {"", "_put_signal_nbi", {}, kSignalOnStream},
    {"", "_wait_until", {}, kDeviceOnly},
    {"", "_wait_until_all", {}, kDeviceOnly},
    {"", "_wait_until_any", {}, kDeviceOnly},
    {"", "_wait_until_some", {}, kDeviceOnly},
    {"", "_wait_until_all_vector", {}, kDeviceOnly},
    {"", "_wait_until_any_vector", {}, kDeviceOnly},
    {"", "_wait_until_some_vector", {}, kDeviceOnly},
    {"", "_test", {}, kDeviceOnly},
    {"", "_test_all", {}, kDeviceOnly},
    {"", "_test_any", {}, kDeviceOnly},
    {"", "_test_some", {}, kDeviceOnly},
    {"", "_test_all_vector", {}, kDeviceOnly},
    {"", "_test_any_vector", {}, kDeviceOnly},
    {"", "_test_some_vector", {}, kDeviceOnly},
    {"", "_and_reduce", kNoBitwiseTypes, kNoBitwise},
    {"", "_or_reduce", kNoBitwiseTypes, kNoBitwise},
    {"", "_xor_reduce", kNoBitwiseTypes, kNoBitwise},
};

/// Why \p name is not in the host library, or nothing when it is.
StringRef hostlessReason(StringRef name) {
  for (const Hostless &routine : kHostless) {
    if (!routine.exact.empty()) {
      if (name == routine.exact)
        return routine.reason;
      continue;
    }
    StringRef type = name;
    if (!type.consume_front("nvshmem_") || !type.consume_back(routine.suffix))
      continue;
    if (type.empty() || type.contains('_'))
      continue;
    if (routine.types.empty() || llvm::is_contained(routine.types, type))
      return routine.reason;
  }
  return StringRef();
}

//===----------------------------------------------------------------------===//
// The routines NVIDIA's header defines inline
//===----------------------------------------------------------------------===//

constexpr StringLiteral kHostlibInit = "nvshmemx_hostlib_init_attr";
constexpr StringLiteral kHostlibFinalize = "nvshmemx_hostlib_finalize";
constexpr StringLiteral kQueryThread = "nvshmem_query_thread";
constexpr StringLiteral kExit = "exit";

/// Calls \p callee from the body being built, declaring it in the module
/// around \p anchor if it is not there yet.
FailureOr<LLVM::CallOp> call(OpBuilder &builder, LLVM::LLVMFuncOp anchor,
                             StringRef callee, Type result, ValueRange args) {
  return emitRuntimeCall(builder, anchor, anchor.getLoc(), callee, result,
                         args);
}

/// status = nvshmemx_hostlib_init_attr(0, NULL), and exit(-1) unless it is 0,
/// as NVIDIA's inline nvshmem_init does (NONZERO_EXIT).  Leaves the builder in
/// the block that continues after a successful initialisation.
FailureOr<Value> emitHostlibInit(OpBuilder &builder, LLVM::LLVMFuncOp func) {
  MLIRContext *ctx = builder.getContext();
  Location loc = func.getLoc();
  auto i32 = IntegerType::get(ctx, 32);
  auto ptr = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);

  // No bootstrap flags and no attributes: the bootstrap is the one the
  // environment selects (NVSHMEM_BOOTSTRAP), as for nvshmem_init.
  Value noFlags = createI32Constant(builder, loc, 0);
  Value noAttributes = LLVM::ZeroOp::create(builder, loc, ptr);
  FailureOr<LLVM::CallOp> init =
      call(builder, func, kHostlibInit, i32, {noFlags, noAttributes});
  if (failed(init))
    return failure();
  Value status = init->getResult();

  Block *current = builder.getInsertionBlock();
  Region *body = current->getParent();
  Block *failedBlock = builder.createBlock(body, body->end());
  Block *initialised = builder.createBlock(body, body->end());

  builder.setInsertionPointToEnd(current);
  Value failedInit =
      LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne, status,
                           createI32Constant(builder, loc, 0));
  LLVM::CondBrOp::create(builder, loc, failedInit, failedBlock, initialised);

  builder.setInsertionPointToEnd(failedBlock);
  if (failed(call(builder, func, kExit, voidType,
                  {createI32Constant(builder, loc, -1)})))
    return failure();
  LLVM::UnreachableOp::create(builder, loc);

  builder.setInsertionPointToEnd(initialised);
  return status;
}

/// void nvshmem_init(void)
LogicalResult defineInit(OpBuilder &builder, LLVM::LLVMFuncOp func) {
  if (failed(emitHostlibInit(builder, func)))
    return failure();
  LLVM::ReturnOp::create(builder, func.getLoc(), ValueRange());
  return success();
}

/// int nvshmem_init_thread(int requested, int *provided).  requested does not
/// reach NVSHMEM, which provides NVSHMEM_THREAD_SERIALIZED whatever it is
/// asked for; nvshmem_query_thread says so.
LogicalResult defineInitThread(OpBuilder &builder, LLVM::LLVMFuncOp func) {
  Location loc = func.getLoc();
  MLIRContext *ctx = builder.getContext();
  auto ptr = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);

  FailureOr<Value> status = emitHostlibInit(builder, func);
  if (failed(status))
    return failure();

  // The program may not want to know the level: NULL is left alone.
  Value provided = func.getArgument(1);
  Block *current = builder.getInsertionBlock();
  Region *body = current->getParent();
  Block *query = builder.createBlock(body, body->end());
  Block *done = builder.createBlock(body, body->end());

  builder.setInsertionPointToEnd(current);
  Value wanted =
      LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne, provided,
                           LLVM::ZeroOp::create(builder, loc, ptr));
  LLVM::CondBrOp::create(builder, loc, wanted, query, done);

  builder.setInsertionPointToEnd(query);
  if (failed(call(builder, func, kQueryThread, voidType, {provided})))
    return failure();
  LLVM::BrOp::create(builder, loc, done);

  builder.setInsertionPointToEnd(done);
  LLVM::ReturnOp::create(builder, loc, *status);
  return success();
}

/// void nvshmem_finalize(void)
LogicalResult defineFinalize(OpBuilder &builder, LLVM::LLVMFuncOp func) {
  auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
  if (failed(call(builder, func, kHostlibFinalize, voidType, {})))
    return failure();
  LLVM::ReturnOp::create(builder, func.getLoc(), ValueRange());
  return success();
}

//===----------------------------------------------------------------------===//
// The stand-in's device stubs
//===----------------------------------------------------------------------===//

/// Whether \p func is a body the stand-in headers define for an NVSHMEM
/// routine: an inline definition, which only a header gives a program, of a
/// name in NVSHMEM's namespace, in host code.
bool isStandInStub(LLVM::LLVMFuncOp func) {
  if (func.isExternal() || isInDeviceCode(func))
    return false;
  StringRef name = func.getName();
  if (!name.starts_with("nvshmem_") && !name.starts_with("nvshmemx_"))
    return false;
  switch (func.getLinkage()) {
  case LLVM::Linkage::Linkonce:
  case LLVM::Linkage::LinkonceODR:
  case LLVM::Linkage::AvailableExternally:
  case LLVM::Linkage::Weak:
  case LLVM::Linkage::WeakODR:
    return true;
  default:
    return false;
  }
}

/// Makes \p func the declaration of the library routine it stands in for.
void dropStubBody(LLVM::LLVMFuncOp func) {
  func.eraseBody();
  func.setLinkage(LLVM::Linkage::External);
  func.removeComdatAttr();
  // The routine is in a shared library, not in this module.
  func.removeDsoLocalAttr();
}

/// A routine NVIDIA's header defines inline, and what the pass defines it as.
struct InlineRoutine {
  StringLiteral name;
  /// The prototype the stand-in headers give it; a declaration of any other
  /// type is reported rather than given a body it does not match.
  LLVM::LLVMFunctionType (*type)(MLIRContext *);
  LogicalResult (*define)(OpBuilder &, LLVM::LLVMFuncOp);
};

LLVM::LLVMFunctionType voidOfNothing(MLIRContext *ctx) {
  return LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {});
}
LLVM::LLVMFunctionType initThreadType(MLIRContext *ctx) {
  auto i32 = IntegerType::get(ctx, 32);
  return LLVM::LLVMFunctionType::get(i32,
                                     {i32, LLVM::LLVMPointerType::get(ctx)});
}

constexpr InlineRoutine kInlineRoutines[] = {
    {"nvshmem_init", voidOfNothing, defineInit},
    {"nvshmem_init_thread", initThreadType, defineInitThread},
    {"nvshmem_finalize", voidOfNothing, defineFinalize},
};

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct NVSHMEMToHostABIPass
    : public impl::NVSHMEMToHostABIBase<NVSHMEMToHostABIPass> {
  using NVSHMEMToHostABIBase::NVSHMEMToHostABIBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;

    // 1. The stand-in's stub bodies go, so the calls reach the library.
    for (auto func : module.getOps<LLVM::LLVMFuncOp>())
      if (isStandInStub(func))
        dropStubBody(func);

    // 2. The inline routines get their body.
    for (const InlineRoutine &routine : kInlineRoutines) {
      Operation *symbol = SymbolTable::lookupSymbolIn(module, routine.name);
      if (!symbol)
        continue;
      if (auto funcOp = dyn_cast<func::FuncOp>(symbol)) {
        if (!funcOp.isDeclaration())
          continue;
        funcOp.emitError() << "'" << routine.name
                           << "' is declared as a func.func; the pass defines "
                              "it as an llvm.func, and does not convert calls";
        anyError = true;
        continue;
      }
      auto func = dyn_cast<LLVM::LLVMFuncOp>(symbol);
      // A module that defines the routine itself keeps its definition.
      if (!func || !func.isExternal())
        continue;
      if (func.getFunctionType() != routine.type(&getContext())) {
        func.emitError() << "'" << routine.name << "' is declared as "
                         << func.getFunctionType()
                         << ", not as the NVSHMEM headers declare it ("
                         << routine.type(&getContext()) << ")";
        anyError = true;
        continue;
      }
      // Static inline in NVIDIA's header, so internal here.
      func.setLinkage(LLVM::Linkage::Internal);
      OpBuilder builder(func.getContext());
      Block *entry = func.addEntryBlock(builder);
      builder.setInsertionPointToStart(entry);
      if (failed(routine.define(builder, func)))
        anyError = true;
    }

    // 3. A call of a routine the host library does not have.  Only a
    // declaration needs the library -- a definition in the module is what the
    // call reaches -- and only a call from host code: a GPU kernel calls
    // NVSHMEM's device routines, which this pass has nothing to do with.
    for (Operation &op : module.getBody()->getOperations()) {
      auto symbol = dyn_cast<SymbolOpInterface>(&op);
      if (!symbol)
        continue;
      // isExternal, not isDeclaration: an llvm.func does not implement the
      // latter, and SymbolOpInterface's default says false for every one.
      auto func = dyn_cast<FunctionOpInterface>(&op);
      if (!func || !func.isExternal())
        continue;
      StringRef reason = hostlessReason(symbol.getName());
      if (reason.empty())
        continue;
      std::optional<SymbolTable::UseRange> uses =
          SymbolTable::getSymbolUses(&op, module);
      if (!uses)
        continue;
      for (const SymbolTable::SymbolUse &use : *uses) {
        if (isInDeviceCode(use.getUser()))
          continue;
        use.getUser()->emitError()
            << "'" << symbol.getName()
            << "' is declared by the stand-in NVSHMEM headers, but NVIDIA's "
               "host library has no such routine: "
            << reason;
        anyError = true;
      }
    }

    if (anyError)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::spmd
