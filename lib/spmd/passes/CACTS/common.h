/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include <functional>
#include <optional>
#include <vector>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/JSON.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "spmd/SPMDOps.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "utils.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "spmd/TargetABI/MPICH.h"
#include "llvm/ADT/DenseSet.h"

#define DEBUG_TYPE "convert-apiCalls-to-spmd"

#define PATTERN "main"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

namespace mlir::spmd {

/// MPI_COMM_WORLD's handle value in the MPI ABI our bundled headers describe.
inline constexpr int64_t kMpiCommWorldHandle = abi::mpich::kCommWorld;

/// Look through chains of single-operand builtin.unrealized_conversion_cast.
/// CACTS hooks run *during* convert-apiCalls-to-spmd, where converted operands
/// are still wrapped in materialisation casts; spmd-reconcile-unrealized-cast
/// only folds them away later in the pipeline.
inline Value peelUnrealizedCasts(Value v) {
  while (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1)
      break;
    v = cast.getInputs()[0];
  }
  return v;
}

/// True if \p v denotes the world communicator, in either form it can take
/// mid-conversion: already lowered to spmd.commWorld, or still the raw
/// MPI_COMM_WORLD constant because the call producing it has not been
/// converted yet. Which one you get depends on the order CACTS happens to
/// visit the calls in, so both must be accepted.
inline bool isWorldComm(Value v) {
  v = peelUnrealizedCasts(v);
  if (v.getDefiningOp<spmd::CommWorldOp>())
    return true;
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto i = dyn_cast<IntegerAttr>(c.getValueAttr()))
      return i.getInt() == kMpiCommWorldHandle;
  return false;
}

struct ResultEntry {
  llvm::StringRef name;
  std::optional<unsigned> apiCallArgIdx;
  std::optional<unsigned> apiCallResultIdx;
  std::function<Value(OpBuilder &, Location, Value, int)> conversion = nullptr;
  /// The spmd result type, for an out-parameter the API description leaves
  /// unclassified (no "kind", e.g. MPI_Waitsome's outcount). Without it the
  /// result type is derived from the kind, and an unknown kind falls back to
  /// the C argument's type -- a pointer, which no spmd op declares.
  std::function<Type(OpBuilder &)> type = nullptr;
};

// name: name of the operand in the spmd.op
// apiCallArgIdx: index of the operand in the original llvm.call, if it is
// generated from an existing operand apiCallResIdx: index of the result in the
// original llvm.call, if it is generated from an existing result? not correct?
// conversion: optional function to convert the old operand value to the new
// operand value, if it is generated from an existing operand or result
// functionForDefaultValue: optional function to generate the operand value if
// it is not generated from an existing operand or result, takes only the
// builder and location as arguments, used for operands with default values

struct OperandEntry {
  llvm::StringRef name;
  std::optional<unsigned> apiCallArgIdx =
      std::nullopt; // new operand will come from the idx-th argument of the
                    // original llvm.call
  std::optional<unsigned> apiCallResIdx =
      std::nullopt; // new operand will come in form of the idx-th result of the
                    // new operation?
  std::function<Value(OpBuilder &, Location, Value, int)> conversion = nullptr;
  std::function<Value(OpBuilder &, Location, int)> functionForDefaultValue =
      nullptr;
};

struct APICallEntry {
  llvm::StringRef spmdOpName;
  SmallVector<OperandEntry> operands;
  SmallVector<ResultEntry> results = {};
  SmallVector<std::function<NamedAttribute(OpBuilder &)>> newAttributes = {};
  SmallVector<std::function<void(Operation *, Operation *)>> postProcessing =
      {};
  bool producesError = true;
};

// Operand generated directly from existing parameter
inline OperandEntry
O(StringRef name, unsigned apiCallArgIdx,
  std::function<Value(OpBuilder &, Location, Value, int)> conv = nullptr) {
  return OperandEntry{name, apiCallArgIdx, std::nullopt, conv, nullptr};
}

// Operand generated without existing paremeter
inline OperandEntry
D(StringRef name, std::function<Value(OpBuilder &, Location, int)> defaultFn) {
  return OperandEntry{name, std::nullopt, std::nullopt, nullptr, defaultFn};
}

// Operand generated from existing result value
inline OperandEntry
V(StringRef name, unsigned apiCallResIdx,
  std::function<Value(OpBuilder &, Location, int)> defaultFn = nullptr) {
  return OperandEntry{name, std::nullopt, apiCallResIdx, nullptr, defaultFn};
}

//  // missing =>  optional operand without default function
// inline OperandEntry M(StringRef name) {
//     auto emptyFn = [](OpBuilder &builder, Location loc, int modelGroup) ->
//     Value {
//         return Value();
//     };
//     return OperandEntry{
//         name,
//         std::nullopt,
//         std::nullopt,
//         nullptr,
//         emptyFn
//     };
// }

// So irgendwie die Result zu Parameter transformation
/* inline OperandEntry OR(StringRef name,
                     std::optional<unsigned> mpiResIdx,
                     std::function<Value(OpBuilder&, Location, Value, int)> conv
= nullptr) { return OperandEntry{ name, std::nullopt, nullptr, emptyFn
  };
}*/

// Result generated from existing operand value
inline ResultEntry
R(StringRef name, std::optional<unsigned> mpiArgIdx,
  std::function<Value(OpBuilder &, Location, Value, int)> conv = nullptr,
  std::function<Type(OpBuilder &)> type = nullptr) {
  return ResultEntry{name, mpiArgIdx, std::nullopt, conv, type};
}

inline Type i32Type(OpBuilder &builder) { return builder.getI32Type(); }

/// Operand conversion for an `int *` array argument the API description leaves
/// unclassified (MPI_Waitsome's array_of_indices): the op takes it as a
/// memref<?xi32> view of the caller's array, which the callee fills in place.
inline Value viewAsI32Array(OpBuilder &builder, Location loc, Value address,
                            int modelGroup) {
  auto memrefType =
      MemRefType::get({ShapedType::kDynamic}, builder.getI32Type());
  return UnrealizedConversionCastOp::create(builder, loc, memrefType, address)
      .getResult(0);
}

// Result generated from existing result value
inline ResultEntry
RR(StringRef name, std::optional<unsigned> mpiResIdx,
   std::function<Value(OpBuilder &, Location, Value, int)> conv = nullptr) {
  return ResultEntry{name, std::nullopt, mpiResIdx, conv};
}

template <typename... Ops>
inline SmallVector<OperandEntry> Operands(Ops... ops) {
  return {ops...};
}

template <typename... Res> inline SmallVector<ResultEntry> Results(Res... res) {
  return {res...};
}

template <typename... F>
inline SmallVector<std::function<NamedAttribute(OpBuilder &)>>
newAttributes(F... funcs) {
  return {funcs...};
}

template <typename... F>
inline SmallVector<std::function<void(Operation *, Operation *)>>
postProcessing(F... funcs) {
  return {funcs...};
}

inline APICallEntry copyAPICallEntry(const APICallEntry &entry) {
  APICallEntry copy = entry; // flache Kopie
  return copy;
}

template <typename... Res>
inline APICallEntry addToResults(APICallEntry entry, Res... res) {
  entry.results.insert(entry.results.end(), {res...});
  return entry;
}

template <typename... Ops>
inline APICallEntry addToOperands(APICallEntry entry, Ops... ops) {
  entry.operands.insert(entry.operands.end(), {ops...});
  return entry;
}

template <typename... Attrs>
inline APICallEntry addToAttributes(APICallEntry entry, Attrs... attrs) {
  entry.newAttributes.insert(entry.newAttributes.end(), {attrs...});
  return entry;
}

template <typename... Proc>
inline APICallEntry addToPostProcessing(APICallEntry entry, Proc... proc) {
  entry.postProcessing.insert(entry.postProcessing.end(), {proc...});
  return entry;
}

inline Value zeroPtr(OpBuilder &builder, Location loc, int modelGroup) {
  return LLVM::ZeroOp::create(builder, loc,
                              builder.getType<LLVM::LLVMPointerType>())
      .getResult();
}

inline Value oneI64(OpBuilder &builder, Location loc, int modelGroup) {
  return arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(1))
      .getResult();
}

inline Value zeroI64(OpBuilder &builder, Location loc, int modelGroup) {
  return arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(0))
      .getResult();
}

inline Value zeroI32(OpBuilder &builder, Location loc, int modelGroup) {
  return arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0))
      .getResult();
}

inline Value getWorldComm(OpBuilder &builder, Location loc, int modelGroup) {
  return spmd::CommWorldOp::create(builder, loc, modelGroup).getResult();
}

inline Value getNodeComm(OpBuilder &builder, Location loc, int modelGroup) {
  return spmd::CommNodeOp::create(builder, loc, modelGroup).getResult();
}

/// Creates spmd.constThreadLevel("SINGLE").
/// Used as the $threadRequired default operand when lowering plain MPI_Init.
inline Value getSingleThreadLevel(OpBuilder &builder, Location loc,
                                  int modelGroup) {
  return spmd::ConstThreadLevelOp::create(builder, loc, StringRef("SINGLE"),
                                          modelGroup)
      .getResult();
}

inline Value convertI32ToI64(OpBuilder &builder, Location loc, Value val,
                             int modelGroup) {
  return arith::ExtUIOp::create(builder, loc, builder.getI64Type(), val)
      .getResult();
}

inline Value convertI1ToI32(OpBuilder &builder, Location loc, Value val,
                            int modelGroup) {
  return arith::ExtUIOp::create(builder, loc, builder.getI32Type(), val)
      .getResult();
}

inline NamedAttribute getAtomicAttr(OpBuilder &builder) {
  return builder.getNamedAttr("isAtomic", builder.getBoolAttr(true));
}

inline NamedAttribute getNotAtomicAttr(OpBuilder &builder) {
  return builder.getNamedAttr("isAtomic", builder.getBoolAttr(false));
}

inline NamedAttribute getReadyModeAttr(OpBuilder &builder) {
  return builder.getNamedAttr("isReadyMode", builder.getBoolAttr(true));
}

inline NamedAttribute getNotReadyModeAttr(OpBuilder &builder) {
  return builder.getNamedAttr("isReadyMode", builder.getBoolAttr(false));
}

inline NamedAttribute getBufferedAttr(OpBuilder &builder) {
  return builder.getNamedAttr("isBuffered", builder.getBoolAttr(true));
}

inline NamedAttribute getNotBufferedAttr(OpBuilder &builder) {
  return builder.getNamedAttr("isBuffered", builder.getBoolAttr(false));
}

inline NamedAttribute getLocalAttr(OpBuilder &builder) {
  return builder.getNamedAttr("isLocal", builder.getBoolAttr(true));
}

inline NamedAttribute getNotLocalAttr(OpBuilder &builder) {
  return builder.getNamedAttr("isLocal", builder.getBoolAttr(false));
}

inline APICallEntry
makeNonBlocking(const APICallEntry &blocking,
                std::optional<unsigned> reqArgIdx = std::nullopt) {
  if (!reqArgIdx.has_value()) {
    return addToResults(copyAPICallEntry(blocking));
  } else {
    return addToResults(copyAPICallEntry(blocking),
                        R("req", reqArgIdx.value()));
  }
}

/// Maps an integer thread-level constant to a spmd.constThreadLevel SSA value.
/// The encoding 0=SINGLE, 1=FUNNELED, 2=SERIALIZED, 3=MULTIPLE is identical
/// across MPI, OpenSHMEM, and NVSHMEM — so this lives in common.h.
inline Value getThreadLevelFromInt(Value arg, int modelGroup) {
  auto constantOp = dyn_cast_or_null<arith::ConstantOp>(arg.getDefiningOp());
  if (!constantOp)
    llvm::reportFatalUsageError(
        "init_thread: 'required' is not a compile-time constant");

  auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr());
  if (!intAttr)
    llvm::reportFatalUsageError(
        "init_thread: 'required' is not an integer constant");

  StringRef levelStr;
  switch (intAttr.getInt()) {
  case 0:
    levelStr = "SINGLE";
    break;
  case 1:
    levelStr = "FUNNELED";
    break;
  case 2:
    levelStr = "SERIALIZED";
    break;
  case 3:
    levelStr = "MULTIPLE";
    break;
  default:
    llvm::reportFatalUsageError(
        "init_thread: unknown thread level constant value");
  }

  OpBuilder builder(constantOp);
  builder.setInsertionPoint(constantOp.getOperation());
  return spmd::ConstThreadLevelOp::create(builder, constantOp->getLoc(),
                                          levelStr, modelGroup)
      .getResult();
}

/// ResultEntry conversion: inserts spmd.castThreadLevelToI32 so the
/// !spmd.threadLevel result can be stored back to the int* provided pointer.
inline Value castThreadLevelToI32(OpBuilder &builder, Location loc, Value v,
                                  int modelGroup) {
  return spmd::CastThreadLevelToI32Op::create(builder, loc,
                                              builder.getI32Type(), v)
      .getResult();
}

using ApiCallToSpmdMap = llvm::StringMap<APICallEntry>;

enum class APIKind { MPI, SHMEM, NCCL, NVSHMEM, CUDA, OMP };

// --- JSON loader ---
static inline llvm::Expected<llvm::json::Value>
loadJsonDynamic(StringRef filename) {
  auto fileOrErr = llvm::MemoryBuffer::getFile(filename);
  if (!fileOrErr)
    return llvm::errorCodeToError(fileOrErr.getError());

  return llvm::json::parse(fileOrErr.get()->getBuffer());
}

/// Lowercased function name -> its USIS API description entry, for one model.
using ApiEntryIndex = llvm::StringMap<const llvm::json::Object *>;

/// Loads and caches one model's API description, indexed by the lowercased
/// part of each "<model>:<function>" key after the ':' (USIS spells OpenMP
/// "openmp" and keeps camelCase). Returns nullptr after a diagnostic on error.
static inline const ApiEntryIndex *getJsonIndex(llvm::StringRef apiName) {
  static llvm::StringMap<std::unique_ptr<llvm::json::Value>> jsonValues;
  static llvm::StringMap<ApiEntryIndex> jsonIndices;

  // Already loaded?
  auto it = jsonIndices.find(apiName);
  if (it != jsonIndices.end())
    return &it->second;

  // Build ENV var name
  std::string envVar = toUpperStr(apiName) + "_JSON_PATH";
  const char *path = std::getenv(envVar.c_str());
  if (!path) {
    llvm::errs() << "[SPMD] ERROR: " << envVar
                 << " env-var not set — required for API '" << apiName << "'\n";
    return nullptr;
  }

  auto jsonOrErr = loadJsonDynamic(path);
  if (!jsonOrErr) {
    llvm::errs() << "[SPMD] ERROR: Failed to load JSON from '" << path
                 << "' for API '" << apiName
                 << "': " << llvm::toString(jsonOrErr.takeError()) << "\n";
    return nullptr;
  }

  auto value = std::make_unique<llvm::json::Value>(std::move(*jsonOrErr));

  const llvm::json::Object *root = value->getAsObject();
  if (!root) {
    llvm::errs() << "[SPMD] JSON root for '" << apiName
                 << "' is not an object \n";
    return nullptr;
  }

  const llvm::json::Object *entries = root->getObject("entries");
  if (!entries) {
    llvm::errs() << "[SPMD] ERROR: '" << path << "' (API '" << apiName
                 << "') has no 'entries' object — expected a USIS API "
                    "description file\n";
    return nullptr;
  }

  ApiEntryIndex index;
  for (const auto &kv : *entries) {
    const llvm::json::Object *entry = kv.second.getAsObject();
    if (!entry) {
      llvm::errs() << "[SPMD] ERROR: entry '" << kv.first << "' in '" << path
                   << "' is not an object\n";
      return nullptr;
    }
    llvm::StringRef key = kv.first;
    // "mpi:MPI_Send" -> "mpi_send"; a key without ':' is indexed whole.
    llvm::StringRef functionName = key.substr(key.find(':') + 1);
    index[toLowerStr(functionName)] = entry;
  }

  if (index.empty()) {
    llvm::errs() << "[SPMD] ERROR: '" << path << "' (API '" << apiName
                 << "') describes no entries\n";
    return nullptr;
  }

  jsonValues[apiName] = std::move(value);
  return &(jsonIndices[apiName] = std::move(index));
}

/// The USIS kind of one `parameters[]` entry; std::nullopt if USIS leaves it
/// unclassified, in which case callers use their default. MPI_Win_lock's
/// lock_type has no USIS kind of its own and is named here.
inline std::optional<std::string>
getSpmdParamKind(const llvm::json::Object *paramObj) {
  if (!paramObj)
    return std::nullopt;

  if (toLowerStr(getParamString(paramObj, "name").value_or("")) == "lock_type")
    return std::string("LOCK_TYPE");

  return getParamString(paramObj, "kind");
}

inline llvm::StringRef toString(APIKind kind) {
  switch (kind) {
  case APIKind::MPI:
    return "mpi";
  case APIKind::SHMEM:
    return "shmem";
  case APIKind::NCCL:
    return "nccl";
  case APIKind::NVSHMEM:
    return "nvshmem";
  case APIKind::CUDA:
    return "cuda";
  case APIKind::OMP:
    return "omp";
  }
  llvm_unreachable("unknown APIKind");
}

struct ApiRegistryEntry {
  APIKind apiKind;        // "mpi", "shmem", ...
  llvm::StringRef prefix; // "MPI_"
  const ApiEntryIndex *entryIndex;
  const ApiCallToSpmdMap *apiCallToSpmd; // e.g. mpiToSpmdOperands
  int modelGroup;
};

class ApiRegistry {
public:
  bool registerApi(APIKind apiKind, llvm::StringRef prefix,
                   const ApiCallToSpmdMap &apiCallToSpmd, int modelGroup) {
    const ApiEntryIndex *entryIndex = getJsonIndex(toString(apiKind));

    if (!entryIndex) {
      llvm::errs() << "[SPMD] ERROR: Cannot register API '" << toString(apiKind)
                   << "' — JSON root unavailable. "
                   << "Check that the corresponding _JSON_PATH env-var is set "
                   << "and the file is valid JSON.\n";
      return false;
    }

    // Check if already registered
    for (const auto &entry : entries) {
      if (entry.apiKind == apiKind) {
        LLVM_DEBUG(DBGS() << "Tried to register same model twice: "
                          << toString(apiKind) << "\n");
        return false;
      }
    }

    entries.push_back(ApiRegistryEntry{apiKind, prefix, entryIndex,
                                       &apiCallToSpmd, modelGroup});

    return true;
  }

  const ApiRegistryEntry *matchByCallee(llvm::StringRef callee) const {
    for (const auto &entry : entries) {
      if (callee.starts_with(entry.prefix))
        return &entry;
    }
    return nullptr;
  }

  const SmallVector<ApiRegistryEntry> &getEntries() const { return entries; }

  /// Reports mapped names with no API description entry (such calls are
  /// silently left unconverted) and returns true if all are in
  /// \p knownUnresolved. Enabled by SPMD_CHECK_API_COVERAGE=1.
  bool checkCoverage(llvm::ArrayRef<llvm::StringRef> knownUnresolved,
                     llvm::raw_ostream &os) const {
    llvm::StringSet<> expected;
    for (llvm::StringRef name : knownUnresolved)
      expected.insert(toLowerStr(name));

    bool allExpected = true;
    for (const auto &entry : entries) {
      unsigned resolved = 0;
      SmallVector<std::string> unexpected, expectedMisses;
      for (const auto &kv : *entry.apiCallToSpmd) {
        std::string key = toLowerStr(stripNumericSuffix(kv.getKey()));
        if (entry.entryIndex->contains(key)) {
          ++resolved;
          continue;
        }
        (expected.contains(key) ? expectedMisses : unexpected)
            .push_back(kv.getKey().str());
      }
      os << "[SPMD] coverage " << toString(entry.apiKind) << ": " << resolved
         << "/" << entry.apiCallToSpmd->size() << " mapped names resolve";
      if (!expectedMisses.empty())
        os << ", " << expectedMisses.size() << " expected miss(es)";
      os << "\n";
      llvm::sort(unexpected);
      for (const std::string &name : unexpected) {
        os << "[SPMD]   UNRESOLVED: " << toString(entry.apiKind) << ':' << name
           << "\n";
        allExpected = false;
      }
    }
    return allExpected;
  }

private:
  SmallVector<ApiRegistryEntry> entries;
};

struct PointerInfo {
  Type elemType;
  SmallVector<int64_t> shape;
};

static inline bool sameShape(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

// copied over from Enzyme-JAX/src/enzyme_ad/jax/Passes/LLVMToAffineAccess.cpp -
// from here
static inline std::optional<int64_t> getConstant(Operation *op) {
  if (auto cst = dyn_cast_or_null<arith::ConstantIntOp>(op)) {
    return cst.value();
  } else if (auto cst = dyn_cast_or_null<arith::ConstantIndexOp>(op)) {
    return cst.value();
  } else if (auto cst = dyn_cast_or_null<LLVM::ConstantOp>(op)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getValue().getSExtValue();
  }
  return {};
}

static inline std::optional<int64_t> getConstant(Value v) {
  Operation *op = v.getDefiningOp();
  if (op)
    return getConstant(op);
  return {};
}

static inline std::optional<PointerInfo>
resolvePointerInfoImpl(Value ptr, DenseSet<Value> &visited);

static inline std::optional<PointerInfo>
mergePointerInfos(SmallVectorImpl<PointerInfo> &infos) {
  if (infos.empty())
    return std::nullopt;

  const PointerInfo &first = infos.front();
  for (const PointerInfo &info : infos) {
    if (info.elemType != first.elemType)
      return std::nullopt;
    if (!sameShape(info.shape, first.shape))
      return std::nullopt;
  }
  return first;
}

static inline std::optional<PointerInfo>
resolvePointerInfoFromBlockArg(BlockArgument barg, DenseSet<Value> &visited) {
  Block *block = barg.getOwner();
  if (!block || !block->isEntryBlock()) {
    LLVM_DEBUG(DBGS() << "BlockArgument is not in entry block, cannot resolve "
                         "pointer info\n");
    return std::nullopt;
  }

  auto func = dyn_cast<LLVM::LLVMFuncOp>(block->getParentOp());
  if (!func) {
    LLVM_DEBUG(DBGS() << "BlockArgument does not belong to LLVMFuncOp\n");
    return std::nullopt;
  }

  ModuleOp module = func->getParentOfType<ModuleOp>();
  if (!module) {
    LLVM_DEBUG(DBGS() << "Could not find parent module for function "
                      << func.getSymName() << "\n");
    return std::nullopt;
  }

  unsigned argNo = barg.getArgNumber();
  SmallVector<PointerInfo> callerInfos;

  module.walk([&](LLVM::CallOp callOp) {
    auto callee = callOp.getCallee();
    if (!callee || *callee != func.getSymName())
      return;
    if (argNo >= callOp.getNumOperands())
      return;

    Value actual = callOp.getOperand(argNo);
    if (auto info = resolvePointerInfoImpl(actual, visited))
      callerInfos.push_back(*info);
  });

  if (callerInfos.empty()) {
    LLVM_DEBUG(DBGS() << "No caller information found for argument " << argNo
                      << " of function " << func.getSymName() << "\n");
    return std::nullopt;
  }

  auto merged = mergePointerInfos(callerInfos);
  if (!merged) {
    LLVM_DEBUG(DBGS() << "Conflicting pointer infos for argument " << argNo
                      << " of function " << func.getSymName() << "\n");
  }
  return merged;
}

static inline std::optional<PointerInfo>
resolvePointerInfoImpl(Value ptr, DenseSet<Value> &visited) {
  if (!ptr) {
    LLVM_DEBUG(DBGS() << "resolvePointerInfoImpl called with null Value\n");
    return std::nullopt;
  }

  if (!isa<LLVM::LLVMPointerType>(ptr.getType())) {
    LLVM_DEBUG(
        DBGS() << "resolvePointerInfoImpl called on non-pointer value\n");
    return std::nullopt;
  }

  if (!visited.insert(ptr).second) {
    LLVM_DEBUG(DBGS() << "Already visited pointer value, abort recursion\n");
    return std::nullopt;
  }

  // Case 1: function / block argument
  if (auto barg = dyn_cast<BlockArgument>(ptr)) {
    return resolvePointerInfoFromBlockArg(barg, visited);
  }

  Operation *defOp = ptr.getDefiningOp();
  if (!defOp) {
    LLVM_DEBUG(
        DBGS() << "Pointer has no defining op and is not a BlockArgument\n");
    return std::nullopt;
  }

  // Case 2: alloca
  if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(defOp)) {
    Type elemType = allocaOp.getElemType();
    auto sizeVal = getConstant(allocaOp.getArraySize());
    SmallVector<int64_t> shape;
    if (sizeVal)
      shape.push_back(*sizeVal);
    else
      shape.push_back(ShapedType::kDynamic);

    LLVM_DEBUG(DBGS() << "Resolved pointer info from alloca: elemType="
                      << elemType << ", shape[0]=" << shape[0] << "\n");
    return PointerInfo{elemType, shape};
  }

  // Case 3: global
  if (auto addrOf = dyn_cast<LLVM::AddressOfOp>(defOp)) {
    FlatSymbolRefAttr symRef = addrOf.getGlobalNameAttr();
    Operation *symbol = SymbolTable::lookupNearestSymbolFrom(addrOf, symRef);
    auto global = dyn_cast_or_null<LLVM::GlobalOp>(symbol);
    if (!global) {
      LLVM_DEBUG(DBGS() << "AddressOf symbol not found or not a global\n");
      return std::nullopt;
    }

    Type elemType = global.getGlobalType();

    // Conservative default:
    // a global symbol as pointer-backed memory is treated as one element.
    // Adjust this later if you want arrays/globals modeled more precisely.
    SmallVector<int64_t> shape = {1};

    LLVM_DEBUG(DBGS() << "Resolved pointer info from global: elemType="
                      << elemType << ", shape=1\n");
    return PointerInfo{elemType, shape};
  }

  // Case 4: passthrough pointer ops
  if (auto bitcastOp = dyn_cast<LLVM::BitcastOp>(defOp)) {
    return resolvePointerInfoImpl(bitcastOp.getArg(), visited);
  }

  if (auto addrSpaceCastOp = dyn_cast<LLVM::AddrSpaceCastOp>(defOp)) {
    return resolvePointerInfoImpl(addrSpaceCastOp.getArg(), visited);
  }

  if (auto gepOp = dyn_cast<LLVM::GEPOp>(defOp)) {
    return resolvePointerInfoImpl(gepOp.getBase(), visited);
  }

  // Case 5: enzymexla.ptr2memref — unwrap back to pointer
  if (auto ptr2Memref = dyn_cast<enzymexla::Pointer2MemrefOp>(defOp)) {
    return resolvePointerInfoImpl(ptr2Memref.getOperand(), visited);
  }

  // Case 6: memref.load — only meaningful if it loads a pointer
  if (auto loadOp = dyn_cast<memref::LoadOp>(defOp)) {
    Value memrefVal = loadOp.getMemRef();

    // Unwrap pointer2memref to get the backing pointer, then recurse on that
    if (auto p2m = memrefVal.getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
      Value backingPtr = p2m.getOperand();
      return resolvePointerInfoImpl(backingPtr, visited);
    }

    LLVM_DEBUG(
        DBGS() << "memref.load source is not pointer2memref, cannot trace\n");
    return std::nullopt;
  }

  // Case 7: arith.select — merge info from both arms.
  // Both arms must resolve to compatible pointer info (same element type
  // and shape) for the result to be usable. Otherwise bail conservatively.
  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    SmallVector<PointerInfo> armInfos;
    if (auto trueInfo =
            resolvePointerInfoImpl(selectOp.getTrueValue(), visited))
      armInfos.push_back(*trueInfo);
    if (auto falseInfo =
            resolvePointerInfoImpl(selectOp.getFalseValue(), visited))
      armInfos.push_back(*falseInfo);

    if (armInfos.empty()) {
      LLVM_DEBUG(
          DBGS() << "Neither arm of arith.select resolved pointer info\n");
      return std::nullopt;
    }
    return mergePointerInfos(armInfos);
  }

  // Optional: forward through unrealized casts if they appear in your pipeline.
  if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
    if (castOp->getNumOperands() == 1 &&
        isa<LLVM::LLVMPointerType>(castOp.getOperand(0).getType())) {
      return resolvePointerInfoImpl(castOp.getOperand(0), visited);
    }
  }

  LLVM_DEBUG(DBGS() << "Unhandled defining op in resolvePointerInfoImpl: "
                    << defOp->getName() << "\n");
  return std::nullopt;
}

static inline std::optional<PointerInfo> resolvePointerInfo(Value ptr) {
  DenseSet<Value> visited;
  return resolvePointerInfoImpl(ptr, visited);
}

static inline std::optional<Value>
getMemrefFromPointer(Value ptr, OpBuilder &builder, Location loc) {
  assert(isa<LLVM::LLVMPointerType>(ptr.getType()) &&
         "ptr argument is not of LLVMPointerType");

  auto info = resolvePointerInfo(ptr);
  if (!info) {
    LLVM_DEBUG(
        DBGS() << "Failed to resolve pointer info in getMemrefFromPointer\n");
    return std::nullopt;
  }

  auto memrefType = MemRefType::get(info->shape, info->elemType);
  return enzymexla::Pointer2MemrefOp::create(builder, loc, memrefType, ptr)
      .getResult();
}

/// True for LLVM aggregates whose *whole* type must never be used as the
/// element type of a single-scalar out-parameter store (see below).
inline bool isLLVMAggregateType(Type type) {
  return isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(type);
}

/// True for the plain numeric types a `memref` element may safely have here:
/// later passes (`-llvm-to-affine-access`) ask the DataLayout for the size of
/// every memref element type, which aborts on SPMD handle types
/// (`!spmd.comm`, `!spmd.req`, ...). Those must keep going through the
/// resolved pointee type plus an unrealized cast.
inline bool isDataLayoutScalar(Type type) {
  return isa<IntegerType, FloatType>(type);
}

/// The element type of an array-of-scalars allocation, else a null Type.
///
/// A handle passed by address is often one element of an array of handles:
/// `MPI_Isend(..., &reqs[i])` on `MPI_Request reqs[2]`. `resolvePointerInfo`
/// reports the allocation, `!llvm.array<2 x i32>`, and drops the GEP offset on
/// the way to it (clang spells `&reqs[1]` as a byte GEP), so the handle cannot
/// be stored or loaded as that type. The address itself still names exactly one
/// element -- a C pointer to an element sits on an element boundary -- so a
/// one-element view of the scalar element type at that address is the store or
/// load the call performs. Arrays of structs stay unresolved: which field the
/// address names is not recoverable here.
inline Type getScalarArrayElementType(Type allocType) {
  if (!isa<LLVM::LLVMArrayType>(allocType))
    return Type();
  Type elemType = allocType;
  while (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(elemType))
    elemType = arrayType.getElementType();
  return isDataLayoutScalar(elemType) ? elemType : Type();
}

inline std::optional<Value> loadValueFromMemref(Value addressArg,
                                                Type valueType,
                                                OpBuilder &builder,
                                                Location loc) {
  assert(isa<LLVM::LLVMPointerType>(addressArg.getType()) &&
         "addressArg argument is not of LLVMPointerType");

  auto info = resolvePointerInfo(addressArg);
  if (!info) {
    LLVM_DEBUG(
        DBGS() << "Could not resolve pointer info in loadValueFromMemref\n");
    return std::nullopt;
  }

  // One element of an array of handles (MPI_Wait(&reqs[i], ...)); see
  // getScalarArrayElementType.
  Type elemType = info->elemType;
  SmallVector<int64_t> shape(info->shape.begin(), info->shape.end());
  if (Type arrayElemType = getScalarArrayElementType(info->elemType)) {
    elemType = arrayElemType;
    shape.assign(1, 1);
  }

  auto memrefType = MemRefType::get(shape, elemType);
  auto memref =
      enzymexla::Pointer2MemrefOp::create(builder, loc, memrefType, addressArg)
          .getResult();

  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  auto loadedValue =
      memref::LoadOp::create(builder, loc, memref, c0).getResult();

  auto castedValue =
      UnrealizedConversionCastOp::create(builder, loc, valueType, loadedValue)
          .getResult(0);
  return castedValue;
}

inline LogicalResult storeValueIntoMemref(Value addressArg, Value newValue,
                                          OpBuilder &builder, Location loc) {
  assert(isa<LLVM::LLVMPointerType>(addressArg.getType()) &&
         "addressArg argument is not of LLVMPointerType");

  auto info = resolvePointerInfo(addressArg);

  // Element type for the one-element store `*addressArg = newValue`.
  //
  // `resolvePointerInfo` reports the type of the *allocation* the pointer
  // belongs to, which is not the type being written whenever the pointer names
  // a field inside an aggregate: `MPI_Comm_rank(comm, &H->mype)` on a
  // `struct {int mype; int nproc;} *H` resolves to the struct (the GEP case
  // recurses to its base and drops the field offset), so the i32 rank used to
  // be cast to the whole struct and stored over BOTH fields. A later read of
  // `H->nproc` then aliased the rank store and inherited its non-uniformity —
  // a false "collective not called by all processes" on correct code.
  //
  // An out-parameter writes exactly one value of the value's own type at the
  // address, so for an aggregate allocation the value type is authoritative and
  // no cast is needed. The resolved type is still used when it is a scalar,
  // because the pointee type and the value type legitimately differ there
  // (`MPI_Comm *` is an `int*` while the stored value is a `!spmd.comm`), and
  // the cast is what `-spmd-reconcile-unrealized-cast` later pairs with the
  // matching load.
  Type elemType;
  SmallVector<int64_t> shape;
  if (info && !isLLVMAggregateType(info->elemType)) {
    elemType = info->elemType;
    shape.assign(info->shape.begin(), info->shape.end());
  } else if (isDataLayoutScalar(newValue.getType())) {
    elemType = newValue.getType();
    shape.push_back(1);
  } else if (Type arrayElemType =
                 info ? getScalarArrayElementType(info->elemType) : Type()) {
    // A handle written into one element of an array of handles, e.g. the
    // request of MPI_Isend(..., &reqs[i]); see getScalarArrayElementType.
    elemType = arrayElemType;
    shape.push_back(1);
  } else {
    // Aggregate (or unresolved) allocation holding a non-scalar handle: there
    // is no well-typed one-element view to build here.
    LLVM_DEBUG(
        DBGS() << "Could not resolve pointer info in storeValueIntoMemref\n");
    return failure();
  }

  auto memrefType = MemRefType::get(shape, elemType);
  auto memref =
      enzymexla::Pointer2MemrefOp::create(builder, loc, memrefType, addressArg)
          .getResult();

  Value storedValue = newValue;
  if (elemType != newValue.getType())
    storedValue =
        UnrealizedConversionCastOp::create(builder, loc, elemType, newValue)
            .getResult(0);

  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  memref::StoreOp::create(builder, loc, storedValue, memref, c0);
  return success();
}

} // namespace mlir::spmd
