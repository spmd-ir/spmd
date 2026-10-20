// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- NCCL.h - Predefined constants of the NCCL ABI ------------*- C++ -*-===//
//
// The enumerators and constants of NCCL 2.29, whose header
// includesProgramModels/nccl/include carries (nccl.h:19).  A communicator is a
// pointer to an opaque struct, a datatype and a reduction operation are C
// enums, and there are no predefined communicators at all: every one is made
// by ncclCommInitRank from a unique id the program distributes itself.
// -convert-apiCalls-to-spmd recognises these values in the calls it raises and
// -lower-spmd-to-nccl emits them in the calls it lowers to, so the two read
// them from this one place.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_TARGETABI_NCCL_H
#define SPMD_TARGETABI_NCCL_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>

namespace mlir::spmd::abi::nccl {

/// ncclSuccess (nccl.h:42), what every NCCL routine returns when it succeeds.
inline constexpr int32_t kSuccess = 0;

/// NCCL_UNIQUE_ID_BYTES (nccl.h:38): an ncclUniqueId is a struct holding this
/// many chars, which ncclCommInitRank takes by value.
inline constexpr unsigned kUniqueIdBytes = 128;

/// An enumerator together with the name the dialect gives it: the NCCL name
/// without its prefix, upper case, as spmd.constOpType spells MPI's reductions
/// too.
struct NamedHandle {
  llvm::StringLiteral name;
  int32_t handle;
};

// The built-in reduction operations (nccl.h:310).  NCCL's are the same
// operations as MPI's of the same name, so a translated program keeps its
// spmd.constOpType; AVG is NCCL's own.
inline constexpr NamedHandle kReduceOps[] = {
    {"SUM", 0}, {"PROD", 1}, {"MAX", 2}, {"MIN", 3}, {"AVG", 4},
};

/// What the dialect records of a datatype: spmd.constDatatype keeps an MLIR
/// type.  NCCL's integers all say their signedness, ncclChar too, which nccl.h
/// makes ncclInt8.
enum class DatatypeShape {
  SInt8,
  UInt8,
  SInt32,
  UInt32,
  SInt64,
  UInt64,
  Float16,
  Float32,
  Float64,
  BFloat16,
  Float8E4M3,
  Float8E5M2,
};

struct DatatypeHandle {
  llvm::StringLiteral name;
  int32_t handle;
  DatatypeShape shape;
};

/// The datatypes (nccl.h:328).  NCCL gives several of them two names; the
/// first entry of a shape is the one a datatype of that shape lowers to.
inline constexpr DatatypeHandle kDatatypes[] = {
    {"ncclInt8", 0, DatatypeShape::SInt8},
    {"ncclChar", 0, DatatypeShape::SInt8},
    {"ncclUint8", 1, DatatypeShape::UInt8},
    {"ncclInt32", 2, DatatypeShape::SInt32},
    {"ncclInt", 2, DatatypeShape::SInt32},
    {"ncclUint32", 3, DatatypeShape::UInt32},
    {"ncclInt64", 4, DatatypeShape::SInt64},
    {"ncclUint64", 5, DatatypeShape::UInt64},
    {"ncclFloat16", 6, DatatypeShape::Float16},
    {"ncclHalf", 6, DatatypeShape::Float16},
    {"ncclFloat32", 7, DatatypeShape::Float32},
    {"ncclFloat", 7, DatatypeShape::Float32},
    {"ncclFloat64", 8, DatatypeShape::Float64},
    {"ncclDouble", 8, DatatypeShape::Float64},
    {"ncclBfloat16", 9, DatatypeShape::BFloat16},
    // NCCL's FP8 types are the OCP ones, whose E4M3 has no infinities.
    {"ncclFloat8e4m3", 10, DatatypeShape::Float8E4M3},
    {"ncclFloat8e5m2", 11, DatatypeShape::Float8E5M2},
};

inline std::optional<int32_t> lookupHandle(llvm::ArrayRef<NamedHandle> table,
                                           llvm::StringRef name) {
  for (const NamedHandle &entry : table)
    if (entry.name == name)
      return entry.handle;
  return std::nullopt;
}

inline std::optional<llvm::StringRef>
lookupName(llvm::ArrayRef<NamedHandle> table, int64_t handle) {
  for (const NamedHandle &entry : table)
    if (entry.handle == handle)
      return entry.name;
  return std::nullopt;
}

inline Type getDatatypeShapeType(MLIRContext *ctx, DatatypeShape shape) {
  Builder b(ctx);
  auto sint = [ctx](unsigned width) {
    return IntegerType::get(ctx, width, IntegerType::Signed);
  };
  auto uint = [ctx](unsigned width) {
    return IntegerType::get(ctx, width, IntegerType::Unsigned);
  };
  switch (shape) {
  case DatatypeShape::SInt8:
    return sint(8);
  case DatatypeShape::UInt8:
    return uint(8);
  case DatatypeShape::SInt32:
    return sint(32);
  case DatatypeShape::UInt32:
    return uint(32);
  case DatatypeShape::SInt64:
    return sint(64);
  case DatatypeShape::UInt64:
    return uint(64);
  case DatatypeShape::Float16:
    return b.getF16Type();
  case DatatypeShape::Float32:
    return b.getF32Type();
  case DatatypeShape::Float64:
    return b.getF64Type();
  case DatatypeShape::BFloat16:
    return b.getBF16Type();
  case DatatypeShape::Float8E4M3:
    return Float8E4M3FNType::get(ctx);
  case DatatypeShape::Float8E5M2:
    return Float8E5M2Type::get(ctx);
  }
  llvm_unreachable("unknown datatype shape");
}

/// The type spmd.constDatatype records for the datatype enumerator \p handle.
inline Type getDatatypeType(MLIRContext *ctx, int64_t handle) {
  for (const DatatypeHandle &entry : kDatatypes)
    if (entry.handle == handle)
      return getDatatypeShapeType(ctx, entry.shape);
  return Type();
}

/// The datatype enumerator a spmd.constDatatype of \p type lowers to.  An
/// integer of unknown signedness -- MPI_CHAR's i8 -- lowers to the signed
/// datatype of its width, which moves the same bits; the lowering refuses it
/// where signedness changes the result, for MAX and MIN.
inline std::optional<int32_t> getDatatypeHandle(Type type) {
  for (const DatatypeHandle &entry : kDatatypes)
    if (getDatatypeShapeType(type.getContext(), entry.shape) == type)
      return entry.handle;
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.isSignless())
    return getDatatypeHandle(IntegerType::get(
        type.getContext(), integer.getWidth(), IntegerType::Signed));
  return std::nullopt;
}

} // namespace mlir::spmd::abi::nccl

#endif // SPMD_TARGETABI_NCCL_H
