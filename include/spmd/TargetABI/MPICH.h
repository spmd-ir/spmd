// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- MPICH.h - Predefined handles of the MPICH ABI ------------*- C++ -*-===//
//
// The predefined handles and constants of MPICH 4.2.2, the MPI whose headers
// includesProgramModels/mpi/include carries (mpi.h:47).  In that ABI every
// handle is an int.  -convert-apiCalls-to-spmd recognises these values in the
// calls it raises and -lower-spmd-to-mpi emits them in the calls it lowers
// to, so the two read them from this one place.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_TARGETABI_MPICH_H
#define SPMD_TARGETABI_MPICH_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::spmd::abi::mpich {

// Communicators (mpi.h:235).
inline constexpr int32_t kCommWorld = 0x44000000;
inline constexpr int32_t kCommSelf = 0x44000001;

// Null handles (mpi.h:134, 305).
inline constexpr int32_t kRequestNull = 0x2c000000;
inline constexpr int32_t kInfoNull = 0x1c000000;

// MPI_STATUS_IGNORE and MPI_STATUSES_IGNORE are both (MPI_Status *)1
// (mpi.h:513), and an MPI_Status is five ints (mpi.h:425).
inline constexpr int64_t kStatusIgnore = 1;
inline constexpr unsigned kStatusNumFields = 5;

// MPI_Win_lock lock types (mpi.h:533).
inline constexpr int32_t kLockExclusive = 234;
inline constexpr int32_t kLockShared = 235;

/// A handle together with the name the dialect gives it: the MPI name without
/// its prefix, as spmd.constOpType, spmd.constLockTypeOp and
/// spmd.constThreadLevel spell it.
struct NamedHandle {
  llvm::StringLiteral name;
  int32_t handle;
};

// Predefined reduction operations (mpi.h:259).
inline constexpr NamedHandle kReduceOps[] = {
    {"MAX", 0x58000001},     {"MIN", 0x58000002},    {"SUM", 0x58000003},
    {"PROD", 0x58000004},    {"LAND", 0x58000005},   {"BAND", 0x58000006},
    {"LOR", 0x58000007},     {"BOR", 0x58000008},    {"LXOR", 0x58000009},
    {"BXOR", 0x5800000a},    {"MINLOC", 0x5800000b}, {"MAXLOC", 0x5800000c},
    {"REPLACE", 0x5800000d}, {"NO_OP", 0x5800000e},  {"EQUAL", 0x5800000f},
};

inline constexpr NamedHandle kLockTypes[] = {
    {"EXCLUSIVE", kLockExclusive},
    {"SHARED", kLockShared},
};

// Thread support levels (mpi.h:585).
inline constexpr NamedHandle kThreadLevels[] = {
    {"SINGLE", 0},
    {"FUNNELED", 1},
    {"SERIALIZED", 2},
    {"MULTIPLE", 3},
};

/// What the dialect records of a datatype: spmd.constDatatype keeps an MLIR
/// type, which drops the character/byte distinction but keeps signedness.  An
/// integer is signed (si32) or unsigned (ui32) when the datatype says which,
/// and signless (i8) when it does not: C leaves the signedness of char to the
/// platform, and MPI_BYTE is no number at all.
enum class DatatypeShape {
  Bool,
  Int8,
  SInt8,
  UInt8,
  SInt16,
  UInt16,
  SInt32,
  UInt32,
  SInt64,
  UInt64,
  Float32,
  Float64,
  Float128,
  Float32SInt32,
  Float64SInt32,
  SInt32SInt32,
};

struct DatatypeHandle {
  llvm::StringLiteral name;
  int32_t handle;
  DatatypeShape shape;
};

/// The predefined datatypes the raiser recognises (mpi.h:141-211).  Several
/// share a shape; the first entry of a shape is the one a datatype of that
/// shape lowers to.
inline constexpr DatatypeHandle kDatatypes[] = {
    {"MPI_C_BOOL", 0x4c00013f, DatatypeShape::Bool},
    {"MPI_CHAR", 0x4c000101, DatatypeShape::Int8},
    {"MPI_SIGNED_CHAR", 0x4c000118, DatatypeShape::SInt8},
    {"MPI_UNSIGNED_CHAR", 0x4c000102, DatatypeShape::UInt8},
    {"MPI_BYTE", 0x4c00010d, DatatypeShape::Int8},
    {"MPI_SHORT", 0x4c000203, DatatypeShape::SInt16},
    {"MPI_UNSIGNED_SHORT", 0x4c000204, DatatypeShape::UInt16},
    {"MPI_INT", 0x4c000405, DatatypeShape::SInt32},
    {"MPI_UNSIGNED", 0x4c000406, DatatypeShape::UInt32},
    {"MPI_LONG", 0x4c000807, DatatypeShape::SInt64},
    {"MPI_UNSIGNED_LONG", 0x4c000808, DatatypeShape::UInt64},
    {"MPI_LONG_LONG_INT", 0x4c000809, DatatypeShape::SInt64},
    {"MPI_UNSIGNED_LONG_LONG", 0x4c000819, DatatypeShape::UInt64},
    {"MPI_UINT64_T", 0x4c00083e, DatatypeShape::UInt64},
    {"MPI_FLOAT", 0x4c00040a, DatatypeShape::Float32},
    {"MPI_DOUBLE", 0x4c00080b, DatatypeShape::Float64},
    {"MPI_LONG_DOUBLE", 0x4c00100c, DatatypeShape::Float128},
    {"MPI_FLOAT_INT", static_cast<int32_t>(0x8c000000),
     DatatypeShape::Float32SInt32},
    {"MPI_DOUBLE_INT", static_cast<int32_t>(0x8c000001),
     DatatypeShape::Float64SInt32},
    {"MPI_2INT", 0x4c000816, DatatypeShape::SInt32SInt32},
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
  case DatatypeShape::Bool:
    return b.getI1Type();
  case DatatypeShape::Int8:
    return b.getI8Type();
  case DatatypeShape::SInt8:
    return sint(8);
  case DatatypeShape::UInt8:
    return uint(8);
  case DatatypeShape::SInt16:
    return sint(16);
  case DatatypeShape::UInt16:
    return uint(16);
  case DatatypeShape::SInt32:
    return sint(32);
  case DatatypeShape::UInt32:
    return uint(32);
  case DatatypeShape::SInt64:
    return sint(64);
  case DatatypeShape::UInt64:
    return uint(64);
  case DatatypeShape::Float32:
    return b.getF32Type();
  case DatatypeShape::Float64:
    return b.getF64Type();
  case DatatypeShape::Float128:
    return b.getF128Type();
  case DatatypeShape::Float32SInt32:
    return b.getTupleType({b.getF32Type(), sint(32)});
  case DatatypeShape::Float64SInt32:
    return b.getTupleType({b.getF64Type(), sint(32)});
  case DatatypeShape::SInt32SInt32:
    return b.getTupleType({sint(32), sint(32)});
  }
  llvm_unreachable("unknown datatype shape");
}

/// The type spmd.constDatatype records for the predefined datatype \p handle.
inline Type getDatatypeType(MLIRContext *ctx, int64_t handle) {
  for (const DatatypeHandle &entry : kDatatypes)
    if (entry.handle == handle)
      return getDatatypeShapeType(ctx, entry.shape);
  return Type();
}

/// The predefined datatype whose type is exactly \p type.  MPI_CHAR is the one
/// integer that leaves its signedness open, as C's char does.
inline std::optional<int32_t> getExactDatatypeHandle(Type type) {
  for (const DatatypeHandle &entry : kDatatypes)
    if (getDatatypeShapeType(type.getContext(), entry.shape) == type)
      return entry.handle;
  return std::nullopt;
}

/// The predefined datatype a spmd.constDatatype of \p type lowers to.  An
/// integer of unknown signedness that no datatype leaves open as well -- i32
/// -- lowers to the signed datatype of its width, which moves the same bits;
/// the lowering refuses it where signedness changes the result, for MAX and
/// MIN.
inline std::optional<int32_t> getDatatypeHandle(Type type) {
  if (std::optional<int32_t> handle = getExactDatatypeHandle(type))
    return handle;
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.isSignless())
    return getExactDatatypeHandle(IntegerType::get(
        type.getContext(), integer.getWidth(), IntegerType::Signed));
  return std::nullopt;
}

} // namespace mlir::spmd::abi::mpich

#endif // SPMD_TARGETABI_MPICH_H
