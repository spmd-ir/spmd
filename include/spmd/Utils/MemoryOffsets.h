/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- MemoryOffsets.h - Byte offsets of memory accesses --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Recovering where inside its allocation an address points.
//
// Two kinds of client need this. An analysis keyed on alias classes wants the
// constant byte offset a memory access touches, to tell fields apart
// (baseByteOffset, accessByteOffset). A transformation comparing two buffers
// wants their displacement even when it is not constant, since `&u[i * n]` and
// `&u[i * n + 4]` are still a known distance apart (ByteOffset,
// resolveBufferSlice). Both rest on the same GEP arithmetic, gepByteOffset.
//
// Enzyme's alias classes identify an *allocation*, not a location inside it:
// `AliasAnalysis::visitOperation` gives an `llvm.getelementptr` result the
// base's class verbatim, which is what keeps may-alias queries sound. Two
// fields of the same struct are therefore indistinguishable to the alias
// analysis alone. A client that wants field sensitivity gets it by keying its
// own state on (alias class, byte offset) and using these helpers to compute
// the offset, leaving the alias lattice as coarse — and as sound — as it was.
//
// Anything not understood yields nullopt, and the caller is expected to fall
// back to whole-object behaviour (conservative in both directions: a write it
// cannot pin down taints the object, a read it cannot pin down observes it).
//
// MLIR has no equivalent of LLVM IR's `GEPOperator::accumulateConstantOffset`,
// which is why this exists here.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_UTILS_MEMORYOFFSETS_H
#define SPMD_UTILS_MEMORYOFFSETS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::spmd {

/// The types an address has: `!llvm.ptr` and memrefs.
bool isAddressType(Type type);

/// Byte size of `type` when the data layout gives it a fixed one. Unlike
/// DataLayout::getTypeSize, which aborts on a type without layout information
/// (the `!spmd.req` elements of a request array, say), this answers nullopt.
std::optional<int64_t> getFixedTypeByteSize(Type type, const DataLayout &dl);

/// Byte offset of element `fieldIdx` of `structType`, following the same
/// padding rule LLVMStructType::getTypeSizeInBits uses.
std::optional<int64_t> structFieldByteOffset(LLVM::LLVMStructType structType,
                                             int64_t fieldIdx,
                                             const DataLayout &dl);

/// A byte displacement: a constant plus a sum of symbolic `scale * value`
/// terms.
///
/// It is affine and syntactic. Two displacements are compared by cancelling
/// identical terms, so `&u[i * n]` and `&u[i * n + 4]` differ by a constant
/// although neither is constant - the shape of a halo exchange in a loop. No
/// value is evaluated and no range is consulted: a term cancels or it does not.
struct ByteOffset {
  int64_t constant = 0;
  /// (value, bytes per unit of that value), sorted by the value's opaque
  /// pointer with equal values merged and zero scales dropped, so that equal
  /// displacements are equal field by field.
  SmallVector<std::pair<Value, int64_t>, 2> terms;

  bool isConstant() const { return terms.empty(); }

  /// Adds `scale * index` bytes, folding an index that is a constant. Fails on
  /// signed overflow, leaving the displacement unspecified.
  LogicalResult addScaled(Value index, int64_t scale);

  /// Adds `other`. Fails on signed overflow, leaving the displacement
  /// unspecified.
  LogicalResult add(const ByteOffset &other);

  bool operator==(const ByteOffset &other) const {
    return constant == other.constant && terms == other.terms;
  }
  bool operator!=(const ByteOffset &other) const { return !(*this == other); }

  /// `*this - other` in bytes, when the terms cancel exactly. nullopt means
  /// "not a known constant distance apart", never "far apart".
  std::optional<int64_t> constantDifference(const ByteOffset &other) const;

  /// Prints the constant followed by ` + <scale>*<value>` per term. With more
  /// than one term the order follows the values' addresses and can differ
  /// between runs.
  void print(raw_ostream &os) const;
};

/// Displacement a `llvm.getelementptr` adds to its base. Constant indices fold
/// into the constant, whether they stride over the element type or descend
/// into a struct or array; a dynamic index becomes a term scaled by the size of
/// what it strides over. nullopt when a type involved has no fixed size, or the
/// indices descend into something other than a struct or an array.
std::optional<ByteOffset> gepByteOffset(LLVM::GEPOp gep, const DataLayout &dl);

/// Byte offset a `llvm.getelementptr` adds to its base, if every index it uses
/// is a compile-time constant.
std::optional<int64_t> gepConstantByteOffset(LLVM::GEPOp gep,
                                             const DataLayout &dl);

/// One step of an address computation.
struct AddressStep {
  /// The address the op's result is computed from.
  Value source;
  /// Displacement of the result from `source`, or nullopt when it cannot be
  /// recovered.
  std::optional<ByteOffset> offset;
};

/// When `op` computes an address from another address - a pointer or memref
/// cast, the enzymexla memref/pointer bridges, a GEP, enzymexla.subindex, a
/// memref view - the address it starts from and the displacement it adds.
/// nullopt for any other op, allocations included.
std::optional<AddressStep> getAddressStep(Operation *op, const DataLayout &dl);

/// An address resolved to the address it is computed from.
struct BufferSlice {
  /// The first address along the chain that no AddressStep computes: an
  /// allocation, a global, a block argument, a loaded pointer.
  Value base;
  /// Displacement of the address from `base`; nullopt once a step's
  /// displacement cannot be recovered.
  std::optional<ByteOffset> offset;
};

/// Follows `address` back through getAddressStep as far as it goes.
BufferSlice resolveBufferSlice(Value address, const DataLayout &dl);

/// Byte offset of `v` relative to the start of the allocation it points into.
/// Walks back through the address-forming ops; reaching a root (an allocation,
/// a global, a block argument, or anything else that starts a fresh alias
/// class) contributes nothing further, since the alias class already names it.
std::optional<int64_t> baseByteOffset(Value v, const DataLayout &dl);

/// Constant byte offset of the memory `op` touches through address `addr`, or
/// nullopt when it cannot be pinned down. `addr` is the value reported by the
/// memory effect, which for memref ops is the memref — the subscript lives on
/// the op, so both have to be combined here.
std::optional<int64_t> accessByteOffset(Operation *op, Value addr,
                                        const DataLayout &dl);

} // namespace mlir::spmd

#endif // SPMD_UTILS_MEMORYOFFSETS_H
