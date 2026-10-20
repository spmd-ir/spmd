// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Utils/MemoryOffsets.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::spmd {

bool isAddressType(Type type) {
  return isa<BaseMemRefType, LLVM::LLVMPointerType>(type);
}

/// Types DataLayout::getTypeSize answers for without aborting.
static bool hasLayoutSize(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<LLVM::LLVMPointerType>(type))
    return true;
  if (auto complex = dyn_cast<ComplexType>(type))
    return hasLayoutSize(complex.getElementType());
  if (auto vector = dyn_cast<VectorType>(type))
    return hasLayoutSize(vector.getElementType());
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type))
    return hasLayoutSize(array.getElementType());
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type))
    return !structType.isOpaque() &&
           llvm::all_of(structType.getBody(), hasLayoutSize);
  return false;
}

std::optional<int64_t> getFixedTypeByteSize(Type type, const DataLayout &dl) {
  if (!type || !hasLayoutSize(type))
    return std::nullopt;
  llvm::TypeSize size = dl.getTypeSize(type);
  if (size.isScalable())
    return std::nullopt;
  return static_cast<int64_t>(size.getFixedValue());
}

/// Byte offset of element `fieldIdx` of `structType`, following the same
/// padding rule LLVMStructType::getTypeSizeInBits uses.
std::optional<int64_t> structFieldByteOffset(LLVM::LLVMStructType structType,
                                             int64_t fieldIdx,
                                             const DataLayout &dl) {
  ArrayRef<Type> body = structType.getBody();
  if (fieldIdx < 0 || static_cast<size_t>(fieldIdx) >= body.size())
    return std::nullopt;

  uint64_t offset = 0;
  for (int64_t i = 0; i <= fieldIdx; ++i) {
    uint64_t alignment =
        structType.isPacked() ? 1 : dl.getTypeABIAlignment(body[i]);
    offset = llvm::alignTo(offset, alignment);
    if (i == fieldIdx)
      return static_cast<int64_t>(offset);
    llvm::TypeSize size = dl.getTypeSize(body[i]);
    if (size.isScalable())
      return std::nullopt;
    offset += size.getFixedValue();
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ByteOffset
//===----------------------------------------------------------------------===//

/// Adds `scale * value` as a term, merging it into an existing term for the
/// same value and dropping terms whose scale reaches zero.
static LogicalResult addTerm(ByteOffset &offset, Value value, int64_t scale) {
  if (scale == 0)
    return success();
  const void *key = value.getAsOpaquePointer();
  auto *it = llvm::lower_bound(
      offset.terms, key,
      [](const std::pair<Value, int64_t> &term, const void *k) {
        return term.first.getAsOpaquePointer() < k;
      });
  if (it != offset.terms.end() && it->first == value) {
    if (llvm::AddOverflow(it->second, scale, it->second))
      return failure();
    if (it->second == 0)
      offset.terms.erase(it);
    return success();
  }
  offset.terms.insert(it, {value, scale});
  return success();
}

LogicalResult ByteOffset::addScaled(Value index, int64_t scale) {
  APInt constantIndex;
  if (matchPattern(index, m_ConstantInt(&constantIndex))) {
    int64_t bytes;
    if (constantIndex.getSignificantBits() > 64 ||
        llvm::MulOverflow(constantIndex.getSExtValue(), scale, bytes) ||
        llvm::AddOverflow(constant, bytes, constant))
      return failure();
    return success();
  }
  return addTerm(*this, index, scale);
}

LogicalResult ByteOffset::add(const ByteOffset &other) {
  if (llvm::AddOverflow(constant, other.constant, constant))
    return failure();
  for (const auto &[value, scale] : other.terms)
    if (failed(addTerm(*this, value, scale)))
      return failure();
  return success();
}

std::optional<int64_t>
ByteOffset::constantDifference(const ByteOffset &other) const {
  int64_t difference;
  if (terms != other.terms ||
      llvm::SubOverflow(constant, other.constant, difference))
    return std::nullopt;
  return difference;
}

void ByteOffset::print(raw_ostream &os) const {
  os << constant;
  for (const auto &[value, scale] : terms) {
    os << " + " << scale << "*";
    value.printAsOperand(os, OpPrintingFlags());
  }
}

//===----------------------------------------------------------------------===//
// GEPs
//===----------------------------------------------------------------------===//

std::optional<ByteOffset> gepByteOffset(LLVM::GEPOp gep, const DataLayout &dl) {
  ByteOffset offset;
  auto indices = gep.getIndices();
  if (indices.empty())
    return offset;

  auto addIndex = [&](auto index, int64_t scale) -> LogicalResult {
    if (auto attr = dyn_cast<IntegerAttr>(index)) {
      int64_t bytes;
      if (llvm::MulOverflow(attr.getInt(), scale, bytes) ||
          llvm::AddOverflow(offset.constant, bytes, offset.constant))
        return failure();
      return success();
    }
    return offset.addScaled(cast<Value>(index), scale);
  };

  // Index 0 strides over the pointee type; the rest descend into it.
  Type current = gep.getElemType();
  std::optional<int64_t> elemSize = getFixedTypeByteSize(current, dl);
  if (!elemSize || failed(addIndex(indices[0], *elemSize)))
    return std::nullopt;

  for (unsigned i = 1, e = indices.size(); i < e; ++i) {
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(current)) {
      // Struct indices are constants by construction of the op.
      auto attr = dyn_cast<IntegerAttr>(indices[i]);
      if (!attr)
        return std::nullopt;
      std::optional<int64_t> fieldOffset =
          structFieldByteOffset(structType, attr.getInt(), dl);
      if (!fieldOffset ||
          llvm::AddOverflow(offset.constant, *fieldOffset, offset.constant))
        return std::nullopt;
      current = structType.getBody()[attr.getInt()];
      continue;
    }
    if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(current)) {
      current = arrayType.getElementType();
      std::optional<int64_t> size = getFixedTypeByteSize(current, dl);
      if (!size || failed(addIndex(indices[i], *size)))
        return std::nullopt;
      continue;
    }
    return std::nullopt;
  }
  return offset;
}

/// Byte offset a `llvm.getelementptr` adds to its base, if every index it uses
/// is a compile-time constant.
std::optional<int64_t> gepConstantByteOffset(LLVM::GEPOp gep,
                                             const DataLayout &dl) {
  // Only indices spelled as attributes count. An index passed as the result
  // of an arith.constant is not folded here, unlike in gepByteOffset: the
  // field-sensitive uniformity state is keyed on this offset, and folding more
  // would change which fields it keeps apart.
  if (llvm::any_of(gep.getIndices(),
                   [](auto index) { return isa<Value>(index); }))
    return std::nullopt;
  std::optional<ByteOffset> offset = gepByteOffset(gep, dl);
  if (!offset)
    return std::nullopt;
  return offset->constant;
}

//===----------------------------------------------------------------------===//
// Offsets of memory accesses
//===----------------------------------------------------------------------===//

/// Byte offset of `v` relative to the start of the allocation it points into.
/// Walks back through the address-forming ops; reaching a root (an allocation,
/// a global, a block argument, or anything else that starts a fresh alias
/// class) contributes nothing further, since the alias class already names it.
std::optional<int64_t> baseByteOffset(Value v, const DataLayout &dl) {
  int64_t total = 0;
  // Guard against a cycle through block arguments / unusual IR.
  for (unsigned steps = 0; steps < 64; ++steps) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return total; // block / function argument: root of this alias class

    if (isa<enzymexla::Pointer2MemrefOp, enzymexla::Memref2PointerOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast->getNumOperands() != 1)
        return std::nullopt;
      v = cast->getOperand(0);
      continue;
    }
    if (auto bitcast = dyn_cast<LLVM::BitcastOp>(def)) {
      v = bitcast.getArg();
      continue;
    }
    if (auto addrCast = dyn_cast<LLVM::AddrSpaceCastOp>(def)) {
      v = addrCast.getArg();
      continue;
    }
    if (auto gep = dyn_cast<LLVM::GEPOp>(def)) {
      std::optional<int64_t> delta = gepConstantByteOffset(gep, dl);
      if (!delta)
        return std::nullopt;
      total += *delta;
      v = gep.getBase();
      continue;
    }
    // Allocations, globals and calls (malloc & friends) each start their own
    // alias class, so the walk stops with the offset accumulated so far.
    if (isa<LLVM::AllocaOp, LLVM::AddressOfOp, memref::AllocaOp,
            memref::AllocOp>(def) ||
        isa<CallOpInterface>(def))
      return total;

    // Pointer arithmetic we cannot follow (ptrtoint/inttoptr, a loaded
    // pointer, ...): the offset is not statically known.
    return std::nullopt;
  }
  return std::nullopt;
}

/// Constant byte offset of the memory `op` touches through address `addr`, or
/// nullopt when it cannot be pinned down. `addr` is the value reported by the
/// memory effect, which for memref ops is the memref — the subscript lives on
/// the op, so both have to be combined here.
std::optional<int64_t> accessByteOffset(Operation *op, Value addr,
                                        const DataLayout &dl) {
  std::optional<int64_t> base = baseByteOffset(addr, dl);
  if (!base)
    return std::nullopt;

  auto memrefIndexOffset = [&](Value memref,
                               ValueRange indices) -> std::optional<int64_t> {
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType || memrefType.getRank() != 1 || indices.size() != 1)
      return std::nullopt;
    APInt constIdx;
    if (!matchPattern(indices[0], m_ConstantInt(&constIdx)))
      return std::nullopt;
    llvm::TypeSize elemSize = dl.getTypeSize(memrefType.getElementType());
    if (elemSize.isScalable())
      return std::nullopt;
    return constIdx.getSExtValue() *
           static_cast<int64_t>(elemSize.getFixedValue());
  };

  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    if (load.getMemRef() != addr)
      return std::nullopt;
    std::optional<int64_t> delta =
        memrefIndexOffset(load.getMemRef(), load.getIndices());
    if (!delta)
      return std::nullopt;
    return *base + *delta;
  }
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    if (store.getMemRef() != addr)
      return std::nullopt;
    std::optional<int64_t> delta =
        memrefIndexOffset(store.getMemRef(), store.getIndices());
    if (!delta)
      return std::nullopt;
    return *base + *delta;
  }
  if (isa<LLVM::LoadOp, LLVM::StoreOp>(op))
    return base;

  // Any other effecting op (calls, intrinsics, ...) may touch the object at an
  // offset we have no handle on.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Displacements of buffer addresses
//===----------------------------------------------------------------------===//

/// Displacement of a memref.subview from its source: each offset times that
/// dimension's stride, in elements of the source.
static std::optional<ByteOffset> subviewByteOffset(memref::SubViewOp subview,
                                                   const DataLayout &dl) {
  auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!sourceType)
    return std::nullopt;
  std::optional<int64_t> elemSize =
      getFixedTypeByteSize(sourceType.getElementType(), dl);
  SmallVector<int64_t> strides;
  int64_t sourceOffset = 0;
  if (!elemSize ||
      failed(sourceType.getStridesAndOffset(strides, sourceOffset)))
    return std::nullopt;

  SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();
  if (offsets.size() != strides.size())
    return std::nullopt;

  ByteOffset offset;
  for (auto [dimOffset, stride] : llvm::zip_equal(offsets, strides)) {
    int64_t scale;
    if (ShapedType::isDynamic(stride) ||
        llvm::MulOverflow(stride, *elemSize, scale))
      return std::nullopt;
    if (auto value = dyn_cast<Value>(dimOffset)) {
      if (failed(offset.addScaled(value, scale)))
        return std::nullopt;
      continue;
    }
    auto attr = dyn_cast<IntegerAttr>(cast<Attribute>(dimOffset));
    int64_t bytes;
    if (!attr || llvm::MulOverflow(attr.getInt(), scale, bytes) ||
        llvm::AddOverflow(offset.constant, bytes, offset.constant))
      return std::nullopt;
  }
  return offset;
}

/// Displacement of an enzymexla.subindex from its source: the index times the
/// size of one slab below the leading dimension.
static std::optional<ByteOffset> subindexByteOffset(enzymexla::SubIndexOp sub,
                                                    const DataLayout &dl) {
  auto sourceType = dyn_cast<MemRefType>(sub.getSource().getType());
  if (!sourceType)
    return std::nullopt;
  std::optional<int64_t> slab =
      getFixedTypeByteSize(sourceType.getElementType(), dl);
  if (!slab)
    return std::nullopt;
  for (int64_t dim : sourceType.getShape().drop_front())
    if (ShapedType::isDynamic(dim) || llvm::MulOverflow(*slab, dim, *slab))
      return std::nullopt;

  ByteOffset offset;
  if (failed(offset.addScaled(sub.getIndex(), *slab)))
    return std::nullopt;
  return offset;
}

std::optional<AddressStep> getAddressStep(Operation *op, const DataLayout &dl) {
  if (!op)
    return std::nullopt;

  // Casts and bridges that only re-type the address.
  if (isa<enzymexla::Pointer2MemrefOp, enzymexla::Memref2PointerOp,
          LLVM::BitcastOp, LLVM::AddrSpaceCastOp, memref::CastOp>(op) ||
      (isa<UnrealizedConversionCastOp>(op) && op->getNumOperands() == 1 &&
       op->getNumResults() == 1 && isAddressType(op->getResult(0).getType()))) {
    Value source = op->getOperand(0);
    if (!isAddressType(source.getType()))
      return std::nullopt;
    return AddressStep{source, ByteOffset()};
  }

  if (auto gep = dyn_cast<LLVM::GEPOp>(op))
    return AddressStep{gep.getBase(), gepByteOffset(gep, dl)};
  if (auto sub = dyn_cast<enzymexla::SubIndexOp>(op))
    return AddressStep{sub.getSource(), subindexByteOffset(sub, dl)};
  if (auto subview = dyn_cast<memref::SubViewOp>(op))
    return AddressStep{subview.getSource(), subviewByteOffset(subview, dl)};

  // Any other view - memref.view's byte shift, memref.reinterpret_cast, a view
  // of another dialect - is known to point into its source, but not where.
  if (auto view = dyn_cast<ViewLikeOpInterface>(op))
    return AddressStep{view.getViewSource(), std::nullopt};
  return std::nullopt;
}

BufferSlice resolveBufferSlice(Value address, const DataLayout &dl) {
  BufferSlice slice{address, ByteOffset()};
  while (std::optional<AddressStep> step =
             getAddressStep(slice.base.getDefiningOp(), dl)) {
    if (slice.offset &&
        (!step->offset || failed(slice.offset->add(*step->offset))))
      slice.offset = std::nullopt;
    slice.base = step->source;
  }
  return slice;
}

} // namespace mlir::spmd
