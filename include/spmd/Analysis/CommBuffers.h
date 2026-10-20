/*
 * SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- CommBuffers.h - Memory moved by communication ops --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What a transformation of communication needs to know about the memory a
// communication op moves: which ops move data at all, which of their declared
// effects concern program memory, whether a buffer lives in an allocation
// private to its function, and every operation that reads or writes it.
//
// These are local facts computed by walking SSA uses, not a dataflow analysis.
// Whatever is not understood makes the query fail, and the caller then has to
// treat the buffer as both read and written.
//
// Which ranks execute an op, and whether they agree on a value, is answered by
// ExecutionContextAnalysis (isUniformlyExecuted, isUniformValue); which
// communicator or request a handle denotes, by CommOriginsAnalysis and
// HandleOriginsAnalysis.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_ANALYSIS_COMMBUFFERS_H
#define SPMD_ANALYSIS_COMMBUFFERS_H

#include "spmd/SPMDOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// Classification
//===----------------------------------------------------------------------===//

/// A collective that moves data. spmd.barrier moves none, and the OpenMP
/// constructs that carry the Collective trait are no transfer of their own.
bool isDataMovingCollective(Operation *op);

/// The op is ordered relative to a device stream (NCCL, NVSHMEM, CUDA) rather
/// than to program order, so when it reads and writes cannot be read off the
/// CFG.
bool hasStreamOperand(Operation *op);

/// The op is a non-blocking form: it hands out a request, and its buffers are
/// read and written by the time the request completes rather than at the op.
bool isNonBlocking(Operation *op);

/// Byte size of one element of the datatype `datatype` denotes, when it is a
/// spmd.constDatatype of a type with a fixed size.
std::optional<int64_t> getDatatypeByteSize(Value datatype,
                                           const DataLayout &dl);

//===----------------------------------------------------------------------===//
// Memory effects
//===----------------------------------------------------------------------===//

/// Appends the effects `op` declares on program memory to `effects`.
///
/// The SPMD dialect also declares effects on its own resources - the runtime,
/// communicators, requests, windows, streams, messages. Those order operations
/// against each other; none of them is memory a buffer lives in, and a
/// communicator effect is attached to the communicator value, not to an
/// address. They are dropped here, so that an optimization reasoning about
/// buffers does not have to give up on every communication op. An effect on
/// program memory with no value attached is kept: it may touch any memory.
///
/// Fails when `op` does not describe its memory effects at all.
LogicalResult getProgramMemoryEffects(
    Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects);

/// Appends the addresses `op` writes or frees through to `written`. Fails when
/// `op` does not describe its memory effects, or declares a write to program
/// memory that is not tied to an address.
LogicalResult collectWrittenAddresses(Operation *op,
                                      SmallVectorImpl<Value> &written);

//===----------------------------------------------------------------------===//
// Buffers
//===----------------------------------------------------------------------===//

/// An allocation private to its function (llvm.alloca, memref.alloca,
/// memref.alloc), whose every access is a use of its SSA value. Symmetric
/// memory from spmd.malloc or spmd.winAlloc is not: another rank reads and
/// writes it without any local use showing.
bool isPrivateAllocation(Value value);

/// An intrinsic that annotates an allocation - its lifetime, its invariance,
/// its debug information - without reading, writing or capturing it. Clang
/// emits lifetime markers around every stack buffer, so a use-walk that did
/// not recognise them would give up on essentially all real input.
bool isPointerAnnotation(Operation *op);

/// Everything the uses of a private allocation tell about the memory behind
/// one buffer.
struct BufferFacts {
  /// The private allocation the buffer points into.
  Value allocation;
  /// Every address computed from `allocation`, `allocation` included.
  llvm::SetVector<Value> aliases;
  /// Every operation that reads through one of `aliases`.
  llvm::SetVector<Operation *> readers;
  /// Every operation that writes or frees through one of `aliases`.
  llvm::SetVector<Operation *> writers;
};

/// Computes the facts for the allocation behind `buffer`. Fails when that is
/// not a private allocation, or when an address computed from it reaches an
/// operation whose effect on it is not declared - a call, a store of the
/// address itself, a region capturing it - since the memory can then be
/// reached in ways no use shows.
LogicalResult computeBufferFacts(Value buffer, const DataLayout &dl,
                                 BufferFacts &facts);

/// `op`, or an operation nested in it, reads the buffer `facts` describes.
bool readsBuffer(Operation *op, const BufferFacts &facts);

/// `op`, or an operation nested in it, writes the buffer `facts` describes.
bool writesBuffer(Operation *op, const BufferFacts &facts);

//===----------------------------------------------------------------------===//
// Accesses away from the op
//===----------------------------------------------------------------------===//

/// `op` may access memory it names at another time than where it sits in the
/// program: a non-blocking or persistent communication op reads and writes its
/// buffers by the time its request completes, and an RMA op exposes memory to
/// other ranks, which access it at any time.
bool accessesMemoryAwayFromOp(Operation *op);

/// A reader or writer of the buffer `facts` describes accesses it away from
/// where that reader or writer sits (see accessesMemoryAwayFromOp).
bool isAccessedAwayFromOps(const BufferFacts &facts);

/// `op` runs inside a thread team, a thread-level spmd.parallel, whose other
/// threads execute the same region concurrently.
bool isInThreadTeam(Operation *op);

} // namespace mlir::spmd

#endif // SPMD_ANALYSIS_COMMBUFFERS_H
