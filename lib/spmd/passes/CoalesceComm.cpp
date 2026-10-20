// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- CoalesceComm.cpp - SPMD Pass -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Point-to-point message coalescing: several blocking sends (or receives) to
// one peer become a single larger message.
//
// This is the first transformation in these passes that is not a whole-op
// textual removal, so the argument of -eliminate-dead-comm and
// -eliminate-redundant-comm does not carry over.  docs/optimizations.md,
// "Coalescing", gives the one that replaces it.  In short:
//
//   * A merge is sound only if the peer merges to match, and the two sides are
//     transformed independently.  So the grouping is a function of the tags
//     alone: members are ordered by ascending tag, the merged operation
//     carries the least tag, and a group is legal only when every operation in
//     the module using one of its tags, on its communicator, belongs to a
//     surviving group of exactly those tags.  Per-side maximality does not
//     give this - `send(t0); send(t1)` against `recv(t0); store; recv(t1)`
//     groups on one side only - which is why this pass is module-scoped, and
//     why it assumes the module is the whole program.
//
//   * Uniform execution is not required.  A group is a run inside a single
//     block, so any rank executing the first member executes all of them, and
//     collapsing the run keeps the ranks' sequences of messages aligned.  The
//     grouping keys on the SSA identity of the peer expression, not on its
//     value: real halo code sends to `rank +/- 1` under `if (rank < size-1)`,
//     where a uniformity gate would reject every realistic input.
//
// Two rewrite tiers:
//
//   * zero-copy, when the pieces are already adjacent in one allocation: only
//     the count changes.
//   * staged, which packs through a scratch buffer.
//
// Either way the merged message has boundaries between its pieces that each
// endpoint computes on its own, from its own counts.  MPI lets a receive post
// more elements than arrive, so a receiver whose first piece is larger than
// the sender's would find the second piece at the sender's boundary, not its
// own.  Both tiers therefore need every use of a tag, module-wide, to agree on
// one compile-time-constant count and element type.  The zero-copy tier is
// tried first because it needs no copies.
//
//===----------------------------------------------------------------------===//

#include "spmd/Analysis/CommBuffers.h"
#include "spmd/Analysis/CommOriginsAnalysis.h"
#include "spmd/Analysis/HandleSlotAnalysis.h"
#include "spmd/SPMDOps.h"
#include "spmd/SPMDPasses.h"
#include "spmd/Utils/MemoryOffsets.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

namespace mlir::spmd {

#define GEN_PASS_DEF_COALESCECOMM
#include "spmd/SPMDPasses.h.inc"

namespace {

/// The usedModel of MPI operations (CACTS' APIKind::MPI).
static constexpr int32_t kMPIModel = 0;

//===----------------------------------------------------------------------===//
// Point-to-point classification
//===----------------------------------------------------------------------===//

enum class MessageDirection { Send, Recv };

/// One side of a point-to-point transfer, in the form a coalescing or motion
/// pass reasons about it.
struct P2PMessage {
  MessageDirection direction;
  Value comm;
  Value buffer;
  Value count;
  Value datatype;
  /// destRank for a send, srcRank for a recv.
  ///
  /// Deliberately *not* required to be uniform.  In a halo exchange the peer is
  /// `rank +/- 1`, so its runtime value differs on every rank; what a rewrite
  /// needs is that all ranks evaluate the same *expression*, which is SSA
  /// identity.  Compare this field with `==`, never through a uniformity query.
  Value peer;
  Value tag;
  /// Protocol attributes.  They change what the call does on the wire, so a
  /// group may not mix them.
  bool blocking = true;
  bool isBuffered = false;
  bool isReadyMode = false;
  int32_t usedModel = 0;
  /// The operands of the buffer and the count, which a rewrite changes.
  OpOperand *bufferOperand = nullptr;
  OpOperand *countOperand = nullptr;
};

/// The point-to-point transfer `op` performs, if it performs exactly one.
///
/// Answers nullopt for spmd.sendrecv (two transfers in one op), for the
/// persistent spmd.sendInit/recvInit (spmd.start re-arms the request, so the
/// transfer's lifetime is not the block's), and for spmd.mRecv/imRecv (matched
/// by message handle rather than by tag).  Those ops can never be part of a
/// coalesced group - but their tags still have to be *seen*, which is what
/// collectP2PTagUses is for.
static std::optional<P2PMessage> getP2PMessage(Operation *op) {
  if (auto send = dyn_cast<SendOp>(op)) {
    CommBufferOperands operands = send.getCommBufferOperands();
    return P2PMessage{MessageDirection::Send,
                      send.getComm(),
                      CommBufferOperands::valueOf(operands.sendBuffer),
                      CommBufferOperands::valueOf(operands.sendCount),
                      CommBufferOperands::valueOf(operands.sendDatatype),
                      send.getDestRank(),
                      send.getTag(),
                      send.getIsBlocking(),
                      send.getIsBuffered(),
                      send.getIsReadyMode(),
                      static_cast<int32_t>(send.getUsedModel()),
                      operands.sendBuffer,
                      operands.sendCount};
  }

  if (auto recv = dyn_cast<RecvOp>(op)) {
    CommBufferOperands operands = recv.getCommBufferOperands();
    return P2PMessage{MessageDirection::Recv,
                      recv.getComm(),
                      CommBufferOperands::valueOf(operands.recvBuffer),
                      CommBufferOperands::valueOf(operands.recvCount),
                      CommBufferOperands::valueOf(operands.recvDatatype),
                      recv.getSrcRank(),
                      recv.getTag(),
                      recv.getIsBlocking(),
                      /*isBuffered=*/false,
                      /*isReadyMode=*/false,
                      static_cast<int32_t>(recv.getUsedModel()),
                      operands.recvBuffer,
                      operands.recvCount};
  }

  // Everything else either performs two transfers (sendrecv), outlives its
  // block (the persistent pair), or is matched by something other than a tag
  // (mRecv/imMRecv, via mProbe).  See collectP2PTagUses.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Communicator identity
//===----------------------------------------------------------------------===//

/// A communicator, identified structurally rather than by SSA value.
///
/// The tag census is module-wide, and `spmd.commWorld` in two functions is two
/// SSA values denoting one communicator - so keying the census on Value would
/// let a tag be merged in one function while the matching operation in another
/// function was never seen.  Operation pointers are at least 4-byte aligned, so
/// the small integers are safe sentinels.
using CommId = uintptr_t;
static constexpr CommId kCommWorld = 1;
static constexpr CommId kCommNode = 2;

/// The communicators `comm` may denote at `op`, as CommOriginsAnalysis and
/// HandleSlotAnalysis resolve it through arguments, casts and memory.  Empty
/// when it may denote one they do not know.
static SmallVector<CommId> getCommIds(DataFlowSolver &solver, Value comm,
                                      Operation *op) {
  HandleFactsValue facts =
      resolveHandleFacts(solver, comm, op, HandleKind::Communicator);
  // No atoms at all means the analysis never reached the value.
  if (facts.unknown || facts.atoms.empty())
    return {};

  SmallVector<CommId> ids;
  for (Value atom : facts.atoms) {
    Operation *def = atom.getDefiningOp();
    if (!def)
      return {};
    if (isa<CommWorldOp>(def))
      ids.push_back(kCommWorld);
    else if (isa<CommNodeOp>(def))
      ids.push_back(kCommNode);
    else
      ids.push_back(reinterpret_cast<CommId>(def));
  }
  llvm::sort(ids);
  ids.erase(llvm::unique(ids), ids.end());
  return ids;
}

//===----------------------------------------------------------------------===//
// Census
//===----------------------------------------------------------------------===//

/// What one tag is used for, module-wide.
struct TagShape {
  /// Compile-time element count, when every use agrees on one.
  std::optional<int64_t> count;
  /// Element type behind the datatype operand, when every use agrees.
  Type elementType;
  /// Cleared as soon as two uses disagree, or one cannot be read.
  bool consistent = true;
};

struct CommCensus {
  /// No merging at all on this communicator.  Set by a wildcard or
  /// non-constant tag, or a wildcard peer.
  bool poisoned = false;
  llvm::DenseMap<int64_t, TagShape> shapes;
  /// Every operation that matches an envelope carrying this tag - including
  /// the ones that can never be grouped.  This is what validateGlobally reads.
  llvm::DenseMap<int64_t, SmallVector<Operation *>> users;
};

//===----------------------------------------------------------------------===//
// Groups
//===----------------------------------------------------------------------===//

struct Member {
  Operation *op;
  /// Index of the member in the run, in the order of the block.
  unsigned position;
  P2PMessage msg;
  int64_t tag;
  BufferSlice slice;
  /// count * sizeof(datatype), symbolic when the count is not constant.
  std::optional<ByteOffset> extent;
  /// The same, when it is a compile-time constant.
  std::optional<int64_t> constantExtent;
};

struct Group {
  SmallVector<Member> members; ///< sorted by ascending tag
  Operation *lastInBlock = nullptr;
  CommId comm = 0;
  int64_t elemSize = 0;
  bool contiguous = false; ///< the zero-copy tier applies
  bool valid = true;

  SmallVector<int64_t> tagSet() const {
    SmallVector<int64_t> tags;
    for (const Member &m : members)
      tags.push_back(m.tag);
    return tags;
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Every (communicator, tag) pair `op` participates in, appended to `uses`.
///
/// Covers every operation that can match a point-to-point envelope, including
/// the ones getP2PMessage rejects: a tag belonging to an op that cannot be
/// grouped must still prevent that tag from being merged into someone else's
/// message, or the two endpoints stop agreeing on the wire format.
///
/// A tag that is not a compile-time constant is reported as nullopt, which
/// poisons its communicator.  MPI_ANY_TAG and MPI_ANY_SOURCE raise to negative
/// constants, so the caller treats those the same way.
static void collectP2PTagUses(
    Operation *op,
    SmallVectorImpl<std::pair<Value, std::optional<int64_t>>> &uses) {
  auto record = [&](Value comm, Value tag) {
    uses.emplace_back(comm, getConstantIntValue(tag));
  };

  if (auto send = dyn_cast<SendOp>(op))
    return record(send.getComm(), send.getTag());
  if (auto recv = dyn_cast<RecvOp>(op))
    return record(recv.getComm(), recv.getTag());

  // Two envelopes, one op: both tags are spoken for, and neither can be
  // rewritten, so both block any partition containing them.
  if (auto sendrecv = dyn_cast<SendrecvOp>(op)) {
    record(sendrecv.getComm(), sendrecv.getSendTag());
    record(sendrecv.getComm(), sendrecv.getRecvTag());
    return;
  }

  if (auto sendInit = dyn_cast<SendInitOp>(op))
    return record(sendInit.getComm(), sendInit.getTag());
  if (auto recvInit = dyn_cast<RecvInitOp>(op))
    return record(recvInit.getComm(), recvInit.getTag());

  // A matching probe consumes a message with this envelope; the mRecv that
  // follows carries only the resulting handle, so this is the op that speaks
  // for both.
  if (auto probe = dyn_cast<MProbeOp>(op))
    return record(probe.getComm(), probe.getTag());
}

/// `count` elements of `elemSize` bytes, as a displacement.  A non-constant
/// count becomes a symbolic term rather than a bail-out: that is what lets
/// `&u[0]` of n elements and `&u[n]` of m be recognised as adjacent.
static std::optional<ByteOffset> extentOf(Value count, int64_t elemSize) {
  ByteOffset extent;
  if (failed(extent.addScaled(count, elemSize)))
    return std::nullopt;
  return extent;
}

/// The function whose entry block holds a staged group's scratch buffer: the
/// nearest op isolated from above that encloses `op`, when that is a function
/// with a body.  A scratch buffer anywhere else would either not exist or not
/// be visible from the group.
static FunctionOpInterface getStagingScope(Operation *op) {
  auto func = dyn_cast_or_null<FunctionOpInterface>(
      op->getParentWithTrait<OpTrait::IsIsolatedFromAbove>());
  if (!func || func.isExternal())
    return {};
  return func;
}

/// Whether the staged tier can copy to and from `buffer`: a pointer, or a
/// memref of host memory laid out as the identity, whose elements start where
/// it does.
static bool hasStagingAddress(Value buffer) {
  if (isa<LLVM::LLVMPointerType>(buffer.getType()))
    return true;
  auto memref = dyn_cast<MemRefType>(buffer.getType());
  return memref && (!memref.getLayout() || memref.getLayout().isIdentity()) &&
         !memref.getMemorySpace();
}

/// The address the staged tier copies to or from; see hasStagingAddress.
static Value getStagingAddress(RewriterBase &rewriter, Location loc,
                               Value buffer) {
  if (isa<LLVM::LLVMPointerType>(buffer.getType()))
    return buffer;
  return rewriter.createOrFold<enzymexla::Memref2PointerOp>(
      loc, LLVM::LLVMPointerType::get(rewriter.getContext()), buffer);
}

/// Can `op` be a member of a group at all?
static bool isEligibleMember(Operation *op) {
  std::optional<P2PMessage> msg = getP2PMessage(op);
  if (!msg)
    return false;
  // Tag matching is MPI's; the other models' send/recv carry a tag operand
  // without those semantics.
  if (msg->usedModel != kMPIModel)
    return false;
  if (!msg->blocking || isNonBlocking(op) || hasStreamOperand(op))
    return false;
  // Other threads run the same block concurrently, so what lies between two
  // members in the block says nothing about what happens between them.
  if (isInThreadTeam(op))
    return false;
  // An observed error code is an observation of the individual message.
  for (Value result : op->getResults())
    if (!result.use_empty())
      return false;
  return true;
}

/// The envelope: everything two messages must share to be merged.  Compared by
/// SSA identity throughout - notably including the peer, which is `rank +/- 1`
/// in any real halo exchange and therefore not uniform.
static bool sameEnvelope(const P2PMessage &a, const P2PMessage &b) {
  return a.direction == b.direction && a.comm == b.comm && a.peer == b.peer &&
         a.datatype == b.datatype && a.blocking == b.blocking &&
         a.isBuffered == b.isBuffered && a.isReadyMode == b.isReadyMode &&
         a.usedModel == b.usedModel;
}

/// May `op` sit between two members without preventing the merge?
///
/// `facts` holds the escape analysis for each member's buffer when it was
/// available.  When it was not, `strict` demands the operation be free of
/// memory effects entirely - a coarser test that still lets textually adjacent
/// messages merge.
static bool isTransparentBetween(Operation *op, ArrayRef<BufferFacts> facts,
                                 bool strict) {
  if (isPointerAnnotation(op))
    return true;

  // Merging moves the earlier members past `op`, so nothing inside it may be
  // ordered against them.  Communication is: MPI's non-overtaking guarantee
  // holds within an envelope, and moving a blocking send across a receive can
  // deadlock outright.  So is every other op of the dialect that does anything
  // at all - a wait, a lock, an abort, a finalize - and so is a call or an op
  // with undescribed effects, which may do any of these.  This holds for ops
  // nested in `op`'s regions as much as for `op` itself.
  WalkResult result = op->walk([&](Operation *nested) {
    if (isa<CallOpInterface>(nested) ||
        (isSpmdOp(nested) && !isMemoryEffectFree(nested)) ||
        (!isa<MemoryEffectOpInterface>(nested) &&
         !nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return false;

  if (strict)
    return isMemoryEffectFree(op);

  for (const BufferFacts &fact : facts)
    if (readsBuffer(op, fact) || writesBuffer(op, fact))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct CoalesceComm : impl::CoalesceCommBase<CoalesceComm> {
  using CoalesceCommBase::CoalesceCommBase;

  void runOnOperation() override;

private:
  void takeCensus(ModuleOp moduleOp);
  void recordTagUse(CommCensus &entry, Operation *op,
                    std::optional<int64_t> tag);
  void discoverGroups(ModuleOp moduleOp, const DataLayout &dl);
  void collectBlockGroups(Block &block, const DataLayout &dl);
  void applyCostHeuristic();
  void validateGlobally();
  void rewrite(RewriterBase &rewriter, Group &group);
  void rewriteContiguous(RewriterBase &rewriter, Group &group);
  void rewriteStaged(RewriterBase &rewriter, Group &group);

  /// Solved CommOriginsAnalysis, for getCommIds.
  DataFlowSolver *solver = nullptr;
  llvm::DenseMap<CommId, CommCensus> census;
  /// A message uses a communicator that could be any of them.
  bool poisonAll = false;
  SmallVector<Group, 4> groups;
  unsigned fragmented = 0;
};

//===----------------------------------------------------------------------===//
// Phase A - census
//===----------------------------------------------------------------------===//

void CoalesceComm::takeCensus(ModuleOp moduleOp) {
  moduleOp.walk([&](Operation *op) {
    SmallVector<std::pair<Value, std::optional<int64_t>>> uses;
    collectP2PTagUses(op, uses);
    if (uses.empty())
      return;

    for (auto &[comm, tag] : uses) {
      // An unrecognisable communicator could be any of them, so it has to
      // poison every communicator - also the ones the walk has not reached.
      SmallVector<CommId> ids = getCommIds(*solver, comm, op);
      if (ids.empty()) {
        poisonAll = true;
        continue;
      }
      // A message that may use one of several communicators speaks for its tag
      // on each; getting no group of its own, it keeps the tag from merging.
      for (CommId id : ids)
        recordTagUse(census[id], op, tag);
    }
  });
}

void CoalesceComm::recordTagUse(CommCensus &entry, Operation *op,
                                std::optional<int64_t> tag) {
  // MPI_ANY_TAG is -1 and MPI_ANY_SOURCE is -2, so a negative constant is a
  // complete wildcard test for this shim's headers.
  if (!tag || *tag < 0) {
    entry.poisoned = true;
    return;
  }

  entry.users[*tag].push_back(op);

  TagShape &shape = entry.shapes[*tag];
  std::optional<P2PMessage> msg = getP2PMessage(op);
  if (!msg) {
    // A tag whose shape cannot be read is still a tag that is spoken for.
    shape.consistent = false;
    return;
  }
  if (getConstantIntValue(msg->peer).value_or(0) < 0)
    entry.poisoned = true;

  std::optional<int64_t> count = getConstantIntValue(msg->count);
  // Compared without signedness: an MPI_INT send packs the same bytes as the
  // MPI_UNSIGNED receive that matches it.
  Type elementType;
  if (auto constDatatype = msg->datatype.getDefiningOp<ConstDatatypeOp>()) {
    elementType = constDatatype.getTypeAttr();
    if (auto integer = dyn_cast<IntegerType>(elementType))
      elementType = IntegerType::get(integer.getContext(), integer.getWidth());
  }

  if (!count || !elementType) {
    shape.consistent = false;
    return;
  }
  if (!shape.count) {
    shape.count = count;
    shape.elementType = elementType;
  } else if (*shape.count != *count || shape.elementType != elementType) {
    shape.consistent = false;
  }
}

//===----------------------------------------------------------------------===//
// Phase B - discovery
//===----------------------------------------------------------------------===//

void CoalesceComm::discoverGroups(ModuleOp moduleOp, const DataLayout &dl) {
  moduleOp.walk([&](Block *block) { collectBlockGroups(*block, dl); });
}

void CoalesceComm::collectBlockGroups(Block &block, const DataLayout &dl) {
  SmallVector<Operation *> ops;
  for (Operation &op : block)
    ops.push_back(&op);

  for (size_t start = 0; start < ops.size(); ++start) {
    if (!isEligibleMember(ops[start]))
      continue;

    P2PMessage head = *getP2PMessage(ops[start]);
    std::optional<int64_t> headTag = getConstantIntValue(head.tag);
    std::optional<int64_t> elemSize = getDatatypeByteSize(head.datatype, dl);
    if (!headTag || *headTag < 0 || !elemSize || *elemSize <= 0)
      continue;
    // Members share the communicator value, so they share what it resolves to.
    SmallVector<CommId> headComms = getCommIds(*solver, head.comm, ops[start]);
    if (headComms.size() != 1)
      continue;

    SmallVector<Operation *> chosen{ops[start]};
    SmallVector<int64_t> tags{*headTag};
    SmallVector<BufferFacts> facts;
    bool precise = true;
    {
      BufferFacts fact;
      if (succeeded(computeBufferFacts(head.buffer, dl, fact))) {
        // Merging moves where the buffer is read or written; that is only
        // unobservable if every other access sits where its op does.
        if (isAccessedAwayFromOps(fact))
          continue;
        facts.push_back(std::move(fact));
      } else {
        precise = false;
      }
    }

    // Grow the run.  Members need not be textually adjacent; anything
    // transparent may sit between them.
    size_t cursor = start + 1;
    for (; cursor < ops.size(); ++cursor) {
      Operation *candidate = ops[cursor];

      if (isEligibleMember(candidate)) {
        P2PMessage msg = *getP2PMessage(candidate);
        std::optional<int64_t> tag = getConstantIntValue(msg.tag);
        // A duplicate tag makes the intra-group order observable through MPI's
        // non-overtaking guarantee, so it cannot be a shared ordering key.
        if (!sameEnvelope(head, msg) || !tag || *tag < 0 ||
            llvm::is_contained(tags, *tag))
          break;

        BufferFacts fact;
        if (succeeded(computeBufferFacts(msg.buffer, dl, fact))) {
          if (isAccessedAwayFromOps(fact))
            break;
          facts.push_back(std::move(fact));
        } else {
          precise = false;
        }

        chosen.push_back(candidate);
        tags.push_back(*tag);
        continue;
      }

      if (!isTransparentBetween(candidate, facts, !precise))
        break;
    }

    if (chosen.size() < std::max<unsigned>(2u, minGroupSize))
      continue;

    Group group;
    group.comm = headComms.front();
    group.elemSize = *elemSize;
    group.lastInBlock = chosen.back();

    for (auto [position, op] : llvm::enumerate(chosen)) {
      Member member;
      member.op = op;
      member.position = position;
      member.msg = *getP2PMessage(op);
      member.tag = *getConstantIntValue(member.msg.tag);
      member.slice = resolveBufferSlice(member.msg.buffer, dl);
      member.extent = extentOf(member.msg.count, *elemSize);
      int64_t bytes;
      if (std::optional<int64_t> count = getConstantIntValue(member.msg.count);
          count && *count >= 0 && !llvm::MulOverflow(*count, *elemSize, bytes))
        member.constantExtent = bytes;
      group.members.push_back(std::move(member));
    }

    // Ascending tag is the ordering key both endpoints compute locally.
    llvm::sort(group.members,
               [](const Member &a, const Member &b) { return a.tag < b.tag; });

    // Adjacency in ascending-tag order.  The comparison is on ByteOffsets,
    // so a symbolic-but-cancelling pair counts as adjacent.
    group.contiguous = llvm::all_of(group.members, [&](const Member &m) {
      return m.slice.offset && m.extent &&
             m.slice.base == group.members[0].slice.base;
    });
    if (group.contiguous) {
      for (size_t i = 0; i + 1 < group.members.size(); ++i) {
        ByteOffset expected = *group.members[i].slice.offset;
        if (failed(expected.add(*group.members[i].extent)) ||
            expected != *group.members[i + 1].slice.offset) {
          group.contiguous = false;
          break;
        }
      }
    }
    if (!group.contiguous)
      ++fragmented;

    groups.push_back(std::move(group));
    start = cursor - 1;
  }
}

//===----------------------------------------------------------------------===//
// Phase C - cost, then global validation
//===----------------------------------------------------------------------===//

void CoalesceComm::applyCostHeuristic() {
  for (Group &group : groups) {
    if (!group.valid)
      continue;

    int64_t total = 0;
    bool totalKnown = true;
    for (const Member &member : group.members) {
      if (!member.constantExtent ||
          llvm::AddOverflow(total, *member.constantExtent, total)) {
        totalKnown = false;
        break;
      }
    }

    // The eager-limit guard.  Two blocking sends that each fit under the
    // transport's eager threshold can merge into one that does not, turning a
    // program that never blocked into one that waits for its match.  That
    // holds for the zero-copy tier as much as for the staged one: the bytes on
    // the wire are the same, but they now travel as one message.  A total that
    // is not a compile-time constant cannot be shown to stay under the limit,
    // so it is not merged while the limit is set.
    if (maxMergedBytes > 0 &&
        (!totalKnown || total > static_cast<int64_t>(maxMergedBytes))) {
      group.valid = false;
      continue;
    }

    if (group.contiguous)
      continue;

    // Everything below is the staged tier.
    if (!allowStaging || !totalKnown) {
      group.valid = false;
      continue;
    }
    if (maxStagedBytes > 0 && total > static_cast<int64_t>(maxStagedBytes)) {
      group.valid = false;
      continue;
    }
    // Decided here and not at the rewrite: a group dropped after the global
    // validation would leave its peer's group merged alone.
    if (!getStagingScope(group.members.front().op)) {
      group.valid = false;
      continue;
    }
    // The pack and unpack copies take addresses.  A pointer is one; the
    // raiser's memref views of host memory start at theirs, which
    // enzymexla.memref2pointer names.  A strided layout or a memory space
    // would put the elements somewhere else.
    if (!llvm::all_of(group.members, [](const Member &m) {
          return hasStagingAddress(m.msg.buffer);
        }))
      group.valid = false;
  }
}

void CoalesceComm::validateGlobally() {
  // Removing one group can invalidate another that shared a tag with it, so
  // this runs to a fixpoint rather than once.
  bool changed = true;
  while (changed) {
    changed = false;

    llvm::DenseMap<Operation *, unsigned> memberOf;
    for (auto [index, group] : llvm::enumerate(groups)) {
      if (!group.valid)
        continue;
      for (const Member &member : group.members)
        memberOf[member.op] = index;
    }

    for (Group &group : groups) {
      if (!group.valid)
        continue;

      const CommCensus &entry = census[group.comm];
      if (poisonAll || entry.poisoned) {
        group.valid = false;
        changed = true;
        continue;
      }

      SmallVector<int64_t> block = group.tagSet();
      for (int64_t tag : block) {
        // Each endpoint places the pieces by its own counts, so every endpoint
        // has to agree on each piece - one shape per tag, module-wide.  That
        // holds for the zero-copy tier too: a receive may post more than
        // arrives, and a merged one would then land the next piece at the
        // sender's boundary instead of its own.
        auto shape = entry.shapes.find(tag);
        if (shape == entry.shapes.end() || !shape->second.consistent) {
          group.valid = false;
          break;
        }

        auto users = entry.users.find(tag);
        if (users == entry.users.end())
          continue;
        for (Operation *user : users->second) {
          auto owner = memberOf.find(user);
          if (owner == memberOf.end() ||
              groups[owner->second].tagSet() != block) {
            group.valid = false;
            break;
          }
        }
        if (!group.valid)
          break;
      }
      if (!group.valid)
        changed = true;
    }
  }
}

//===----------------------------------------------------------------------===//
// Phase D - rewrite
//===----------------------------------------------------------------------===//

/// The merged element count.  Every member's count is a constant: validation
/// requires one constant count per tag.
static Value buildTotalCount(OpBuilder &builder, Location loc,
                             const Group &group) {
  int64_t total = 0;
  for (const Member &member : group.members)
    total += *getConstantIntValue(member.msg.count);
  return arith::ConstantOp::create(builder, loc,
                                   builder.getI64IntegerAttr(total));
}

void CoalesceComm::rewriteContiguous(RewriterBase &rewriter, Group &group) {
  const Member &head = group.members.front();
  Operation *survivor = head.op;

  // The merged operation goes where the *last* member was.  Every member's
  // operands dominate that point, and because no communication operation lies
  // between members, collapsing the run anywhere inside the window leaves
  // the program's order of communication unchanged.
  if (survivor != group.lastInBlock)
    rewriter.moveOpAfter(survivor, group.lastInBlock);

  rewriter.setInsertionPoint(survivor);
  Value total = buildTotalCount(rewriter, survivor->getLoc(), group);

  // The buffer and tag operands already belong to the least-tag member, which
  // by the adjacency chain is also the lowest address - so nothing but the
  // count changes.
  rewriter.modifyOpInPlace(survivor,
                           [&] { head.msg.countOperand->set(total); });

  for (const Member &member : group.members)
    if (member.op != survivor)
      rewriter.eraseOp(member.op);
}

void CoalesceComm::rewriteStaged(RewriterBase &rewriter, Group &group) {
  const Member &head = group.members.front();
  Operation *survivor = head.op;
  if (survivor != group.lastInBlock)
    rewriter.moveOpAfter(survivor, group.lastInBlock);

  Location loc = survivor->getLoc();
  MLIRContext *context = survivor->getContext();
  auto ptrType = LLVM::LLVMPointerType::get(context);
  auto i8Type = IntegerType::get(context, 8);
  auto i64Type = IntegerType::get(context, 64);

  int64_t total = 0;
  for (const Member &member : group.members)
    total += *member.constantExtent;

  // The scratch buffer goes at the top of the entry block, not at the merge
  // point: a merge point inside a loop would grow the stack on every iteration.
  FunctionOpInterface func = getStagingScope(survivor);
  rewriter.setInsertionPointToStart(&func.getFunctionBody().front());
  Value size = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                        rewriter.getI64IntegerAttr(total));
  // All members share a datatype, so its size is the alignment the message
  // wants.  Clamped to a power of two no larger than 16.
  uint64_t alignment = 1;
  while (alignment < static_cast<uint64_t>(group.elemSize) && alignment < 16)
    alignment *= 2;
  Value staging =
      LLVM::AllocaOp::create(rewriter, loc, ptrType, i8Type, size, alignment);

  auto sliceAt = [&](int64_t offset) -> Value {
    if (offset == 0)
      return staging;
    // A constant GEP index is 32 bits wide; a larger offset is a value.
    if (offset <= std::numeric_limits<int32_t>::max())
      return LLVM::GEPOp::create(
          rewriter, loc, ptrType, i8Type, staging,
          ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(static_cast<int32_t>(offset))});
    Value index = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                           rewriter.getI64IntegerAttr(offset));
    return LLVM::GEPOp::create(rewriter, loc, ptrType, i8Type, staging,
                               ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(index)});
  };

  bool sending = head.msg.direction == MessageDirection::Send;
  if (sending)
    rewriter.setInsertionPoint(survivor);
  else
    rewriter.setInsertionPointAfter(survivor);

  // The layout is a pure concatenation in ascending-tag order, with no
  // padding, so both endpoints compute it identically without having exchanged
  // anything.
  SmallVector<std::pair<const Member *, int64_t>> pieces;
  int64_t cursor = 0;
  for (const Member &member : group.members) {
    pieces.emplace_back(&member, cursor);
    cursor += *member.constantExtent;
  }
  // The copies follow the members' order in the block, not the layout.  Two
  // receive buffers may overlap, and the overlap has to end up holding the
  // bytes of the receive that came last in the program.
  llvm::sort(pieces, [](const auto &a, const auto &b) {
    return a.first->position < b.first->position;
  });

  for (auto [member, offset] : pieces) {
    Value slot = sliceAt(offset);
    Value length = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(*member->constantExtent));
    Value address = getStagingAddress(rewriter, loc, member->msg.buffer);
    // Pack before the send; unpack after the receive.
    if (sending)
      LLVM::MemcpyOp::create(rewriter, loc, slot, address, length,
                             /*isVolatile=*/false);
    else
      LLVM::MemcpyOp::create(rewriter, loc, address, slot, length,
                             /*isVolatile=*/false);
  }

  rewriter.setInsertionPoint(survivor);
  Value totalCount = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI64IntegerAttr(total / group.elemSize));
  rewriter.modifyOpInPlace(survivor, [&] {
    head.msg.bufferOperand->set(staging);
    head.msg.countOperand->set(totalCount);
  });

  for (const Member &member : group.members)
    if (member.op != survivor)
      rewriter.eraseOp(member.op);
}

void CoalesceComm::rewrite(RewriterBase &rewriter, Group &group) {
  if (group.contiguous)
    rewriteContiguous(rewriter, group);
  else
    rewriteStaged(rewriter, group);
}

void CoalesceComm::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  DataLayout dl(moduleOp);

  // Which communicator each message uses, through arguments, casts and memory.
  DataFlowSolver commSolver;
  commSolver.load<dataflow::DeadCodeAnalysis>();
  commSolver.load<dataflow::SparseConstantPropagation>();
  commSolver.load<CommOriginsAnalysis>();
  commSolver.load<HandleSlotAnalysis>(HandleKind::Communicator);
  if (failed(commSolver.initializeAndRun(moduleOp))) {
    moduleOp.emitError("-coalesce-comm: the communicator analysis failed");
    return signalPassFailure();
  }
  solver = &commSolver;

  takeCensus(moduleOp);
  discoverGroups(moduleOp, dl);
  applyCostHeuristic();
  validateGlobally();

  IRRewriter rewriter(&getContext());
  unsigned merged = 0, removed = 0;
  for (Group &group : groups) {
    if (!group.valid)
      continue;
    removed += group.members.size() - 1;
    ++merged;
    rewrite(rewriter, group);
  }

  numGroups = merged;
  numMessagesRemoved = removed;
  numFragmentedGroups = fragmented;
}

} // namespace

} // namespace mlir::spmd
