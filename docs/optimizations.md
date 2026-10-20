<!--
SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Communication optimizations

`-eliminate-dead-comm`, `-eliminate-redundant-comm` and
`-eliminate-redundant-barriers` delete communication ops; `-coalesce-comm`
merges point-to-point messages. This note explains why each transformation is
sound and states the limits it holds within. The code comments in those passes,
`AvailableCollectives` and `CommCompletions` refer to its sections by name.

## Never before the checkers

These passes remove the very things `-check-collectives`, `-check-dataRace`
and `-check-thread-safety` look for. A collective under a rank-dependent branch
whose result nobody reads is exactly a mismatch `-check-collectives` must
report, and `-eliminate-dead-comm` deletes it. Run the checkers on the IR as
raised, never after these passes. `spmd-verify` runs them only when it builds
an executable and is asked to (`spmd-verify compile opt=<pass>`), and never
together with a checker.

For the same reason none of them is a canonicalization pattern: `spmd-verify`
runs `-canonicalize` before the checkers, and `test/spmd/effects/` pins that it
never removes a communication op.

## Two ways to get it wrong

- **Silent corruption.** Some rank still needed the data the removed op moved,
  and now computes with stale values. Nothing crashes.
- **Mismatch.** The ranks no longer agree on the sequence of communication
  they execute, and deadlock or match the wrong operations.

## Removal is textual and whole-op

Every rank runs the same module. Erasing an op erases it on every rank, at the
same place in each rank's sequence of communication, so the ranks keep agreeing
on that sequence without any uniformity precondition. That argument only holds
because nothing is moved, and nothing is removed on one path only: an op goes
entirely or stays. A non-blocking op goes together with everything that
completes its request (see [Completions](#completions)).

The whole-program premise is that the module is the program and every rank
runs it. A pass that sees only part of the program must not rely on it.
[Coalescing](#coalescing) relies on it more heavily still.

## Buffers: private and intraprocedural

"Who reads or writes this buffer" is answered by `computeBufferFacts`
(`include/spmd/Analysis/CommBuffers.h`) and only for a private allocation:
an `llvm.alloca`, `memref.alloca` or `memref.alloc` whose every use is an op
that declares its effect on it. A buffer that reaches a call, is stored as a
pointer, or is captured by an op without declared effects escapes, and nothing
is concluded about it. Symmetric memory (`spmd.malloc`, `spmd.winAlloc`) is
never private, since another rank accesses it without any local use showing.

Everything is intraprocedural. `AvailableCollectives` forgets every fact at a
call, and neither carries facts into a callee nor back.

## Accesses away from the op

An op's place in the use list says where it touches memory, not when:

- a **non-blocking or persistent** op reads its send buffer and writes its
  receive buffer at some point up to its completion, not at the op;
- a **window** (`spmd.winCreate`) lets other ranks read and write the exposed
  memory at any time;
- inside a **thread team** (a thread-level `spmd.parallel`), other threads run
  the same region concurrently.

A removal that argues from the order of ops in a block therefore gives up on a
buffer that any such op uses, and on an op inside a thread team.

## Overwritten destinations

`-eliminate-dead-comm` removes a blocking collective when a later one in the
same block overwrites its whole destination before anything reads it. Staying
in one block makes the overwrite unconditional. The later op covers the
earlier one when both use the same programming model and communicator, the
same count counted the same way (for the whole buffer or per rank) and the
same datatype, into the base of the same allocation, and the later op writes

- on every rank that executes it, or
- on the same ranks as the earlier one: the same op with the same root.

A root decides which ranks write: `reduce(root=0) -> x; reduce(root=1) -> x`
leaves rank 0 with the first result, so the first reduce stays.

## Dead resources

`-eliminate-dead-comm` removes a window (`spmd.winCreate`, `spmd.winAlloc`
with an unused base address) whose only users are `spmd.winFree` and the ops
that merely synchronize it (fences, flushes, locks), and an `spmd.malloc`
whose address only reaches `spmd.free`. Creating and freeing either is
collective, and every rank runs the same code, so if no op here moves data
through the window, no other rank does either. `spmd.winStart` and
`spmd.winComplete` keep a window alive: their target half has no op in the
dialect to remove with them.

## Available collectives

`-eliminate-redundant-comm` removes a blocking collective when an identical one
(same op, communicator, model, send and receive allocation, and remaining
operands by SSA value) has already run on every path to it, and neither buffer
has been written since. `AvailableCollectives` is the must-analysis behind it:

- **Establishing a fact** needs an op that every rank executes, equally often
  (`isUniformlyExecuted`), on a communicator and operands every rank agrees on
  (`isUniformValue`). Otherwise one rank would hold a fact another does not,
  while the removal happens on all of them.
- **Invalidating a fact** is unconditional: any write that may reach either
  buffer, on any rank, an op whose effects are unknown, a call, and a region op
  the dataflow framework does not enter.
- The join is intersection; an unreached point is bottom, not empty.
- An in-place collective (send and receive buffer in one allocation) is never
  available. Repeating it reduces the already reduced value.
- Both buffers must be private allocations that no op in
  [Accesses away from the op](#accesses-away-from-the-op) uses.

## Completions

A non-blocking initiation is erased together with every op that completes its
request, and only if all of them can go. Completion matching
(`include/spmd/Analysis/CommCompletions.h`) follows the request through casts,
selects, stores into a private request array and loads out of it, and gives up
on anything else: a call, a comparison, `spmd.test`, `waitAny`, `waitSome`.
A completion that may complete requests of other initiations can only go if
those go too, so the passes shrink their set of candidates to a fixpoint
before erasing anything.

## Coalescing

`-coalesce-comm` merges a run of blocking MPI sends, or of blocking MPI
receives, into one message. Unlike the passes above it does not delete whole
ops, so their argument does not carry over. It rests on the following instead.

### The whole program, and every tag in it

A merged send only arrives intact if the matching receives were merged the
same way, and the two sides are transformed independently. The pass therefore
makes the grouping a function of the tags alone: members are ordered by
ascending tag, the merged message carries the least tag, and a group is merged
only when every op in the module that uses one of its tags, on its
communicator, belongs to a merged group of exactly the same tags. Maximal runs
on each side are not enough: `send(t0); send(t1)` against
`recv(t0); store; recv(t1)` gives a group on one side only.

That check assumes much more of the module than the removals do:

- the module is the whole program, and every rank runs it;
- every operation matching a message by tag is in the dialect. An MPI call the
  raiser left as a call is invisible to the check, and so is anything reading a
  receive's status, such as `MPI_Get_count`, which would see the merged count;
- every communicator resolves (`CommOriginsAnalysis`). A message on one that
  does not could be on any, and stops merging in the whole module. So do a
  wildcard source or tag, a negative peer and a tag that is not a constant, on
  their communicator.

### Within one block

A group is a run of ops inside one block, with nothing between its members that
communicates, has unknown effects or touches a member's buffer. Any rank that
reaches the first member reaches all of them, so replacing the run keeps the
ranks agreeing on their sequence of messages without any uniformity
precondition. The peer is compared as an SSA value, not for a uniform value: in
a halo exchange it is `rank ± 1`, and every rank computes that same expression.

The merged message goes where the last member was, which moves where the
earlier members' buffers are read or written. Nothing between the members
shows that move, but that only covers accesses at their ops, so the
restrictions of [Accesses away from the op](#accesses-away-from-the-op) apply:
no member inside a thread team, none whose private buffer a non-blocking,
persistent or RMA op uses.

### One count per tag

Each endpoint places the pieces of a merged message by its own counts, and MPI
lets a receive post more elements than arrive. A receiver posting 8 elements
for tag 0 and 4 for tag 1 against a sender of 4 and 4 would find tag 1's data
where it expects the rest of tag 0's. So every use of a tag in the module must
agree on one compile-time-constant count and one element type. Element types
are compared without their signedness: an `MPI_INT` send and the
`MPI_UNSIGNED` receive matching it pack the same bytes. A count that is not a
constant is never merged: the ranks evaluating it may disagree.

### The eager limit

Two sends that each travel eagerly can merge into one that waits for its
receive, and a program in which two ranks send to each other before receiving
then deadlocks. `max-merged-bytes` keeps merged messages below a size, and with
a count that is not a constant a group is not merged while the limit is set.

### The two rewrites

- **Zero-copy**, when the pieces are adjacent in one allocation in tag order:
  the least-tag member's count becomes the total. The adjacency test compares
  symbolic byte offsets (`resolveBufferSlice`), so `&u[i]` and `&u[i + 4]` are
  adjacent although neither offset is constant.
- **Staged**, otherwise: the pieces are copied into a scratch buffer allocated
  in the entry block of the enclosing function, in tag order. Receives are
  unpacked in the order the program received them, since receive buffers may
  overlap. A group outside a function body is not staged. A buffer the raiser
  sees through a memref is copied at the address `enzymexla.memref2pointer`
  gives it, which needs a memref of host memory with the identity layout.

Out of scope: non-blocking messages, `spmd.sendrecv`, persistent requests,
other models' point-to-point ops and collectives.
