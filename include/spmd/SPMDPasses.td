//===- SPMDPasses.td - SPMD dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_PASSES
#define SPMD_PASSES

include "mlir/Pass/PassBase.td"

def CheckDataRace : Pass<"check-dataRace", "ModuleOp"> {
  let summary = "Checks statically if the given IR includes a data race.";
  let description = [{
    This pass checks for data races in the given IR. Optionally, it can output
    the result as a JSON report if the flag `--write-json-report` is set.
  }];
  let dependentDialects = [];
  let statistics = [];

  let options = [
    Option<"writeJsonReport", "write-json-report", "bool", /*default=*/"false",
           "Write the data race report as a JSON file.">
  ];
}


def ConvertApiCallsToSPMD : Pass<"convert-apiCalls-to-spmd", "ModuleOp"> { 
  let summary = "Converts api calls related to collectives of different programming models such as MPI or OpenSHMEM to a so-called SPMD dialect.";
  let description = [{
    sample Text
  }];
  // enzymexla: pointer operands become enzymexla.pointer2memref views. Raised
  // input normally contains that op already, which is the only reason a module
  // without it did not crash the builder.
  let dependentDialects = ["spmd::SPMDDialect", "memref::MemRefDialect", "arith::ArithDialect", "LLVM::LLVMDialect", "ub::UBDialect", "enzymexla::EnzymeXLADialect"];
  let statistics = [];
}

def ConvertLLVMToFunc : Pass<"convert-llvm-to-func", "ModuleOp"> { 
  let summary = "Converts funcOps and callOps of the llvm dialect to those of the func dialect";
  let description = [{
    sample Text
  }];
  let dependentDialects = ["func::FuncDialect"]; 
  let options = [];
  let statistics = [];
}

// def PrepareLLVMFuncInline : Pass<"prepare-llvm-func-inline", "ModuleOp"> { 
//   let summary = "Marks alls llvm funcs as always_inline so that they get inlined with the standard inline pass.";
//   let description = [{
//     sample Text
//   }];
//   let dependentDialects = ["LLVM::LLVMDialect"]; 
//   let options = [];
//   let statistics = [];
// }

// def HandleLLVMUnreachable : Pass<"handle-llvm-unreachable", "ModuleOp"> { 
//   let summary = "Convert all llvm.unreachable ops to llvm.return ops (assuming all functions being inlined).";
//   let description = [{
//     sample Text
//   }];
//   let dependentDialects = ["LLVM::LLVMDialect"]; 
//   let options = [];
//   let statistics = [];
// }

// def DeleteInlinedFuncs : Pass<"delete-inlined-funcs", "ModuleOp"> { 
//   let summary = "Erase all llvm.func (in future also func.func) ops that were previously inlined and not needed in the main module.";
//   let description = [{
//     sample Text
//   }];
//   let dependentDialects = ["LLVM::LLVMDialect"]; 
//   let options = [];
//   let statistics = [];
// }

def ConvertSCFSwitchToIf : Pass<"convert-scf-switch-to-if", "ModuleOp"> { 
  let summary = "Convert all scf.index_switch ops to a chain of scf.if ops.";
  let description = [{
    sample Text
  }];
  let dependentDialects = ["scf::SCFDialect"]; 
  let options = [];
  let statistics = [];
}

def PrivatizeSymbolsPass : Pass<"privatize-symbols"> {
  let summary = "Mark non-external symbols private";
  let description = [{
    This pass marks all non-external top-level symbols of the operation run as `private`
    except if listed in `exclude` pass option.
  }];
  let options = [
    ListOption<"exclude", "exclude", "std::string",
       "Comma separated list of symbols that should not be marked private">
  ];
}

def SPMDReconcileUnrealizedCast : Pass<"spmd-reconcile-unrealized-cast", "ModuleOp"> { 
  let summary = "Reconcile unrealized casts by hoisting them out of scf.if ops if possible.";
  let description = [{
    sample Text
  }];
  let dependentDialects = ["scf::SCFDialect"]; 
  let options = [];
  let statistics = [];
}

def ConvertArithSelectToIf : Pass<"convert-arith-select-to-if", "ModuleOp"> { 
  let summary = "Convert all scf.select ops to an scf.if op and insert depending operations.";
  let description = [{
    sample Text
  }];
  let dependentDialects = ["scf::SCFDialect"]; 
  let options = [];
  let statistics = [];
}

def SimplifyCmpOps : Pass<"simplify-cmp-ops", "ModuleOp"> { 
  let summary = "If possible, simplify cmp ops represented by additional if ops by cmp ops only.";
  let description = [{
    sample Text
  }];
  let dependentDialects = ["arith::ArithDialect"]; 
  let options = [];
  let statistics = [];
}

// def InlineGPULaunchFuncs : Pass<"inline-gpu-launch-funcs", "ModuleOp"> { 
//   let summary = "Inlines the gpu functions in gpu module into its calling site at gpu launch func operations and move exteral functions to the main module.";
//   let description = [{
//     sample Text
//   }];
//   let dependentDialects = []; 
//   let options = [];
//   let statistics = [];
// }

def CheckCollectives : Pass<"check-collectives", "ModuleOp"> {
  let summary = "Checks statically if the given IR includes a collective communication error.";
  let description = [{
    This pass checks for collective communication errors in the given IR. Optionally, it can output
    the result as a JSON report if the flag `--write-json-report` is set.
  }];
  let dependentDialects = [];

  let options = [
    Option<"numRanks", "num-ranks", "uint64_t",
           /*default=*/"0",
           "Set number of ranks with which the SPMD program shall be run (default: 0, meaning unspecified)">,
    Option<"writeJsonReport", "write-json-report", "bool", /*default=*/"false",
           "Write the collective communication error report as a JSON file.">
  ];

  let statistics = [];
}

def CheckThreadSafety : Pass<"check-thread-safety", "ModuleOp"> {
  let summary = "Verify that SPMD communication ops inside thread-parallel "
                "regions respect the thread-support level declared at "
                "initialisation time";
  let description = [{
    For each process-level spmd.parallel carrying a threadLevel attribute,
    checks that every Communication-trait op nested inside a thread-level
    spmd.parallel is safe under that thread-support level.

    The thread-support levels follow the nomenclature established by MPI
    (MPI_Init_Thread) and adopted equivalently by OpenSHMEM (shmem_init_thread)
    and NVSHMEM (nvshmemx_init_thread):

      SINGLE:     Only one thread executes. No SPMD communication op may
                  appear inside any thread-level parallel region.
      FUNNELED:   The process may be multithreaded, but only the main thread
                  (OMP thread 0 / master) may issue communication calls.
                  The ExecutionContextAnalysis must show executors == {0}.
      SERIALIZED: Multiple threads may issue communication calls, but not
                  concurrently. Calls must be inside spmd.critical or
                  spmd.single, or the analysis must show a single executor.
      MULTIPLE:   No restrictions.

    When threadLevel is absent on a process-level parallel, SINGLE is assumed
    (consistent with the MPI specification §12.4.3, which states that
    MPI_Init is equivalent to MPI_Init_Thread with required = SINGLE; the
    equivalent holds for shmem_init and nvshmem_init).
  }];
  let options = [
    Option<"writeJsonReport", "write-json-report", "bool", "false",
           "Write violations to thread_safety_report.json instead of stderr">
  ];
  let dependentDialects = ["mlir::spmd::SPMDDialect"];
}

def OMPToSPMD : Pass<"omp-to-spmd", "ModuleOp"> {
  let summary = "Converts OpenMP dialect ops to the SPMD dialect";
  let description = [{
    Converts structural OpenMP dialect operations to their SPMD dialect
    equivalents. Run before ConvertApiCallsToSPMD, followed by a
    canonicalization pass to fold spmd.getParallelComm ops.

    OpenMP support is a subset. The exact list of lowered and deliberately
    unsupported ops, and the modelling limitations of the lowered subset
    (task scheduling order, taskwait, worksharing-loop uniformity, ignored
    clauses), are documented at the top of lib/spmd/passes/OMPToSPMD.cpp.
    An unsupported op fails the conversion, so the module gets no
    verification rather than an unsound result.
  }];
  let dependentDialects = [
    "mlir::omp::OpenMPDialect",
    "mlir::spmd::SPMDDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect"
  ];
  let options = [];
  let statistics = [];
}

def EliminateRedundantBarriers : Pass<"eliminate-redundant-barriers", "ModuleOp"> {
  let summary = "Eliminates redundant adjacent spmd.barrier operations.";
  let description = [{
    This pass removes a barrier when it is immediately preceded by an
    equivalent barrier in the same block and the later barrier's results are
    unused. It is intentionally local and conservative.
  }];
  let dependentDialects = ["spmd::SPMDDialect"];
  let options = [];
  let statistics = [
    Statistic<"numFolded", "num-folded",
              "Barriers removed because an identical one ran right before">
  ];
}

def EliminateDeadComm : Pass<"eliminate-dead-comm", "ModuleOp"> {
  let summary = "Eliminates communication whose result nothing can observe.";
  let description = [{
    Dead communication elimination.  Three independent removals, each of which
    deletes whole operations textually (so every rank loses the same operation
    and per-rank operation counts stay aligned by construction):

    * `dead-dest` - a data-moving collective whose destination buffer is never
      read is removed, together with its completion (`spmd.wait`/`spmd.reqFree`)
      when it is non-blocking.
    * `overwritten-dest` - a data-moving collective whose destination buffer is
      completely overwritten by a later collective in the same block, before
      anything reads it, is removed.
    * `dead-resource` - an RMA window that no data-movement operation ever
      targets is removed together with its whole lifecycle (fences, locks,
      `spmd.winFree`); likewise an `spmd.malloc` whose address only ever reaches
      `spmd.free`.

    Buffer reasoning is restricted to private, function-local allocations
    (`memref.alloca`/`memref.alloc`/`llvm.alloca`) that provably do not escape.
    Symmetric memory, streams (NCCL/NVSHMEM), P2P operations and RMA data
    movement are deliberately left alone.

    Note this is an *optimisation*, not a verification pass, and the two do not
    commute: a dead collective sitting under rank-dependent control is removed
    here, which is exactly what `-check-collectives` exists to report.  Run the
    check passes on the un-optimised IR.
  }];
  let dependentDialects = ["spmd::SPMDDialect"];

  let options = [
    Option<"deadDest", "dead-dest", "bool", /*default=*/"true",
           "Remove collectives whose destination buffer is never read.">,
    Option<"overwrittenDest", "overwritten-dest", "bool", /*default=*/"true",
           "Remove collectives whose destination buffer is fully overwritten "
           "before it is read.">,
    Option<"deadResource", "dead-resource", "bool", /*default=*/"true",
           "Remove RMA windows and symmetric allocations nothing ever uses.">,
    Option<"maxIterations", "max-iterations", "unsigned", /*default=*/"4",
           "Upper bound on the number of times the removals are re-run to pick "
           "up cascades.">
  ];

  let statistics = [
    Statistic<"numDeadDest", "num-dead-dest",
              "Collectives removed because their destination is never read">,
    Statistic<"numOverwrittenDest", "num-overwritten-dest",
              "Collectives removed because their destination is overwritten">,
    Statistic<"numDeadResource", "num-dead-resource",
              "Windows and symmetric allocations removed as unused">
  ];
}

def EliminateRedundantComm : Pass<"eliminate-redundant-comm", "ModuleOp"> {
  let summary = "Eliminates collectives whose result is already in place.";
  let description = [{
    Redundant collective elimination.  A blocking, data-moving collective is
    removed when an identical one - same operation, same communicator, same
    send and receive allocations, same count/datatype/operator/root by SSA
    identity - has already run on every path to it, with nothing written to
    either buffer in between.

    Where `-eliminate-dead-comm` only ever needs local buffer liveness, this
    pass needs a must-analysis, and therefore a soundness condition
    (`docs/optimizations.md`, "Available collectives"): a fact may only be
    *established* by an
    operation every rank executes, on operands every rank agrees on, while a
    write *invalidates* a fact unconditionally.  The removal itself is textual
    and whole-op, so all ranks lose the same operation and per-rank operation
    counts stay aligned by construction.

    Two shapes are deliberately never available:

    * in-place collectives (`MPI_IN_PLACE`, i.e. the send and receive buffers
      resolve to one allocation).  Those are not idempotent - a second
      allreduce reduces the already-reduced value - and at small rank counts
      the wrong answer can look plausible.
    * buffers that do not denote an allocation's base address, since the key
      carries no offset.

    Non-blocking collectives (the write lands at the completion, not here),
    stream-ordered collectives (NCCL/NVSHMEM/CUDA), P2P and RMA are all out of
    scope, as is interprocedural availability: a call kills everything.

    Compare operands by SSA identity, so run `-canonicalize -cse` first;
    otherwise two unmerged `arith.constant` operations produce two keys and
    nothing is removed.  Like `-eliminate-dead-comm` this is an optimisation,
    not a verification pass - run the check passes on the un-optimised IR.
  }];
  let dependentDialects = ["spmd::SPMDDialect"];

  let options = [
    Option<"maxIterations", "max-iterations", "unsigned", /*default=*/"4",
           "Upper bound on the number of times the analysis is re-run to pick "
           "up cascades.">
  ];

  let statistics = [
    Statistic<"numRedundant", "num-redundant",
              "Collectives removed because an identical one already ran">
  ];
}

def CoalesceComm : Pass<"coalesce-comm", "ModuleOp"> {
  let summary = "Merges several point-to-point messages to one peer into one.";
  let description = [{
    Point-to-point message coalescing.  A run of blocking `spmd.send`s (or
    `spmd.recv`s) sharing a communicator, a peer, a datatype and the protocol
    attributes becomes a single larger message.

    Unlike `-eliminate-dead-comm` and `-eliminate-redundant-comm`, this pass
    does not merely delete whole operations, so their structural argument
    against deadlock does not carry over.  Two properties replace it.

    **Global tag-partition consistency.**  A merge is sound only if the peer
    merges to match, and the two sides are transformed independently.  So the
    grouping is made a function of the *tag block* alone: members are ordered by
    ascending tag, the merged operation carries the least tag of its block, and
    a block is legal only when every operation in the module using one of its
    tags belongs to a surviving group realising exactly that block.  Merging a
    group whose peer could not form the matching group is what would otherwise
    truncate a message, and per-side maximality does not prevent it - which is
    why this pass is module-scoped rather than function-scoped.

    **Uniform execution is not required.**  A group is a run inside one block,
    so any rank that executes the first member executes all of them; collapsing
    it is a textual whole-group replacement and per-rank message counts stay
    aligned.  The grouping keys on the SSA identity of the peer *expression*,
    not on its value - in a halo exchange the peer is `rank +/- 1` and is
    therefore never uniform, so a uniformity gate would decline every realistic
    input.

    Two tiers.  When the pieces are already adjacent inside one allocation the
    rewrite is zero-copy: only the count operand changes.  Otherwise, with
    `allow-staging`, the pieces are packed through a scratch buffer allocated in
    the function entry block.  Both tiers need every use of a tag in the
    module to agree on one compile-time-constant count and element type: each
    endpoint places the pieces by its own counts, and a receive may post more
    elements than arrive.

    Scope in this version: blocking MPI point-to-point only.  Non-blocking
    groups need the request array rebuilt and the `spmd.waitAll` count rewritten;
    collectives and one-sided operations need their own grouping rules.

    The tag check assumes the module is the whole program, run by every rank,
    and that every operation matching a message by tag is in the dialect: an
    MPI call left unraised, or a status query such as `MPI_Get_count`, is
    invisible to it.  Like the other communication optimisations this pass must
    never run before the checkers; `spmd-verify` runs it only for
    `compile opt=coalesce-comm`, which builds an executable.  See
    docs/optimizations.md, "Coalescing".
  }];
  let dependentDialects = ["spmd::SPMDDialect", "arith::ArithDialect",
                           "LLVM::LLVMDialect", "memref::MemRefDialect",
                           "enzymexla::EnzymeXLADialect"];

  let options = [
    Option<"allowStaging", "allow-staging", "bool", /*default=*/"true",
           "Pack non-adjacent pieces through a scratch buffer.  When false, "
           "only groups that are already contiguous are merged.">,
    Option<"maxStagedBytes", "max-staged-bytes", "unsigned", /*default=*/"8192",
           "Do not stage a group whose total payload exceeds this, where the "
           "copies cost more than the saved per-message overhead.  0 disables "
           "the check.">,
    Option<"maxMergedBytes", "max-merged-bytes", "unsigned",
           /*default=*/"16384",
           "Do not merge blocking sends into a message larger than this.  Two "
           "sends that each fit under the transport's eager threshold can merge "
           "into one that does not, turning a program that never blocked into "
           "one that waits for its match.  A total that is not a compile-time "
           "constant is never merged while the check is on.  0 disables the "
           "check.">,
    Option<"minGroupSize", "min-group-size", "unsigned", /*default=*/"2",
           "Smallest run worth merging.">
  ];

  let statistics = [
    Statistic<"numGroups", "num-groups", "Groups merged">,
    Statistic<"numMessagesRemoved", "num-messages-removed",
              "Messages eliminated by merging">,
    Statistic<"numFragmentedGroups", "num-fragmented-groups",
              "Groups whose pieces were not already adjacent.  This is a "
              "measurement of the *input*: it is the share of the opportunity "
              "that only the staging tier can take.">
  ];
}

def LowerSPMDtoMPI : Pass<"lower-spmd-to-mpi", "ModuleOp"> {
  let summary = "Lower SPMD operations tagged for MPI to MPI calls";
  let description = [{
    Model-to-model translation, the inverse of `-convert-apiCalls-to-spmd` for
    MPI: every `spmd` operation whose `usedModel` is MPI (0) becomes the call
    to the MPI function it stands for, which is declared in the module if it
    is not already.  Operations tagged for another programming model are left
    alone, so a mixed-model module is lowered one model at a time; so are the
    `spmd.castSPMDErrorToI32` and `spmd.castThreadLevelToI32` of their
    results.  The process-level `spmd.parallel` that `-convert-apiCalls-to-spmd`
    wraps around `main` is spliced back into `main`, without the barrier the
    raiser puts in front of each `spmd.parallel_yield`.

    The calls follow the MPICH ABI of the headers in
    `includesProgramModels/mpi/include` (MPICH 4.2.2): communicators,
    datatypes, reduction operations, requests, windows and message handles are
    `int`s, an `MPI_Status` is five `int`s passed by address, and the
    predefined handles are MPICH's constants, from
    `include/spmd/TargetABI/MPICH.h`.  The output links against MPICH or an MPI
    with the same ABI, not against Open MPI.  A datatype, reduction operation,
    lock type or thread level with no predefined MPICH handle is an error, and
    so is an operand MPI has no parameter for: a stride other than 1, a stream,
    a communicator config, an alignment, separate bcast buffers, a window
    without a size.

    Element counts are `MPI_Count`s in the dialect and `int`s in MPI's classic
    calls.  A count that provably fits an `int` (a widened `int`, or a small
    constant) is passed as one; a call with any other count becomes its MPI 4
    large-count form (`MPI_Send_c`, ...), never a truncation.

    The conversion is total for the MPI model: an MPI-tagged operation without
    a lowering fails the pass instead of being left behind.  Out-parameters
    (new communicators, requests, flags, statuses, message handles) are
    written to stack storage and read back from it.
  }];
  let dependentDialects = ["LLVM::LLVMDialect", "enzymexla::EnzymeXLADialect"];
}

def LowerSPMDtoSHMEM : Pass<"lower-spmd-to-shmem", "ModuleOp"> {
  let summary = "Lower SPMD operations tagged for OpenSHMEM to OpenSHMEM calls";
  let description = [{
    Model-to-model translation, the inverse of `-convert-apiCalls-to-spmd` for
    OpenSHMEM: every `spmd` operation whose `usedModel` is OpenSHMEM (1)
    becomes the call to the OpenSHMEM routine it stands for, which is declared
    in the module if it is not already.  Operations tagged for another
    programming model are left alone, and so is everything
    `-lower-spmd-to-mpi` documents about the process-level `spmd.parallel`
    and the casts of the results.

    The calls are derived from the CACTS map the raiser reads and the USIS API
    description, like MPI's.  OpenSHMEM spells the element type and the
    reduction operation in the function name, so an `spmd.allreduce` of
    `i32` with `sum` becomes `shmem_int_sum_reduce` and one of `f64` with
    `max` becomes `shmem_double_max_reduce`; a reduction given MPI's spelling
    (`SUM`, `BAND`) is the same one, except the logical `LAND`, `LOR` and
    `LXOR`, which OpenSHMEM does not have.

    The calls follow the Sandia OpenSHMEM ABI of the headers in
    `includesProgramModels/shmem/include`, with the constants of
    `include/spmd/TargetABI/OpenSHMEM.h`: a team is a pointer and the
    predefined teams are the extern variables `SHMEM_TEAM_WORLD` and
    `SHMEM_TEAM_SHARED`, which the lowering loads.

    A window is the symmetric address it is over: `spmd.winAlloc` becomes
    `shmem_malloc`, `spmd.winCreate` its base address, and a transfer's target
    address the window.  A target displacement other than zero, and a window
    provably over the stack, are errors: OpenSHMEM reaches only the symmetric
    heap and global variables, and the displacement unit does not survive.
    Freeing a window completes what is outstanding with `shmem_quiet()`, and
    frees its memory only when `spmd.winAlloc` allocated it.  Whether an
    address of unknown provenance is symmetric is the program's business, as
    it is in OpenSHMEM itself.

    OpenSHMEM hands out no request.  A non-blocking transfer becomes its
    `_nbi` form, its request one constant, and `spmd.wait`/`spmd.waitAll`
    `shmem_quiet()`, which completes every outstanding operation of the PE;
    `spmd.fence` is `shmem_barrier_all()`.

    Everything OpenSHMEM cannot express is an error, reported for the whole
    module before anything is rewritten: a rooted reduction, a scatter or
    gather, a communicator split by colour and key, a non-blocking
    collective, two-sided messaging, window locks and epochs, a wait asking
    for a status, and an operand no call has a parameter for.
  }];
  let dependentDialects = ["LLVM::LLVMDialect", "enzymexla::EnzymeXLADialect"];
}

def LowerSPMDtoNVSHMEM : Pass<"lower-spmd-to-nvshmem", "ModuleOp"> {
  let summary = "Lower SPMD operations tagged for NVSHMEM to NVSHMEM host calls";
  let description = [{
    Model-to-model translation, the inverse of `-convert-apiCalls-to-spmd` for
    NVSHMEM: every `spmd` operation whose `usedModel` is NVSHMEM (2) becomes
    the call to the NVSHMEM host routine it stands for, which is declared in
    the module if it is not already.  Operations tagged for another
    programming model are left alone, and so is everything
    `-lower-spmd-to-mpi` documents about the process-level `spmd.parallel`
    and the casts of the results.

    NVSHMEM's host API is OpenSHMEM's with an `nvshmem_` prefix, and this is
    `-lower-spmd-to-shmem`'s lowering in every respect but the ABI: the calls
    are derived from the CACTS map and the USIS API description, the element
    type and the reduction operation are spelled in the callee's name, a
    window is the symmetric address it is over, a non-blocking transfer is the
    `_nbi` call, and `spmd.wait` is `nvshmem_quiet()`.  The ABI is that of the
    headers in `includesProgramModels/mynvshmem/include`, with the constants of
    `include/spmd/TargetABI/NVSHMEM.h`: a team is an `int32_t`, and the
    predefined teams are the constants `NVSHMEM_TEAM_WORLD` (0) and
    `NVSHMEMX_TEAM_NODE` (2), which `spmd.commWorld` and `spmd.commNode`
    become.

    Those headers are a stand-in for NVIDIA's, which the raiser reads, so the
    calls emitted here are the ones the program made, and a raised call lowers
    back to itself.  Not all of them are what NVIDIA's library exports;
    `-nvshmem-to-host-abi` maps them to it afterwards.

    Only host-initiated NVSHMEM is lowered.  Everything NVSHMEM cannot express
    is an error, reported for the whole module before anything is rewritten:
    what `-lower-spmd-to-shmem` reports for OpenSHMEM, an operation inside a
    GPU kernel, and a window over a global variable, which NVSHMEM does not
    make symmetric -- only what `nvshmem_malloc` allocates on the GPU is.
  }];
  let dependentDialects = ["LLVM::LLVMDialect", "enzymexla::EnzymeXLADialect"];
}

def NVSHMEMToHostABI : Pass<"nvshmem-to-host-abi", "ModuleOp"> {
  let summary = "Map the stand-in NVSHMEM API to NVIDIA's host library";
  let description = [{
    The NVSHMEM headers the frontend compiles against
    (`includesProgramModels/mynvshmem/include`) stand in for NVIDIA's: they
    declare every host routine as an ordinary function, which is how the
    raiser sees the calls, and some that NVIDIA's host library does not have.
    This pass maps a module calling them -- whether the calls were lowered by
    `-lower-spmd-to-nvshmem` or never raised -- to what `libnvshmem_host`
    exports, so that the program links without NVIDIA's device library.
    `spmd-verify compile` runs it after the lowering.

    `nvshmem_init`, `nvshmem_init_thread` and `nvshmem_finalize` are inline
    functions of NVIDIA's header.  A declaration of one of them gets a body,
    with internal linkage: `nvshmemx_hostlib_init_attr(0, NULL)`, exiting with
    -1 when it fails as the header's version does, and for
    `nvshmem_init_thread` `nvshmem_query_thread` for the granted level;
    `nvshmemx_hostlib_finalize()`.  This initialises the host library without
    the state of the device library, which only the program's own GPU kernels
    need -- and none is compiled here.  NVSHMEM provides
    `NVSHMEM_THREAD_SERIALIZED` whatever level is requested, so the level is
    what it would have been.

    A call of a routine the host library does not have at all -- `realloc`,
    the device-only waits and tests, the signalling puts without a stream, the
    bitwise reductions over signed and floating-point types -- is an error
    that names the reason.  A declaration nothing calls is left alone, and so
    is a module that uses no NVSHMEM.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def LowerSPMDtoCUDA : Pass<"lower-spmd-to-cuda", "ModuleOp"> {
  let summary = "Lower SPMD operations tagged for CUDA to CUDA runtime calls";
  let description = [{
    Model-to-model translation, the inverse of `-convert-apiCalls-to-spmd` for
    CUDA: every `spmd` operation whose `usedModel` is CUDA (4) becomes the call
    to the CUDA runtime routine it stands for, which is declared in the module
    if it is not already.  Operations tagged for another programming model are
    left alone, and so is everything `-lower-spmd-to-mpi` documents about the
    casts of the results.  CUDA's map has no init and no finalize, so no
    process-level `spmd.parallel` is ever tagged for it.

    The calls are derived from the CACTS map the raiser reads and the USIS API
    description, like MPI's, under the ABI of CUDA's `cuda_runtime_api.h`
    (`include/spmd/TargetABI/CUDA.h`): a buffer is a pointer, a count the
    `size_t` the dialect holds as an `i64`, a direction and a fill byte the
    `int`s the program computed, and every routine returns a `cudaError_t`,
    which is an `int`.

    The scope is what the raiser raises, which is two of CUDA's ~330 runtime
    routines: `cudaMemcpy` and `cudaMemset`.  Every other CUDA call in the
    program is still the `llvm.call` the frontend produced and needs no
    lowering at all.  A routine added to
    `lib/spmd/passes/CACTS/cudaSpmdEntries.cpp` gets its lowering from the
    table with no change here, unless it has an argument the CUDA ABI has no
    rule for -- an out-parameter such as `cudaMalloc`'s -- in which case the
    call is left out of the table and reported rather than emitted wrongly.

    Everything CUDA cannot express is an error, reported for the whole module
    before anything is rewritten: an operation the raiser's map raises nothing
    to, an operand no call has a parameter for, and a constant `$kind` on an
    `spmd.memcpy` that is no direction of `enum cudaMemcpyKind`.

    CUDA kernels are out of scope: nothing here outlines or launches one.
  }];
  let dependentDialects = ["LLVM::LLVMDialect", "enzymexla::EnzymeXLADialect"];
}

def LowerSPMDtoNCCL : Pass<"lower-spmd-to-nccl", "ModuleOp"> {
  let summary = "Lower SPMD operations tagged for NCCL to NCCL calls";
  let description = [{
    Model-to-model translation, the inverse of `-convert-apiCalls-to-spmd` for
    NCCL: every `spmd` operation whose `usedModel` is NCCL (3) becomes the call
    to the NCCL routine it stands for, which is declared in the module if it is
    not already.  Operations tagged for another programming model are left
    alone, and so is everything `-lower-spmd-to-mpi` documents about the
    process-level `spmd.parallel` and the casts of the results.

    The calls are derived from the CACTS map the raiser reads and the USIS API
    description, like MPI's, under the ABI of
    `includesProgramModels/nccl/include/nccl.h`: a communicator and a CUDA
    stream are pointers, and a datatype and a reduction operation are the
    enumerators of `include/spmd/TargetABI/NCCL.h`.  `spmd.castStream` is the
    program's own `cudaStream_t`.

    NCCL has no world communicator, and the raiser keeps none of what
    `ncclCommInitRank` was given.  `spmd.init` therefore bootstraps it the way
    NCCL programs do: the lowering adds a function that has rank 0 call
    `ncclGetUniqueId`, broadcasts the id over the module's process model --
    `MPI_Bcast`, or `shmem_broadcastmem` when the module starts no MPI -- and
    calls `ncclCommInitRank` with the real rank and size, leaving the
    communicator in a global that `spmd.commWorld` loads.  A module that starts
    neither MPI nor OpenSHMEM, one that initialises NCCL twice, and an
    `spmd.commWorld` whose communicator is used where no `spmd.init` of NCCL
    dominates it, are errors.

    NCCL has no requests either.  An operation with a request is one of a
    group, as the raiser makes the calls between `ncclGroupStart` and
    `ncclGroupEnd`: it becomes `ncclGroupStart()` and its call, and a wait
    becomes one `ncclGroupEnd()` per request it completes.  NCCL runs nested
    groups at the outermost `ncclGroupEnd`, so a wait has to complete every
    operation still open in its block, with a constant count, and nothing else
    may communicate in between; anything else is an error.

    Everything NCCL cannot express is an error, reported for the whole module
    before anything is rewritten: finalize, abort, a barrier, a duplicated
    communicator, a request freed, tested or completed in part, a wait asking
    for a status, a communicator kept in an `int`, and an operand no call has a
    parameter for -- a tag other than 0, or a missing stream.
  }];
  let dependentDialects = ["LLVM::LLVMDialect", "enzymexla::EnzymeXLADialect"];
}

def LowerSPMDtoOMP : Pass<"lower-spmd-to-omp", "ModuleOp"> {
  let summary = "Lower SPMD operations tagged for OpenMP back to OpenMP";
  let description = [{
    The inverse of `-omp-to-spmd` and of the OpenMP half of
    `-convert-apiCalls-to-spmd`: every `spmd` operation whose `usedModel` is
    OpenMP (5) goes back to what it was raised from.  Operations tagged for
    another programming model are left alone.

    The runtime queries are calls: `spmd.getRankInComm` and
    `spmd.getSizeOfComm` become `omp_get_thread_num()` and
    `omp_get_num_threads()`, derived from the CACTS map and the USIS API
    description like MPI's.  The directives are not: a thread-level
    `spmd.parallel`, `spmd.wsloop`/`spmd.loop_nest`, `spmd.single`,
    `spmd.sections`/`spmd.section`, `spmd.critical`, `spmd.barrier`, the task
    operations and `spmd.declare_reduction` become the `omp` dialect operations
    they mirror, with their terminators, which `-convert-polygeist-to-llvm` and
    `mlir-translate` turn into `__kmpc_*` calls.

    What the raiser adds goes away.  A team (`spmd.commWorld`, the
    `spmd.commDup` before a parallel region, `spmd.getParallelComm`) has no
    OpenMP object and becomes a constant nothing reads; a worksharing construct
    followed by a barrier of its team is the construct without `nowait`, and
    one that is not gets `nowait`; error results are 0.

    Everything OpenMP cannot express is an error, reported for the whole module
    before anything is rewritten: an operation of another kind (a message, a
    split, a window), a process team, a parallel region with private variables
    or an exit code, a directive or query on a team other than the current one,
    and a team used by another model's operation.
  }];
  let dependentDialects = ["LLVM::LLVMDialect", "mlir::omp::OpenMPDialect",
                           "enzymexla::EnzymeXLADialect"];
}

def LowerSPMD : Pass<"lower-spmd", "ModuleOp"> {
  let summary = "Lower every SPMD operation to the calls of a programming model";
  let description = [{
    The whole module, under a table that says what each programming model is
    lowered to: `models="mpi=mpi,shmem=keep"`.  A pair `<source>=<target>`
    naming two different models is a translation -- an MPI-raised program
    lowered to OpenSHMEM calls -- and `keep` leaves a model's operations for
    whatever runs after this pass.  The default table lowers every model to
    itself.  NVSHMEM and NCCL can be neither the source nor the target of a
    translation: their buffers are GPU memory, where MPI's and OpenSHMEM's
    are host memory, and no lowering moves a program between the two, nor
    between NVSHMEM and NCCL.

    Four phases, in this order, so that no diagnostic arrives after half the
    module has been rewritten:

    1. *retarget*: re-tag `usedModel` by the table, which is
       `-spmd-retarget-model` on its own.  A handle producer the raiser left
       untagged follows the operations that use it; a handle left crossing two
       models is an error.
    2. *check*: every target model's own check over the operations now tagged
       for it -- all of them, before anything is rewritten.
    3. *structure*: the process-level `spmd.parallel` of every model about to
       be lowered, spliced back into the function it wraps in one step,
       together with the self-matching barrier the raiser put in front of each
       yield.  The process team belongs to the launcher rather than to a
       model, and undoing that structure before the calls are emitted keeps it
       out of them: a parallel still standing during a conversion would need
       its communicator materialised back into an `!spmd.comm`, and that cast
       would outlive it.  The parallel of a model that is kept stays with its
       operations.
    4. *lower*: each target model's lowering, one after the other.  The handle
       types they lower to differ (an MPI communicator is an `i32`, an
       `ncclComm_t` a pointer) and a type carries no model, so the models are
       lowered sequentially and unrealized casts bridge between them.

    A `-spmd-reconcile-unrealized-cast` afterwards folds the bridges away, as
    it does after the single-model passes.

    A target for one class of operations ("the collectives to NCCL, the rest
    to MPI") needs an NCCL communicator for every MPI communicator the
    collectives use, and a CUDA stream MPI's operations do not carry.  The
    NCCL lowering bootstraps only NCCL's own world communicator over MPI's, so
    a source naming one, such as `mpi.collectives`, is still an error rather
    than silently the whole model.
  }];
  let options = [
    Option<"models", "models", "std::string", /*default=*/"\"\"",
           "Comma separated <source>=<target> pairs, naming a programming "
           "model (mpi, shmem, nvshmem, nccl, cuda, omp), or 'keep' as the "
           "target.  Empty: every model that can be lowered, to itself.">
  ];
  let dependentDialects = ["LLVM::LLVMDialect", "enzymexla::EnzymeXLADialect"];
}

def SPMDRetargetModel : Pass<"spmd-retarget-model", "ModuleOp"> {
  let summary = "Re-tag SPMD operations for another programming model";
  let description = [{
    The first phase of `-lower-spmd` on its own: every operation tagged for
    `from` is tagged for `to` instead, so that the lowering of `to` is the one
    that picks it up.  This is what makes a translation possible at all --
    nothing else re-tags `usedModel`, so lowering an MPI-raised module to
    OpenSHMEM would otherwise convert nothing.

    `usedModel` is a required attribute of every SPMD operation, the handle
    producers included, so `spmd.commWorld` and the `spmd.const*` operations
    move with the calls that use them.  A handle that ends up crossing two
    models -- because the module mixed them and only one end moved -- is an
    error, and so is an operation the target model cannot express, reported
    here rather than by a lowering that has already rewritten half the module.

    Only `usedModel` changes; the operations keep their form until a lowering
    runs.
  }];
  let options = [
    Option<"from", "from", "std::string", /*default=*/"\"\"",
           "The programming model whose operations are re-tagged">,
    Option<"to", "to", "std::string", /*default=*/"\"\"",
           "The programming model they are tagged for">
  ];
}

#endif // SPMD_PASSES
