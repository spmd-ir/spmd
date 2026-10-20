// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===-- SPMDEffects.td - SPMD side effect helpers --------*- tablegen -*-===//
//
// Reusable side-effect helpers for the SPMD dialect.
//
// Naming convention:
//   - FooRead / FooWrite / FooAlloc / FooFree
//       => for Arg<...,[...]> / Res<...,[...]>
//
//   - FooReadT / FooWriteT / FooAllocT / FooFreeT
//       => for op trait lists
//
// Important:
//   The strings in Resource<"..."> must name the corresponding C++ resource
//   classes in your implementation.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_EFFECTS
#define SPMD_EFFECTS

include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Resource bindings actually used by SPMDOps.td
//===----------------------------------------------------------------------===//

def SPMDRuntimeResource : Resource<"::mlir::spmd::RuntimeResource">;
def SPMDCommResource    : Resource<"::mlir::spmd::CommResource">;
def SPMDWinResource     : Resource<"::mlir::spmd::WinResource">;
def SPMDReqResource     : Resource<"::mlir::spmd::ReqResource">;
def SPMDStreamResource  : Resource<"::mlir::spmd::StreamResource">;
def SPMDMsgResource     : Resource<"::mlir::spmd::MsgResource">;

//===----------------------------------------------------------------------===//
// Generic decorators for Arg / Res
//===----------------------------------------------------------------------===//

class ReadOn<Resource r, int stage = 0,
             EffectRange range = PartialEffect>
    : MemRead<r, stage, range>;

class WriteOn<Resource r, int stage = 0,
              EffectRange range = PartialEffect>
    : MemWrite<r, stage, range>;

class AllocOn<Resource r, int stage = 0,
              EffectRange range = PartialEffect>
    : MemAlloc<r, stage, range>;

class FreeOn<Resource r, int stage = 0,
             EffectRange range = PartialEffect>
    : MemFree<r, stage, range>;

//===----------------------------------------------------------------------===//
// Generic op-wide traits
//===----------------------------------------------------------------------===//

class ReadOnT<Resource r, int stage = 0,
              EffectRange range = PartialEffect>
    : MemoryEffects<[MemRead<r, stage, range>]>;

class WriteOnT<Resource r, int stage = 0,
               EffectRange range = PartialEffect>
    : MemoryEffects<[MemWrite<r, stage, range>]>;

class AllocOnT<Resource r, int stage = 0,
               EffectRange range = PartialEffect>
    : MemoryEffects<[MemAlloc<r, stage, range>]>;

class FreeOnT<Resource r, int stage = 0,
              EffectRange range = PartialEffect>
    : MemoryEffects<[MemFree<r, stage, range>]>;

//===----------------------------------------------------------------------===//
// Runtime
//===----------------------------------------------------------------------===//

class RuntimeRead<int stage = 0, EffectRange range = PartialEffect>
    : ReadOn<SPMDRuntimeResource, stage, range>;
class RuntimeWrite<int stage = 0, EffectRange range = PartialEffect>
    : WriteOn<SPMDRuntimeResource, stage, range>;
class RuntimeAlloc<int stage = 0, EffectRange range = PartialEffect>
    : AllocOn<SPMDRuntimeResource, stage, range>;
class RuntimeFree<int stage = 0, EffectRange range = PartialEffect>
    : FreeOn<SPMDRuntimeResource, stage, range>;

class RuntimeReadT<int stage = 0, EffectRange range = PartialEffect>
    : ReadOnT<SPMDRuntimeResource, stage, range>;
class RuntimeWriteT<int stage = 0, EffectRange range = PartialEffect>
    : WriteOnT<SPMDRuntimeResource, stage, range>;
class RuntimeAllocT<int stage = 0, EffectRange range = PartialEffect>
    : AllocOnT<SPMDRuntimeResource, stage, range>;
class RuntimeFreeT<int stage = 0, EffectRange range = PartialEffect>
    : FreeOnT<SPMDRuntimeResource, stage, range>;

//===----------------------------------------------------------------------===//
// Communicator
//===----------------------------------------------------------------------===//

class CommRead<int stage = 0, EffectRange range = PartialEffect>
    : ReadOn<SPMDCommResource, stage, range>;
class CommWrite<int stage = 0, EffectRange range = PartialEffect>
    : WriteOn<SPMDCommResource, stage, range>;
class CommAlloc<int stage = 0, EffectRange range = PartialEffect>
    : AllocOn<SPMDCommResource, stage, range>;
class CommFree<int stage = 0, EffectRange range = PartialEffect>
    : FreeOn<SPMDCommResource, stage, range>;

class CommReadT<int stage = 0, EffectRange range = PartialEffect>
    : ReadOnT<SPMDCommResource, stage, range>;
class CommWriteT<int stage = 0, EffectRange range = PartialEffect>
    : WriteOnT<SPMDCommResource, stage, range>;
class CommAllocT<int stage = 0, EffectRange range = PartialEffect>
    : AllocOnT<SPMDCommResource, stage, range>;
class CommFreeT<int stage = 0, EffectRange range = PartialEffect>
    : FreeOnT<SPMDCommResource, stage, range>;

//===----------------------------------------------------------------------===//
// Window
//===----------------------------------------------------------------------===//

class WinRead<int stage = 0, EffectRange range = PartialEffect>
    : ReadOn<SPMDWinResource, stage, range>;
class WinWrite<int stage = 0, EffectRange range = PartialEffect>
    : WriteOn<SPMDWinResource, stage, range>;
class WinAlloc<int stage = 0, EffectRange range = PartialEffect>
    : AllocOn<SPMDWinResource, stage, range>;
class WinFree<int stage = 0, EffectRange range = PartialEffect>
    : FreeOn<SPMDWinResource, stage, range>;

class WinReadT<int stage = 0, EffectRange range = PartialEffect>
    : ReadOnT<SPMDWinResource, stage, range>;
class WinWriteT<int stage = 0, EffectRange range = PartialEffect>
    : WriteOnT<SPMDWinResource, stage, range>;
class WinAllocT<int stage = 0, EffectRange range = PartialEffect>
    : AllocOnT<SPMDWinResource, stage, range>;
class WinFreeT<int stage = 0, EffectRange range = PartialEffect>
    : FreeOnT<SPMDWinResource, stage, range>;

//===----------------------------------------------------------------------===//
// Request
//===----------------------------------------------------------------------===//

class ReqRead<int stage = 0, EffectRange range = PartialEffect>
    : ReadOn<SPMDReqResource, stage, range>;
class ReqWrite<int stage = 0, EffectRange range = PartialEffect>
    : WriteOn<SPMDReqResource, stage, range>;
class ReqAlloc<int stage = 0, EffectRange range = PartialEffect>
    : AllocOn<SPMDReqResource, stage, range>;
class ReqFree<int stage = 0, EffectRange range = PartialEffect>
    : FreeOn<SPMDReqResource, stage, range>;

class ReqReadT<int stage = 0, EffectRange range = PartialEffect>
    : ReadOnT<SPMDReqResource, stage, range>;
class ReqWriteT<int stage = 0, EffectRange range = PartialEffect>
    : WriteOnT<SPMDReqResource, stage, range>;
class ReqAllocT<int stage = 0, EffectRange range = PartialEffect>
    : AllocOnT<SPMDReqResource, stage, range>;
class ReqFreeT<int stage = 0, EffectRange range = PartialEffect>
    : FreeOnT<SPMDReqResource, stage, range>;

//===----------------------------------------------------------------------===//
// Stream
//===----------------------------------------------------------------------===//

class StreamRead<int stage = 0, EffectRange range = PartialEffect>
    : ReadOn<SPMDStreamResource, stage, range>;
class StreamWrite<int stage = 0, EffectRange range = PartialEffect>
    : WriteOn<SPMDStreamResource, stage, range>;
class StreamAlloc<int stage = 0, EffectRange range = PartialEffect>
    : AllocOn<SPMDStreamResource, stage, range>;
class StreamFree<int stage = 0, EffectRange range = PartialEffect>
    : FreeOn<SPMDStreamResource, stage, range>;

class StreamReadT<int stage = 0, EffectRange range = PartialEffect>
    : ReadOnT<SPMDStreamResource, stage, range>;
class StreamWriteT<int stage = 0, EffectRange range = PartialEffect>
    : WriteOnT<SPMDStreamResource, stage, range>;
class StreamAllocT<int stage = 0, EffectRange range = PartialEffect>
    : AllocOnT<SPMDStreamResource, stage, range>;
class StreamFreeT<int stage = 0, EffectRange range = PartialEffect>
    : FreeOnT<SPMDStreamResource, stage, range>;

//===----------------------------------------------------------------------===//
// Message
//===----------------------------------------------------------------------===//

class MsgRead<int stage = 0, EffectRange range = PartialEffect>
    : ReadOn<SPMDMsgResource, stage, range>;
class MsgWrite<int stage = 0, EffectRange range = PartialEffect>
    : WriteOn<SPMDMsgResource, stage, range>;
class MsgAlloc<int stage = 0, EffectRange range = PartialEffect>
    : AllocOn<SPMDMsgResource, stage, range>;
class MsgFree<int stage = 0, EffectRange range = PartialEffect>
    : FreeOn<SPMDMsgResource, stage, range>;

class MsgReadT<int stage = 0, EffectRange range = PartialEffect>
    : ReadOnT<SPMDMsgResource, stage, range>;
class MsgWriteT<int stage = 0, EffectRange range = PartialEffect>
    : WriteOnT<SPMDMsgResource, stage, range>;
class MsgAllocT<int stage = 0, EffectRange range = PartialEffect>
    : AllocOnT<SPMDMsgResource, stage, range>;
class MsgFreeT<int stage = 0, EffectRange range = PartialEffect>
    : FreeOnT<SPMDMsgResource, stage, range>;

#endif // SPMD_EFFECTS