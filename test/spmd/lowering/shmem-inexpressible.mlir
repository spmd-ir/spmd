// RUN: spmd-opt %s -split-input-file -lower-spmd-to-shmem -verify-diagnostics
//
// Everything OpenSHMEM cannot express is reported before the module is
// rewritten, each with its own reason rather than as the conversion's "failed
// to legalize".  Several are what the roundtrip lowering silently emitted
// something else for.

// A rooted reduction: OpenSHMEM's reductions deliver to every PE.  roundtrip
// lowered this to shmem_int_sum_reduce, an allreduce.
func.func @rooted_reduce(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 1 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.reduce' op is a rooted reduction; every PE of an OpenSHMEM team receives the result of shmem_<T>_<op>_reduce}}
  %e = "spmd.reduce"(%world, %sbuf, %rbuf, %n, %dt, %op, %root) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  return
}

// -----

// MPI's logical and.  OpenSHMEM's and, or and xor are bitwise, so only BAND,
// BOR and BXOR are the same operation.  With every typed reduction a
// candidate, the diagnostic names the value no call takes, not one of them.
func.func @logical_and(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "LAND", usedModel = 1 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.allreduce' op has a reduce op 'LAND' that no OpenSHMEM call of this shape takes}}
  %e = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %dt, %op) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}

// -----

// A datatype OpenSHMEM has no type name for.
func.func @no_type_name(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = i1, usedModel = 1 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 1 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.allreduce' op has a datatype i1 that no OpenSHMEM call of this shape takes}}
  %e = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %dt, %op) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}

// -----

// OpenSHMEM's collectives are all blocking.  Making this one blocking would
// move the synchronisation to a point the program did not choose.
func.func @nonblocking_collective(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 1 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.allreduce' op is a non-blocking collective, which OpenSHMEM does not have}}
  %req, %e = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %dt, %op) <{isBlocking = false, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  return
}

// -----

// A split by colour and key.  roundtrip passed the colour as the start PE and
// the key as the team size.
func.func @split(%color: i32, %key: i32) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.commSplit' op splits by a colour and a key; OpenSHMEM splits a team by a triplet of PE numbers (shmem_team_split_strided), which spmd.commSplitStrided is}}
  %new, %e = "spmd.commSplit"(%world, %color, %key) <{usedModel = 1 : i32}> : (!spmd.comm, i32, i32) -> (!spmd.comm, !spmd.error)
  return
}

// -----

// Two-sided messaging, which OpenSHMEM does not have.
func.func @send(%buf: !llvm.ptr, %n: i64, %dest: i32, %tag: i32) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  // expected-error @+1 {{'spmd.send' op is two-sided messaging, which OpenSHMEM does not have; its transfers name an address on the symmetric heap, not a peer that receives}}
  %e = "spmd.send"(%world, %buf, %n, %dt, %dest, %tag) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// -----

// The target address is the window, so the displacement has to be zero: its
// unit is the window's, and nothing of it survives the conversion.
func.func @displacement(%src: !llvm.ptr, %n: i64, %pe: i32, %win: !spmd.win) {
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  // expected-error @+1 {{'spmd.put' op has a target disp other than 0}}
  %e = "spmd.put"(%src, %n, %dt, %c1, %pe, %win, %c4, %n, %dt, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  return
}

// -----

// A window over the stack, which no other PE can reach.
func.func @stack_window() {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %c1 = arith.constant 1 : i32
  %buf = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // expected-error @+1 {{'spmd.winCreate' op creates a window over the stack, which is not symmetric; OpenSHMEM reaches only the symmetric heap and global variables from another PE}}
  %win, %e = "spmd.winCreate"(%world, %buf, %c1) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  return
}

// -----

// A window over an element of a stack array is over the stack all the same.
func.func @stack_element_window() {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %buf = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr
  %elem = llvm.getelementptr %buf[2] : (!llvm.ptr) -> !llvm.ptr, i32
  // expected-error @+1 {{'spmd.winCreate' op creates a window over the stack, which is not symmetric}}
  %win, %e = "spmd.winCreate"(%world, %elem, %c1) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  return
}

// -----

// A window freed where the lowering cannot see what made it.  One from
// spmd.winAlloc owns shmem_malloc'd memory and one from spmd.winCreate is the
// program's own, so freeing it or not is a guess either way.
func.func @free_unknown_window(%win: !spmd.win) {
  // expected-error @+1 {{'spmd.winFree' op frees a window made neither by spmd.winAlloc nor by spmd.winCreate in sight; whether shmem_free has to free its memory depends on which, and guessing either way frees memory twice or leaks it}}
  %e = "spmd.winFree"(%win) <{usedModel = 1 : i32}> : (!spmd.win) -> !spmd.error
  return
}

// -----

// shmem_quiet() completes everything and reports on nothing.
func.func @wait_status(%req: !spmd.req) {
  // expected-error @+1 {{'spmd.wait' op asks for a status; shmem_quiet() reports nothing about the operation it completes}}
  %status, %e = "spmd.wait"(%req) <{usedModel = 1 : i32}> : (!spmd.req) -> (!spmd.status, !spmd.error)
  return
}

// -----

// A thread level OpenSHMEM does not define.
func.func @thread_level() {
  // expected-error @+1 {{'spmd.constThreadLevel' op 'OVERSUBSCRIBED' is not an OpenSHMEM thread support level}}
  %l = "spmd.constThreadLevel"() <{level = "OVERSUBSCRIBED", usedModel = 1 : i32}> : () -> !spmd.threadLevel
  %e = "spmd.init"(%l) <{usedModel = 1 : i32}> : (!spmd.threadLevel) -> !spmd.error
  return
}

// -----

// A team kept in an int, as a program written against MPICH keeps its
// communicator: a team is a pointer here.
func.func @team_in_int(%store: !llvm.ptr) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{the program keeps an OpenSHMEM team in 'i32', but a team is a pointer in this ABI; a variable declared for an MPI handle is too narrow for it}}
  %asInt = builtin.unrealized_conversion_cast %world : !spmd.comm to i32
  llvm.store %asInt, %store : i32, !llvm.ptr
  return
}

// -----

// Two different counts, which a put that takes one cannot express.
func.func @different_counts(%src: !llvm.ptr, %n: i32, %m: i32, %pe: i32, %win: !spmd.win) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %i32 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %originCount = arith.extui %n : i32 to i64
  %targetCount = arith.extui %m : i32 to i64
  // expected-error @+1 {{'spmd.put' op has distinct origin and target counts; shmem_int_put has one}}
  %e = "spmd.put"(%src, %originCount, %i32, %c1, %pe, %win, %c0, %targetCount, %i32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  return
}

// -----

// A maximum over an integer of unknown signedness: shmem_int_max_reduce and
// shmem_uint_max_reduce compute different ones, and i32 does not say which the
// program meant.  An i8 is char's, which OpenSHMEM has a maximum of.
func.func @unknown_signedness(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %i32 = "spmd.constDatatype"() <{typeAttr = i32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %i8 = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 1 : i32}> : () -> !spmd.datatype
  %max = "spmd.constOpType"() <{stringAttr = "max", usedModel = 1 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.allreduce' op reduces with 'max' over i32, whose signedness the result depends on and the datatype does not record}}
  %e0 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %i32, %max) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e1 = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %i8, %max) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}
