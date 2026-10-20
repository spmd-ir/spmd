// RUN: spmd-opt %s -split-input-file -lower-spmd-to-nccl -verify-diagnostics
//
// Everything NCCL cannot express is reported before the module is rewritten,
// each with its own reason rather than as the conversion's "failed to
// legalize".  Several are what the roundtrip lowering silently emitted
// something else for.

// NCCL keeps no process-wide state; roundtrip lowered this to a constant 0.
func.func @finalize() {
  // expected-error @+1 {{'spmd.finalize' op has no NCCL counterpart; NCCL keeps no process-wide state, and each communicator is destroyed by ncclCommDestroy, which spmd.commDestroy is}}
  %e = "spmd.finalize"() <{usedModel = 3 : i32}> : () -> !spmd.error
  return
}

// -----

// roundtrip lowered a barrier to an empty group, which waits for nothing.
func.func @barrier(%comm: !spmd.comm) {
  // expected-error @+1 {{'spmd.barrier' op has no NCCL counterpart; NCCL has no barrier, and an empty group synchronises nothing}}
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 3 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// -----

func.func @comm_dup(%comm: !spmd.comm) {
  // expected-error @+1 {{'spmd.commDup' op has no NCCL counterpart; NCCL duplicates a communicator by splitting it with one colour, which spmd.commSplit is}}
  %dup, %e = "spmd.commDup"(%comm) <{usedModel = 3 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  return
}

// -----

func.func @no_call_at_all() {
  // expected-error @+1 {{'spmd.commNode' op has no NCCL counterpart}}
  %node = "spmd.commNode"() <{usedModel = 3 : i32}> : () -> !spmd.comm
  return
}

// -----

// NCCL has no tag; the raiser gives the operation 0.  roundtrip dropped the
// tag, so two messages the program told apart by it were matched in order.
func.func @tag(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64, %peer: i32) {
  %tag = arith.constant 5 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  // expected-error @+1 {{'spmd.send' op has a tag other than 0}}
  %e = "spmd.send"(%comm, %buf, %n, %dt, %peer, %tag, %s) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> !spmd.error
  return
}

// -----

// A program raised from a model without streams has none to pass.
func.func @no_stream(%comm: !spmd.comm, %send: !llvm.ptr, %recv: !llvm.ptr, %n: i64) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 3 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.allreduce' op has no CUDA stream; every NCCL communication call takes one}}
  %e = "spmd.allreduce"(%comm, %send, %recv, %n, %dt, %op) <{isBlocking = true, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}

// -----

// roundtrip rounded every datatype it did not know to ncclInt32 and every
// reduction to ncclSum.
func.func @enumerators() {
  // expected-error @+1 {{'spmd.constDatatype' op has no NCCL datatype for 'si16'}}
  %dt = "spmd.constDatatype"() <{typeAttr = si16, usedModel = 3 : i32}> : () -> !spmd.datatype
  // expected-error @+1 {{'spmd.constOpType' op 'BAND' is not a built-in NCCL reduction operation}}
  %op = "spmd.constOpType"() <{stringAttr = "BAND", usedModel = 3 : i32}> : () -> !spmd.op
  return
}

// -----

func.func @stream_storage(%slot: memref<1x!llvm.ptr>) {
  // expected-error @+1 {{'spmd.castStream' op casts 'memref<1x!llvm.ptr>' to a stream; NCCL takes the cudaStream_t itself, which is a pointer}}
  %s = "spmd.castStream"(%slot) <{usedModel = 3 : i32}> : (memref<1x!llvm.ptr>) -> !spmd.stream
  return
}

// -----

// A wait that completes one of two open operations: ncclGroupEnd would run
// both, or neither.
func.func @wait_part_of_group(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64, %peer: i32) {
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %r0, %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %tag, %s) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> (!spmd.req, !spmd.error)
  %r1, %e1 = "spmd.recv"(%comm, %buf, %n, %dt, %peer, %tag, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> (!spmd.req, !spmd.error)
  // expected-error @+1 {{'spmd.wait' op completes 1 request(s), but 2 grouped NCCL operation(s) are open in its block; ncclGroupEnd runs every operation of its group, so a wait has to complete all of them}}
  %e2 = "spmd.wait"(%r0) <{usedModel = 3 : i32}> : (!spmd.req) -> !spmd.error
  %e3 = "spmd.wait"(%r1) <{usedModel = 3 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// -----

// The wait is in another block, so the lowering does not see where the group
// ends.
func.func @group_across_blocks(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64, %cond: i1) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  // expected-error @+1 {{'spmd.allgather' op is an operation of a group that no spmd.wait in its block completes; NCCL runs a group at its ncclGroupEnd, which the lowering can place only where it sees every operation of the group}}
  %r, %e0 = "spmd.allgather"(%comm, %buf, %n, %dt, %buf, %n, %dt, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, !spmd.stream) -> (!spmd.req, !spmd.error)
  scf.if %cond {
    // expected-error @+1 {{'spmd.wait' op completes 1 request(s), but 0 grouped NCCL operation(s) are open in its block}}
    %e1 = "spmd.wait"(%r) <{usedModel = 3 : i32}> : (!spmd.req) -> !spmd.error
  }
  return
}

// -----

// An operation outside the group that runs while it is open would be deferred
// into it.
func.func @communication_inside_open_group(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %r, %e0 = "spmd.allgather"(%comm, %buf, %n, %dt, %buf, %n, %dt, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, !spmd.stream) -> (!spmd.req, !spmd.error)
  // expected-error @+1 {{'spmd.allgather' op runs while 1 grouped NCCL operation(s) wait for their ncclGroupEnd, and would be grouped with them}}
  %e1 = "spmd.allgather"(%comm, %buf, %n, %dt, %buf, %n, %dt, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, !spmd.stream) -> !spmd.error
  %e2 = "spmd.wait"(%r) <{usedModel = 3 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// -----

func.func @wait_all_counted_at_run_time(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64, %count: i32, %reqs: memref<1x!spmd.req>) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %r, %e0 = "spmd.allgather"(%comm, %buf, %n, %dt, %buf, %n, %dt, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, !spmd.stream) -> (!spmd.req, !spmd.error)
  // expected-error @+1 {{'spmd.waitAll' op completes a number of requests that is not a constant; the lowering emits one ncclGroupEnd per request}}
  %e1 = "spmd.waitAll"(%count, %reqs) <{usedModel = 3 : i32}> : (i32, memref<1x!spmd.req>) -> !spmd.error
  return
}

// -----

func.func @wait_status(%comm: !spmd.comm, %stream: !llvm.ptr, %buf: !llvm.ptr, %n: i64) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  %r, %e0 = "spmd.allgather"(%comm, %buf, %n, %dt, %buf, %n, %dt, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, !spmd.stream) -> (!spmd.req, !spmd.error)
  // expected-error @+1 {{'spmd.wait' op asks for a status; ncclGroupEnd reports nothing about a single operation}}
  %st, %e1 = "spmd.wait"(%r) <{usedModel = 3 : i32}> : (!spmd.req) -> (!spmd.status, !spmd.error)
  return
}

// -----

func.func @req_free(%req: !spmd.req) {
  // expected-error @+1 {{'spmd.reqFree' op lets an operation complete on its own; an NCCL group runs only at the ncclGroupEnd that closes it, which spmd.wait is}}
  %e = "spmd.reqFree"(%req) <{usedModel = 3 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// -----

// The world communicator is read before the initialisation that makes it.
func.func @world_before_init() -> i32 {
  %mpiSingle = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  %e0 = "spmd.init"(%mpiSingle) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
  // expected-error @+1 {{'spmd.commWorld' op reads NCCL's world communicator where no spmd.init of NCCL has run before it; NCCL has no world communicator until ncclCommInitRank makes one, which spmd.init does}}
  %world = "spmd.commWorld"() <{usedModel = 3 : i32}> : () -> !spmd.comm
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 3 : i32}> : () -> !spmd.threadLevel
  %e1 = "spmd.init"(%single) <{usedModel = 3 : i32}> : (!spmd.threadLevel) -> !spmd.error
  %rank, %e2 = "spmd.getRankInComm"(%world) <{usedModel = 3 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  return %rank : i32
}

// -----

// Nothing to send the unique id over.
func.func @nccl_alone() {
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 3 : i32}> : () -> !spmd.threadLevel
  // expected-error @+1 {{'spmd.init' op starts NCCL, whose world communicator is made from a unique id that rank 0 sends the other processes over another model's collectives; the module starts neither MPI nor OpenSHMEM}}
  %e = "spmd.init"(%single) <{usedModel = 3 : i32}> : (!spmd.threadLevel) -> !spmd.error
  return
}

// -----

func.func @thread_provided() {
  %mpiSingle = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  %e0 = "spmd.init"(%mpiSingle) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
  %multiple = "spmd.constThreadLevel"() <{level = "MULTIPLE", usedModel = 3 : i32}> : () -> !spmd.threadLevel
  // expected-error @+1 {{'spmd.init' op asks for the thread level granted; NCCL has no thread support levels}}
  %provided, %e1 = "spmd.init"(%multiple) <{usedModel = 3 : i32}> : (!spmd.threadLevel) -> (!spmd.threadLevel, !spmd.error)
  return
}

// -----

// A program written against MPICH keeps a communicator in an int, and an NCCL
// communicator is a pointer.
func.func @comm_in_int(%handle: i32) -> i32 {
  // expected-error @+1 {{the program keeps an NCCL communicator in 'i32', but a communicator is a pointer in this ABI; a variable declared for an MPI handle is too narrow for it}}
  %comm = builtin.unrealized_conversion_cast %handle : i32 to !spmd.comm
  %rank, %e = "spmd.getRankInComm"(%comm) <{usedModel = 3 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  return %rank : i32
}

// -----

// The dialect has one world communicator, so the spmd.commWorld after either
// initialisation is the same one -- -cse makes the two a single operation --
// and neither can be told which communicator it reads.
func.func @initialised_twice() {
  %mpiSingle = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  %e0 = "spmd.init"(%mpiSingle) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 3 : i32}> : () -> !spmd.threadLevel
  // expected-note @+1 {{NCCL is first initialised here}}
  %e1 = "spmd.init"(%single) <{usedModel = 3 : i32}> : (!spmd.threadLevel) -> !spmd.error
  // expected-error @+1 {{'spmd.init' op initialises NCCL a second time; the dialect has one world communicator, so the spmd.commWorld after either initialisation reads the same one}}
  %e2 = "spmd.init"(%single) <{usedModel = 3 : i32}> : (!spmd.threadLevel) -> !spmd.error
  return
}

// -----

// A declaration of MPI_Init is no MPI: the raiser leaves the program's
// declarations behind, and a program translated from MPI to OpenSHMEM still has
// this one.  Only an initialisation counts.
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
func.func @mpi_only_declared() {
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 3 : i32}> : () -> !spmd.threadLevel
  // expected-error @+1 {{'spmd.init' op starts NCCL, whose world communicator is made from a unique id that rank 0 sends the other processes over another model's collectives; the module starts neither MPI nor OpenSHMEM}}
  %e = "spmd.init"(%single) <{usedModel = 3 : i32}> : (!spmd.threadLevel) -> !spmd.error
  return
}

// -----

// ncclMax and ncclAvg over an integer of unknown signedness: ncclInt8 and
// ncclUint8 give different results, and i8 -- MPI_CHAR's, translated -- does
// not say which the program meant.  A sum is the same either way.
func.func @unknown_signedness(%comm: !spmd.comm, %stream: !llvm.ptr, %send: !llvm.ptr, %recv: !llvm.ptr, %n: i64) {
  %dt = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 3 : i32}> : () -> !spmd.datatype
  %max = "spmd.constOpType"() <{stringAttr = "MAX", usedModel = 3 : i32}> : () -> !spmd.op
  %avg = "spmd.constOpType"() <{stringAttr = "AVG", usedModel = 3 : i32}> : () -> !spmd.op
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 3 : i32}> : () -> !spmd.op
  %s = "spmd.castStream"(%stream) <{usedModel = 3 : i32}> : (!llvm.ptr) -> !spmd.stream
  // expected-error @+1 {{'spmd.allreduce' op reduces with 'MAX' over i8, whose signedness the result depends on and the datatype does not record}}
  %e0 = "spmd.allreduce"(%comm, %send, %recv, %n, %dt, %max, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, !spmd.stream) -> !spmd.error
  // expected-error @+1 {{'spmd.allreduce' op reduces with 'AVG' over i8, whose signedness the result depends on and the datatype does not record}}
  %e1 = "spmd.allreduce"(%comm, %send, %recv, %n, %dt, %avg, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, !spmd.stream) -> !spmd.error
  %e2 = "spmd.allreduce"(%comm, %send, %recv, %n, %dt, %sum, %s) <{isBlocking = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, !spmd.stream) -> !spmd.error
  return
}
