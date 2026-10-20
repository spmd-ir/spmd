// RUN: spmd-opt %s -split-input-file -check-thread-safety -verify-diagnostics
//
// -check-thread-safety on hand-written SPMD IR: the thread-support level an
// MPI (or OpenSHMEM, or NVSHMEM) program declared at initialisation time,
// against the constructs it then issues communication from. Every case is one
// `main` with a process-level spmd.parallel carrying the level, and an
// OpenMP-style thread-level spmd.parallel over a duplicate of the world inside
// it — the shape -convert-apiCalls-to-spmd and -omp-to-spmd leave behind.
//
// The contract, MPI 5.0 §12.4.3 and the same wording in OpenSHMEM and NVSHMEM:
//
//   SINGLE      no thread-parallel region may exist at all; the region itself
//               is the finding, whatever it contains. Also the default when
//               the attribute is absent, since MPI_Init == MPI_Init_thread
//               with required = SINGLE.
//   FUNNELED    only the master thread may communicate, so only a guard on
//               thread number 0 is safe. spmd.single and spmd.critical are
//               not: they pick *a* thread, not the master.
//   SERIALIZED  calls must not be concurrent, so anything that runs on one
//               thread at a time is safe — the master guard, spmd.single and
//               spmd.critical alike.
//   MULTIPLE    no restriction.
//
// Model group 0 is MPI, 5 is OpenMP; the checker only looks at communication
// ops of the process-level parallel's own model.

// SINGLE, the default: the thread-parallel region itself is the finding, and
// it is reported once regardless of what is inside.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    // expected-warning @below {{[CheckThreadSafety] thread-parallel region may violate thread-support level SINGLE — the declared thread-support level requires the process to remain fully single-threaded}}
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// The same with the level spelled out (threadLevel = 0), and with the
// communication already inside a spmd.critical: still the whole region.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 0 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    // expected-warning @below {{thread-parallel region may violate thread-support level SINGLE}}
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      %ec = "spmd.critical"() <{usedModel = 5 : i32}> ({
        %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
        "spmd.terminator"() : () -> ()
      }) : () -> !spmd.error
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// FUNNELED: an unguarded call runs on every thread.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 1 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      // expected-warning @below {{[CheckThreadSafety] barrier may violate thread-support level FUNNELED — only the master thread may issue communication calls}}
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// FUNNELED: a guard on thread number 0 in the team is exactly the master
// thread, which is what the level allows.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 1 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      %tid, %et = "spmd.getRankInComm"(%team) <{usedModel = 5 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
      %is0 = arith.cmpi eq, %tid, %c0 : i32
      scf.if %is0 {
        %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      }
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// FUNNELED: spmd.single runs the body on one thread, but not necessarily the
// master one, so it does not satisfy this level.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 1 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      "spmd.single"(%team) <{usedModel = 5 : i32}> ({
        // expected-warning @below {{barrier may violate thread-support level FUNNELED}}
        %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
        "spmd.terminator"() : () -> ()
      }) : (!spmd.comm) -> ()
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// FUNNELED: nor does spmd.critical, for the same reason — it serialises the
// threads but does not select the master.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 1 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      %ec = "spmd.critical"() <{usedModel = 5 : i32}> ({
        // expected-warning @below {{barrier may violate thread-support level FUNNELED}}
        %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
        "spmd.terminator"() : () -> ()
      }) : () -> !spmd.error
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// SERIALIZED: an unguarded call is still concurrent.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 2 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      // expected-warning @below {{[CheckThreadSafety] barrier may violate thread-support level SERIALIZED — communication calls must not be issued concurrently}}
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// SERIALIZED: spmd.single and spmd.critical both make the call non-concurrent,
// and so does the master guard. Nothing here is reported.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 2 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      "spmd.single"(%team) <{usedModel = 5 : i32}> ({
        %e0 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
        "spmd.terminator"() : () -> ()
      }) : (!spmd.comm) -> ()
      %ec = "spmd.critical"() <{usedModel = 5 : i32}> ({
        %e1 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
        "spmd.terminator"() : () -> ()
      }) : () -> !spmd.error
      %tid, %et = "spmd.getRankInComm"(%team) <{usedModel = 5 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
      %is0 = arith.cmpi eq, %tid, %c0 : i32
      scf.if %is0 {
        %e2 = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      }
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// SERIALIZED: a task body runs on one thread at a time locally, but several
// tasks run concurrently, so the construct as a whole is the finding — it is
// reported on the spmd.task, not on the call inside it.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 2 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      // expected-warning @below {{task may violate thread-support level SERIALIZED}}
      "spmd.task"() <{mergeable = false, operandSegmentSizes = array<i32: 0, 0, 0>, untied = true, usedModel = 5 : i32}> ({
        %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
        "spmd.terminator"() : () -> ()
      }) : () -> ()
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// The check follows call edges: the communication is in a callee, and the
// finding lands on the operation there.
func.func @send_it(%w: !spmd.comm) {
  // expected-warning @below {{barrier may violate thread-support level SERIALIZED}}
  %e = "spmd.barrier"(%w) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 2 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      func.call @send_it(%world) : (!spmd.comm) -> ()
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// MULTIPLE lifts every restriction: the unguarded call is fine.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 3 : i32, usedModel = 0 : i32}> ({
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// One model group may have only one process-level parallel; two is an IR
// invariant violation the pass refuses to analyse.
func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  // expected-error @below {{[CheckThreadSafety] multiple process-level spmd.parallel ops for model group 0 — IR invariant violated}}
  %b = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %a : i32
}

// -----

// A translation unit without `main` is still analysed, from a whole-module
// walk, and says so.
// expected-warning @below {{[CheckThreadSafety] no 'main' function found; falling back to whole-module analysis}}
module {
  func.func @lib() -> i32 {
    %c0 = arith.constant 0 : i32
    %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
    %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 1 : i32, usedModel = 0 : i32}> ({
      %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
      %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
        // expected-warning @below {{barrier may violate thread-support level FUNNELED}}
        %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
        "spmd.parallel_yield"(%c0) : (i32) -> ()
      }) : (!spmd.comm) -> i32
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    return %r : i32
  }
}
