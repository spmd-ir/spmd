// RUN: env SPMD_IR_EXITCODE=0 spmd-opt %s -check-collectives 2>&1 >/dev/null | FileCheck %s --check-prefix=COLL
// RUN: env SPMD_IR_EXITCODE=0 spmd-opt %s -check-dataRace 2>&1 >/dev/null | FileCheck %s --check-prefix=RACE
// RUN: env SPMD_IR_EXITCODE=0 spmd-opt %s -check-thread-safety 2>&1 >/dev/null | FileCheck %s --check-prefix=THREAD
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && env SPMD_IR_EXITCODE=0 spmd-opt %s -check-collectives=write-json-report >/dev/null
// RUN: cd %t && env SPMD_IR_EXITCODE=0 spmd-opt %s -check-dataRace=write-json-report >/dev/null
// RUN: cd %t && env SPMD_IR_EXITCODE=0 spmd-opt %s -check-thread-safety=write-json-report >/dev/null
// RUN: FileCheck %s --check-prefix=COLL-JSON < %t/collective_mismatch_report.json
// RUN: FileCheck %s --check-prefix=RACE-JSON < %t/local_data_race_report.json
// RUN: FileCheck %s --check-prefix=THREAD-JSON < %t/thread_safety_report.json
//
// One program with one finding for each of the three checkers, to pin the two
// report formats against each other: the human-readable one on stderr and the
// JSON the suites and the IDE integration consume. The findings themselves are
// covered case by case in collectives.mlir, data-race.mlir and
// thread-safety.mlir; what is checked here is that a finding reaches both
// reports, with a location attached.
//
// SPMD_IR_EXITCODE=0 is what makes these runnable: a checker with findings
// signals pass failure, and spmd-opt turns that into a non-zero exit unless
// the variable says otherwise. lit.cfg.py deliberately does not set it
// globally, so it is spelled out per RUN line.
//
// The program declares MPI_THREAD_FUNNELED (threadLevel = 1) and then
//   * calls a barrier only on rank 0,
//   * writes the buffer of a receive that is still pending, and
//   * calls a barrier from every thread of an OpenMP region.

// COLL:      Static Verification of Collective Communication:
// COLL:      warning: [CheckCollectives] barrier operation/call may not be called by all processes
// COLL:      Probably due to the conditionals:
// COLL:      remark: probable causing conditional number 0

// COLL-JSON:      "messages": [
// COLL-JSON:        "references": [
// COLL-JSON:            "column":
// COLL-JSON:            "file":
// COLL-JSON:            "line":
// COLL-JSON:        "text": "Detected conflict involving 1 operation(s) and 1 causing conditional(s)."
// COLL-JSON:        "type": "Collective Mismatch"

// RACE:      Static Verification of Data Races:
// RACE:      warning: [CheckDataRace]store operation/call may access the same memory as the previous operation without proper synch. (Data Race)
// RACE:      Previous operation:
// RACE:      warning: recv operation/call
// RACE:      With the alias values:
// RACE-NEXT: Causing Access on:
// RACE-NEXT: Previous Access on:

// RACE-JSON:      "text": "Found recv operation/call at {{.*}} which is in conflict with store operation/call at {{.*}} before release"
// RACE-JSON:      "type": "Local Data Race"

// THREAD:      Static Verification of Thread-Support Level Compliance:
// THREAD:      warning: [CheckThreadSafety] barrier may violate thread-support level FUNNELED

// THREAD-JSON:      "text": "barrier at {{.*}} may violate MPI_THREAD_FUNNELED"
// THREAD-JSON:      "type": "Thread Safety Violation"

func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 1 : i32, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<1xi32>
    %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    %is0 = arith.cmpi eq, %rank, %c0 : i32
    scf.if %is0 {
      %eb = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    }
    %req, %err = "spmd.recv"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    memref.store %c0, %buf[%idx] : memref<1xi32>
    %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    %team, %ed = "spmd.commDup"(%world) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %t = "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
      "spmd.parallel_yield"(%c0) : (i32) -> ()
    }) : (!spmd.comm) -> i32
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
