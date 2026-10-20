// RUN: env SPMD_IR_EXITCODE=0 spmd-opt %s -check-collectives 2>&1 >/dev/null | FileCheck %s --check-prefix=COLL
// RUN: env SPMD_IR_EXITCODE=0 spmd-opt %s -check-dataRace 2>&1 >/dev/null | FileCheck %s --check-prefix=RACE
// RUN: env SPMD_IR_EXITCODE=0 spmd-opt %s -check-thread-safety 2>&1 >/dev/null | FileCheck %s --check-prefix=THREAD
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && env SPMD_IR_EXITCODE=0 spmd-opt %s -check-collectives=write-json-report >/dev/null
// RUN: cd %t && env SPMD_IR_EXITCODE=0 spmd-opt %s -check-dataRace=write-json-report >/dev/null
// RUN: cd %t && env SPMD_IR_EXITCODE=0 spmd-opt %s -check-thread-safety=write-json-report >/dev/null
// RUN: FileCheck %s --check-prefix=EMPTY < %t/collective_mismatch_report.json
// RUN: FileCheck %s --check-prefix=EMPTY < %t/local_data_race_report.json
// RUN: FileCheck %s --check-prefix=EMPTY < %t/thread_safety_report.json
//
// What each checker prints when it finds nothing, and the JSON report it
// writes then. Both are what spmd-verify's exit status and the benchmark
// suites read, so the wording is part of the interface and not an
// implementation detail.
//
// The report goes to the file named by the checker in the *current* directory,
// which is why the JSON runs above change into %t first. SPMD_IR_EXITCODE=0
// is not needed on a clean program, but is kept so the two halves of the file
// differ only in the program — see report-findings.mlir, where the pass does
// fail.
//
// The program: one collective every rank reaches, a non-blocking receive
// completed before its buffer is touched, and no thread-parallel region at all.

// COLL: Static Verification of Collective Communication:
// COLL: No Collective Communication Error Found

// RACE: Static Verification of Data Races:
// RACE: No Data Race Found

// THREAD: Static Verification of Thread-Support Level Compliance:
// THREAD: No thread-safety violations found

// EMPTY:      "messages": []

func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %idx = arith.constant 0 : index
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 1 : i32, usedModel = 0 : i32}> ({
    %buf = memref.alloca() : memref<1xi32>
    %eb = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    %req, %err = "spmd.recv"(%world, %buf, %count, %dt, %c0, %c0) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    memref.store %c0, %buf[%idx] : memref<1xi32>
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
