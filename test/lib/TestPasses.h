/*
 * SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- TestPasses.h - Test-only passes of spmd-opt --------------*- C++ -*-===//
//
// Passes that exist only to make an analysis observable from a lit test, by
// writing what it computed into discardable `test.*` attributes. They are
// linked into the //test/lib:spmd-opt binary the unit tests run, and not into
// //spmd-opt:spmd-opt, which spmd-verify ships.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_TEST_LIB_TESTPASSES_H
#define SPMD_TEST_LIB_TESTPASSES_H

namespace mlir::spmd::test {

void registerTestBufferSlicesPass();
void registerTestCUDALoweringTablePass();
void registerTestExecutionContextPasses();
void registerTestHandleOriginsPasses();
void registerTestMPILoweringTablePass();
void registerTestNCCLLoweringTablePass();
void registerTestNVSHMEMLoweringTablePass();
void registerTestOMPLoweringTablePass();
void registerTestSHMEMLoweringTablePass();
void registerTestUniformityPass();

/// Registers every test pass.
inline void registerTestPasses() {
  registerTestBufferSlicesPass();
  registerTestCUDALoweringTablePass();
  registerTestExecutionContextPasses();
  registerTestHandleOriginsPasses();
  registerTestMPILoweringTablePass();
  registerTestNCCLLoweringTablePass();
  registerTestNVSHMEMLoweringTablePass();
  registerTestOMPLoweringTablePass();
  registerTestSHMEMLoweringTablePass();
  registerTestUniformityPass();
}

} // namespace mlir::spmd::test

#endif // SPMD_TEST_LIB_TESTPASSES_H
