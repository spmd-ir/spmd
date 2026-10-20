// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- nvshmem_abi_test.cpp - NVSHMEM.h against the vendored nvshmem.h ----===//
//
// include/spmd/TargetABI/NVSHMEM.h copies NVSHMEM's predefined constants by
// hand, so that the passes never see nvshmem.h's enumerators.  This test
// compiles the table against includesProgramModels/mynvshmem/include/nvshmem.h
// and checks every entry against the enumerator it copies, so a value that
// drifts fails the build.  It also pins the ABI fact the lowering's type
// converter rests on: a team handle is an int32_t, not a pointer as in
// OpenSHMEM.
//
// The vendored headers are a stand-in for NVIDIA's (NVSHMEM.h says why), so
// this checks the raiser's view of NVSHMEM.  That the values are NVIDIA's too
// was checked by hand against NVSHMEM 3.8's headers; nothing in the build can
// check it, since they are not vendored.
//
//===----------------------------------------------------------------------===//

#include "spmd/TargetABI/NVSHMEM.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "nvshmem.h"

namespace nvshmem = mlir::spmd::abi::nvshmem;

namespace {

// Not a value NVSHMEM uses, so a name missing from the table never compares
// equal to an enumerator.
constexpr int64_t kMissing = INT64_MIN;

constexpr bool sameName(llvm::StringLiteral name, std::string_view expected) {
  return std::string_view(name.data(), name.size()) == expected;
}

template <std::size_t N>
constexpr int64_t handleNamed(const nvshmem::NamedHandle (&table)[N],
                              std::string_view name) {
  for (const nvshmem::NamedHandle &entry : table)
    if (sameName(entry.name, name))
      return entry.handle;
  return kMissing;
}

#define CHECK_NAMED(TABLE, NAME, ENUMERATOR)                                   \
  static_assert(handleNamed(nvshmem::TABLE, NAME) == (ENUMERATOR), #ENUMERATOR)

// Adding a table entry without checking it here fails too.
static_assert(std::size(nvshmem::kThreadLevels) == 4);
static_assert(std::size(nvshmem::kSignalOps) == 2);

CHECK_NAMED(kThreadLevels, "SINGLE", NVSHMEM_THREAD_SINGLE);
CHECK_NAMED(kThreadLevels, "FUNNELED", NVSHMEM_THREAD_FUNNELED);
CHECK_NAMED(kThreadLevels, "SERIALIZED", NVSHMEM_THREAD_SERIALIZED);
CHECK_NAMED(kThreadLevels, "MULTIPLE", NVSHMEM_THREAD_MULTIPLE);

CHECK_NAMED(kSignalOps, "SET", NVSHMEM_SIGNAL_SET);
CHECK_NAMED(kSignalOps, "ADD", NVSHMEM_SIGNAL_ADD);

// A team handle is an int32_t, which is why the lowering converts !spmd.comm
// to i32, and the predefined teams are its constants.
static_assert(std::is_same_v<nvshmem_team_t, int32_t>);
static_assert(nvshmem::kTeamWorld == NVSHMEM_TEAM_WORLD);
static_assert(nvshmem::kTeamShared == NVSHMEM_TEAM_SHARED);
static_assert(nvshmem::kTeamNode == NVSHMEMX_TEAM_NODE);
static_assert(nvshmem::kTeamInvalid == NVSHMEM_TEAM_INVALID);

// nvshmem_team_split_strided takes its configuration by address, so a lowering
// with nothing to configure passes a null pointer and an empty mask.
static_assert(sizeof(nvshmem_team_config_t) == 64);
static_assert(nvshmem::kTeamConfigMaskNone == 0);

} // namespace

// Nothing here needs the runtime: every check is a static_assert, so the test
// passing is the test having compiled.
int main() { return 0; }
