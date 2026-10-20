// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- openshmem_abi_test.cpp - OpenSHMEM.h against the vendored shmem.h --===//
//
// include/spmd/TargetABI/OpenSHMEM.h copies OpenSHMEM's predefined constants
// by hand, so that the passes never see shmem.h's macros.  This test compiles
// the table against includesProgramModels/shmem/include/shmem.h and checks
// every entry against the macro it copies, so a value that drifts fails the
// build.  It also pins the two ABI facts the lowering's type converter rests
// on: a team handle is a pointer, and the predefined teams are variables of
// that type rather than constants.
//
//===----------------------------------------------------------------------===//

#include "spmd/TargetABI/OpenSHMEM.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "shmem.h"

namespace openshmem = mlir::spmd::abi::openshmem;

namespace {

// Not a value OpenSHMEM uses, so a name missing from the table never compares
// equal to a macro.
constexpr int64_t kMissing = INT64_MIN;

constexpr bool sameName(llvm::StringLiteral name, std::string_view expected) {
  return std::string_view(name.data(), name.size()) == expected;
}

template <std::size_t N>
constexpr int64_t handleNamed(const openshmem::NamedHandle (&table)[N],
                              std::string_view name) {
  for (const openshmem::NamedHandle &entry : table)
    if (sameName(entry.name, name))
      return entry.handle;
  return kMissing;
}

#define CHECK_NAMED(TABLE, NAME, MACRO)                                        \
  static_assert(handleNamed(openshmem::TABLE, NAME) == (MACRO), #MACRO)

// Adding a table entry without checking it here fails too.
static_assert(std::size(openshmem::kThreadLevels) == 4);
static_assert(std::size(openshmem::kSignalOps) == 2);

CHECK_NAMED(kThreadLevels, "SINGLE", SHMEM_THREAD_SINGLE);
CHECK_NAMED(kThreadLevels, "FUNNELED", SHMEM_THREAD_FUNNELED);
CHECK_NAMED(kThreadLevels, "SERIALIZED", SHMEM_THREAD_SERIALIZED);
CHECK_NAMED(kThreadLevels, "MULTIPLE", SHMEM_THREAD_MULTIPLE);

CHECK_NAMED(kSignalOps, "SET", SHMEM_SIGNAL_SET);
CHECK_NAMED(kSignalOps, "ADD", SHMEM_SIGNAL_ADD);

static_assert(openshmem::kTeamNumContexts == SHMEM_TEAM_NUM_CONTEXTS);

// A team handle is a pointer, which is why the lowering converts !spmd.comm to
// !llvm.ptr rather than to an integer, and SHMEM_TEAM_INVALID is the null one.
static_assert(std::is_pointer_v<shmem_team_t>);
static_assert(SHMEM_TEAM_INVALID == nullptr);
static_assert(openshmem::kTeamInvalid == 0);

// The predefined teams are variables of that pointer type, so a call that
// takes SHMEM_TEAM_WORLD loads the variable these names spell.
static_assert(std::is_same_v<decltype(SHMEM_TEAM_WORLD), shmem_team_t>);
static_assert(std::is_same_v<decltype(SHMEM_TEAM_SHARED), shmem_team_t>);
static_assert(sameName(openshmem::kTeamWorld, "SHMEM_TEAM_WORLD"));
static_assert(sameName(openshmem::kTeamShared, "SHMEM_TEAM_SHARED"));

// shmem_team_split_strided takes its configuration by address, so a lowering
// with nothing to configure passes a null pointer and an empty mask.
static_assert(sizeof(shmem_team_config_t) == sizeof(int));
static_assert(openshmem::kTeamConfigMaskNone == 0);

// Where this implementation's return type differs from the specification's,
// which the lowering declares otherwise.
static_assert(std::size(openshmem::kReturnTypeOverrides) == 1);
static_assert(sameName(openshmem::kReturnTypeOverrides[0].callee,
                       "shmem_team_destroy") &&
              sameName(openshmem::kReturnTypeOverrides[0].cType, "int"));
static_assert(
    std::is_same_v<decltype(shmem_team_destroy(SHMEM_TEAM_WORLD)), int>);

} // namespace

// Nothing here needs the runtime: every check is a static_assert, so the test
// passing is the test having compiled.
int main() { return 0; }
