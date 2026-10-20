# SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Lit configuration for the execution tests.
#
# Each test is a program that checks its own results.  Its RUN lines compile it
# through the SPMD dialect and back (`spmd-verify compile`), launch it, and
# FileCheck what it prints: a wrong value makes the program abort or exit
# non-zero, which fails the test, and the program has to print that it passed,
# so one that exits 0 without having run its checks fails as well.  Unlike the
# unit tests next door (test/lit.cfg.py), these need a real programming-model
# runtime, so they are not part of //test:spmd_tests and run outside Bazel:
#
#   ml purge; ml GCC/14.2.0 MPICH/4.3.0 LLVM/20.1.7
#   lit -sv test/exec
#
# Build //utility:spmd-verify first.  What the RUN lines use:
#   %spmd-verify  -D spmd_verify=, else $SPMD_VERIFY, else
#                 bazel-bin/utility/spmd-verify of this checkout
#   %mpiexec      mpiexec under `timeout`, so a deadlocked program fails
#                 instead of hanging the run; $SPMD_EXEC_TIMEOUT seconds
#                 (default 120)
#   %run          `timeout` alone, for a program that is not launched by MPI
# OpenMP programs (omp/) link against LLVM's libomp, which `compile` finds
# through the clang on PATH, so load the LLVM module too:
#   ml purge; ml GCC/14.2.0 MPICH/4.3.0 LLVM/20.1.7
#   FileCheck     bazel-bin's, when //test:spmd_tests has built it, else PATH
# Outputs go to -D build_dir=, default build/exec-tests/ of this checkout.
#
# proxyapps/ builds real applications from the benchmark submodule's
# proxyApps/ into one bitcode file each and compiles that.  Those tests
# REQUIRE `proxyapps` (the submodule is checked out and Bazel built
# reactant-clang and lld) and, for a CMake project, `cmake`.  They use:
#   %proxyapps       externals/data-race-detection-benchmark-suite/proxyApps
#   %reactant-clang  Bazel's reactant-clang, used as a plain clang
#   %lto-cflags      compile a translation unit for linking to bitcode
#   %lto-ldflags     link the translation units to one bitcode file, at
#                    --lto-O1 (see spmd-verify.sh's compile)
#   %mpi-include     the MPICH headers the raising is written against

import os
import shlex
import shutil

import lit.formats

config.name = "SPMD-exec"
config.test_format = lit.formats.ShTest()
config.suffixes = [".c", ".cpp", ".test"]
config.excludes = ["Inputs"]
config.test_source_root = os.path.dirname(os.path.abspath(__file__))

repo_root = os.path.dirname(os.path.dirname(config.test_source_root))
config.test_exec_root = lit_config.params.get(
    "build_dir", os.path.join(repo_root, "build", "exec-tests")
)

spmd_verify = lit_config.params.get("spmd_verify") or os.environ.get(
    "SPMD_VERIFY", os.path.join(repo_root, "bazel-bin", "utility", "spmd-verify")
)
if not os.access(spmd_verify, os.X_OK):
    lit_config.fatal(
        "spmd-verify not found at %s; build //utility:spmd-verify, or pass "
        "-D spmd_verify=<path>" % spmd_verify
    )

mpiexec = "mpiexec"
timeout = os.environ.get("SPMD_EXEC_TIMEOUT", "120")

config.substitutions.append(("%spmd-verify", shlex.quote(spmd_verify)))

proxyapps = os.path.join(
    repo_root, "externals", "data-race-detection-benchmark-suite", "proxyApps"
)
reactant_clang = os.path.join(repo_root, "bazel-bin", "external", "reactant", "reactant-clang")
reactant_resource = os.path.join(
    repo_root, "bazel-out", "k8-opt", "bin", "external", "reactant", "reactant-clang-resource"
)
lld = os.path.join(repo_root, "bazel-bin", "external", "llvm-project", "lld", "ld.lld")
if (
    os.path.isdir(os.path.join(proxyapps, "TeaLeaf"))
    and os.access(reactant_clang, os.X_OK)
    and os.access(lld, os.X_OK)
):
    config.available_features.add("proxyapps")
if shutil.which("cmake"):
    config.available_features.add("cmake")
config.substitutions.append(("%proxyapps", shlex.quote(proxyapps)))
config.substitutions.append(("%reactant-clang", shlex.quote(reactant_clang)))
config.substitutions.append(
    ("%lto-cflags", "-resource-dir %s -O1 -flto -fno-exceptions" % shlex.quote(reactant_resource))
)
config.substitutions.append(
    (
        "%lto-ldflags",
        "-resource-dir %s -fuse-ld=%s -flto -fno-exceptions -Wl,--lto-O1 "
        "-Wl,--lto-emit-llvm" % (shlex.quote(reactant_resource), shlex.quote(lld)),
    )
)
config.substitutions.append(
    ("%mpi-include", shlex.quote(os.path.join(repo_root, "includesProgramModels", "mpi", "include")))
)
config.substitutions.append(("%mpiexec", "timeout %s %s" % (timeout, mpiexec)))
config.substitutions.append(("%run", "timeout %s" % timeout))

path = [os.environ.get("PATH", "")]
filecheck_dir = os.path.join(repo_root, "bazel-bin", "external", "llvm-project", "llvm")
if os.access(os.path.join(filecheck_dir, "FileCheck"), os.X_OK):
    path.insert(0, filecheck_dir)
config.environment["PATH"] = os.pathsep.join(path)

# lit starts the tests from a reduced environment.  The compile step needs the
# compiler and runtime a module loaded, and the launcher its own settings.
_PASS_PREFIXES = (
    "CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "CC", "CXX", "MPICC", "MPICXX",
    "OSHCC", "MPIEXEC", "SPMD_", "MPIR_", "MPICH_", "HYDRA_", "UCX_", "FI_",
    "I_MPI_", "PMI", "SLURM_", "NCCL_", "CUDA_", "OMP_", "KMP_", "EBROOT",
    "HOME", "USER", "TMPDIR",
)
for name, value in os.environ.items():
    if name.startswith(_PASS_PREFIXES):
        config.environment[name] = value
