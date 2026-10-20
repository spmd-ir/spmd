<!--
SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# SPMD

An MLIR dialect and toolchain for statically verifying and optimising SPMD
(single-program, multiple-data) parallel programs -- data races,
collective-communication mismatches, thread-safety violations and more --
across MPI, OpenSHMEM, NVSHMEM, NCCL, CUDA and OpenMP.

SPMD is the PhD project of Semih Burak at RWTH Aachen University; see
*Authors* below.

## Directory Structure
- tried to resemble https://mlir.llvm.org/docs/Tutorials/CreatingADialect/ and structure of https://github.com/llvm/Polygeist


## How to Build

This project builds with **Bazel only**. The CMake build was removed: it had
gone stale (it still referenced deleted sources and missed most current ones,
so it could not even configure), and nothing used it -- CI, the README and the
test suites all drive Bazel. CMake still appears below only as the build system
of the two external benchmark suites, which are separate repositories.

0. Set up the environment. The versions in brackets are the ones CI uses:
- Clang/LLVM 20 [20.1.7, on GCC 14.2's libstdc++], used as the compiler:
  `export CC=clang CXX=clang++`
- Bazel 7.4.1 -- pinned in `.bazelversion`, so
  [Bazelisk](https://github.com/bazelbuild/bazelisk) picks the right one
- Python 3 [3.13.5]
- CUDA [13.1.0] -- its headers are needed for GPU programming model support
  (CUDA, NCCL, NVSHMEM)
- CMake and Ninja [4.0.3, 1.12.1] -- only for the benchmark suites, not for
  SPMD itself

On RWTH Aachen University's CLAIX cluster, all of these come from environment
modules:
```
ml purge
ml GCCcore/14.2.0
ml LLVM/20.1.7
ml CUDA/13.1.0
ml CMake/4.0.3
ml Ninja/1.12.1
ml Python/3.13.5
ml Bazel/7.4.1-Java-21
export CC=clang CXX=clang++
```

1. Clone the project:
```
git clone https://github.com/spmd-ir/spmd.git
```

The two benchmark suites under `externals/` are git submodules. SPMD builds
and its unit tests run without them; only the benchmark suites (see *Running
test suites*) and the proxy-app tests need them. **They are not public yet.**
Their submodule URLs are relative (`../<suite>.git`), so they resolve next to
whichever copy of this repository you cloned, and on GitHub they will start to
work once the suites are published beside this repository. Until then,
`git clone --recursive` and `git submodule update --init` fail for these two
submodules; clone without `--recursive` as above. Where the suites are
available, `git submodule update --init --recursive` fetches them and
`git submodule update --remote` moves them to their newest state.

Bazel fetches every other dependency itself -- LLVM/MLIR, Enzyme-JAX, Enzyme
and Reactant -- so none of them needs a separate clone. Their pinned commits
live in the build files, not here, so that they cannot go stale:
- Enzyme-JAX and Reactant: `ENZYME_JAX_COMMIT` and `REACTANT_COMMIT` in
  `workspace.bzl`. Reactant comes from the fork
  https://github.com/burakSemih/Reactant (see `externals/reactant/workspace.bzl`).
- Enzyme, which provides the alias analysis
  (`enzyme/Enzyme/MLIR/Analysis/DataFlowAliasAnalysis.h`): `ENZYME_COMMIT` in
  Enzyme-JAX's own `workspace.bzl`, at the Enzyme-JAX commit above.
- LLVM/MLIR: `LLVM_COMMIT` in XLA's `third_party/llvm/workspace.bzl`, reached
  through Enzyme-JAX -> XLA. After a build it is at
  `external/xla/third_party/llvm/workspace.bzl` under Bazel's output base.
  Unlike the others, it has no `OVERRIDE_..._PATH` for a local checkout.
  As of Enzyme-JAX `80c6fad` (October 2026) it is
  [`293623ce`](https://github.com/llvm/llvm-project/tree/293623ce99d4d6819378311c1506c5ab08d1d860),
  for browsing the matching sources; it moves with every Enzyme-JAX bump, so
  trust the file above over this line.

For following the steps easier, hold all additional repositories in the parent directory of the SPMD project repository.


2. To build the SPMD Project along with LLVM, MLIR, ENZYME-JAX, ENZYME, Reactant an other external projects needed by ENZYME-JAX:
```sh
cd spmd
bazel --output_base=/tmp/$USER/bazelOut build -c opt --spawn_strategy=local //utility:spmd-verify
```

This will give us most-importantly the wrapper script spmd-verify that allows applying data race detection and collectives verificaiton.
We also obtain the spmd-opt to run passes separately. 
Both are located in the `spmd/bazel-bin` directory (sym-link to the respective directory in spmd/bazel-out).

You can also replace the target with `//spmd-opt:spmd-opt` if you only need spmd-opt.

Note if you want to change build behavior such as enabling assertions for the release build look at the `.bazelrc` file (e.g. `build:dev_opt --copt=-DLLVM_ENABLE_ASSERTIONS`).

This matters more than it looks: in a plain `-c opt` build NDEBUG is defined, so
`assert` and `llvm_unreachable` compile away. The passes report genuinely
unsupported input via `llvm::reportFatalUsageError`, which is always enabled,
but if you are chasing an internal invariant use `--config=dev_opt`
(assertions on an optimised build) or `--config=dev_dbg`.

`--output_base=` moves Bazel's output tree off the home directory, whose disk space / file count quota is easily exceeded on HPC systems; drop it, or pick another directory, if that does not apply to you. `--spawn_strategy=local` makes builds faster, and is how CI builds, so it is the configuration that is tested.
If you want to build a debug build, use `dbg` instead of `opt`.

Use `--config=dev_opt` or directly `--copt=-UNDEBUG` etc., if you want certain debug functionality in spmd-opt in release builds.


Notes for shared resources (e.g., Polaris):
On login nodes that implement per user limits on processes, ninja and bazel would try to probe the “available” hardware processes and launch tasks accordingly without accounting for the per user limits, this could lead to unexpected hang. One may append option “-j #procs” to ninja or “–-local_resources=cpu=#procs” to bazel to avoid the hang. Similarly, if there is also per user RAM limit implemented, one may append option “--local_resources=memory=#MB” to limit the RAM bazel would consume.
bazel would write to ~/.cache/bazel by default, this is typically on home directory, which may be subject to strict disk quota and lead to compilation error. To avoid this issue, add option  “--output_base=<directory>” to bazel where <directory> is a location where the quota is less strict, such as a project directory.

Take care for disk quota exceedings, especially for filecount.

3. Developing on Enzyme-JAX and Enzyme
If you want to alter one of both projects, you can make use of `OVERRIDE_ENZYME_JAX_PATH` option in `spmd/workspace.bzl` and then of `OVERRIDE_ENZYME_PATH` in `Enzyme-JAX/workspace.bzl`.
By setting them, bazel will use the local repositories instead of cloning them from git. Make sure to checkout the correct commits.


4. Usage
All commands in this step and under *Running test suites* are run from the
root of the spmd checkout -- the `spmd/` directory that `git clone` created
and that you built in -- so `$PWD` in them expands to that directory.

Specify the path to the includesProgrammModels directory in the spmd repository by setting `PROG_MODEL_INCLUDES`, before using `bazel-bin/spmd-opt/spmd-opt`.

Usage informations can also be found in the spmd-verify file.

Flags may be given in any order; an unrecognised flag is an error rather than
being silently passed on as an input file. `spmd-verify` exits non-zero when a
verification pass reports findings (the passes call `signalPassFailure()`) or
when the toolchain fails -- set `SPMD_IR_EXITCODE=0` if you want findings
without a non-zero status.
For spmd-opt see:
`bazel-bin/spmd-opt/spmd-opt --help`

Example use, on one of the sample inputs under `test/spmd/` (a collective mismatch, checked for 2 ranks):
`bazel-bin/utility/spmd-verify check-collectives 2 emitSPMDIR test/spmd/mpi/024-mismatch_reduceScatter_nb-yes.c`

or

```sh
alias spmdVerify="$PWD/bazel-bin/utility/spmd-verify"

spmdVerify check-collectives 2 emitSPMDIR test/spmd/mpi/024-mismatch_reduceScatter_nb-yes.c
```

The API description JSON files themselves are **generated by [USIS](https://github.com/RWTH-HPC/USIS)** from each programming model's own standard or headers, and copied into `includesProgramModels/<model>/api.json` verbatim — do not hand-edit them. See [includesProgramModels/README.md](includesProgramModels/README.md) for how to refresh them, which fields the pass actually reads, and how to check that every mapped call still resolves.

To run the pass convert-apiCalls-to-spmd directly via spmd-opt (without using the wrapper spmd-verify), the paths to the API description JSON files must be provided manually through environment variables.

Each supported programming model requires its corresponding *_JSON_PATH variable to be set before invoking the pass.

Example:
```sh
export PROG_MODEL_INCLUDES=$PWD/includesProgramModels
export MPI_JSON_PATH="$PROG_MODEL_INCLUDES/mpi/api.json"
export NCCL_JSON_PATH="$PROG_MODEL_INCLUDES/nccl/api.json"
export SHMEM_JSON_PATH="$PROG_MODEL_INCLUDES/shmem/api.json"
export NVSHMEM_JSON_PATH="$PROG_MODEL_INCLUDES/mynvshmem/api.json"
export CUDA_JSON_PATH="$PROG_MODEL_INCLUDES/cuda/api.json"
export OMP_JSON_PATH="$PROG_MODEL_INCLUDES/omp/api.json"
```

```sh
alias spmdOpt="$PWD/bazel-bin/spmd-opt/spmd-opt"
spmdOpt -convert-apiCalls-to-spmd test/spmd/raise/isendrecv.mlir
```

5. General Hint
For reference: with Clang 20.1.7, building SPMD takes about 6 minutes on a
96-core node of RWTH's CLAIX-2023 cluster, once the dependencies are cached.

## Running test suites

### Unit tests

Hermetic `spmd-opt | FileCheck` tests under `test/spmd/`, one Bazel `lit_test`
per `.mlir` file (CI's `unit-test` stage):

```bash
bazel --output_base=/tmp/$USER/bazelOut test -c opt --spawn_strategy=local //test:spmd_tests
```

A new `.mlir` file with `RUN:` lines anywhere under `test/spmd/` is picked up
automatically.

### Benchmark suites

Make sure the submodules are checked out using `git submodule update --init --recursive`
(the suites are not public yet; see step 1 of *How to Build*).

From the root of the spmd checkout:

```bash
cmake -B spmdir-coll-tests -S externals/collective-verification-benchmark-suite -DSPMD_PATH=$PWD/bazel-bin/utility
cmake --build spmdir-coll-tests --target check-collective
```

For the data race suite, the targets are `check-race` and
`check-thread-safety` (both live in the data-race suite; CI runs all three).
Tests are automatically annotated with lit-annotations when building the check-target. To explicitly trigger annotation, the `annotate-tests` target can be used.

## Developer setup

Three one-time steps per clone. None is done for you by cloning, and the first
two are what keep CI from telling you things you could have known locally:

```bash
# 1. Run the formatters automatically on every commit (see "Code formatting")
git config core.hooksPath utility/git-hooks

# 2. Let `git blame` see through formatting-only commits
git config blame.ignoreRevsFile .git-blame-ignore-revs

# 3. Fetch buildifier once (only needed if you touch BUILD.bazel / *.bzl)
utility/run-buildifier.sh --bootstrap --check
```

Per shell, before formatting or building, have the tools from step 0 of *How
to Build* on `PATH`. On CLAIX:

```bash
ml GCCcore/14.2.0 LLVM/20.1.7   # ... plus the rest of the list under "How to Build"
```

`utility/README.md` describes each script in that directory.

## Code formatting

Two formatters are enforced, both by CI's `formatting` job:

| Files | Tool | Style |
|---|---|---|
| C/C++ in `include/`, `lib/`, `spmd-opt/` | `clang-format` 20.1.7 | plain upstream LLVM (`.clang-format`) |
| `BUILD.bazel`, `WORKSPACE.bazel`, `*.bzl` | `buildifier` 8.2.1 | buildifier's canonical layout |

Format before you push:

```bash
ml GCCcore/14.2.0 LLVM/20.1.7           # CLAIX; elsewhere, clang-format 20.x on PATH
utility/run-clang-format.sh             # reformat C/C++ in place
utility/run-buildifier.sh               # reformat Bazel files in place
utility/run-clang-format.sh --check     # what CI runs; changes nothing
utility/run-buildifier.sh --check
```

Both take optional pathspecs (`utility/run-clang-format.sh lib/spmd/passes`) if
you only want to touch part of the tree.

`buildifier` is rarely installed (and is not an environment module on CLAIX),
so the first run needs `--bootstrap` (step 3 of *Developer setup*), which
downloads the pinned 8.2.1 into `~/.cache/spmd-buildifier` and verifies its
sha256. (It is fetched from the `@bazel/buildifier` npm package rather than the
GitHub release: the binaries are the same, and the npm registry is reachable
from more of the networks in use here. `go install` is not an option — the
buildtools `go.mod` declares a module path without the `/vN` suffix its
`v6`/`v7`/`v8` tags need, and its master branch now requires Go 1.25 while
CLAIX tops out at 1.23.6.) `--lint` additionally prints buildifier's lint
suggestions; those are advisory and never affect the exit code.

### Catching it before you push

`git config core.hooksPath utility/git-hooks` (step 1 of *Developer setup*)
enables `utility/git-hooks/pre-commit`, which runs both checks over just the
files a commit touches. A missing tool warns instead of blocking — CI is
the real gate, and a hook that made `git commit` impossible without
clang-format on `PATH` would only get switched off. Bypass it with
`git commit --no-verify` or `SPMD_SKIP_FORMAT_HOOK=1`.

**The clang-format version matters.** clang-format's output changes between
major releases, so the script insists on 20.x -- the version that comes with
LLVM 20.1.7, the same LLVM whose `clang`/`clang++` the build uses (on CLAIX,
the `LLVM/20.1.7` module). A different version would produce a diff that CI
then rejects. (20.1.5 and 20.1.7 agree on every file in this tree, so patch-level drift is not a
concern; major-version drift is.)

This is deliberately *not* the LLVM the dialect links against: Bazel fetches
its own `llvm-project` through XLA's pin, currently `23.0.0git` — unreleased
main, which moves with every Enzyme-JAX/XLA bump. Tying the formatter to it
would let an unrelated dependency bump reformat the tree or fail CI, and would
be far more expensive, since Clang is not built in this workspace at all
(`@llvm-project//clang:clang-format` pulls in ~2000 extra Clang targets).
Enzyme-JAX draws the same line — its format workflow pins clang-format 16 while
building against LLVM main. Re-pin here as a deliberate commit, together with a
re-sweep, when CI moves to a newer LLVM worth moving to.

Exclusions live in `.clang-format-ignore` (vendored `includesProgramModels/`
headers, `test/` inputs, `patches/`, `externals/` submodules); the script
itself has no second list. `.td` files are not formatted at all -- clang-format
has no TableGen mode and would parse them as C++, which is why llvm-project
does not format its own either.

Commits that only reformat go into `.git-blame-ignore-revs`. Step 2 of
*Developer setup* tells `git blame` to look through them; GitHub's and GitLab's
blame views honour the file without any configuration.

## Authors

SPMD is developed as the PhD project of Semih Burak at RWTH Aachen
University. Students and student workers contributed to it, at least at
draft level.

## License

SPMD is licensed under the Apache License v2.0 with LLVM Exceptions; see
[LICENSE](LICENSE). Every file states its license and copyright, either in an
SPDX header or in [REUSE.toml](REUSE.toml), and the repository is compliant
with the [REUSE](https://reuse.software) specification (check with
`reuse lint`).

Third-party files keep their own licenses and are not covered by the Apache
License: the vendored programming-model headers under
`includesProgramModels/*/include/` (MPICH, Sandia OpenSHMEM, NVIDIA NCCL and
NVIDIA NVSHMEM) and the Enzyme-JAX patches under `patches/`. `LICENSE` lists
them, and their full license texts are in [LICENSES/](LICENSES/).
