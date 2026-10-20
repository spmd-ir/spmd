<!--
SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# `utility/`

Developer-facing scripts. One of them is a build target; the rest are run by
hand (and by CI, and by the commit hook).

| Entry | What it is |
|---|---|
| `spmd-verify.sh` | The source-to-verification wrapper: compiles a source file to MLIR and runs a check pass over it, or (`compile`) lowers it back and links an executable. **Not run from here** — it is a `genrule` input whose `@…@` placeholders are substituted at build time, so run the built copy at `bazel-bin/utility/spmd-verify`. Its header comment is the reference for the flag list. |
| `run-clang-format.sh` | Formats, or checks, every tracked C/C++ file against `.clang-format`. |
| `run-buildifier.sh` | The same for `BUILD.bazel` / `WORKSPACE.bazel` / `*.bzl`, against buildifier. |
| `git-hooks/pre-commit` | Runs both checks over the files a commit touches. |
| `check-setup.sh` | Reports which programming-model runtimes (MPI, OpenSHMEM, NCCL, NVSHMEM) are installed, for `spmd-verify compile`. |
| `BUILD.bazel` | The `//utility:spmd-verify` genrule. |

## Formatting

Both formatters take the same shape — no arguments means the whole tree,
`--check` reports without editing and is byte for byte what CI's `formatting`
job runs, and any pathspecs narrow the scope:

```sh
ml GCCcore/14.2.0 LLVM/20.1.7    # provides clang-format 20.1.7

utility/run-clang-format.sh                       # format everything
utility/run-clang-format.sh lib/spmd/passes       # or just a subtree
utility/run-clang-format.sh --check               # what CI runs

utility/run-buildifier.sh --bootstrap --check     # first use: fetch the binary
utility/run-buildifier.sh                         # format the Bazel files
```

Both pin their tool's version, because clang-format and buildifier each change
their output between releases and an unpinned check would reject a tree
somebody had just formatted. `CLANG_FORMAT` / `BUILDIFIER` point at a different
binary; `CLANG_FORMAT_SKIP_VERSION_CHECK=1` / `BUILDIFIER_SKIP_VERSION_CHECK=1`
lift the guard when you mean to. Each script's header comment explains where
its tool comes from and why that source was chosen — worth reading once before
changing either pin.

Exit codes are shared: `0` clean, `1` violations (or the wrong tool version),
`2` bad usage, `127` tool not found. The commit hook keys on `127` to warn
rather than block.

## The commit hook

```sh
git config core.hooksPath utility/git-hooks
```

Once per clone. Then every `git commit` runs both checks over just that
commit's files. A missing tool warns instead of blocking — CI is the real gate,
and a hook that made `git commit` impossible without `ml LLVM/20.1.7` would
only get switched off. It reads the working tree rather than the index, and
says so when a staged file has unstaged changes. Bypass with
`git commit --no-verify` or `SPMD_SKIP_FORMAT_HOOK=1`.

## `spmd-verify compile`: back to an executable

The verification pipeline raises a program into the SPMD dialect; `compile`
takes the same SPMD IR back down to a binary:

```
C/C++ --reactant-clang--> MLIR --convert-apiCalls-to-spmd--> SPMD IR
      --[opt=... passes]--> --lower-spmd--> --convert-polygeist-to-llvm-->
      --mlir-translate--> LLVM IR --clang--> object --mpicc/oshcc--> executable
```

```sh
ml purge; ml GCC/14.2.0 MPICH/4.3.0 LLVM/20.1.7  # an MPICH-ABI MPI, and libomp; see below
bazel-bin/utility/spmd-verify compile -o ring ring.c
mpiexec -n 2 ./ring

# with an optimisation, keeping every stage's output
bazel-bin/utility/spmd-verify compile opt=coalesce-comm intermediates=stages -o ring ring.c
```

- `opt=<pass>` runs `spmd-opt -<pass>` on the SPMD IR before it is lowered,
  repeatable and in order (`opt=coalesce-comm=max-merged-bytes=0`).
- `models=<table>` is `-lower-spmd`'s table, e.g. `models=shmem=mpi`.
- `intermediates=<dir>` keeps `00-frontend.mlir` … `05-module.o`.
- Every stage's exit status is checked, and the first failure stops the build
  with the stage's name.

The linker is chosen by the calls in the lowered module: `$MPICC` (`mpicc`, or
`$MPICXX`/`mpicxx` for C++) for MPI, `$OSHCC` (`oshcc`) for OpenSHMEM, plus
`-lnccl -lcudart` for NCCL, `-lnvshmem_host -lcudart` for NVSHMEM (with
`-L$NVSHMEM_HOME/lib` and an rpath when `NVSHMEM_HOME` is set) and `-lomp` for
OpenMP. `SPMD_LINK` overrides the choice, `SPMD_LDFLAGS` adds flags.

**NVSHMEM is compiled against stand-in headers.**
`includesProgramModels/mynvshmem/include` is not NVIDIA's `nvshmem.h`: it
declares every host routine as an ordinary function, which is how the raiser
sees `nvshmem_init` as a call, and the lowering emits the same routines back.
NVIDIA's header defines `nvshmem_init`, `nvshmem_init_thread` and
`nvshmem_finalize` inline, over its device library, which only `nvcc -rdc`
links; and the stand-in declares some 270 host routines NVIDIA's host library
does not have. `compile` therefore runs `-nvshmem-to-host-abi` after the
lowering: it defines the three over `nvshmemx_hostlib_init_attr` and
`nvshmemx_hostlib_finalize`, and reports a call of any routine the host library
lacks, with the reason. A host-initiated NVSHMEM program then links with the
C compiler and `libnvshmem_host` alone.

**OpenMP links against LLVM's libomp.** `-lower-spmd` turns OpenMP's directives
back into the `omp` dialect, and `mlir-translate` outlines those into calls to
LLVM's OpenMP runtime (`__kmpc_fork_call`, ...). GCC's libgomp does not have
them, so `-fopenmp` on a GCC-based `mpicc` does not link. `compile` adds
`-L<dir> -lomp` and an rpath for the directory in `SPMD_OMP_LIBDIR`, or else
the one the `clang` on `PATH` reports for `libomp.so` — with the `LLVM` module
loaded, that is the module's. A hybrid MPI+OpenMP program links with `mpicc`
and that libomp.

**The MPI lowering emits the MPICH ABI** (handles are `int`s with MPICH's
values), so an MPI program has to be linked against MPICH or an ABI-compatible
MPI (Intel MPI, MVAPICH). Open MPI would link and then fail at run time, so
`compile` asks the wrapper's own `mpi.h` for `MPI_COMM_WORLD` first and refuses
anything but MPICH's. Counts that may not fit an `int` lower to the MPI 4
large-count calls, so the MPI has to be 4.0 or newer.

What does not compile: a program with SPMD operations of a model the table
keeps stops right after the lowering with those operations named. NVSHMEM
called from a GPU kernel is not lowered, and a translation to or from NVSHMEM
is refused: its buffers are GPU memory. CUDA is lowered: of its ~330 runtime routines the raiser raises
`cudaMemcpy` and `cudaMemset`, and those two go back to their calls, which is
what the NCCL benchmark programs need. Every other CUDA call in a program is
still the `llvm.call` the frontend produced and is compiled as such; CUDA
kernels are a separate matter, as `compile` does no GPU kernel outlining. Of
OpenMP, what the frontend raises compiles and runs (thread teams, the thread
number and count, worksharing loops and sections, single, critical, master,
masked, barriers); tasks crash `reactant-clang` itself, and a C `reduction`
clause is raised to a form that does not translate to LLVM IR, before any
SPMD pass runs. OpenSHMEM, NCCL and NVSHMEM
programs take the same path, but none has been run end to end: there is no
OpenSHMEM installation or GPU on the machines this was developed on.

## `check-setup.sh`

Reports which programming-model runtimes (MPI, OpenSHMEM, NCCL, NVSHMEM) are installed.

```sh
./check-setup.sh                       # detect headers, libraries, launchers
./check-setup.sh --compile             # also build a probe program per model
./check-setup.sh --run                 # also launch it on 2 ranks (MPI/OpenSHMEM)
./check-setup.sh --models=mpi,nccl     # restrict the check
./check-setup.sh --strict              # exit nonzero if a checked model is missing
```

**None of these runtimes are required for verification.** `spmd-verify` and
`spmd-opt` compile against the shim headers in `includesProgramModels/`, so a
machine with no MPI at all can still run `check-dataRace`/`check-collectives`.
The runtimes matter only for the round-trip path, where lowered IR is linked and
executed (`spmd-verify compile`, and the execution tests in `test/exec/`). The
script therefore exits 0 by default and treats every model as optional; the
summary lists what each missing one would have unlocked. For MPI it also checks
that the installation has the MPICH ABI the MPI lowering emits (MPICH, Intel
MPI, MVAPICH; not Open MPI).

The search deliberately ignores anything inside the repository, and the probes
and the ABI check run with repository entries dropped from `CPATH`, so the
shipped shim headers are never mistaken for an installation — a `CPATH` that
includes `includesProgramModels/mpi/include` will not turn into a false "ok".

Point it at non-standard prefixes with the usual variables: `MPI_HOME`,
`SHMEM_HOME`/`OSHMEM_HOME`/`SOS_HOME`, `NCCL_HOME`, `NVSHMEM_HOME`, `CUDA_HOME`
(or `MPICC`/`OSHCC` for the wrappers directly). A wrapper found on `PATH`, as a
module puts it there, is searched first.
