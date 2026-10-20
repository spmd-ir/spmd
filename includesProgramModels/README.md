<!--
SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# API descriptions

One `api.json` per programming model, describing the calls
`convert-apiCalls-to-spmd` (CACTS) knows how to lower into the SPMD dialect,
plus the vendored headers each model needs to compile its own benchmarks.

## Do not hand-edit `api.json`

These files are **generated**, and copied here verbatim. They come from
[USIS](https://github.com/RWTH-HPC/USIS) — the Unified
Semantic Interface Specification — which extracts them from each model's own
primary source (the MPI Forum's LaTeX standard and `apis.json`, the OpenSHMEM
standard, the NCCL/NVSHMEM headers and documentation, the OpenMP ARB's
interface definition, CUDA's runtime headers). A fix belongs upstream, in the
stage that produced the wrong value; edited in place here it is silently
reverted by the next copy.

To refresh them, build USIS in a clone beside `spmd/` and copy its per-model
artifacts:

```sh
git clone https://github.com/RWTH-HPC/USIS.git
cd USIS
python3 workflow/fetch/fetch.py                 # once: the inputs USIS does not vendor
python3 workflow/build.py                       # needs Python >= 3.10
cd ../spmd
for p in mpi:mpi shmem:shmem nccl:nccl mynvshmem:nvshmem omp:openmp cuda:cuda; do
  cp ../USIS/instances/final-${p##*:}-api.json \
     includesProgramModels/${p%%:*}/api.json
done
```

The directory names are this project's, not USIS's, and the mapping above is
the whole of the difference — `mynvshmem` ← `nvshmem` and `omp` ← `openmp`.
Record the USIS commit in the commit message that updates them.

## `mynvshmem/include` is a stand-in

The MPI, OpenSHMEM and NCCL headers here are their implementations' own
(MPICH, Sandia OpenSHMEM, NCCL), so what the raiser reads is also what a
binary links against. The NVSHMEM headers are not NVIDIA's: they were written
for the verification. Every host routine is an ordinary function, which is how
the raiser sees `nvshmem_init` as a call, where NVIDIA's header defines it
inline over its device library; and some of their host routines are device
routines, or do not exist, in NVIDIA's library. `-nvshmem-to-host-abi` maps
the one to the other for `spmd-verify compile`.

The empty bodies they give NVSHMEM's device routines are for a GPU kernel, so
only a CUDA compile sees them (`NVSHMEMI_DEVICE_STUB`, on `__CUDACC__`). A host
compile that saw them would take each for the definition of the host routine
of the same name. Zero-argument routines are declared `(void)`, so that a C
program gets their prototype.

## Shape

```jsonc
{
  "schema_version": "v3.14",
  "entries": {
    "mpi:MPI_Send": {
      "identity":   { ... },     // name, api_group, description, version
      "bindings":   { ... },     // the C/Fortran declarations
      "execution":  { "blocking": true, ... },
      "semantics":  { ... },     // including the operational `formal` layer
      "parameters": [ { "name": "buf", "kind": "BUFFER", "direction": "in",
                        "length": null, "root_only": false, ... } ],
      "return":     { "kind": "ERROR_CODE", ... },
      "relationships": { ... },
      "tool_integration": { ... }
    }
  }
}
```

Entry keys are `"<usis-model>:<function>"` — CACTS indexes them by the
lowercased part after the colon, which is why the two directory-name
mismatches above cost nothing.

CACTS reads a small fraction of this. Everything else is payload it ignores,
and no analysis pass reads the JSON at all. The fields that are actually
consumed, all in `lib/spmd/passes/ConvertApiCallsToSPMD.cpp` and
`lib/spmd/passes/CACTS/mpiTransformations.cpp`:

| Read | What it decides |
|---|---|
| the entry itself | whether the call is converted at all |
| `parameters[].kind` | how each operand is converted (via `getSpmdParamKind`) |
| `parameters[].name` | `sendbuf`/`recvbuf` in-place detection (MPI) |
| `parameters[].root_only` | whether to check for `MPI_IN_PLACE` |
| `parameters[].length` | array-valued status/request parameters (MPI) |
| `parameters[].direction` | whether a communicator was passed by address (MPI) |
| `execution.blocking` | the `isBlocking` attribute on the emitted op |
| `return.value_kind` | whether a returned value is typed `i32`: `RANK`, `TEAM_SIZE` or `DEVICE` (`shmem_my_pe`, `omp_get_num_threads`, `omp_get_device_num`) |

`execution.blocking` is a **hard requirement**: 25 SPMD ops declare
`isBlocking` as a required attribute, so CACTS errors out by name if an entry
lacks it rather than emitting an op that fails verification later.

## Checking coverage

A name in a CACTS map with no entry here is *silently* skipped — the call is
left as an ordinary `llvm.call`, no diagnostic — which turns a true positive
into a false negative and lets a negative test pass vacuously. To check:

```sh
SPMD_CHECK_API_COVERAGE=1 bazel-bin/utility/spmd-verify check-dataRace anyfile.c
```

It prints a per-model resolved/total line and names every unresolved entry.
The expected misses are listed, with the reason for each, in
`ConvertApiCallsToSPMD.cpp`'s `kKnownUnresolved`.
