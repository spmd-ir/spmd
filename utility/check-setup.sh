#!/bin/bash

# SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Setup check for the SPMD project's programming-model runtimes.
#
# None of these runtimes are needed to *verify* code: spmd-verify / spmd-opt
# compile against the shim headers shipped in includesProgramModels/. They are
# only needed for the round-trip path, where lowered SPMD IR is linked and run
# against a real implementation (spmd-verify compile, test/exec/).
#
# Usage: ./check-setup.sh [options]
#   --models=mpi,shmem,nccl,nvshmem  restrict the check (default: all)
#   --compile                        also compile+link a probe per model
#   --run                            also launch the probe on 2 ranks
#                                    (implies --compile; MPI/OpenSHMEM only)
#   --strict                         exit nonzero if a checked model is missing
#   --no-color                       plain output
#   -h | --help                      this text
#
# Exit code: 0 unless --strict (this is an informational check, and every
# runtime is optional). Note this is the conventional shell meaning, *not*
# spmd-opt's inverted "nonzero means success".

set -uo pipefail

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
: "${REPO_ROOT:=$(realpath -s "$SCRIPT_PATH/..")}"

MODELS="mpi shmem nccl nvshmem"
DO_COMPILE=0
DO_RUN=0
STRICT=0
USE_COLOR=1

for arg in "$@"; do
  case "$arg" in
    --models=*) MODELS="${arg#--models=}"; MODELS="${MODELS//,/ }" ;;
    --compile)  DO_COMPILE=1 ;;
    --run)      DO_RUN=1; DO_COMPILE=1 ;;
    --strict)   STRICT=1 ;;
    --no-color) USE_COLOR=0 ;;
    -h|--help)  sed -n '7,26p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# Validate the model list before printing anything.
for model in $MODELS; do
  case "$model" in
    mpi|shmem|openshmem|nccl|nvshmem) ;;
    *) echo "Unknown model: $model (expected mpi, shmem, nccl, nvshmem)" >&2; exit 2 ;;
  esac
done

if [[ $USE_COLOR -eq 1 && -t 1 ]]; then
  C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'
  C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
  C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_BOLD=""; C_OFF=""
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

# Per-model verdict, filled in by the check_* functions.
declare -A VERDICT   # ok | partial | missing
declare -A NOTE

ok()      { echo "  ${C_OK}[ok]${C_OFF}      $*"; }
warn()    { echo "  ${C_WARN}[warn]${C_OFF}    $*"; }
missing() { echo "  ${C_ERR}[missing]${C_OFF} $*"; }
info()    { echo "  ${C_DIM}          $*${C_OFF}"; }
section() { echo; echo "${C_BOLD}== $* ==${C_OFF}"; }

# The repo ships shim headers for every model; they satisfy #include but are
# not an installation, so never let a search path inside the repo count.
is_repo_path() {
  [[ "$(realpath -m "$1" 2>/dev/null)" == "$REPO_ROOT"/* ]]
}

# The same for the compilers this script runs: a CPATH entry inside the repo
# (as spmd-verify's own environment has) would let a probe or the MPICH ABI
# check read a shim header instead of the installation's.
if [[ -n "${CPATH:-}" ]]; then
  _cpath=""
  IFS=: read -r -a _entries <<< "$CPATH"
  for _entry in "${_entries[@]}"; do
    [[ -z "$_entry" ]] || is_repo_path "$_entry" || _cpath="${_cpath:+$_cpath:}$_entry"
  done
  export CPATH="$_cpath"
fi

# find_header <header> <dir>...  -> prints the include dir that contains it
find_header() {
  local header="$1"; shift
  local dir
  for dir in "$@"; do
    [[ -n "$dir" && -d "$dir" ]] || continue
    is_repo_path "$dir" && continue
    if [[ -f "$dir/$header" ]]; then
      echo "$dir"
      return 0
    fi
  done
  return 1
}

# find_lib <libname-without-lib-prefix> <dir>...  -> prints the .so/.a found
find_lib() {
  local lib="$1"; shift
  local dir f
  for dir in "$@"; do
    [[ -n "$dir" && -d "$dir" ]] || continue
    is_repo_path "$dir" && continue
    for f in "$dir/lib$lib.so" "$dir/lib$lib.so".* "$dir/lib$lib.a"; do
      [[ -e "$f" ]] && { echo "$f"; return 0; }
    done
  done
  # Fall back to the dynamic linker's own view.
  local from_ldconfig
  from_ldconfig=$(ldconfig -p 2>/dev/null | awk -v l="lib$lib.so" '$1 ~ "^"l {print $NF; exit}')
  [[ -n "$from_ldconfig" ]] && { echo "$from_ldconfig"; return 0; }
  return 1
}

# Common install prefixes, plus whatever the user pointed us at.
prefix_dirs() {
  local var prefix
  for var in "$@"; do
    prefix="${!var:-}"
    [[ -n "$prefix" ]] && printf '%s\n' "$prefix"
  done
  printf '%s\n' /usr /usr/local /opt
}

expand_prefixes() {
  local p
  while read -r p; do
    printf '%s\n%s\n%s\n' "$p/include" "$p/lib" "$p/lib64"
  done
}

# compile_probe <label> <source-file> <compiler> [flags...]
# On success sets PROBE_BIN to the built binary (status text goes to the
# terminal, so it cannot be returned through stdout).
PROBE_BIN=""
compile_probe() {
  local label="$1" src="$2" cc="$3"; shift 3
  local out="$WORKDIR/${label}.bin" log="$WORKDIR/${label}.log"
  PROBE_BIN=""
  if "$cc" "$src" -o "$out" "$@" >"$log" 2>&1; then
    ok "probe compiles and links ($cc)"
    PROBE_BIN="$out"
    return 0
  fi
  warn "probe failed to compile/link with $cc"
  info "$(tail -n 3 "$log" | sed 's/^/> /')"
  return 1
}

# run_probe <binary> <launcher> [launcher-args...]
run_probe() {
  local bin="$1"; shift
  local log="$WORKDIR/run.log"
  if "$@" "$bin" >"$log" 2>&1; then
    ok "2-rank run succeeded: $(tr '\n' ' ' <"$log" | cut -c1-80)"
    return 0
  fi
  warn "2-rank run failed (launcher present but the job did not complete)"
  info "$(tail -n 3 "$log" | sed 's/^/> /')"
  return 1
}

# ---------------------------------------------------------------- MPI --------

check_mpi() {
  section "MPI  (round-trip target of -lower-spmd-to-mpi)"
  local wrapper="" inc="" lib="" found=0

  wrapper=$(command -v "${MPICC:-mpicc}" 2>/dev/null) || wrapper=""
  if [[ -n "$wrapper" ]]; then
    ok "compiler wrapper: $wrapper"
    found=1
    local ver
    ver=$("$wrapper" -show 2>/dev/null | head -n1)
    [[ -n "$ver" ]] && info "wraps: $ver"
  else
    warn "no mpicc in PATH (set MPICC=, or 'module load' your MPI)"
  fi

  local -a dirs
  mapfile -t dirs < <(prefix_dirs MPI_HOME MPI_ROOT MPICH_DIR OPAL_PREFIX | expand_prefixes)
  # A module system puts the wrapper on PATH without setting any of those; the
  # installation it belongs to is the one to look in first.
  [[ -n "$wrapper" ]] && dirs=("$(dirname "$(dirname "$wrapper")")"/{include,lib,lib64} "${dirs[@]}")
  dirs+=(/usr/lib64/mpich/include /usr/lib64/mpich/lib
         /usr/lib64/openmpi/include /usr/lib64/openmpi/lib
         /usr/include/mpich-x86_64 /usr/lib64/mpi/gcc/openmpi/include)

  if inc=$(find_header mpi.h "${dirs[@]}"); then
    ok "mpi.h: $inc/mpi.h"
    found=1
  else
    missing "mpi.h not found outside the repo's shim headers"
  fi

  if lib=$(find_lib mpi "${dirs[@]}"); then
    ok "runtime library: $lib"
    found=1
  else
    missing "libmpi not found"
  fi

  local launcher
  launcher=$(command -v mpirun 2>/dev/null || command -v mpiexec 2>/dev/null) || launcher=""
  if [[ -n "$launcher" ]]; then
    ok "launcher: $launcher"
  else
    warn "no mpirun/mpiexec (the execution tests in test/exec/ launch 2 ranks)"
  fi

  if [[ $found -eq 0 ]]; then
    VERDICT[mpi]=missing
    NOTE[mpi]="no MPI installation detected"
    return
  fi

  # -lower-spmd-to-mpi writes MPICH's handle values into the calls, so an MPI
  # with another ABI (Open MPI) links and then fails at run time.  Ask the
  # installation's own mpi.h, as spmd-verify compile does before it links.
  if [[ -n "$wrapper" ]]; then
    local comm_world
    comm_world=$(printf '#include <mpi.h>\nMPI_COMM_WORLD\n' |
                 "$wrapper" -E -P -x c - 2>/dev/null | tail -n 1)
    if [[ "$comm_world" == *0x44000000* ]]; then
      ok "MPICH ABI (MPI_COMM_WORLD is $comm_world)"
    else
      warn "not the MPICH ABI the MPI lowering emits (MPI_COMM_WORLD is '${comm_world:-?}')"
      VERDICT[mpi]=partial
      NOTE[mpi]="found, but not MPICH-ABI compatible; load MPICH, Intel MPI or MVAPICH"
      return
    fi
  fi

  if [[ $DO_COMPILE -eq 1 ]]; then
    cat >"$WORKDIR/mpi.c" <<'EOF'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
  int rank = -1, size = -1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) printf("MPI ok: %d ranks\n", size);
  MPI_Finalize();
  return 0;
}
EOF
    PROBE_BIN=""
    if [[ -n "$wrapper" ]]; then
      compile_probe mpi "$WORKDIR/mpi.c" "$wrapper"
    elif [[ -n "$inc" && -n "$lib" ]]; then
      compile_probe mpi "$WORKDIR/mpi.c" "${CC:-cc}" -I"$inc" -L"$(dirname "$lib")" -lmpi
    fi
    local bin="$PROBE_BIN"
    if [[ -z "$bin" ]]; then
      VERDICT[mpi]=partial
      NOTE[mpi]="found, but the probe did not build"
      return
    fi
    if [[ $DO_RUN -eq 1 && -n "$launcher" ]]; then
      run_probe "$bin" "$launcher" -n 2 || {
        VERDICT[mpi]=partial; NOTE[mpi]="builds, but the 2-rank run failed"; return; }
    fi
  fi

  VERDICT[mpi]=ok
  NOTE[mpi]="usable for spmd-verify compile of MPI programs"
}

# ------------------------------------------------------------ OpenSHMEM ------

check_shmem() {
  section "OpenSHMEM  (round-trip target of -lower-spmd-to-shmem, expects 1.6)"
  local wrapper="" inc="" lib="" found=0

  wrapper=$(command -v "${OSHCC:-oshcc}" 2>/dev/null \
            || command -v shmemcc 2>/dev/null) || wrapper=""
  if [[ -n "$wrapper" ]]; then
    ok "compiler wrapper: $wrapper"
    found=1
  else
    warn "no oshcc/shmemcc in PATH"
  fi

  local -a dirs
  mapfile -t dirs < <(prefix_dirs SHMEM_HOME OSHMEM_HOME SOS_HOME OPAL_PREFIX | expand_prefixes)
  [[ -n "$wrapper" ]] && dirs=("$(dirname "$(dirname "$wrapper")")"/{include,lib,lib64} "${dirs[@]}")

  if inc=$(find_header shmem.h "${dirs[@]}"); then
    ok "shmem.h: $inc/shmem.h"
    found=1
  else
    missing "shmem.h not found outside the repo's shim headers"
  fi

  local l
  for l in sma oshmem shmem; do
    if lib=$(find_lib "$l" "${dirs[@]}"); then
      ok "runtime library: $lib"
      found=1
      break
    fi
    lib=""
  done
  [[ -z "$lib" ]] && missing "no libsma / liboshmem / libshmem found"

  local launcher
  launcher=$(command -v oshrun 2>/dev/null || command -v shmemrun 2>/dev/null) || launcher=""
  if [[ -n "$launcher" ]]; then
    ok "launcher: $launcher"
  else
    warn "no oshrun/shmemrun (some implementations launch via mpirun instead)"
  fi

  if [[ $found -eq 0 ]]; then
    VERDICT[shmem]=missing
    NOTE[shmem]="no OpenSHMEM installation detected"
    return
  fi

  if [[ $DO_COMPILE -eq 1 ]]; then
    cat >"$WORKDIR/shmem.c" <<'EOF'
#include <shmem.h>
#include <stdio.h>
int main(void) {
  shmem_init();
  if (shmem_my_pe() == 0) printf("SHMEM ok: %d PEs\n", shmem_n_pes());
  shmem_finalize();
  return 0;
}
EOF
    PROBE_BIN=""
    if [[ -n "$wrapper" ]]; then
      compile_probe shmem "$WORKDIR/shmem.c" "$wrapper"
    elif [[ -n "$inc" && -n "$lib" ]]; then
      compile_probe shmem "$WORKDIR/shmem.c" "${CC:-cc}" \
        -I"$inc" -L"$(dirname "$lib")" "-l$(basename "$lib" | sed 's/^lib//; s/\.\(so\|a\).*$//')"
    fi
    local bin="$PROBE_BIN"
    if [[ -z "$bin" ]]; then
      VERDICT[shmem]=partial
      NOTE[shmem]="found, but the probe did not build"
      return
    fi
    if [[ $DO_RUN -eq 1 && -n "$launcher" ]]; then
      run_probe "$bin" "$launcher" -n 2 || {
        VERDICT[shmem]=partial; NOTE[shmem]="builds, but the 2-PE run failed"; return; }
    fi
  fi

  VERDICT[shmem]=ok
  NOTE[shmem]="usable for OpenSHMEM round-trip compile/link"
}

# ---------------------------------------------------------------- CUDA -------

# Shared prerequisite for NCCL and NVSHMEM; reported once, reused by both.
CUDA_INC=""
CUDA_LIBDIR=""
check_cuda_prereq() {
  local nvcc
  nvcc=$(command -v nvcc 2>/dev/null) || nvcc=""
  if [[ -n "$nvcc" ]]; then
    ok "nvcc: $nvcc ($("$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/CUDA \1/p'))"
  else
    warn "no nvcc in PATH ('ml CUDA/12.4.0' on the cluster)"
  fi

  local -a dirs
  mapfile -t dirs < <(prefix_dirs CUDA_HOME CUDA_PATH CUDA_ROOT | expand_prefixes)
  dirs+=(/usr/local/cuda/include /usr/local/cuda/lib64)

  CUDA_INC=$(find_header cuda_runtime.h "${dirs[@]}") || CUDA_INC=""
  local rt
  rt=$(find_lib cudart "${dirs[@]}") || rt=""
  [[ -n "$rt" ]] && CUDA_LIBDIR=$(dirname "$rt")

  if [[ -n "$CUDA_INC" && -n "$rt" ]]; then
    ok "CUDA toolkit: $CUDA_INC, $rt"
    return 0
  fi
  missing "CUDA toolkit (cuda_runtime.h / libcudart) incomplete"
  return 1
}

# ---------------------------------------------------------------- NCCL -------

check_nccl() {
  section "NCCL  (round-trip target of -lower-spmd-to-nccl)"
  local have_cuda=0
  check_cuda_prereq && have_cuda=1

  local -a dirs
  mapfile -t dirs < <(prefix_dirs NCCL_HOME NCCL_ROOT NCCL_DIR CUDA_HOME | expand_prefixes)
  dirs+=(/usr/local/cuda/include /usr/local/cuda/lib64 /usr/include/nccl)

  local inc lib
  inc=$(find_header nccl.h "${dirs[@]}") || inc=""
  lib=$(find_lib nccl "${dirs[@]}") || lib=""

  if [[ -n "$inc" ]]; then
    local major minor patch
    major=$(sed -n 's/^#define NCCL_MAJOR \([0-9]*\).*/\1/p' "$inc/nccl.h")
    minor=$(sed -n 's/^#define NCCL_MINOR \([0-9]*\).*/\1/p' "$inc/nccl.h")
    patch=$(sed -n 's/^#define NCCL_PATCH \([0-9]*\).*/\1/p' "$inc/nccl.h")
    ok "nccl.h: $inc/nccl.h ${major:+(v$major.$minor.$patch)}"
  else
    missing "nccl.h not found outside the repo's shim headers"
  fi
  [[ -n "$lib" ]] && ok "runtime library: $lib" || missing "libnccl not found"

  if [[ -z "$inc" && -z "$lib" ]]; then
    VERDICT[nccl]=missing
    NOTE[nccl]="no NCCL installation detected"
    return
  fi
  if [[ $have_cuda -eq 0 ]]; then
    VERDICT[nccl]=partial
    NOTE[nccl]="NCCL present but the CUDA toolkit is incomplete"
    return
  fi

  if [[ $DO_COMPILE -eq 1 ]]; then
    if [[ -z "$inc" || -z "$lib" ]]; then
      VERDICT[nccl]=partial; NOTE[nccl]="header/library pair incomplete"; return
    fi
    cat >"$WORKDIR/nccl.c" <<'EOF'
#include <nccl.h>
#include <stdio.h>
int main(void) {
  int v = 0;
  if (ncclGetVersion(&v) != ncclSuccess) return 1;
  printf("NCCL ok: runtime version %d\n", v);
  return 0;
}
EOF
    compile_probe nccl "$WORKDIR/nccl.c" "${CC:-cc}" \
      -I"$inc" ${CUDA_INC:+-I"$CUDA_INC"} \
      -L"$(dirname "$lib")" ${CUDA_LIBDIR:+-L"$CUDA_LIBDIR"} -lnccl -lcudart
    local bin="$PROBE_BIN"
    if [[ -z "$bin" ]]; then
      VERDICT[nccl]=partial; NOTE[nccl]="found, but the probe did not build"; return
    fi
    # ncclGetVersion needs no device, so this is safe on a login node.
    if [[ $DO_RUN -eq 1 ]]; then
      run_probe "$bin" env || { VERDICT[nccl]=partial; NOTE[nccl]="builds, but the probe did not run"; return; }
    fi
  fi

  VERDICT[nccl]=ok
  NOTE[nccl]="usable for NCCL round-trip compile/link"
}

# ------------------------------------------------------------- NVSHMEM ------

check_nvshmem() {
  section "NVSHMEM  (device-side model raised by -convert-apiCalls-to-spmd)"
  local have_cuda=0
  check_cuda_prereq && have_cuda=1

  local -a dirs
  mapfile -t dirs < <(prefix_dirs NVSHMEM_HOME NVSHMEM_PREFIX NVSHMEM_DIR CUDA_HOME | expand_prefixes)
  dirs+=(/usr/local/nvshmem/include /usr/local/nvshmem/lib)

  local inc host dev
  inc=$(find_header nvshmem.h "${dirs[@]}") || inc=""
  host=$(find_lib nvshmem_host "${dirs[@]}") || host=""
  dev=$(find_lib nvshmem_device "${dirs[@]}") || dev=$(find_lib nvshmem "${dirs[@]}") || dev=""

  [[ -n "$inc" ]] && ok "nvshmem.h: $inc/nvshmem.h" \
                  || missing "nvshmem.h not found outside the repo's shim headers"
  [[ -n "$host" ]] && ok "host library: $host" || missing "libnvshmem_host not found"
  [[ -n "$dev" ]] && ok "device library: $dev" || missing "libnvshmem_device / libnvshmem not found"

  local launcher
  launcher=$(command -v nvshmrun 2>/dev/null || command -v nvshmrun.hydra 2>/dev/null) || launcher=""
  [[ -n "$launcher" ]] && ok "launcher: $launcher" \
                       || warn "no nvshmrun (NVSHMEM jobs are often launched via mpirun instead)"

  if [[ -z "$inc" && -z "$host" && -z "$dev" ]]; then
    VERDICT[nvshmem]=missing
    NOTE[nvshmem]="no NVSHMEM installation detected"
    return
  fi
  if [[ $have_cuda -eq 0 ]]; then
    VERDICT[nvshmem]=partial
    NOTE[nvshmem]="NVSHMEM present but the CUDA toolkit is incomplete"
    return
  fi

  if [[ $DO_COMPILE -eq 1 ]]; then
    if [[ -z "$inc" || -z "$host" ]]; then
      VERDICT[nvshmem]=partial; NOTE[nvshmem]="header/library pair incomplete"; return
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
      warn "skipping probe: NVSHMEM sources need nvcc"
      VERDICT[nvshmem]=partial; NOTE[nvshmem]="found, probe skipped (no nvcc)"; return
    fi
    cat >"$WORKDIR/nvshmem.cu" <<'EOF'
#include <nvshmem.h>
#include <stdio.h>
int main(void) {
  nvshmem_init();
  if (nvshmem_my_pe() == 0) printf("NVSHMEM ok: %d PEs\n", nvshmem_n_pes());
  nvshmem_finalize();
  return 0;
}
EOF
    compile_probe nvshmem "$WORKDIR/nvshmem.cu" nvcc -rdc=true -ccbin "${CXX:-c++}" \
      -I"$inc" -L"$(dirname "$host")" -lnvshmem_host -lnvshmem_device -lcudart
    local bin="$PROBE_BIN"
    if [[ -z "$bin" ]]; then
      VERDICT[nvshmem]=partial; NOTE[nvshmem]="found, but the probe did not build"; return
    fi
    # Deliberately not run: nvshmem_init() needs actual GPUs, not a login node.
    [[ $DO_RUN -eq 1 ]] && info "not running the NVSHMEM probe (needs GPUs); launch it yourself on a compute node"
  fi

  VERDICT[nvshmem]=ok
  NOTE[nvshmem]="usable for NVSHMEM compile/link"
}

# ---------------------------------------------------------------- main -------

echo "${C_BOLD}SPMD programming-model setup check${C_OFF}"
echo "${C_DIM}repo: $REPO_ROOT${C_OFF}"
echo
echo "Verification (spmd-verify / spmd-opt) uses the shim headers in"
echo "includesProgramModels/ and needs none of the runtimes below. They matter"
echo "only for the round-trip path, where lowered IR is linked and run."

for model in $MODELS; do
  case "$model" in
    mpi)     check_mpi ;;
    shmem|openshmem) check_shmem ;;
    nccl)    check_nccl ;;
    nvshmem) check_nvshmem ;;
  esac
done

section "Summary"
rc=0
for model in $MODELS; do
  [[ "$model" == openshmem ]] && model=shmem
  case "${VERDICT[$model]:-missing}" in
    ok)      printf "  %-9s ${C_OK}ok${C_OFF}       %s\n" "$model" "${NOTE[$model]}" ;;
    partial) printf "  %-9s ${C_WARN}partial${C_OFF}  %s\n" "$model" "${NOTE[$model]}"; rc=1 ;;
    *)       printf "  %-9s ${C_ERR}missing${C_OFF}  %s\n" "$model" "${NOTE[$model]:-not found}"; rc=1 ;;
  esac
done

echo
if [[ $DO_COMPILE -eq 0 ]]; then
  echo "${C_DIM}Detection only. Re-run with --compile (or --run) to actually build and"
  echo "launch a probe per model.${C_OFF}"
fi
if [[ $rc -ne 0 ]]; then
  echo "${C_DIM}Missing runtimes are fine unless you use the matching lowering:"
  for model in $MODELS; do
    [[ "$model" == openshmem ]] && model=shmem
    [[ "${VERDICT[$model]:-missing}" == ok ]] && continue
    case "$model" in
      mpi)     echo "  mpi     -> spmd-verify compile of MPI programs, test/exec/" ;;
      shmem)   echo "  shmem   -> spmd-verify compile of OpenSHMEM programs" ;;
      nccl)    echo "  nccl    -> spmd-verify compile of NCCL programs (needs CUDA)" ;;
      nvshmem) echo "  nvshmem -> NVSHMEM inputs and GPU runs (needs CUDA)" ;;
    esac
  done
  echo "${C_OFF}"
fi

[[ $STRICT -eq 1 ]] && exit $rc
exit 0
