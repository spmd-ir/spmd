#!/bin/bash

# SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pipefail: the real work happens in "clang | spmd-opt" pipelines, whose exit
# status would otherwise be clang's output being discarded in favour of the
# last stage only.
set -o pipefail

# This script is used to run the SPMD verification passes on the input files.
# It takes the following arguments:
# 1. verPass: The verification pass to run (e.g., check-dataRace, check-collectives).
# 2. numRanks: Optional number of ranks for the check-collectives pass.
# 3. emitMLIR: Optional flag to emit mlir directly after the transformation.
# 4. emitPRESPMDIR: Optional flag to emit mlir before -convert-apiCalls-to-spmd passes.
# 5. emitSPMDIR: Optional flag to emit the program in SPMD IR.
# 6. emitPREMV: Optional flag to emit the program in SPMD IR without the MV-related Passes.
# 7. emitJson: Optional flag to emit JSON output.
# 8. printDebugInfo: Optional flag to print debug information.
# 9. time: Optional flag to enable timing.
# 10. input files: The input files to process.
# 11. noDebugInfo: Optional flag to disable debug information.
# 12. acceptMLIR: Optional flag to skip reactant-clang and feed an existing .mlir file directly to spmd-opt.
# 13. useOpt: Optional flag to use reactant-opt on a .bc file.
# 14. useOptChain: Optional flag to use reactant-opt on a .bc file got through reactant-clang.
# 15. noOMP: Optional flag to left out the llvm-to-omp pass and the accompanying ones
# 16. inline: Optional flag to enable the inliner in the raising pipeline.
#
# Flags may be given in any order, and an unrecognised flag is a hard error
# rather than being silently treated as an input file.
# Usage: ./spmd-verify.sh <verPass> [numRanks] [flags...] <input files>
# verPass: check-dataRace | check-collectives | check-thread-safety
# Example: ./spmd-verify.sh check-dataRace emitSPMDIR file1.cpp file2.cu
# Example: ./spmd-verify.sh check-collectives 4 file1.cpp
# Example: ./spmd-verify.sh check-thread-safety emitJson file1.cpp
# Example: ./spmd-verify.sh check-thread-safety acceptMLIR file1.mlir
#
# In place of a verification pass, "compile" builds an executable from the
# program's SPMD IR instead of checking it: the same raising, then any
# optimisation passes, -lower-spmd, -nvshmem-to-host-abi,
# -convert-polygeist-to-llvm, mlir-translate, clang and the programming
# model's linker.
# Usage: ./spmd-verify.sh compile [flags...] [-o executable] <input files>
# The input is either source files with their compiler flags, or a single
# LLVM bitcode file (.bc) of the whole program, which reactant-opt raises as
# useOpt does. A program of several translation units needs the latter: link
# it with clang -flto -fuse-ld=lld -Wl,--lto-O1 -Wl,--lto-emit-llvm.
# Keep --lto-O1: at O2 LTO vectorises, and the raised module no longer
# satisfies the alignment its vector loads and stores claim, so the
# executable crashes.
# Flags besides the common ones above:
#   opt=<pass>          run spmd-opt's -<pass> on the SPMD IR before it is
#                       lowered; repeatable, in order, e.g.
#                       opt=coalesce-comm=max-merged-bytes=0
#   models=<table>      -lower-spmd's models table, e.g. models=shmem=mpi;
#                       default: every model to itself
#   intermediates=<dir> keep each stage's output in <dir> (00-frontend.mlir,
#                       01-spmd.mlir, 02-lowered.mlir, 03-llvm.mlir,
#                       04-module.ll, 05-module.o) instead of a temporary
#                       directory
#   -o <executable>     default: the first input's name without its suffix,
#                       in the current directory
# The link step picks its linker from the calls the lowered module makes:
# $MPICC (default mpicc; $MPICXX/mpicxx for C++) for MPI, $OSHCC (default
# oshcc) for OpenSHMEM, adding -lnccl for NCCL, -lnvshmem_host for NVSHMEM
# (from $NVSHMEM_HOME/lib when that is set) and -lcudart for NCCL, NVSHMEM or
# the CUDA runtime calls the lowering emits. NVSHMEM programs are compiled
# against stand-in headers; -nvshmem-to-host-abi maps them to NVIDIA's host
# library, so no device library and no nvcc are needed for host-initiated
# NVSHMEM, and a routine the host library lacks is an error before the link.
# $SPMD_LINK replaces
# that choice and $SPMD_LDFLAGS is appended. The MPI lowering emits MPICH ABI
# handles, so MPI programs must link against MPICH or an ABI-compatible MPI
# (Intel MPI, MVAPICH); an MPI whose mpi.h does not agree is refused.
# OpenMP directives are lowered back to the omp dialect, which mlir-translate
# turns into calls to LLVM's OpenMP runtime (__kmpc_*), so an OpenMP program
# links against libomp -- not GCC's libgomp, which -fopenmp on a GCC-based
# mpicc would pick. The library is found in $SPMD_OMP_LIBDIR, or else where
# the clang on PATH has it (clang -print-file-name=libomp.so), e.g. with the
# LLVM module loaded.
# Every stage's exit status is checked; the first failure stops the build.
# Example: ./spmd-verify.sh compile -o ring ring.c && mpiexec -n 2 ./ring

verPass=$1

# Defaults
emitSPMDIR=false
emitPRESPMDIR=false
emitPREMV=false
emitJson=false
emitMLIR=false
acceptMLIR=false
useOpt=false
useOptChain=false
noOMP=false
optLevel=O1
numRanks=0
inline=false
compileOptPasses=()
lowerModels=""
intermediatesDir=""
executable=""

echo "start spmd-verify"

shift

# Parse numRanks if applicable
if [[ "$verPass" == "check-collectives" && "${1:-}" =~ ^[0-9]+$ ]]; then
  numRanks=$1
  shift
fi

# Optional flags, in any order. Anything that is not a known flag ends the
# flag section and is treated as an input file.
DEBUGFLAG=-gmlt # due to some bug/limitation in upstream mlir use gmlt instead of g
while [[ $# -gt 0 ]]; do
  case "$1" in
    emitMLIR)        emitMLIR=true ;;
    emitPRESPMDIR)   emitPRESPMDIR=true ;;
    emitSPMDIR)      emitSPMDIR=true ;;
    emitPREMV)       emitPREMV=true ;;
    emitJson)        emitJson=true ;;
    printDebugInfo)  PRINTDEBUGINFO=-mlir-print-debuginfo ;;
    time)            time=-mlir-timing ;;
    noDebugInfo)     DEBUGFLAG= ;;
    acceptMLIR)      acceptMLIR=true ;;
    useOpt)          useOpt=true ;;
    useOptChain)     useOptChain=true ;;
    noOMP)           noOMP=true ;;
    inline)          inline=true ;;
    -o)
      # The verification passes hand a leading -o to clang with the rest, as
      # they always have.
      [[ "$verPass" != compile ]] && break
      if [[ $# -lt 2 ]]; then
        echo "Error: -o needs an executable name" >&2
        exit 2
      fi
      executable="$2"
      shift
      ;;
    opt=*|models=*|intermediates=*)
      if [[ "$verPass" != compile ]]; then
        echo "Error: '$1' is a flag of 'compile' only" >&2
        exit 2
      fi
      case "$1" in
        opt=*)           compileOptPasses+=("-${1#opt=}") ;;
        models=*)        lowerModels="${1#models=}" ;;
        intermediates=*) intermediatesDir="${1#intermediates=}" ;;
      esac
      ;;
    -*)              break ;;   # compiler flag: hand the rest to clang
    *)
      # First non-flag token. If it names an existing file it is the start of
      # the input list; otherwise it is a mistyped flag, which used to be
      # passed to the compiler as a filename and fail confusingly there.
      if [[ -e "$1" ]]; then
        break
      fi
      echo "Error: unknown flag or missing input file: '$1'" >&2
      echo "Flags: emitMLIR emitPRESPMDIR emitSPMDIR emitPREMV emitJson" >&2
      echo "       printDebugInfo time noDebugInfo acceptMLIR useOpt" >&2
      echo "       useOptChain noOMP inline" >&2
      echo "compile: opt=<pass> models=<table> intermediates=<dir> -o <file>" >&2
      exit 2
      ;;
  esac
  shift
done

# Add emitJson as pass option
emitJsonFlag=""
if [ "$emitJson" == true ]; then
  emitJsonFlag="write-json-report"
fi

# Compose verPass string
if [[ "$verPass" == "check-dataRace" ]]; then
  verPass="-check-dataRace"
  if [ "$emitJson" == true ]; then
    verPass="-check-dataRace=$emitJsonFlag"
  fi
elif [[ "$verPass" == "check-collectives" ]]; then
  verPass="-check-collectives"
  options="num-ranks=$numRanks"
  if [ "$emitJson" == true ]; then
    options+=" $emitJsonFlag"
  fi
  verPass+="=$options"
  optLevel=O0
elif [[ "$verPass" == "check-thread-safety" ]]; then
  verPass="-check-thread-safety"
  if [ "$emitJson" == true ]; then
    verPass="-check-thread-safety=$emitJsonFlag"
  fi
elif [[ "$verPass" == "compile" ]]; then
  if [ "$emitSPMDIR" == true ] || [ "$emitPRESPMDIR" == true ] ||
     [ "$emitPREMV" == true ] || [ "$emitMLIR" == true ] ||
     [ "$emitJson" == true ] || [ "$acceptMLIR" == true ] ||
     [ "$useOpt" == true ] || [ "$useOptChain" == true ]; then
    echo "Error: compile keeps its stages with intermediates=<dir>; the emit*, acceptMLIR and useOpt* flags do not apply" >&2
    exit 2
  fi
else
  echo "Invalid input for verPass. Please set it to one of: 'check-dataRace', 'check-collectives', 'check-thread-safety', 'compile'."
  exit 1
fi

# Collect input files
if [[ "$verPass" == "compile" ]] && [[ " $* " == *" -o "* ]]; then
  echo "Error: compile takes -o before the input files and compiler flags" >&2
  exit 2
fi
inputFiles=()
for arg in "$@"; do
  if [[ "$arg" != -* ]]; then
    inputFile=$(realpath "$arg")
    inputFiles+=("$inputFile")
    if [[ -z "$inputDir" ]]; then
      inputDir="$(dirname "$arg")"
      file=$(basename "$inputFile")
      filename="${file%.*}"         # strip extension like .cpp/.cu
    fi
  fi
done

if [[ -z "$inputDir" ]]; then
  echo "Error: No valid input file provided." >&2
  exit 1
fi

mlirOutput="${inputDir}/${filename}.mlir"
tmpFile=$(mktemp --suffix=.mlir)
trap 'rm -f "$tmpFile"' EXIT
outputFile="$tmpFile"

if [ "$emitSPMDIR" == true ] || [ "$emitPRESPMDIR" == true ] || [ "$emitPREMV" == true ] || [ "$emitMLIR" == true ]; then
  outputFile="$mlirOutput"
fi

if [ "$emitMLIR" == true ]; then
  outputFile_realPath=$(realpath "$outputFile")
else
  outputFile_realPath=""
fi

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
: "${SPMD_WORKSPACE_DIR:=$(realpath -s "$SCRIPT_PATH/../..")}"
PROG_MODEL_INCLUDES="$SPMD_WORKSPACE_DIR/includesProgramModels"
# Prepended, not appended: these headers are what convert-apiCalls-to-spmd's
# api.json descriptions are written against, so they have to win over whatever
# a site MPI/SHMEM module happens to have put on CPATH. 
# compile's link step has to see the installation's headers, not these.
if [[ -n "${CPATH+set}" ]]; then
  callerCPATH=$CPATH
fi
for progModel in mpi nccl shmem mynvshmem omp; do
  CPATH=${PROG_MODEL_INCLUDES}/${progModel}/include${CPATH:+:$CPATH}
done

spmdOpt=${SPMD_WORKSPACE_DIR}/@SPMD_OPT_PATH@

# Each of these is only defaulted, so a caller can point one at its own API
# description. (These used to be assigned with ":=" and then unconditionally
# re-exported, which silently discarded any caller-supplied value.)
: "${MPI_JSON_PATH:=$PROG_MODEL_INCLUDES/mpi/api.json}"
: "${NCCL_JSON_PATH:=$PROG_MODEL_INCLUDES/nccl/api.json}"
: "${SHMEM_JSON_PATH:=$PROG_MODEL_INCLUDES/shmem/api.json}"
: "${NVSHMEM_JSON_PATH:=$PROG_MODEL_INCLUDES/mynvshmem/api.json}"
: "${CUDA_JSON_PATH:=$PROG_MODEL_INCLUDES/cuda/api.json}"
: "${OMP_JSON_PATH:=$PROG_MODEL_INCLUDES/omp/api.json}"
export MPI_JSON_PATH NCCL_JSON_PATH SHMEM_JSON_PATH NVSHMEM_JSON_PATH
export CUDA_JSON_PATH OMP_JSON_PATH

REACTANT_CLANG_BIN=${SPMD_WORKSPACE_DIR}/@REACTANT_CLANG_BIN@
MLIR_TRANSLATE_BIN=${SPMD_WORKSPACE_DIR}/@MLIR_TRANSLATE_PATH@
REACTANT_OPT_BIN=${SPMD_WORKSPACE_DIR}/@REACTANT_OPT_BIN@
clang=$REACTANT_CLANG_BIN
opt=$REACTANT_OPT_BIN

CLANG_OFFLOAD_BUNDLER_PATH=$(realpath -s "$SPMD_WORKSPACE_DIR/@CLANG_OFFLOAD_BUNDLER_PATH@")
LLVM_OFFLOAD_BINARY_PATH=$(realpath -s "$SPMD_WORKSPACE_DIR/@LLVM_OFFLOAD_BINARY_PATH@")
CLANG_OFFLOAD_BUNDLER_DIR=$(dirname "$CLANG_OFFLOAD_BUNDLER_PATH")
LLVM_OFFLOAD_BINARY_DIR=$(dirname "$LLVM_OFFLOAD_BINARY_PATH")
PATH=$CLANG_OFFLOAD_BUNDLER_DIR:$LLVM_OFFLOAD_BINARY_DIR:$PATH


REACTANT_CLANG_RESOURCE="@REACTANT_CLANG_RESOURCE@"
REACTANT_CLANG_RESOURCE_FIRST=$(echo $REACTANT_CLANG_RESOURCE | cut -d' ' -f1)
REACTANT_CLANG_RESOURCE_DIR=$(dirname "$(dirname "$(realpath -s "$SPMD_WORKSPACE_DIR/$REACTANT_CLANG_RESOURCE_FIRST")")")

CLANG_LINKER_WRAPPER_PATH=$(realpath -s "$SPMD_WORKSPACE_DIR/@CLANG_LINKER_WRAPPER_PATH@")
CLANG_LINKER_WRAPPER_DIR=$(dirname "$CLANG_LINKER_WRAPPER_PATH")
PATH=$CLANG_LINKER_WRAPPER_DIR:$PATH

LLD_PATH=$(realpath -s "$SPMD_WORKSPACE_DIR/@LLD_PATH@")

# Source type flags
if echo "$inputFile" | grep -qE "\.cu$"; then
  S="-c --cuda-gpu-arch=sm_89" # -c -fgpu-rdc--cuda-device-only 
else
  S="-c" #-S
fi

clangFlags="-Xclang -add-plugin -Xclang enzyme -resource-dir $REACTANT_CLANG_RESOURCE_DIR -o /dev/null -fopenmp "
clangFlags+="-$optLevel $S $DEBUGFLAG -fno-exceptions -lcudart -lstdc++ -fuse-ld=$CLANG_LINKER_WRAPPER_PATH -Wl,--linker-path=$LLD_PATH" # 
export EXPORT_REACTANT=True
export EXPORT_MLIR_ONLY=True
# export DEBUG_REACTANT=True 

# passPipeline="inline{default-pipeline=canonicalize max-iterations=4},"
if [ "$inline" == true ] ; then
  passPipeline+="inline{default-pipeline=canonicalize max-iterations=4},"
fi
passPipeline+="sroa-wrappers{set_private=false attributor=false}," 
# deliberately left out "lift-tessera-annotations,func-attr-to-tessera-attr,parse-optimization-rules,"
# myPasses, make all symbols except the main private, could be done also by sroa-wrappers setting set_private to true, but it does not allow for excepting main

passPipeline+="symbol-privatize{exclude=main},symbol-dce," # symbol-dce, added , nbot given originally

passPipeline+="gpu-launch-recognition{backend=cuda},canonicalize,"
passPipeline+="libdevice-funcs-raise,canonicalize,symbol-dce," #inline   parallel-lower{wrapParallelOps=true},
if [ "$inline" == true ] ; then
  passPipeline+="inline{max-iterations=4},"
fi
passPipeline+="llvm-to-memref-access,polygeist-mem2reg,canonicalize,convert-llvm-to-cf,canonicalize,polygeist-mem2reg,"
#myPasses
#passPipeline+="handle-llvm-unreachable," #prepare-llvm-func-inline,inline,inline-gpu-launch-funcs,delete-inlined-funcs,

passPipeline+="canonicalize,enzyme-lift-cf-to-scf,canonicalize,"
passPipeline+="func.func(canonicalize-loops),llvm.func(canonicalize-loops),"
passPipeline+="canonicalize-scf-for,canonicalize,"
# # passPipeline+="affine-cfg,canonicalize,"
passPipeline+="func.func(canonicalize-loops),llvm.func(canonicalize-loops),canonicalize,"
passPipeline+="llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,"
# passPipeline+="affine-cfg,canonicalize,"
passPipeline+="llvm-to-affine-access,canonicalize,"
passPipeline+="func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,"
passPipeline+="symbol-dce,lower-affine,parallel-serialization,canonicalize," # normally ,enzyme,remove-unnecessary-enzyme-ops, between both 

if [ "$noOMP" == false ]; then
  passPipeline+="llvm-to-omp,canonicalize,symbol-dce,canonicalize,"

  # repeat passes after llvm-to-omp, some of them could be left out
  if [ "$inline" == true ] ; then
    passPipeline+="inline{default-pipeline=canonicalize max-iterations=4},"
  fi
  # passPipeline+="sroa-wrappers{set_private=false attributor=false},"
  passPipeline+="symbol-dce,canonicalize," 
  passPipeline+="llvm-to-memref-access,polygeist-mem2reg,canonicalize,polygeist-mem2reg,"
  passPipeline+="func.func(canonicalize-loops),llvm.func(canonicalize-loops),canonicalize-scf-for,canonicalize,"
# passPipeline+="affine-cfg,canonicalize,"
  passPipeline+="func.func(canonicalize-loops),llvm.func(canonicalize-loops),canonicalize,"
  passPipeline+="llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,"
  # passPipeline+="affine-cfg,canonicalize,"
  passPipeline+="llvm-to-affine-access,canonicalize,"
  passPipeline+="func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,"
  passPipeline+="symbol-dce,lower-affine,parallel-serialization,canonicalize,"
fi
#teressa and enzyme left out, backend set to cuda instead of variable

passPipeline+="print{stdout=true debug=true filename=$outputFile_realPath}"

# for generating a binary, but not for verification
# passPipeline+="convert-parallel-to-gpu1{arch=sm89},"
# passPipeline+="gpu-kernel-outlining"
# passPipeline+="canonicalize,convert-parallel-to-gpu2{backend=cuda},lower-affine,parallel-serialization," #at beginning after canonic: 
# passPipeline+="canonicalize,convert-polygeist-to-llvm{backend=cuda},strip-gpu-info,gpu-module-to-binary"

# An executable has to behave like the program, and canonicalize-loops on
# llvm.func does not preserve that: in TeaLeaf's decompose_field it folds the
# check after the search loop, if (!x_chunks || !y_chunks) die(...), to true,
# so every run dies, and miniWeather corrupts its heap on two ranks. Both run
# correctly without it. The verification passes keep it for now.
# llvm-to-omp, however, needs it: without it, it cannot pair __kmpc_single
# with __kmpc_end_single (or master, masked) in an outlined function. A program
# that uses OpenMP therefore keeps the pass, and with it the miscompile, until
# canonicalize-loops is fixed. Whether it does is read off the input: a
# directive or runtime call in a source, a runtime symbol in bitcode.
usesOpenMP() {
  local f
  for f in "${inputFiles[@]}"; do
    if [[ "$f" == *.bc ]]; then
      grep -aqE '__kmpc_|omp_(get|set|in)_' "$f" && return 0
    else
      grep -qE '^[[:space:]]*#[[:space:]]*pragma[[:space:]]+omp|omp_(get|set|in)_' "$f" && return 0
    fi
  done
  return 1
}
if [[ "$verPass" == "compile" ]] && ! usesOpenMP; then
  passPipeline=${passPipeline//llvm.func(canonicalize-loops),/}
fi

export OVERRIDE_PASS_PIPELINE=$passPipeline

# ── Build pass sequences ────────────────────────────────────────────────────
commonPasses=""
spmdConvPasses="-omp-to-spmd -canonicalize -convert-apiCalls-to-spmd" 
# not given in llvm/main: -promote-scf-while

intermediatesPasses="-canonicalize -spmd-reconcile-unrealized-cast -canonicalize -reconcile-unrealized-casts -canonicalize -llvm-to-affine-access -canonicalize -cse -canonicalize -polygeist-mem2reg -canonicalize -simplify-cmp-ops -canonicalize -spmd-reconcile-unrealized-cast"

if [ "$emitSPMDIR" == true ]; then
  spmdPasses="$commonPasses $spmdConvPasses $intermediatesPasses $intermediatesPasses" #  -debug-only=convert-apiCalls-to-spmd
elif [ "$emitPRESPMDIR" == true ]; then
  spmdPasses="$commonPasses"
else
  spmdPasses="$commonPasses $spmdConvPasses $intermediatesPasses $intermediatesPasses $verPass" #-debug-only=${verPass#-}
fi

# ── Compile ─────────────────────────────────────────────────────────────────
if [[ "$verPass" == "compile" ]]; then
  # A compile has no findings: a failing spmd-opt is a failed build, whatever
  # the caller wants verification findings to exit with.
  unset SPMD_IR_EXITCODE

  if [[ -z "$executable" ]]; then
    executable="$filename"
  fi
  if [[ -n "$intermediatesDir" ]]; then
    mkdir -p "$intermediatesDir" || exit 1
    stageDir=$(realpath "$intermediatesDir")
  else
    stageDir=$(mktemp -d)
    trap 'rm -f "$tmpFile"; rm -rf "$stageDir"' EXIT
  fi

  # stage <name> <command...>: runs one step, and stops the build with the
  # step's own status and its name if it fails.
  stage() {
    local name=$1
    shift
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "Error: compile: $name failed (exit $rc)" >&2
      [[ -n "$intermediatesDir" ]] && echo "Stages so far are in $stageDir" >&2
      exit $rc
    fi
  }

  isBitcode=false
  [[ "$inputFile" == *.bc ]] && isBitcode=true
  if [ "$isBitcode" == true ] && [[ $# -ne 1 ]]; then
    echo "Error: compile: a bitcode input is the whole program; pass it alone, without other inputs or compiler flags" >&2
    exit 2
  fi
  frontend() {
    if [ "$isBitcode" == true ]; then
      $opt "$inputFile" > "$stageDir/00-frontend.mlir"
    else
      PATH="$PATH" CPATH="$CPATH" $clang $clangFlags "$@" > "$stageDir/00-frontend.mlir"
    fi
  }
  if [ "$isBitcode" == true ]; then
    stage "reactant-opt" frontend
  else
    stage "reactant-clang" frontend "$@"
  fi
  # Only -cse of the cleanup the verification passes get
  # ($intermediatesPasses): the optimisations compare handles such as
  # datatypes by value, so they need one spmd.constDatatype per datatype.
  # MPI writes the null handle back into a request, communicator or window it
  # completes or frees, and the lowering passes MPI the program's variable
  # when the raiser loaded the handle from it; -polygeist-mem2reg would
  # forward the stale handle past the call instead, since the SPMD operation
  # takes the handle and not its address.
  stage "raising to SPMD IR" $spmdOpt $commonPasses $spmdConvPasses -cse \
    $PRINTDEBUGINFO $time \
    -o "$stageDir/01-spmd.mlir" "$stageDir/00-frontend.mlir"

  lowerPass="-lower-spmd"
  [[ -n "$lowerModels" ]] && lowerPass="-lower-spmd=models=$lowerModels"
  # -nvshmem-to-host-abi leaves a module without NVSHMEM as it is.
  stage "lowering" $spmdOpt "${compileOptPasses[@]}" "$lowerPass" \
    -nvshmem-to-host-abi $time \
    -o "$stageDir/02-lowered.mlir" "$stageDir/01-spmd.mlir"
  # -lower-spmd keeps the operations of a model the table says to keep.
  # Nothing after it lowers them, and -convert-polygeist-to-llvm fails on them
  # with a verifier error or a crash that does not say so.
  kept=$(grep -oE '(^|[^!@A-Za-z0-9_.])spmd\.[A-Za-z_]+' "$stageDir/02-lowered.mlir" |
    sed -E 's/.*(spmd\.[A-Za-z_]+)/\1/' | sort -u | tr '\n' ' ')
  if [[ -n "$kept" ]]; then
    echo "Error: compile: the lowered module still has SPMD operations, of a model kept by models=: $kept" >&2
    [[ -n "$intermediatesDir" ]] && echo "Stages so far are in $stageDir" >&2
    exit 1
  fi
  stage "conversion to the LLVM dialect" $spmdOpt -convert-polygeist-to-llvm \
    $time -o "$stageDir/03-llvm.mlir" "$stageDir/02-lowered.mlir"
  stage "mlir-translate" $MLIR_TRANSLATE_BIN --mlir-to-llvmir \
    -o "$stageDir/04-module.ll" "$stageDir/03-llvm.mlir"
  # The frontend's plugin environment would make clang export MLIR again.
  stage "clang" env -u EXPORT_REACTANT -u EXPORT_MLIR_ONLY \
    -u OVERRIDE_PASS_PIPELINE $clang -c -$optLevel \
    -o "$stageDir/05-module.o" "$stageDir/04-module.ll"

  # The models the program calls, read off the module's declarations.
  declares() { grep -qE "^declare .*@$1" "$stageDir/04-module.ll"; }
  isCxx=false
  [[ "$inputFile" =~ \.(cpp|cc|cxx|C)$ ]] && isCxx=true
  # Bitcode does not say its language; a C++ program calls mangled functions.
  [ "$isBitcode" == true ] && declares '_Z' && isCxx=true
  linkFlags=()
  if [[ -n "${SPMD_LINK:-}" ]]; then
    read -r -a linker <<< "$SPMD_LINK"
  elif declares 'MPI_' && declares 'shmem_'; then
    echo "Error: compile: the program calls both MPI and OpenSHMEM; set SPMD_LINK to a linker that provides both" >&2
    exit 1
  elif declares 'MPI_'; then
    if [ "$isCxx" == true ]; then
      linker=("${MPICXX:-mpicxx}")
    else
      linker=("${MPICC:-mpicc}")
    fi
  elif declares 'shmem_'; then
    linker=("${OSHCC:-oshcc}")
  elif [ "$isCxx" == true ]; then
    linker=("${CXX:-c++}")
  else
    linker=("${CC:-cc}")
  fi
  if declares 'nccl'; then
    linkFlags+=(-lnccl)
  fi
  # NVSHMEM's host routines are in libnvshmem_host; -nvshmem-to-host-abi has
  # made the header's inline ones calls of it too.
  if declares 'nvshmem'; then
    if [[ -n "${NVSHMEM_HOME:-}" ]]; then
      linkFlags+=("-L$NVSHMEM_HOME/lib" "-Wl,-rpath,$NVSHMEM_HOME/lib")
    fi
    linkFlags+=(-lnvshmem_host)
  fi
  # NCCL's own calls are in libnccl, but its header's inline helpers and every
  # raised cudaMemcpy/cudaMemset are in libcudart, so an NCCL program, an
  # NVSHMEM one -- which fills its device buffers with them -- and a plain CUDA
  # one all need it.
  if declares 'nccl' || declares 'nvshmem' || declares 'cuda'; then
    linkFlags+=(-lcudart)
  fi
  if declares '__kmpc_' || declares 'omp_'; then
    ompLibDir=${SPMD_OMP_LIBDIR:-}
    if [[ -z "$ompLibDir" ]] && command -v clang > /dev/null; then
      ompLib=$(clang -print-file-name=libomp.so 2> /dev/null)
      # clang prints the bare name back when it has no such library.
      [[ "$ompLib" == /* ]] && ompLibDir=$(dirname "$ompLib")
    fi
    if [[ -z "$ompLibDir" || ! -e "$ompLibDir/libomp.so" ]]; then
      echo "Error: compile: the program uses OpenMP, which links against LLVM's libomp, and none was found; load an LLVM module (e.g. LLVM/20.1.7) or set SPMD_OMP_LIBDIR" >&2
      exit 1
    fi
    linkFlags+=("-L$ompLibDir" -lomp "-Wl,-rpath,$ompLibDir")
  fi
  read -r -a extraLinkFlags <<< "${SPMD_LDFLAGS:-}"

  # The shim headers prepended to CPATH above are MPICH's; an MPI wrapper that
  # finds its mpi.h through CPATH would read them instead of its own.
  withCallerCPATH() {
    if [[ -n "${callerCPATH+set}" ]]; then
      CPATH="$callerCPATH" "$@"
    else
      env -u CPATH "$@"
    fi
  }

  if ! command -v "${linker[0]}" > /dev/null; then
    echo "Error: compile: linker '${linker[0]}' not found; load the programming model's module or set SPMD_LINK" >&2
    exit 127
  fi
  # The lowering writes MPICH's handle values into the calls. An MPI with
  # another ABI links without complaint and then fails at run time, so ask its
  # own mpi.h what MPI_COMM_WORLD is before linking against it.
  if declares 'MPI_' && [[ -z "${SPMD_LINK:-}" ]]; then
    commWorld=$(printf '#include <mpi.h>\nMPI_COMM_WORLD\n' |
      withCallerCPATH "${linker[0]}" -E -P -x c - 2> /dev/null | tail -n 1)
    if [[ "$commWorld" != *0x44000000* ]]; then
      echo "Error: compile: ${linker[0]}'s MPI_COMM_WORLD is '$commWorld', not MPICH's 0x44000000; the MPI lowering targets the MPICH ABI (MPICH, Intel MPI, MVAPICH)" >&2
      exit 1
    fi
  fi
  stage "linking" withCallerCPATH "${linker[@]}" -o "$executable" "$stageDir/05-module.o" \
    "${linkFlags[@]}" "${extraLinkFlags[@]}"
  exit 0
fi

# ── Execute ─────────────────────────────────────────────────────────────────
# Compile and run passes
if [ "$acceptMLIR" == true ]; then
  # Skip reactant-clang entirely: feed the pre-existing MLIR file directly to spmd-opt
  $spmdOpt $spmdPasses $PRINTDEBUGINFO $time -o "${outputFile}2.mlir" "$inputFile"
elif [ "$useOpt" == true ]; then
  if [ "$emitMLIR" == true ]; then
    # OVERRIDE_PASS_PIPELINE already ends with print{filename=outputFile_realPath},
    # so opt handles the file write; no need to pipe into spmdOpt
    $opt "$@" 
  else
    # emitSPMDIR / emitPRESPMDIR / default: opt emits MLIR to stdout,
    # spmdOpt applies the SPMD pass sequence
    $opt "$@" | $spmdOpt $spmdPasses $PRINTDEBUGINFO $time - -o "$outputFile"
  fi
elif [ "$useOptChain" == true ]; then
  PATH="$PATH" CPATH="$CPATH" $clang $clangFlags "$@"  | $opt | $spmdOpt $spmdPasses $PRINTDEBUGINFO $time - -o "$outputFile"
elif [ "$emitMLIR" == true ]; then
  PATH="$PATH" CPATH="$CPATH" $clang $clangFlags "$@" 
else
  PATH="$PATH" CPATH="$CPATH" $clang $clangFlags "$@"  | $spmdOpt $spmdPasses $PRINTDEBUGINFO $time - -o "$outputFile"
fi
status=$?

#cleanup
for inputFile in "${inputFiles[@]}"; do
  re_export_file="${inputFile}.re_export"
  if [ -f "$re_export_file" ]; then
    rm -f "$re_export_file"
  fi
done

# Propagate the toolchain's status instead of the cleanup loop's. A verification
# pass that finds something calls signalPassFailure(), so spmd-opt exits with
# SPMD_IR_EXITCODE (default 1) -- i.e. non-zero means "findings or failure",
# the way a linter behaves. Set SPMD_IR_EXITCODE=0 to get findings without a
# non-zero status. Previously this script always exited 0, so a segfault or a
# failed compile was indistinguishable from a clean verification.
exit $status
