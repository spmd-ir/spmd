#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Format, or check the formatting of, every C/C++ source tracked by this
# repository.
#
#   utility/run-clang-format.sh            reformat in place
#   utility/run-clang-format.sh --check    report violations, change nothing
#
# The set of files is `git ls-files` filtered to C/C++ extensions; the set of
# *exclusions* lives in .clang-format-ignore, which clang-format applies on its
# own, so there is no second list here to drift out of sync. .td files are
# deliberately absent: clang-format has no TableGen mode and would treat them
# as C++ (llvm-project does not format its own .td files either).
#
# clang-format's output is version dependent, so the version is pinned: CI and
# every developer on the cluster get it from `ml LLVM/20.1.7` -- the module
# that also supplies the clang/clang++ the build runs as its host compiler.
#
# Note this is NOT the LLVM the dialect links against. Bazel fetches its own
# llvm-project through XLA's pin (@llvm-project, currently 23.0.0git, i.e.
# unreleased main), and taking clang-format from there was considered and
# rejected:
#
#   * It moves. Every Enzyme-JAX/XLA dependency bump can change the formatter,
#     which would reformat the tree or fail this check for reasons that have
#     nothing to do with the change under review. A released 20.1.x is a fixed
#     point; "main" is not a version.
#   * It is expensive. Clang is not built in this workspace at all (llvm and
#     mlir are), and @llvm-project//clang:clang-format pulls in ~2000 further
#     clang targets -- the whole frontend. A check that today needs no build
#     would have to wait for one.
#
# Enzyme-JAX draws the same line: its format workflow pins clang-format 16
# while the project builds against LLVM main.
#
# Set CLANG_FORMAT to point at a different binary, and
# CLANG_FORMAT_SKIP_VERSION_CHECK=1 if you knowingly want another version.

set -euo pipefail

readonly REQUIRED_MAJOR=20

usage() {
  cat >&2 <<'USAGE'
usage: utility/run-clang-format.sh [--check] [--] [path...]

  --check     do not edit; print every violation and exit non-zero if any
  path...     restrict to these pathspecs (passed to git ls-files)

environment:
  CLANG_FORMAT                    clang-format binary to use
  CLANG_FORMAT_SKIP_VERSION_CHECK=1   accept a version other than 20.x
USAGE
}

check_only=0
paths=()
while [ $# -gt 0 ]; do
  case "$1" in
    --check) check_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; paths+=("$@"); break ;;
    -*) echo "error: unknown option '$1'" >&2; usage; exit 2 ;;
    *) paths+=("$1"); shift ;;
  esac
done

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

clang_format=${CLANG_FORMAT:-clang-format}
if ! command -v "$clang_format" >/dev/null 2>&1; then
  echo "error: '$clang_format' not found. On the cluster:" \
       "ml GCCcore/14.2.0 LLVM/20.1.7" >&2
  exit 127
fi

version=$("$clang_format" --version)
major=$(printf '%s\n' "$version" | sed -n 's/.*version \([0-9]\{1,\}\).*/\1/p')
if [ "${CLANG_FORMAT_SKIP_VERSION_CHECK:-0}" != "1" ] &&
   [ "$major" != "$REQUIRED_MAJOR" ]; then
  echo "error: need clang-format ${REQUIRED_MAJOR}.x, found: ${version}" >&2
  echo "       clang-format's output differs between major versions, so a" >&2
  echo "       mismatch would produce a diff CI then rejects. On the cluster:" >&2
  echo "         ml GCCcore/14.2.0 LLVM/20.1.7" >&2
  echo "       Set CLANG_FORMAT_SKIP_VERSION_CHECK=1 to override." >&2
  exit 1
fi

# No pathspec restriction means the whole repository.
if [ ${#paths[@]} -eq 0 ]; then
  paths=(':/')
fi

# git pathspecs cannot express "these extensions under these directories"
# without magic pathspecs that older git rejects, so list the tracked files
# under the requested paths and filter by extension here.
mapfile -d '' -t files < <(git ls-files -z -- "${paths[@]}")
filtered=()
for f in "${files[@]}"; do
  case "$f" in
    *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp|*.cu|*.cuh) filtered+=("$f") ;;
  esac
done

if [ ${#filtered[@]} -eq 0 ]; then
  echo "no C/C++ sources matched" >&2
  exit 0
fi

if [ "$check_only" -eq 1 ]; then
  echo "${version}: checking ${#filtered[@]} files (exclusions from .clang-format-ignore)"
  if ! "$clang_format" --dry-run -Werror "${filtered[@]}"; then
    echo >&2
    echo "error: the files above are not formatted." >&2
    echo "       Fix them with: utility/run-clang-format.sh" >&2
    exit 1
  fi
  echo "all files are correctly formatted"
else
  echo "${version}: formatting ${#filtered[@]} files (exclusions from .clang-format-ignore)"
  "$clang_format" -i "${filtered[@]}"
  echo "done; review with: git diff"
fi
