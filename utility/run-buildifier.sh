#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Format, or check the formatting of, every Bazel file tracked by this
# repository -- BUILD.bazel, WORKSPACE.bazel and *.bzl.
#
#   utility/run-buildifier.sh              reformat in place
#   utility/run-buildifier.sh --check      report violations, change nothing
#   utility/run-buildifier.sh --lint       additionally print lint warnings
#   utility/run-buildifier.sh --bootstrap  fetch the pinned buildifier first
#
# The counterpart of run-clang-format.sh, and pinned for the same reason:
# buildifier's output changes between releases, so an unpinned check would
# reject files somebody had just formatted.
#
# Unlike clang-format, buildifier is not on the cluster's module system, so
# --bootstrap fetches it. It is a Go binary published as a GitHub release, but
# the release assets are not reachable from every network here; the identical
# official binaries are also shipped inside the @bazel/buildifier npm package,
# which is, so that is what we pull -- with the sha256 of the exact binary
# checked, so it does not matter which mirror hands it over.
#
# `externals/` needs no exclusion. The two benchmark suites there are
# submodules, whose contents are not files of this repository and so are never
# listed by `git ls-files`; the enzyme_ad/ and reactant/ repository macros
# beside them are this project's own Bazel code and are formatted like the
# rest.

set -euo pipefail

readonly REQUIRED_VERSION="8.2.1"
readonly TARBALL_URL="https://registry.npmjs.org/@bazel/buildifier/-/buildifier-${REQUIRED_VERSION}.tgz"
readonly BINARY_IN_TARBALL="package/buildifier-linux_amd64"
readonly EXPECTED_SHA256="6ceb7b0ab7cf66fceccc56a027d21d9cc557a7f34af37d2101edb56b92fcfa1a"

cache_dir=${BUILDIFIER_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/spmd-buildifier}

usage() {
  cat >&2 <<'USAGE'
usage: utility/run-buildifier.sh [--check] [--lint] [--bootstrap] [--] [path...]

  --check      do not edit; list every file that needs reformatting, exit non-zero
  --lint       also print buildifier's lint warnings (never affects the exit code)
  --bootstrap  download the pinned buildifier into the cache if it is missing
  path...      restrict to these pathspecs (passed to git ls-files)

environment:
  BUILDIFIER                    buildifier binary to use
  BUILDIFIER_DIR                where --bootstrap caches it
                                (default: ${XDG_CACHE_HOME:-$HOME/.cache}/spmd-buildifier)
  BUILDIFIER_SKIP_VERSION_CHECK=1   accept a version other than the pinned one
USAGE
}

check_only=0
lint=0
bootstrap=0
paths=()
while [ $# -gt 0 ]; do
  case "$1" in
    --check) check_only=1; shift ;;
    --lint) lint=1; shift ;;
    --bootstrap) bootstrap=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; paths+=("$@"); break ;;
    -*) echo "error: unknown option '$1'" >&2; usage; exit 2 ;;
    *) paths+=("$1"); shift ;;
  esac
done

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

# Verify a candidate binary is the pinned version. Kept separate from the
# download so a buildifier already on PATH is held to the same standard.
version_of() {
  "$1" --version 2>/dev/null | sed -n 's/^buildifier version: *\([^ ]*\).*/\1/p'
}

fetch_buildifier() {
  local dest=$1 tmp
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN
  echo "fetching buildifier ${REQUIRED_VERSION} from ${TARBALL_URL}" >&2
  if ! curl -sSL --fail --max-time 300 -o "$tmp/bf.tgz" "$TARBALL_URL"; then
    echo "error: download failed" >&2
    return 1
  fi
  tar -xzf "$tmp/bf.tgz" -C "$tmp" "$BINARY_IN_TARBALL"
  local got
  got=$(sha256sum "$tmp/$BINARY_IN_TARBALL" | cut -d' ' -f1)
  if [ "$got" != "$EXPECTED_SHA256" ]; then
    echo "error: sha256 mismatch for the downloaded buildifier" >&2
    echo "       expected $EXPECTED_SHA256" >&2
    echo "       got      $got" >&2
    return 1
  fi
  mkdir -p "$(dirname "$dest")"
  install -m 0755 "$tmp/$BINARY_IN_TARBALL" "$dest"
  echo "installed $dest" >&2
}

cached=$cache_dir/buildifier-$REQUIRED_VERSION
if [ -n "${BUILDIFIER:-}" ]; then
  buildifier=$BUILDIFIER
elif command -v buildifier >/dev/null 2>&1; then
  buildifier=$(command -v buildifier)
elif [ -x "$cached" ]; then
  buildifier=$cached
elif [ "$bootstrap" -eq 1 ]; then
  fetch_buildifier "$cached"
  buildifier=$cached
else
  echo "error: buildifier not found." >&2
  echo "       Run with --bootstrap to fetch the pinned ${REQUIRED_VERSION}" >&2
  echo "       into ${cache_dir}, or set BUILDIFIER to your own binary." >&2
  exit 127
fi

version=$(version_of "$buildifier")
if [ "${BUILDIFIER_SKIP_VERSION_CHECK:-0}" != "1" ] &&
   [ "$version" != "$REQUIRED_VERSION" ]; then
  echo "error: need buildifier ${REQUIRED_VERSION}, found: ${version:-unknown} ($buildifier)" >&2
  echo "       buildifier's output differs between releases, so a mismatch" >&2
  echo "       would produce a diff CI then rejects. Run with --bootstrap to" >&2
  echo "       fetch the pinned version, or set" >&2
  echo "       BUILDIFIER_SKIP_VERSION_CHECK=1 to override." >&2
  exit 1
fi

if [ ${#paths[@]} -eq 0 ]; then
  paths=(':/')
fi

# Multiple git pathspecs are a union, not an intersection, so "everything under
# these paths that is also a Bazel file" cannot be asked for directly -- listing
# the tracked files and filtering by name here is what run-clang-format.sh does
# for its extensions too.
mapfile -d '' -t tracked < <(git ls-files -z -- "${paths[@]}")
files=()
for f in "${tracked[@]}"; do
  case "${f##*/}" in
    *.bzl|*.bazel|BUILD|WORKSPACE|MODULE) files+=("$f") ;;
  esac
done

if [ ${#files[@]} -eq 0 ]; then
  echo "no Bazel files matched" >&2
  exit 0
fi

if [ "$lint" -eq 1 ]; then
  # Advisory only: lint findings are style suggestions, not a gate, and its
  # exit code is deliberately discarded so --check keeps meaning "formatting".
  echo "buildifier ${version}: lint warnings"
  "$buildifier" -mode=check -lint=warn "${files[@]}" || true
  echo
fi

if [ "$check_only" -eq 1 ]; then
  echo "buildifier ${version}: checking ${#files[@]} Bazel files"
  if ! "$buildifier" -mode=check "${files[@]}"; then
    echo >&2
    echo "error: the files above are not formatted." >&2
    echo "       Fix them with: utility/run-buildifier.sh" >&2
    exit 1
  fi
  echo "all Bazel files are correctly formatted"
else
  echo "buildifier ${version}: formatting ${#files[@]} Bazel files"
  "$buildifier" -mode=fix "${files[@]}"
  echo "done; review with: git diff"
fi
