# SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loads Enzyme-JAX."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "ENZYME_JAX_COMMIT", "ENZYME_JAX_SHA256", "OVERRIDE_ENZYME_JAX_PATH")

def repo():
    if len(OVERRIDE_ENZYME_JAX_PATH) != 0:
        native.local_repository(
            name = "enzyme_ad",
            path = OVERRIDE_ENZYME_JAX_PATH,
        )
    else:
        http_archive(
            name = "enzyme_ad",
            sha256 = ENZYME_JAX_SHA256,
            strip_prefix = "Enzyme-JAX-" + ENZYME_JAX_COMMIT,
            urls = ["https://github.com/EnzymeAD/Enzyme-JAX/archive/{commit}.tar.gz".format(commit = ENZYME_JAX_COMMIT)],
        )
