# SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loads Reactant"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "OVERRIDE_REACTANT_PATH", "REACTANT_COMMIT", "REACTANT_SHA256")

def repo():
    if len(OVERRIDE_REACTANT_PATH) != 0:
        native.local_repository(
            name = "reactant",
            path = OVERRIDE_REACTANT_PATH + "/enzyme",
        )
    else:
        http_archive(
            name = "reactant",
            sha256 = REACTANT_SHA256,
            strip_prefix = "Reactant-" + REACTANT_COMMIT + "/enzyme",
            urls = ["https://github.com/burakSemih/Reactant/archive/{commit}.tar.gz".format(commit = REACTANT_COMMIT)],
        )
