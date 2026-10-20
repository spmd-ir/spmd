# SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Lit configuration for the SPMD dialect unit tests.
#
# The tools are located through lit parameters rather than a generated
# lit.site.cfg.py: the Bazel rules in test/BUILD.bazel pass
# -Dspmd_opt=$(rootpath //test/lib:spmd-opt) and the equivalent for FileCheck,
# and this file puts their directories on PATH. That keeps the bare `spmd-opt`
# and `FileCheck` spellings in the RUN lines working, with no substitution
# machinery to get wrong.

import os

import lit.formats

config.name = "SPMD"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.environ.get(
    "TEST_UNDECLARED_OUTPUTS_DIR", config.test_source_root
)


def _runfile(param):
    """Absolute path of the file passed as -D<param>=<rootpath>, or None.

    $(rootpath) is relative to the runfiles root of the main repository, which
    Bazel exposes as $TEST_SRCDIR/$TEST_WORKSPACE. External repositories come
    through as "../<repo>/...", which joins correctly against that same base.
    """
    path = lit_config.params.get(param)
    if not path:
        return None
    srcdir = os.environ.get("TEST_SRCDIR")
    workspace = os.environ.get("TEST_WORKSPACE")
    if srcdir and workspace:
        path = os.path.join(srcdir, workspace, path)
    resolved = os.path.realpath(path)
    if not os.path.exists(resolved):
        lit_config.fatal("file for -D%s not found at %s" % (param, resolved))
    return resolved


def _tool_dir(param):
    """Absolute directory holding the tool passed as -D<param>=<rootpath>."""
    path = _runfile(param)
    return os.path.dirname(path) if path else None


tool_dirs = [d for d in (_tool_dir("spmd_opt"), _tool_dir("filecheck")) if d]

config.environment["PATH"] = os.pathsep.join(tool_dirs + [os.environ.get("PATH", "")])

# The API descriptions -convert-apiCalls-to-spmd needs, one -D<MODEL>_JSON_PATH
# per model (see _API_JSONS in test/BUILD.bazel). Exported under the same name
# spmd-verify.sh uses, so the RUN lines need no per-test setup.
for param in lit_config.params:
    if param.endswith("_JSON_PATH"):
        config.environment[param] = _runfile(param)

# Deliberately *not* set: SPMD_IR_EXITCODE. spmd-opt returns it in place of its
# normal nonzero failure code, so setting it to 0 makes a parse error or a pass
# failure look like success.
config.environment.pop("SPMD_IR_EXITCODE", None)
