# SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# llvm_openmp_repo.bzl  (in your main spmd project root)
def _llvm_openmp_headers_impl(rctx):
    llvm_root = rctx.path(Label("@llvm-raw//:CMakeLists.txt")).dirname
    omp_include = str(llvm_root) + "/openmp/runtime/src/include"

    rctx.execute(["mkdir", "-p", "staging/include"])

    result = rctx.execute(["sh", "-c", """
        set -e
        sed \
          -e 's/@LIBOMP_VERSION_MAJOR@/5/g' \
          -e 's/@LIBOMP_VERSION_MINOR@/2/g' \
          -e 's/@LIBOMP_VERSION_BUILD@/20231101/g' \
          -e 's/@LIBOMP_LIB_FILE@/libomp.so/g' \
          -e 's/@LIBOMP_HAVE_QUAD_PRECISION@/0/g' \
          -e 's/@LIBOMP_QUAD_PRECISION_TYPEOF_SPECIFIER@//g' \
          -e 's/@LIBOMP_HAVE_HIDDEN_HELPER_TASK@/1/g' \
          -e 's/@LIBOMP_HAVE_OMPT_SUPPORT@/0/g' \
          -e 's/@LIBOMP_OMPT_OPTIONAL@/0/g' \
          -e 's/@LIBOMP_USE_ITT_NOTIFY@/0/g' \
          '{src}/omp.h.var' > staging/include/omp.h
        for h in omp-tools.h ompx.h omp_lib.h; do
            test -f "{src}/$h" && cp "{src}/$h" "staging/include/$h" || true
        done
    """.format(src = omp_include)])

    if result.return_code != 0:
        fail("Failed to generate omp headers: " + result.stderr)

    rctx.file("BUILD.bazel", """
filegroup(
    name = "omp_headers_staged",
    srcs = glob(["staging/include/**"]),
    visibility = ["//visibility:public"],
)
""")

llvm_openmp_headers = repository_rule(
    implementation = _llvm_openmp_headers_impl,
    attrs = {},
)
