# SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration for SPMD workspace."""

# Define these at the top level of your workspace or in a separate file
ENZYME_JAX_COMMIT = "80c6fad13c6bebc4f2fa08ff608723c95c259f13"  # or specific commit hash
ENZYME_JAX_SHA256 = ""  # Add if using http_archive
OVERRIDE_ENZYME_JAX_PATH = ""  # Set this for local development ../Enzyme-JAX

REACTANT_COMMIT = "6dd9d9832a4691a2457e28317750f61488c9e383"
REACTANT_SHA256 = ""  # Add if using http_archive
OVERRIDE_REACTANT_PATH = ""  # Set this for local development
