<!--
SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# `patches/`

These are **not** spmd's own patches, and they are not optional: they are
patches against `jax` and `xla` that Enzyme-JAX applies on our behalf.

## Why they live here and not in Enzyme-JAX

Enzyme-JAX's repository macros name them with *main-repository* labels:

```python
# @enzyme_ad//third_party/jax:workspace.bzl
http_archive(name = "jax", patches = ["//:patches/jax.patch"], patch_args = ["-p1"], ...)

# @enzyme_ad//third_party/xla:workspace.bzl
http_archive(name = "xla", patches = ["//:patches/xla.patch", "//:patches/xla_win.patch"], ...)
```

A bare `//:` label resolves against whichever repository is the main one. When
Enzyme-JAX is built standalone that is Enzyme-JAX, and its own `patches/`
directory supplies them. When **spmd** is the main repository — which is the
only way this project is built — `//:patches/...` resolves *here*, so
`WORKSPACE.bazel`'s `jax_workspace()` / `xla_workspace()` calls fetch these
files from `spmd/patches/`.

Consequence: nothing in spmd's own build files mentions these filenames, so
they look unreferenced to any grep of this repository. They are not. Deleting
`jax.patch`, `xla.patch` or `xla_win.patch` breaks the build at fetch time.

## Contents

| file | required | applied to |
| --- | --- | --- |
| `jax.patch` | yes — `//:patches/jax.patch` | `@jax` |
| `xla.patch` | yes — `//:patches/xla.patch` | `@xla` |
| `xla_win.patch` | yes — `//:patches/xla_win.patch` | `@xla` |

All three are required. A fourth file, `xla_dedup_comm.patch`, used to sit here
in the slot where Enzyme-JAX's own `patches/` carries `xla_group_cp.patch`; no
macro referenced it, so it was removed. If you need an extra patch, both macros
accept an `extra_patches` argument rather than picking files up implicitly.

## Keeping them in sync

When `ENZYME_JAX_COMMIT` in `workspace.bzl` is bumped, check
`Enzyme-JAX/patches/` for changes to the three required files and to the set of
`//:patches/...` labels the `third_party/*/workspace.bzl` macros reference — a
patch that no longer applies, or a newly required label with no file here,
shows up as a fetch failure rather than a compile error.
