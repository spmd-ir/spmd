<!--
SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# SPMD IR logo

The mark is one program that forks into four ranks, which meet at a barrier
and continue. In the logo, the name takes the place of the lanes that leave
the barrier; the mark on its own keeps them. The lettering is IBM Plex Sans
Bold (SIL Open Font License), converted to outlines, so the files show the
same everywhere, including through `<img>`, where an SVG cannot load fonts.

| File | Use |
|---|---|
| `spmd-ir-logo.svg` | The logo (mark and name), for light backgrounds: README, slides, papers |
| `spmd-ir-logo-dark.svg` | The same, for dark backgrounds |
| `spmd-ir-mark.svg` | The full mark alone, for light backgrounds, where the name is already written next to it |
| `spmd-ir-mark-dark.svg` | The same, for dark backgrounds |
| `spmd-ir-avatar.svg`, `.png` | Group, organisation and project avatars (512 px); a solid tile, so one file works in light and dark themes |
| `spmd-ir-social-preview.svg`, `.png` | GitHub's social preview image (1280 x 640) |

The PNGs are rendered from the SVGs beside them:

```sh
rsvg-convert -w 512 -h 512 spmd-ir-avatar.svg -o spmd-ir-avatar.png
rsvg-convert spmd-ir-social-preview.svg -o spmd-ir-social-preview.png
```

Keep the name *SPMD IR*; `spmd` is the name of the repository, the dialect and
the tools (`spmd-opt`, `spmd-verify`), not of the project.

## Using the logo

The logo is not under the Apache License that covers the rest of SPMD IR. Its
terms are in
[LICENSES/LicenseRef-SPMD-IR-Logo.txt](../../LICENSES/LicenseRef-SPMD-IR-Logo.txt).
In short: use it, unmodified, to refer to SPMD IR (papers, slides, posters,
articles, forks and mirrors of this repository) without asking; ask first
before changing it, using it for something else, or suggesting an
endorsement.
