/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- common.h - Shared includes for SPMD passes ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements miscellaneous functionality  useful for the CACTS pass
// and related components that do not fit into one component.
// Include several headers useful for multiple components.
//
//===----------------------------------------------------------------------===//

#ifndef COMMON_H
#define COMMON_H

#include "mlir/IR/Location.h"
#include "llvm/Support/Debug.h"

// for json writing
#include "llvm/ADT/SmallString.h"    // for llvm::SmallString
#include "llvm/Support/FileSystem.h" // for real_path
#include "llvm/Support/JSON.h"
#include <filesystem> // C++17

namespace mlir::spmd {

static inline llvm::json::Object getLocationJson(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    std::string file = fileLoc.getFilename().str();

    llvm::SmallString<256> absPath(file);
    if (!llvm::sys::fs::real_path(file, absPath)) {
      file = absPath.str().str();
    }

    return llvm::json::Object{{"file", file},
                              {"line", fileLoc.getLine()},
                              {"column", fileLoc.getColumn()}};
  }

  return llvm::json::Object{{"file", "<unknown>"}, {"line", 0}, {"column", 0}};
}

static inline std::string locToString(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return (llvm::Twine(fileLoc.getFilename()) + ":" +
            llvm::Twine(fileLoc.getLine()) + ":" +
            llvm::Twine(fileLoc.getColumn()))
        .str();
  return "<unknown location>";
}

} // namespace mlir::spmd

#endif // COMMON_H