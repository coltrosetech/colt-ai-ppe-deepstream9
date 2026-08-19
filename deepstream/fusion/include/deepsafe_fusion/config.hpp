#pragma once

#include <string>

#include "deepsafe_fusion/core.hpp"

namespace deepsafe::fusion {

inline constexpr const char* kConfigSchemaVersion =
    "deepsafe.fusion-runtime-config/v1";

struct LoadedConfig {
  bool ok{false};
  std::string error{};
  std::string sha256{};
  RuntimeConfig runtime{};
};

// Reads one stable, single-link, non-symlink snapshot. expected_sha256 must be
// exactly 64 lowercase hexadecimal characters. The line-oriented config has an
// exact required key set and rejects duplicates, unknown keys and noncanonical
// scalar encodings.
[[nodiscard]] LoadedConfig load_hashed_config(const std::string& path,
                                              const std::string& expected_sha256);

}  // namespace deepsafe::fusion
