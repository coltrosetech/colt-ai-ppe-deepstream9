#include "deepsafe_fusion/config.hpp"

#include <fcntl.h>
#include <glib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <string_view>
#include <vector>

namespace deepsafe::fusion {
namespace {

constexpr std::size_t kMaximumConfigBytes = 64U * 1024U;

LoadedConfig failure(std::string error) {
  LoadedConfig result;
  result.error = std::move(error);
  return result;
}

bool lowercase_sha256(std::string_view value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool safe_absolute_path(const std::string& value) {
  if (value.empty() || value.front() != '/' || value.size() > 4096U ||
      value.find("//") != std::string::npos ||
      !std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isalnum(character) != 0 || character == '/' ||
               character == '.' || character == '_' || character == '-';
      })) {
    return false;
  }
  std::size_t start = 1U;
  while (start <= value.size()) {
    const auto end = value.find('/', start);
    const auto part = value.substr(start, end - start);
    if (part.empty() || part == "." || part == "..") {
      return false;
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1U;
  }
  return true;
}

struct Snapshot {
  bool ok{false};
  std::string error{};
  std::vector<unsigned char> bytes{};
};

Snapshot stable_snapshot(const std::string& path) {
  Snapshot result;
  if (!safe_absolute_path(path)) {
    result.error = "config path must be normalized, absolute and injection-safe";
    return result;
  }
  std::array<char, 4097> resolved{};
  if (realpath(path.c_str(), resolved.data()) == nullptr || path != resolved.data()) {
    result.error = "config path is missing, escapes, or contains a symlink";
    return result;
  }
  struct stat initial {};
  if (lstat(path.c_str(), &initial) != 0 || !S_ISREG(initial.st_mode) ||
      S_ISLNK(initial.st_mode) || initial.st_nlink != 1 || initial.st_size <= 0 ||
      static_cast<std::uint64_t>(initial.st_size) > kMaximumConfigBytes) {
    result.error = "config must be a bounded, single-link regular file";
    return result;
  }
  const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0) {
    result.error = "cannot open config snapshot";
    return result;
  }
  struct stat before {};
  struct stat after {};
  bool success = false;
  if (fstat(descriptor, &before) == 0 && before.st_dev == initial.st_dev &&
      before.st_ino == initial.st_ino && before.st_mode == initial.st_mode &&
      before.st_nlink == initial.st_nlink && before.st_size == initial.st_size &&
      before.st_mtim.tv_sec == initial.st_mtim.tv_sec &&
      before.st_mtim.tv_nsec == initial.st_mtim.tv_nsec &&
      before.st_ctim.tv_sec == initial.st_ctim.tv_sec &&
      before.st_ctim.tv_nsec == initial.st_ctim.tv_nsec) {
    result.bytes.resize(static_cast<std::size_t>(before.st_size));
    std::size_t offset = 0U;
    while (offset < result.bytes.size()) {
      const auto count = read(descriptor, result.bytes.data() + offset,
                              result.bytes.size() - offset);
      if (count <= 0) {
        break;
      }
      offset += static_cast<std::size_t>(count);
    }
    struct stat current {};
    success = offset == result.bytes.size() && fstat(descriptor, &after) == 0 &&
              lstat(path.c_str(), &current) == 0 &&
              before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
              before.st_mode == after.st_mode && before.st_nlink == after.st_nlink &&
              before.st_size == after.st_size &&
              before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
              before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
              before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
              before.st_ctim.tv_nsec == after.st_ctim.tv_nsec &&
              before.st_dev == current.st_dev && before.st_ino == current.st_ino &&
              before.st_mode == current.st_mode && before.st_nlink == current.st_nlink &&
              before.st_size == current.st_size &&
              before.st_mtim.tv_sec == current.st_mtim.tv_sec &&
              before.st_mtim.tv_nsec == current.st_mtim.tv_nsec &&
              before.st_ctim.tv_sec == current.st_ctim.tv_sec &&
              before.st_ctim.tv_nsec == current.st_ctim.tv_nsec;
  }
  close(descriptor);
  if (!success) {
    result.bytes.clear();
    result.error = "config changed during stable snapshot";
    return result;
  }
  result.ok = true;
  return result;
}

bool parse_unsigned(const std::string& value, std::uint64_t& output) {
  if (value.empty() || (value.size() > 1U && value.front() == '0')) {
    return false;
  }
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

bool parse_signed(const std::string& value, int& output) {
  if (value.empty() || value.front() == '-' ||
      (value.size() > 1U && value.front() == '0')) {
    return false;
  }
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

bool parse_float(const std::string& value, float& output) {
  static const std::regex canonical(R"(^[0-9]+\.[0-9]+$)");
  if (!std::regex_match(value, canonical)) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const float parsed = std::strtof(value.c_str(), &end);
  if (errno != 0 || end != value.c_str() + value.size() || !std::isfinite(parsed)) {
    return false;
  }
  output = parsed;
  return true;
}

}  // namespace

LoadedConfig load_hashed_config(const std::string& path,
                                const std::string& expected_sha256) {
  if (!lowercase_sha256(expected_sha256)) {
    return failure("expected config SHA-256 must be 64 lowercase hex characters");
  }
  const auto snapshot = stable_snapshot(path);
  if (!snapshot.ok) {
    return failure(snapshot.error);
  }
  gchar* digest = g_compute_checksum_for_data(
      G_CHECKSUM_SHA256, snapshot.bytes.data(), snapshot.bytes.size());
  if (digest == nullptr) {
    return failure("GLib SHA-256 calculation failed");
  }
  const std::string observed(digest);
  g_free(digest);
  if (observed != expected_sha256) {
    return failure("config SHA-256 does not match the explicit expected digest");
  }
  if (snapshot.bytes.back() != '\n' ||
      std::find(snapshot.bytes.begin(), snapshot.bytes.end(), '\r') !=
          snapshot.bytes.end() ||
      std::find(snapshot.bytes.begin(), snapshot.bytes.end(), '\0') !=
          snapshot.bytes.end()) {
    return failure("config must be LF-terminated UTF-8/ASCII without CR or NUL");
  }
  const std::string text(snapshot.bytes.begin(), snapshot.bytes.end());
  const std::vector<std::string> expected_keys = {
      "schema_version",
      "person_gie_id",
      "pose_gie_id",
      "ppe_gie_id",
      "person_class_id",
      "ppe_helmet_present_class_id",
      "ppe_helmet_absent_class_id",
      "ppe_hi_vis_present_class_id",
      "ppe_hi_vis_absent_class_id",
      "max_sources",
      "max_batch_size",
      "max_persons_per_frame",
      "max_pose_detections",
      "max_ppe_observations",
      "pose_detection_threshold",
      "pose_keypoint_threshold",
      "pose_ambiguity_margin",
      "minimum_pose_iou",
      "minimum_pose_coverage",
      "minimum_person_height_px",
      "minimum_tracker_confidence",
      "ppe_minimum_confidence",
      "ppe_minimum_person_coverage",
      "ppe_minimum_zone_coverage",
      "ppe_ambiguity_margin",
      "ppe_minimum_presence_visible_fraction",
      "ppe_minimum_absence_visible_fraction",
      "reject_nonmonotonic_pts",
      "missing_ppe_policy",
      "ambiguous_policy",
      "occluded_policy",
      "duplicate_meta_policy",
  };
  std::map<std::string, std::string> values;
  std::vector<std::string> observed_keys;
  std::size_t start = 0U;
  while (start < text.size()) {
    const auto end = text.find('\n', start);
    if (end == std::string::npos || end == start) {
      return failure("config contains an empty or unterminated line");
    }
    const auto line = text.substr(start, end - start);
    const auto separator = line.find('=');
    if (separator == std::string::npos || separator == 0U ||
        separator == line.size() - 1U || line.find('=', separator + 1U) != std::string::npos) {
      return failure("config line must contain exactly one non-edge '='");
    }
    const auto key = line.substr(0U, separator);
    const auto value = line.substr(separator + 1U);
    if (!std::all_of(key.begin(), key.end(), [](unsigned char character) {
          return (character >= 'a' && character <= 'z') || character == '_';
        }) ||
        !std::all_of(value.begin(), value.end(), [](unsigned char character) {
          return character >= 33U && character <= 126U;
        }) ||
        !values.emplace(key, value).second) {
      return failure("config has an invalid or duplicate key/value");
    }
    observed_keys.push_back(key);
    start = end + 1U;
  }
  if (observed_keys != expected_keys) {
    return failure("config required key set/order drifted");
  }
  if (values.at("schema_version") != kConfigSchemaVersion ||
      values.at("reject_nonmonotonic_pts") != "true" ||
      values.at("missing_ppe_policy") != "unknown" ||
      values.at("ambiguous_policy") != "unknown_unassociated" ||
      values.at("occluded_policy") != "unknown_unassociated" ||
      values.at("duplicate_meta_policy") != "reject_frame") {
    return failure("config schema or fail-closed policy drifted");
  }

  RuntimeConfig config;
  auto parse_u32 = [&](const char* key, std::uint32_t& target) {
    std::uint64_t parsed = 0U;
    if (!parse_unsigned(values.at(key), parsed) ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    target = static_cast<std::uint32_t>(parsed);
    return true;
  };
  auto parse_size = [&](const char* key, std::size_t& target) {
    std::uint64_t parsed = 0U;
    if (!parse_unsigned(values.at(key), parsed) ||
        parsed > std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    target = static_cast<std::size_t>(parsed);
    return true;
  };
  if (!parse_u32("person_gie_id", config.person_gie_id) ||
      !parse_u32("pose_gie_id", config.pose_gie_id) ||
      !parse_u32("ppe_gie_id", config.ppe_gie_id) ||
      !parse_signed(values.at("person_class_id"), config.person_class_id) ||
      !parse_signed(values.at("ppe_helmet_present_class_id"),
                    config.ppe_helmet_present_class_id) ||
      !parse_signed(values.at("ppe_helmet_absent_class_id"),
                    config.ppe_helmet_absent_class_id) ||
      !parse_signed(values.at("ppe_hi_vis_present_class_id"),
                    config.ppe_hi_vis_present_class_id) ||
      !parse_signed(values.at("ppe_hi_vis_absent_class_id"),
                    config.ppe_hi_vis_absent_class_id) ||
      !parse_u32("max_sources", config.max_sources) ||
      !parse_u32("max_batch_size", config.max_batch_size) ||
      !parse_size("max_persons_per_frame", config.max_persons_per_frame) ||
      !parse_size("max_pose_detections", config.max_pose_detections) ||
      !parse_size("max_ppe_observations", config.max_ppe_observations) ||
      !parse_float(values.at("pose_detection_threshold"),
                   config.pose_detection_threshold) ||
      !parse_float(values.at("pose_keypoint_threshold"),
                   config.pose_keypoint_threshold) ||
      !parse_float(values.at("pose_ambiguity_margin"),
                   config.pose_ambiguity_margin) ||
      !parse_float(values.at("minimum_pose_iou"), config.minimum_pose_iou) ||
      !parse_float(values.at("minimum_pose_coverage"),
                   config.minimum_pose_coverage) ||
      !parse_float(values.at("minimum_person_height_px"),
                   config.minimum_person_height_px) ||
      !parse_float(values.at("minimum_tracker_confidence"),
                   config.minimum_tracker_confidence) ||
      !parse_float(values.at("ppe_minimum_confidence"),
                   config.ppe_association.minimum_confidence) ||
      !parse_float(values.at("ppe_minimum_person_coverage"),
                   config.ppe_association.minimum_person_coverage) ||
      !parse_float(values.at("ppe_minimum_zone_coverage"),
                   config.ppe_association.minimum_zone_coverage) ||
      !parse_float(values.at("ppe_ambiguity_margin"),
                   config.ppe_association.ambiguity_margin) ||
      !parse_float(values.at("ppe_minimum_presence_visible_fraction"),
                   config.ppe_association.minimum_presence_visible_fraction) ||
      !parse_float(values.at("ppe_minimum_absence_visible_fraction"),
                   config.ppe_association.minimum_absence_visible_fraction)) {
    return failure("config contains a noncanonical or out-of-range scalar");
  }
  config.reject_nonmonotonic_pts = true;
  if (!config.valid()) {
    return failure("parsed fusion runtime configuration is semantically invalid");
  }
  LoadedConfig result;
  result.ok = true;
  result.sha256 = observed;
  result.runtime = config;
  return result;
}

}  // namespace deepsafe::fusion
