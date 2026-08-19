#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "deepsafe_pose/association.hpp"
#include "deepsafe_pose/decoder.hpp"
#include "deepsafe_ppe/association.hpp"

namespace deepsafe::fusion {

inline constexpr std::uint32_t kAbiVersionV1 = 0x00010000U;
inline constexpr std::uint32_t kCanonicalPersonGieId = 1U;
inline constexpr std::uint32_t kPoseGieId = 2U;
inline constexpr std::uint32_t kPpeGieId = 3U;
inline constexpr std::uint32_t kMaximumSources = 12U;

enum class FrameStatus : std::uint8_t {
  Ok = 0,
  Stale = 1,
  RejectedInput = 2,
};

enum class LinkStatus : std::uint8_t {
  Matched = 0,
  UnknownNoObservation = 1,
  UnknownNoPerson = 2,
  UnknownAmbiguous = 3,
  UnknownOccluded = 4,
  UnknownDuplicate = 5,
  UnknownGeometry = 6,
  UnknownTrackConflict = 7,
  UnknownStale = 8,
  UnknownInvalidInput = 9,
};

struct RuntimeConfig {
  std::uint32_t person_gie_id{kCanonicalPersonGieId};
  std::uint32_t pose_gie_id{kPoseGieId};
  std::uint32_t ppe_gie_id{kPpeGieId};
  int person_class_id{0};
  int ppe_helmet_present_class_id{0};
  int ppe_helmet_absent_class_id{1};
  int ppe_hi_vis_present_class_id{2};
  int ppe_hi_vis_absent_class_id{3};
  std::uint32_t max_sources{kMaximumSources};
  std::uint32_t max_batch_size{kMaximumSources};
  std::size_t max_persons_per_frame{512U};
  std::size_t max_pose_detections{300U};
  std::size_t max_ppe_observations{1024U};
  float pose_detection_threshold{0.25F};
  float pose_keypoint_threshold{0.25F};
  float pose_ambiguity_margin{0.05F};
  float minimum_pose_iou{0.30F};
  float minimum_pose_coverage{0.50F};
  float minimum_person_height_px{24.0F};
  float minimum_tracker_confidence{0.0F};
  deepsafe::ppe::AssociationConfig ppe_association{};
  bool reject_nonmonotonic_pts{true};

  [[nodiscard]] bool valid() const noexcept;
};

struct CanonicalTrack {
  std::uint64_t track_id{deepsafe::pose::kUntrackedObjectId};
  deepsafe::pose::RectF bbox{};
  bool confirmed{true};
  bool occluded{false};
  deepsafe::ppe::RegionVisibility helmet_visibility{};
  deepsafe::ppe::RegionVisibility hi_vis_visibility{};
};

struct FrameInput {
  std::uint32_t source_id{0U};
  std::uint64_t frame_num{0U};
  std::uint64_t pts_ns{0U};
  std::vector<CanonicalTrack> canonical_tracks{};
  std::vector<deepsafe::pose::PoseDetection> poses{};
  std::vector<deepsafe::ppe::EquipmentObservation> ppe_observations{};
  bool duplicate_pose_tensor_meta{false};
  bool invalid_pose_tensor_meta{false};
  bool duplicate_fusion_output_meta{false};
};

struct PoseLink {
  LinkStatus status{LinkStatus::UnknownNoObservation};
  float association_score{0.0F};
  std::size_t pose_index{0U};
  deepsafe::pose::PoseDetection pose{};
};

struct EquipmentLink {
  LinkStatus status{LinkStatus::UnknownNoObservation};
  deepsafe::ppe::Evidence evidence{deepsafe::ppe::Evidence::Unknown};
  float confidence{0.0F};
  std::uint64_t observation_id{deepsafe::ppe::kInvalidObservationId};
};

struct PersonResult {
  std::uint64_t track_id{deepsafe::pose::kUntrackedObjectId};
  deepsafe::pose::RectF bbox{};
  bool occluded{false};
  PoseLink pose{};
  EquipmentLink helmet{};
  EquipmentLink hi_vis{};
};

struct FrameDiagnostics {
  std::size_t canonical_track_count{0U};
  std::size_t pose_input_count{0U};
  std::size_t ppe_input_count{0U};
  std::size_t unassociated_pose_count{0U};
  std::size_t ambiguous_pose_count{0U};
  std::size_t unassociated_ppe_count{0U};
  std::size_t ambiguous_ppe_count{0U};
  bool duplicate_pose_tensor_meta{false};
  bool duplicate_fusion_output_meta{false};
  bool invalid_pose_tensor_meta{false};
  bool stale_pts{false};
  std::string error{};
};

struct FrameResult {
  FrameStatus status{FrameStatus::Ok};
  std::uint32_t source_id{0U};
  std::uint64_t frame_num{0U};
  std::uint64_t pts_ns{0U};
  std::vector<PersonResult> persons{};
  FrameDiagnostics diagnostics{};
};

// Stateful only for per-source frame/PTS monotonicity. Invalid/stale frames do
// not mutate the watermark. Association itself is deterministic and delegates
// geometry scoring to the existing pose and PPE postprocess cores.
class FusionEngine {
 public:
  explicit FusionEngine(RuntimeConfig config);

  [[nodiscard]] const RuntimeConfig& config() const noexcept { return config_; }
  [[nodiscard]] FrameResult process(const FrameInput& input);
  void reset_source(std::uint32_t source_id) noexcept;
  void reset() noexcept;

 private:
  struct Watermark {
    std::uint64_t frame_num{0U};
    std::uint64_t pts_ns{0U};
    bool initialized{false};
  };

  RuntimeConfig config_;
  std::map<std::uint32_t, Watermark> watermarks_{};
};

}  // namespace deepsafe::fusion
