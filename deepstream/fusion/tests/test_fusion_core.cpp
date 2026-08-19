#include "deepsafe_fusion/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: "          \
                << #condition << '\n';                                          \
      ++failures;                                                               \
    }                                                                           \
  } while (false)

deepsafe::ppe::RegionVisibility visible() {
  return {deepsafe::ppe::Visibility::Visible, 1.0F};
}

deepsafe::fusion::CanonicalTrack track(std::uint64_t id, float left,
                                       float right, bool occluded = false) {
  deepsafe::fusion::CanonicalTrack value;
  value.track_id = id;
  value.bbox = {left, 0.0F, right, 200.0F};
  value.confirmed = true;
  value.occluded = occluded;
  value.helmet_visibility = occluded
                                ? deepsafe::ppe::RegionVisibility{
                                      deepsafe::ppe::Visibility::NotVisible, 0.0F}
                                : visible();
  value.hi_vis_visibility = value.helmet_visibility;
  return value;
}

deepsafe::pose::PoseDetection pose(float left, float right, float score = 0.9F) {
  deepsafe::pose::PoseDetection value;
  value.bbox = {left, 0.0F, right, 200.0F};
  value.score = score;
  value.class_id = 0;
  for (std::size_t index = 0; index < value.keypoints.size(); ++index) {
    value.keypoints[index] = {
        static_cast<deepsafe::pose::CocoKeypoint>(index),
        (left + right) * 0.5F,
        50.0F + static_cast<float>(index),
        0.8F,
        true,
    };
  }
  return value;
}

deepsafe::ppe::EquipmentObservation equipment(
    std::uint64_t id, deepsafe::ppe::Equipment type,
    deepsafe::ppe::Evidence evidence, deepsafe::ppe::RectF bbox) {
  return {id, type, evidence, bbox, 0.95F};
}

deepsafe::fusion::FrameInput frame(std::uint32_t source, std::uint64_t number,
                                   std::uint64_t pts) {
  deepsafe::fusion::FrameInput value;
  value.source_id = source;
  value.frame_num = number;
  value.pts_ns = pts;
  return value;
}

void test_matched_pose_and_explicit_ppe() {
  deepsafe::fusion::FusionEngine engine({});
  auto input = frame(0, 1, 100);
  input.canonical_tracks = {track(42, 0.0F, 100.0F)};
  input.poses = {pose(0.0F, 100.0F)};
  input.ppe_observations = {
      equipment(1, deepsafe::ppe::Equipment::Helmet,
                deepsafe::ppe::Evidence::Present, {20, 10, 50, 40}),
      equipment(2, deepsafe::ppe::Equipment::HiVis,
                deepsafe::ppe::Evidence::Absent, {20, 55, 75, 125}),
  };
  const auto result = engine.process(input);
  CHECK(result.status == deepsafe::fusion::FrameStatus::Ok);
  CHECK(result.persons.size() == 1U);
  CHECK(result.persons[0].track_id == 42U);
  CHECK(result.persons[0].pose.status == deepsafe::fusion::LinkStatus::Matched);
  CHECK(result.persons[0].helmet.evidence == deepsafe::ppe::Evidence::Present);
  CHECK(result.persons[0].hi_vis.evidence == deepsafe::ppe::Evidence::Absent);
  CHECK(result.diagnostics.unassociated_pose_count == 0U);
  CHECK(result.diagnostics.unassociated_ppe_count == 0U);
}

void test_missing_ppe_is_unknown_never_absent() {
  deepsafe::fusion::FusionEngine engine({});
  auto input = frame(0, 1, 100);
  input.canonical_tracks = {track(7, 0.0F, 100.0F)};
  const auto result = engine.process(input);
  CHECK(result.persons.size() == 1U);
  CHECK(result.persons[0].helmet.evidence == deepsafe::ppe::Evidence::Unknown);
  CHECK(result.persons[0].hi_vis.evidence == deepsafe::ppe::Evidence::Unknown);
  CHECK(result.persons[0].helmet.status ==
        deepsafe::fusion::LinkStatus::UnknownNoObservation);
  CHECK(result.persons[0].hi_vis.status ==
        deepsafe::fusion::LinkStatus::UnknownNoObservation);
}

void test_ambiguous_pose_and_ppe_are_unassociated() {
  deepsafe::fusion::FusionEngine engine({});
  auto input = frame(0, 1, 100);
  input.canonical_tracks = {
      track(9, 0.0F, 100.0F),
      track(3, 0.0F, 100.0F),
  };
  input.poses = {pose(0.0F, 100.0F)};
  input.ppe_observations = {
      equipment(1, deepsafe::ppe::Equipment::Helmet,
                deepsafe::ppe::Evidence::Absent, {20, 10, 50, 40}),
  };
  const auto result = engine.process(input);
  CHECK(result.diagnostics.ambiguous_pose_count == 1U);
  CHECK(result.diagnostics.unassociated_pose_count == 1U);
  CHECK(result.diagnostics.ambiguous_ppe_count == 1U);
  CHECK(result.diagnostics.unassociated_ppe_count == 1U);
  for (const auto& person : result.persons) {
    CHECK(person.pose.status == deepsafe::fusion::LinkStatus::UnknownNoObservation ||
          person.pose.status == deepsafe::fusion::LinkStatus::UnknownAmbiguous);
    CHECK(person.helmet.evidence == deepsafe::ppe::Evidence::Unknown);
    CHECK(person.helmet.status != deepsafe::fusion::LinkStatus::Matched);
  }
}

void test_occluded_person_is_never_assigned() {
  deepsafe::fusion::FusionEngine engine({});
  auto input = frame(0, 1, 100);
  input.canonical_tracks = {track(5, 0.0F, 100.0F, true)};
  input.poses = {pose(0.0F, 100.0F)};
  input.ppe_observations = {
      equipment(1, deepsafe::ppe::Equipment::Helmet,
                deepsafe::ppe::Evidence::Absent, {20, 10, 50, 40}),
  };
  const auto result = engine.process(input);
  CHECK(result.persons[0].pose.status ==
        deepsafe::fusion::LinkStatus::UnknownOccluded);
  CHECK(result.persons[0].helmet.status ==
        deepsafe::fusion::LinkStatus::UnknownOccluded);
  CHECK(result.persons[0].helmet.evidence == deepsafe::ppe::Evidence::Unknown);
}

void test_stale_pts_is_transactional() {
  deepsafe::fusion::FusionEngine engine({});
  auto first = frame(2, 10, 1000);
  first.canonical_tracks = {track(1, 0.0F, 100.0F)};
  CHECK(engine.process(first).status == deepsafe::fusion::FrameStatus::Ok);
  auto stale = first;
  stale.frame_num = 11;
  stale.pts_ns = 1000;
  const auto rejected = engine.process(stale);
  CHECK(rejected.status == deepsafe::fusion::FrameStatus::Stale);
  CHECK(rejected.diagnostics.stale_pts);
  CHECK(rejected.persons[0].helmet.evidence == deepsafe::ppe::Evidence::Unknown);
  auto next = first;
  next.frame_num = 11;
  next.pts_ns = 1001;
  CHECK(engine.process(next).status == deepsafe::fusion::FrameStatus::Ok);
}

void test_duplicate_meta_and_tracks_fail_closed() {
  deepsafe::fusion::FusionEngine engine({});
  auto duplicate_meta = frame(0, 1, 100);
  duplicate_meta.canonical_tracks = {track(1, 0.0F, 100.0F)};
  duplicate_meta.duplicate_pose_tensor_meta = true;
  auto result = engine.process(duplicate_meta);
  CHECK(result.status == deepsafe::fusion::FrameStatus::RejectedInput);
  CHECK(result.persons[0].helmet.evidence == deepsafe::ppe::Evidence::Unknown);

  auto duplicate_tracks = frame(1, 1, 100);
  duplicate_tracks.canonical_tracks = {
      track(4, 0.0F, 100.0F), track(4, 120.0F, 220.0F)};
  result = engine.process(duplicate_tracks);
  CHECK(result.status == deepsafe::fusion::FrameStatus::RejectedInput);
}

void test_track_conflict_has_no_second_person_fallback() {
  deepsafe::fusion::FusionEngine engine({});
  auto input = frame(0, 1, 100);
  input.canonical_tracks = {track(1, 0.0F, 100.0F)};
  input.poses = {pose(0.0F, 100.0F, 0.95F), pose(2.0F, 98.0F, 0.90F)};
  const auto result = engine.process(input);
  CHECK(result.persons[0].pose.status == deepsafe::fusion::LinkStatus::Matched);
  CHECK(result.diagnostics.unassociated_pose_count == 1U);
}

void test_empty_frame_and_deterministic_track_order() {
  deepsafe::fusion::FusionEngine empty_engine({});
  const auto empty = empty_engine.process(frame(0, 1, 100));
  CHECK(empty.status == deepsafe::fusion::FrameStatus::Ok);
  CHECK(empty.persons.empty());

  auto a = frame(1, 1, 100);
  a.canonical_tracks = {track(20, 120, 220), track(10, 0, 100)};
  a.poses = {pose(0, 100), pose(120, 220)};
  auto b = a;
  std::reverse(b.canonical_tracks.begin(), b.canonical_tracks.end());
  deepsafe::fusion::FusionEngine first({});
  deepsafe::fusion::FusionEngine second({});
  const auto left = first.process(a);
  const auto right = second.process(b);
  CHECK(left.persons.size() == right.persons.size());
  CHECK(left.persons[0].track_id == 10U);
  CHECK(right.persons[0].track_id == 10U);
  CHECK(left.persons[1].track_id == 20U);
  CHECK(right.persons[1].track_id == 20U);
  CHECK(std::fabs(left.persons[0].pose.association_score -
                  right.persons[0].pose.association_score) < 1.0e-6F);
}

}  // namespace

int main() {
  test_matched_pose_and_explicit_ppe();
  test_missing_ppe_is_unknown_never_absent();
  test_ambiguous_pose_and_ppe_are_unassociated();
  test_occluded_person_is_never_assigned();
  test_stale_pts_is_transactional();
  test_duplicate_meta_and_tracks_fail_closed();
  test_track_conflict_has_no_second_person_fallback();
  test_empty_frame_and_deterministic_track_order();
  if (failures != 0) {
    std::cerr << failures << " fusion core checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "fusion core checks passed\n";
  return EXIT_SUCCESS;
}
