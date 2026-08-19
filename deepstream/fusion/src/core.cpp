#include "deepsafe_fusion/core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace deepsafe::fusion {
namespace {

deepsafe::ppe::RectF ppe_rect(const deepsafe::pose::RectF& value) noexcept {
  return {value.left, value.top, value.right, value.bottom};
}

bool finite_unit(float value) noexcept {
  return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
}

LinkStatus ppe_status(deepsafe::ppe::Evidence evidence,
                      deepsafe::ppe::EvidenceReason reason) noexcept {
  if (evidence == deepsafe::ppe::Evidence::Unknown) {
    return reason == deepsafe::ppe::EvidenceReason::InsufficientVisibility
               ? LinkStatus::UnknownOccluded
               : LinkStatus::UnknownNoObservation;
  }
  return LinkStatus::Matched;
}

FrameResult base_result(const FrameInput& input) {
  FrameResult result;
  result.source_id = input.source_id;
  result.frame_num = input.frame_num;
  result.pts_ns = input.pts_ns;
  result.diagnostics.canonical_track_count = input.canonical_tracks.size();
  result.diagnostics.pose_input_count = input.poses.size();
  result.diagnostics.ppe_input_count = input.ppe_observations.size();
  result.diagnostics.duplicate_pose_tensor_meta =
      input.duplicate_pose_tensor_meta;
  result.diagnostics.invalid_pose_tensor_meta = input.invalid_pose_tensor_meta;
  result.diagnostics.duplicate_fusion_output_meta =
      input.duplicate_fusion_output_meta;

  std::vector<const CanonicalTrack*> ordered;
  ordered.reserve(input.canonical_tracks.size());
  for (const auto& track : input.canonical_tracks) {
    ordered.push_back(&track);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const CanonicalTrack* lhs, const CanonicalTrack* rhs) {
              return lhs->track_id < rhs->track_id;
            });
  for (const auto* track : ordered) {
    PersonResult person;
    person.track_id = track->track_id;
    person.bbox = track->bbox;
    person.occluded = track->occluded;
    if (track->occluded) {
      person.pose.status = LinkStatus::UnknownOccluded;
      person.helmet.status = LinkStatus::UnknownOccluded;
      person.hi_vis.status = LinkStatus::UnknownOccluded;
    }
    result.persons.push_back(person);
  }
  return result;
}

void reject(FrameResult& result, std::string error, LinkStatus status) {
  result.status = FrameStatus::RejectedInput;
  result.diagnostics.error = std::move(error);
  result.diagnostics.unassociated_pose_count =
      result.diagnostics.pose_input_count;
  result.diagnostics.unassociated_ppe_count =
      result.diagnostics.ppe_input_count;
  for (auto& person : result.persons) {
    person.pose.status = person.occluded ? LinkStatus::UnknownOccluded : status;
    person.helmet.status =
        person.occluded ? LinkStatus::UnknownOccluded : status;
    person.hi_vis.status =
        person.occluded ? LinkStatus::UnknownOccluded : status;
    person.helmet.evidence = deepsafe::ppe::Evidence::Unknown;
    person.hi_vis.evidence = deepsafe::ppe::Evidence::Unknown;
  }
}

}  // namespace

bool RuntimeConfig::valid() const noexcept {
  const std::set<int> ppe_classes{
      ppe_helmet_present_class_id,
      ppe_helmet_absent_class_id,
      ppe_hi_vis_present_class_id,
      ppe_hi_vis_absent_class_id,
  };
  return person_gie_id == kCanonicalPersonGieId && pose_gie_id == kPoseGieId &&
         ppe_gie_id == kPpeGieId && person_class_id >= 0 &&
         ppe_classes.size() == 4U && *ppe_classes.begin() >= 0 &&
         max_sources == kMaximumSources && max_batch_size == kMaximumSources &&
         max_persons_per_frame > 0U && max_persons_per_frame <= 4096U &&
         max_pose_detections == deepsafe::pose::kDefaultMaxDetections &&
         max_ppe_observations > 0U && max_ppe_observations <= 8192U &&
         finite_unit(pose_detection_threshold) &&
         finite_unit(pose_keypoint_threshold) &&
         finite_unit(pose_ambiguity_margin) && finite_unit(minimum_pose_iou) &&
         finite_unit(minimum_pose_coverage) &&
         std::isfinite(minimum_person_height_px) &&
         minimum_person_height_px > 0.0F &&
         std::isfinite(minimum_tracker_confidence) &&
         minimum_tracker_confidence >= 0.0F &&
         minimum_tracker_confidence <= 1.0F && ppe_association.valid();
}

FusionEngine::FusionEngine(RuntimeConfig config) : config_(std::move(config)) {}

FrameResult FusionEngine::process(const FrameInput& input) {
  auto result = base_result(input);
  if (!config_.valid()) {
    reject(result, "fusion runtime configuration is invalid",
           LinkStatus::UnknownInvalidInput);
    return result;
  }
  if (input.source_id >= config_.max_sources ||
      input.canonical_tracks.size() > config_.max_persons_per_frame ||
      input.poses.size() > config_.max_pose_detections ||
      input.ppe_observations.size() > config_.max_ppe_observations) {
    reject(result, "frame source/count limits are invalid",
           LinkStatus::UnknownInvalidInput);
    return result;
  }

  std::set<std::uint64_t> track_ids;
  for (const auto& track : input.canonical_tracks) {
    if (track.track_id == deepsafe::pose::kUntrackedObjectId ||
        !track.bbox.valid() || !track.confirmed ||
        !track.helmet_visibility.valid() || !track.hi_vis_visibility.valid() ||
        !track_ids.insert(track.track_id).second) {
      reject(result, "canonical person tracks are invalid or duplicated",
             LinkStatus::UnknownDuplicate);
      return result;
    }
  }
  if (input.duplicate_pose_tensor_meta || input.invalid_pose_tensor_meta ||
      input.duplicate_fusion_output_meta) {
    reject(result, "duplicate or invalid frame metadata rejected",
           LinkStatus::UnknownDuplicate);
    return result;
  }

  auto& watermark = watermarks_[input.source_id];
  if (config_.reject_nonmonotonic_pts && watermark.initialized &&
      (input.frame_num <= watermark.frame_num || input.pts_ns <= watermark.pts_ns)) {
    result.status = FrameStatus::Stale;
    result.diagnostics.stale_pts = true;
    result.diagnostics.error = "frame_num/PTS is not strictly increasing";
    result.diagnostics.unassociated_pose_count = input.poses.size();
    result.diagnostics.unassociated_ppe_count = input.ppe_observations.size();
    for (auto& person : result.persons) {
      person.pose.status = LinkStatus::UnknownStale;
      person.helmet.status = LinkStatus::UnknownStale;
      person.hi_vis.status = LinkStatus::UnknownStale;
      person.helmet.evidence = deepsafe::ppe::Evidence::Unknown;
      person.hi_vis.evidence = deepsafe::ppe::Evidence::Unknown;
    }
    return result;
  }

  std::map<std::uint64_t, std::size_t> person_by_track;
  for (std::size_t index = 0; index < result.persons.size(); ++index) {
    person_by_track.emplace(result.persons[index].track_id, index);
  }

  struct PoseCandidate {
    std::size_t pose_index{0U};
    std::uint64_t track_id{deepsafe::pose::kUntrackedObjectId};
    float score{0.0F};
  };
  std::vector<PoseCandidate> proposals;
  proposals.reserve(input.poses.size());
  std::vector<LinkStatus> pose_failure(input.poses.size(),
                                       LinkStatus::UnknownNoPerson);
  deepsafe::pose::AssociationConfig pose_config;
  pose_config.expected_person_class_id = config_.person_class_id;
  pose_config.minimum_iou = config_.minimum_pose_iou;
  pose_config.minimum_pose_coverage = config_.minimum_pose_coverage;
  pose_config.require_confirmed_tracks = true;

  for (std::size_t pose_index = 0; pose_index < input.poses.size(); ++pose_index) {
    std::vector<PoseCandidate> candidates;
    for (const auto& track : input.canonical_tracks) {
      if (track.occluded) {
        continue;
      }
      deepsafe::pose::PersonTrack core_track;
      core_track.track_id = track.track_id;
      core_track.bbox = track.bbox;
      core_track.class_id = config_.person_class_id;
      core_track.confirmed = track.confirmed;
      const auto association = deepsafe::pose::associate_poses_to_person_tracks(
          {input.poses[pose_index]}, {core_track}, pose_config);
      if (!association.ok) {
        reject(result, "pose association core rejected input: " + association.error,
               LinkStatus::UnknownInvalidInput);
        return result;
      }
      if (!association.matches.empty()) {
        candidates.push_back(
            {pose_index, track.track_id,
             association.matches.front().association_score});
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const PoseCandidate& lhs, const PoseCandidate& rhs) {
                if (lhs.score != rhs.score) {
                  return lhs.score > rhs.score;
                }
                return lhs.track_id < rhs.track_id;
              });
    if (candidates.empty()) {
      pose_failure[pose_index] = input.canonical_tracks.empty()
                                     ? LinkStatus::UnknownNoPerson
                                     : LinkStatus::UnknownGeometry;
      continue;
    }
    if (candidates.size() > 1U &&
        candidates[0].score - candidates[1].score <=
            config_.pose_ambiguity_margin) {
      pose_failure[pose_index] = LinkStatus::UnknownAmbiguous;
      ++result.diagnostics.ambiguous_pose_count;
      continue;
    }
    proposals.push_back(candidates.front());
  }

  std::sort(proposals.begin(), proposals.end(),
            [](const PoseCandidate& lhs, const PoseCandidate& rhs) {
              if (lhs.track_id != rhs.track_id) {
                return lhs.track_id < rhs.track_id;
              }
              if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
              }
              return lhs.pose_index < rhs.pose_index;
            });
  std::set<std::uint64_t> occupied_tracks;
  std::size_t pose_matches = 0U;
  for (const auto& proposal : proposals) {
    if (!occupied_tracks.insert(proposal.track_id).second) {
      pose_failure[proposal.pose_index] = LinkStatus::UnknownTrackConflict;
      continue;
    }
    auto& person = result.persons[person_by_track.at(proposal.track_id)];
    person.pose.status = LinkStatus::Matched;
    person.pose.association_score = proposal.score;
    person.pose.pose_index = proposal.pose_index;
    person.pose.pose = input.poses[proposal.pose_index];
    ++pose_matches;
  }
  result.diagnostics.unassociated_pose_count = input.poses.size() - pose_matches;

  std::vector<deepsafe::ppe::PersonTrack> ppe_tracks;
  ppe_tracks.reserve(input.canonical_tracks.size());
  for (const auto& track : input.canonical_tracks) {
    deepsafe::ppe::PersonTrack value;
    value.track_id = track.track_id;
    value.bbox = ppe_rect(track.bbox);
    value.confirmed = track.confirmed;
    value.helmet_region = track.helmet_visibility;
    value.hi_vis_region = track.hi_vis_visibility;
    ppe_tracks.push_back(value);
  }
  const auto ppe_result = deepsafe::ppe::associate_equipment_to_person_tracks(
      ppe_tracks, input.ppe_observations, config_.ppe_association);
  if (!ppe_result.ok) {
    reject(result, "PPE association core rejected input: " + ppe_result.error,
           LinkStatus::UnknownInvalidInput);
    return result;
  }

  for (const auto& evidence : ppe_result.track_evidence) {
    auto& person = result.persons[person_by_track.at(evidence.person_track_id)];
    EquipmentLink* link = evidence.equipment == deepsafe::ppe::Equipment::Helmet
                              ? &person.helmet
                              : &person.hi_vis;
    link->evidence = evidence.evidence;
    link->confidence = evidence.confidence;
    link->observation_id = evidence.observation_id.value_or(
        deepsafe::ppe::kInvalidObservationId);
    link->status = person.occluded
                       ? LinkStatus::UnknownOccluded
                       : ppe_status(evidence.evidence, evidence.reason);
    if (person.occluded) {
      link->evidence = deepsafe::ppe::Evidence::Unknown;
      link->confidence = 0.0F;
      link->observation_id = deepsafe::ppe::kInvalidObservationId;
    }
  }

  std::size_t ppe_matches = 0U;
  for (const auto& record : ppe_result.observations) {
    if (record.status == deepsafe::ppe::AssociationStatus::Matched) {
      ++ppe_matches;
    } else if (record.status == deepsafe::ppe::AssociationStatus::Ambiguous) {
      ++result.diagnostics.ambiguous_ppe_count;
      for (const auto track_id : record.candidate_track_ids) {
        const auto person_iterator = person_by_track.find(track_id);
        if (person_iterator == person_by_track.end()) {
          continue;
        }
        auto& person = result.persons[person_iterator->second];
        auto& link = record.equipment == deepsafe::ppe::Equipment::Helmet
                         ? person.helmet
                         : person.hi_vis;
        if (link.status != LinkStatus::Matched && !person.occluded) {
          link.status = LinkStatus::UnknownAmbiguous;
          link.evidence = deepsafe::ppe::Evidence::Unknown;
        }
      }
    }
  }
  result.diagnostics.unassociated_ppe_count =
      input.ppe_observations.size() - ppe_matches;

  watermark.frame_num = input.frame_num;
  watermark.pts_ns = input.pts_ns;
  watermark.initialized = true;
  return result;
}

void FusionEngine::reset_source(std::uint32_t source_id) noexcept {
  watermarks_.erase(source_id);
}

void FusionEngine::reset() noexcept { watermarks_.clear(); }

}  // namespace deepsafe::fusion
