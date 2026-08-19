#include "deepsafe_fusion/c_api.h"

#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "deepsafe_fusion/config.hpp"
#include "deepsafe_fusion/core.hpp"
#include "deepsafe_pose/deepstream_tensor_adapter.hpp"

namespace {

using deepsafe::fusion::CanonicalTrack;
using deepsafe::fusion::FrameInput;
using deepsafe::fusion::FrameResult;
using deepsafe::fusion::FrameStatus;
using deepsafe::fusion::FusionEngine;
using deepsafe::fusion::LinkStatus;

constexpr std::uint64_t kInvalidObservationId =
    deepsafe::ppe::kInvalidObservationId;

void set_error(char* buffer, std::size_t size, const std::string& message) {
  if (buffer == nullptr || size == 0U) {
    return;
  }
  const auto count = std::min(size - 1U, message.size());
  std::memcpy(buffer, message.data(), count);
  buffer[count] = '\0';
}

struct OwnedFrameMeta {
  DeepsafeFusionFrameMetaV1 view{};
  std::vector<DeepsafeFusionPersonV1> persons{};

  OwnedFrameMeta() = default;
  OwnedFrameMeta(const OwnedFrameMeta& other)
      : view(other.view), persons(other.persons) {
    sync();
  }
  OwnedFrameMeta& operator=(const OwnedFrameMeta& other) {
    if (this != &other) {
      view = other.view;
      persons = other.persons;
      sync();
    }
    return *this;
  }
  void sync() noexcept {
    view.person_count = static_cast<std::uint32_t>(persons.size());
    view.persons = persons.empty() ? nullptr : persons.data();
  }
};

DeepsafeFusionRectV1 wire_rect(const deepsafe::pose::RectF& value) noexcept {
  return {value.left, value.top, value.right, value.bottom};
}

DeepsafeFusionEquipmentV1 wire_equipment(
    const deepsafe::fusion::EquipmentLink& value) noexcept {
  return {
      static_cast<std::uint32_t>(value.status),
      static_cast<std::uint32_t>(value.evidence),
      value.confidence,
      value.observation_id,
  };
}

std::unique_ptr<OwnedFrameMeta> make_owned(const FrameResult& value) {
  auto output = std::make_unique<OwnedFrameMeta>();
  output->view.abi_version = DEEPSAFE_FUSION_ABI_VERSION_V1;
  output->view.struct_size = sizeof(DeepsafeFusionFrameMetaV1);
  output->view.frame_status = static_cast<std::uint32_t>(value.status);
  output->view.source_id = value.source_id;
  output->view.frame_num = value.frame_num;
  output->view.pts_ns = value.pts_ns;
  output->view.unassociated_pose_count =
      static_cast<std::uint32_t>(value.diagnostics.unassociated_pose_count);
  output->view.ambiguous_pose_count =
      static_cast<std::uint32_t>(value.diagnostics.ambiguous_pose_count);
  output->view.unassociated_ppe_count =
      static_cast<std::uint32_t>(value.diagnostics.unassociated_ppe_count);
  output->view.ambiguous_ppe_count =
      static_cast<std::uint32_t>(value.diagnostics.ambiguous_ppe_count);
  output->view.duplicate_pose_tensor_meta =
      value.diagnostics.duplicate_pose_tensor_meta ? 1U : 0U;
  output->view.duplicate_fusion_output_meta =
      value.diagnostics.duplicate_fusion_output_meta ? 1U : 0U;
  output->view.invalid_pose_tensor_meta =
      value.diagnostics.invalid_pose_tensor_meta ? 1U : 0U;
  output->view.stale_pts = value.diagnostics.stale_pts ? 1U : 0U;
  output->persons.reserve(value.persons.size());
  for (const auto& person : value.persons) {
    DeepsafeFusionPersonV1 wire{};
    wire.struct_size = sizeof(DeepsafeFusionPersonV1);
    wire.pose_link_status = static_cast<std::uint32_t>(person.pose.status);
    wire.track_id = person.track_id;
    wire.bbox = wire_rect(person.bbox);
    wire.occluded = person.occluded ? 1U : 0U;
    if (person.pose.status == LinkStatus::Matched) {
      wire.pose_score = person.pose.pose.score;
      wire.pose_association_score = person.pose.association_score;
      wire.pose_keypoint_count = DEEPSAFE_FUSION_COCO_KEYPOINTS_V1;
      for (std::size_t index = 0; index < person.pose.pose.keypoints.size(); ++index) {
        const auto& keypoint = person.pose.pose.keypoints[index];
        wire.keypoints[index] = {
            keypoint.x,
            keypoint.y,
            keypoint.confidence,
            static_cast<std::uint8_t>(keypoint.visible ? 1U : 0U),
            {0U, 0U, 0U},
        };
      }
    }
    wire.helmet = wire_equipment(person.helmet);
    wire.hi_vis = wire_equipment(person.hi_vis);
    output->persons.push_back(wire);
  }
  output->sync();
  return output;
}

deepsafe::pose::RectF pose_rect(const NvOSD_RectParams& value) noexcept {
  return {
      value.left,
      value.top,
      value.left + value.width,
      value.top + value.height,
  };
}

deepsafe::ppe::RectF ppe_rect(const NvOSD_RectParams& value) noexcept {
  return {
      value.left,
      value.top,
      value.left + value.width,
      value.top + value.height,
  };
}

deepsafe::ppe::RegionVisibility visibility(bool occluded) noexcept {
  return occluded
             ? deepsafe::ppe::RegionVisibility{
                   deepsafe::ppe::Visibility::NotVisible, 0.0F}
             : deepsafe::ppe::RegionVisibility{
                   deepsafe::ppe::Visibility::Visible, 1.0F};
}

bool touches_frame_boundary(const NvOSD_RectParams& value, float width,
                            float height) noexcept {
  constexpr float kEpsilon = 0.5F;
  return value.left <= kEpsilon || value.top <= kEpsilon ||
         value.left + value.width >= width - kEpsilon ||
         value.top + value.height >= height - kEpsilon;
}

std::uint32_t fusion_meta_type() {
  static char descriptor[] = "DEEPSAFE.FUSION.CANONICAL.V1";
  static const auto value = static_cast<std::uint32_t>(
      nvds_get_user_meta_type(descriptor));
  return value;
}

bool has_existing_output(const NvDsFrameMeta& frame) {
  for (auto* item = frame.frame_user_meta_list; item != nullptr; item = item->next) {
    const auto* meta = static_cast<const NvDsUserMeta*>(item->data);
    if (meta != nullptr &&
        static_cast<std::uint32_t>(meta->base_meta.meta_type) == fusion_meta_type()) {
      return true;
    }
  }
  return false;
}

FrameInput frame_input_from_meta(const NvDsFrameMeta& frame,
                                 const deepsafe::fusion::RuntimeConfig& config) {
  FrameInput input;
  input.source_id = frame.source_id;
  input.frame_num = frame.frame_num < 0 ? 0U : static_cast<std::uint64_t>(frame.frame_num);
  input.pts_ns = frame.buf_pts;
  input.duplicate_fusion_output_meta = has_existing_output(frame);
  if (frame.frame_num < 0 || frame.source_frame_width == 0U ||
      frame.source_frame_height == 0U) {
    input.invalid_pose_tensor_meta = true;
    return input;
  }

  std::size_t ppe_ordinal = 0U;
  for (auto* item = frame.obj_meta_list; item != nullptr; item = item->next) {
    const auto* object = static_cast<const NvDsObjectMeta*>(item->data);
    if (object == nullptr) {
      input.invalid_pose_tensor_meta = true;
      continue;
    }
    if (object->unique_component_id == static_cast<int>(config.person_gie_id) &&
        object->class_id == config.person_class_id &&
        object->object_id != UNTRACKED_OBJECT_ID) {
      CanonicalTrack track;
      track.track_id = object->object_id;
      track.bbox = pose_rect(object->rect_params);
      track.confirmed = true;
      const bool tracker_unreliable =
          !std::isfinite(object->tracker_confidence) ||
          object->tracker_confidence < config.minimum_tracker_confidence;
      track.occluded =
          tracker_unreliable ||
          object->rect_params.height < config.minimum_person_height_px ||
          touches_frame_boundary(object->rect_params,
                                 static_cast<float>(frame.source_frame_width),
                                 static_cast<float>(frame.source_frame_height));
      track.helmet_visibility = visibility(track.occluded);
      track.hi_vis_visibility = visibility(track.occluded);
      input.canonical_tracks.push_back(track);
      continue;
    }
    if (object->unique_component_id != static_cast<int>(config.ppe_gie_id)) {
      continue;
    }
    deepsafe::ppe::EquipmentObservation observation;
    bool recognized = true;
    if (object->class_id == config.ppe_helmet_present_class_id) {
      observation.equipment = deepsafe::ppe::Equipment::Helmet;
      observation.evidence = deepsafe::ppe::Evidence::Present;
    } else if (object->class_id == config.ppe_helmet_absent_class_id) {
      observation.equipment = deepsafe::ppe::Equipment::Helmet;
      observation.evidence = deepsafe::ppe::Evidence::Absent;
    } else if (object->class_id == config.ppe_hi_vis_present_class_id) {
      observation.equipment = deepsafe::ppe::Equipment::HiVis;
      observation.evidence = deepsafe::ppe::Evidence::Present;
    } else if (object->class_id == config.ppe_hi_vis_absent_class_id) {
      observation.equipment = deepsafe::ppe::Equipment::HiVis;
      observation.evidence = deepsafe::ppe::Evidence::Absent;
    } else {
      recognized = false;
    }
    if (recognized) {
      observation.observation_id = static_cast<std::uint64_t>(ppe_ordinal++);
      observation.bbox = ppe_rect(object->rect_params);
      observation.confidence = object->confidence;
      input.ppe_observations.push_back(observation);
    }
  }

  const NvDsInferTensorMeta* pose_meta = nullptr;
  std::size_t pose_meta_count = 0U;
  for (auto* item = frame.frame_user_meta_list; item != nullptr; item = item->next) {
    const auto* user_meta = static_cast<const NvDsUserMeta*>(item->data);
    if (user_meta == nullptr ||
        user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META ||
        user_meta->user_meta_data == nullptr) {
      continue;
    }
    const auto* candidate =
        static_cast<const NvDsInferTensorMeta*>(user_meta->user_meta_data);
    if (candidate->unique_id == config.pose_gie_id) {
      ++pose_meta_count;
      pose_meta = candidate;
    }
  }
  input.duplicate_pose_tensor_meta = pose_meta_count > 1U;
  if (pose_meta_count == 1U) {
    const auto adapted =
        deepsafe::pose::deepstream::tensor_view_from_meta(*pose_meta);
    if (!adapted.ok || pose_meta->network_info.width == 0U ||
        pose_meta->network_info.height == 0U) {
      input.invalid_pose_tensor_meta = true;
    } else {
      deepsafe::pose::DecoderConfig decoder_config;
      decoder_config.detection_threshold = config.pose_detection_threshold;
      decoder_config.keypoint_threshold = config.pose_keypoint_threshold;
      decoder_config.expected_class_id = config.person_class_id;
      decoder_config.expected_max_detections = config.max_pose_detections;
      const deepsafe::pose::Decoder decoder(decoder_config);
      const auto transform = deepsafe::pose::FrameTransform::symmetric_letterbox(
          static_cast<float>(frame.source_frame_width),
          static_cast<float>(frame.source_frame_height),
          static_cast<float>(pose_meta->network_info.width),
          static_cast<float>(pose_meta->network_info.height));
      const auto decoded = decoder.decode(adapted.tensor, {transform});
      if (!decoded.ok() || decoded.frames.size() != 1U) {
        input.invalid_pose_tensor_meta = true;
      } else {
        input.poses = decoded.frames.front();
      }
    }
  }
  return input;
}

}  // namespace

struct DeepsafeFusionHandle {
  explicit DeepsafeFusionHandle(deepsafe::fusion::LoadedConfig value)
      : loaded(std::move(value)), engine(loaded.runtime) {}

  deepsafe::fusion::LoadedConfig loaded;
  FusionEngine engine;
  std::mutex mutex;
};

extern "C" {

DeepsafeFusionHandle* deepsafe_fusion_create_v1(
    const char* config_path, const char* expected_sha256, char* error_buffer,
    size_t error_buffer_size) {
  if (config_path == nullptr || expected_sha256 == nullptr) {
    set_error(error_buffer, error_buffer_size,
              "config path and expected SHA-256 are mandatory");
    return nullptr;
  }
  auto loaded = deepsafe::fusion::load_hashed_config(config_path, expected_sha256);
  if (!loaded.ok) {
    set_error(error_buffer, error_buffer_size, loaded.error);
    return nullptr;
  }
  try {
    return new DeepsafeFusionHandle(std::move(loaded));
  } catch (const std::exception& error) {
    set_error(error_buffer, error_buffer_size, error.what());
    return nullptr;
  }
}

DeepsafeFusionHandle* deepsafe_fusion_create_from_env_v1(
    char* error_buffer, size_t error_buffer_size) {
  const auto* path = g_getenv("DEEPSAFE_FUSION_CONFIG");
  const auto* digest = g_getenv("DEEPSAFE_FUSION_CONFIG_SHA256");
  if (path == nullptr || *path == '\0' || digest == nullptr || *digest == '\0') {
    set_error(error_buffer, error_buffer_size,
              "DEEPSAFE_FUSION_CONFIG and DEEPSAFE_FUSION_CONFIG_SHA256 are mandatory");
    return nullptr;
  }
  return deepsafe_fusion_create_v1(path, digest, error_buffer, error_buffer_size);
}

void deepsafe_fusion_destroy_v1(DeepsafeFusionHandle* handle) { delete handle; }

uint32_t deepsafe_fusion_abi_version_v1(void) {
  return DEEPSAFE_FUSION_ABI_VERSION_V1;
}

uint32_t deepsafe_fusion_nvds_meta_type_v1(void) { return fusion_meta_type(); }

const DeepsafeFusionFrameMetaV1* deepsafe_fusion_frame_meta_view_v1(
    const void* user_meta_data) {
  if (user_meta_data == nullptr) {
    return nullptr;
  }
  const auto* owned = static_cast<const OwnedFrameMeta*>(user_meta_data);
  return &owned->view;
}

void* deepsafe_fusion_nvds_meta_copy_v1(void* data, void*) {
  if (data == nullptr) {
    return nullptr;
  }
  const auto* user_meta = static_cast<const NvDsUserMeta*>(data);
  const auto* source = static_cast<const OwnedFrameMeta*>(user_meta->user_meta_data);
  if (source == nullptr) {
    return nullptr;
  }
  try {
    return new OwnedFrameMeta(*source);
  } catch (...) {
    return nullptr;
  }
}

void deepsafe_fusion_nvds_meta_release_v1(void* data, void*) {
  if (data == nullptr) {
    return;
  }
  auto* user_meta = static_cast<NvDsUserMeta*>(data);
  delete static_cast<OwnedFrameMeta*>(user_meta->user_meta_data);
  user_meta->user_meta_data = nullptr;
}

int deepsafe_fusion_process_gst_buffer_v1(
    DeepsafeFusionHandle* handle, void* gst_buffer, char* error_buffer,
    size_t error_buffer_size) {
  if (handle == nullptr || gst_buffer == nullptr) {
    set_error(error_buffer, error_buffer_size, "fusion handle/GstBuffer is null");
    return -1;
  }
  auto* batch = gst_buffer_get_nvds_batch_meta(static_cast<GstBuffer*>(gst_buffer));
  if (batch == nullptr) {
    set_error(error_buffer, error_buffer_size, "GstBuffer has no NvDsBatchMeta");
    return -2;
  }
  std::lock_guard<std::mutex> guard(handle->mutex);
  nvds_acquire_meta_lock(batch);
  int return_code = 0;
  try {
    if (batch->num_frames_in_batch > handle->loaded.runtime.max_batch_size ||
        batch->max_frames_in_batch > handle->loaded.runtime.max_batch_size) {
      throw std::runtime_error("NvDsBatchMeta exceeds the exact batch-12 contract");
    }
    std::vector<NvDsFrameMeta*> frames;
    for (auto* item = batch->frame_meta_list; item != nullptr; item = item->next) {
      auto* frame = static_cast<NvDsFrameMeta*>(item->data);
      if (frame == nullptr) {
        throw std::runtime_error("NvDsBatchMeta contains a null frame meta");
      }
      if (has_existing_output(*frame)) {
        throw std::runtime_error("frame already contains canonical fusion user meta");
      }
      frames.push_back(frame);
    }
    if (frames.size() != batch->num_frames_in_batch) {
      throw std::runtime_error("NvDsBatchMeta frame list/count mismatch");
    }

    FusionEngine candidate_engine = handle->engine;
    std::vector<std::unique_ptr<OwnedFrameMeta>> outputs;
    outputs.reserve(frames.size());
    for (const auto* frame : frames) {
      auto input = frame_input_from_meta(*frame, handle->loaded.runtime);
      auto result = candidate_engine.process(input);
      outputs.push_back(make_owned(result));
    }
    for (std::size_t index = 0; index < frames.size(); ++index) {
      auto* user_meta = nvds_acquire_user_meta_from_pool(batch);
      if (user_meta == nullptr) {
        throw std::runtime_error("NvDs user-meta pool is exhausted");
      }
      user_meta->user_meta_data = outputs[index].release();
      user_meta->base_meta.meta_type =
          static_cast<NvDsMetaType>(fusion_meta_type());
      user_meta->base_meta.copy_func = deepsafe_fusion_nvds_meta_copy_v1;
      user_meta->base_meta.release_func = deepsafe_fusion_nvds_meta_release_v1;
      nvds_add_user_meta_to_frame(frames[index], user_meta);
    }
    handle->engine = std::move(candidate_engine);
  } catch (const std::exception& error) {
    set_error(error_buffer, error_buffer_size, error.what());
    return_code = -3;
  } catch (...) {
    set_error(error_buffer, error_buffer_size, "unknown fusion metadata failure");
    return_code = -4;
  }
  nvds_release_meta_lock(batch);
  return return_code;
}

}  // extern "C"
