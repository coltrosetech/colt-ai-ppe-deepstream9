#include "deepsafe_fusion/c_api.h"

#include <gst/gst.h>
#include <gstnvdsinfer.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "deepsafe_pose/decoder.hpp"

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

struct TensorFixture {
  explicit TensorFixture(bool half_precision) : half(half_precision) {
    const std::size_t count = 300U * 57U;
    if (half) {
      fp16.assign(count, 0U);
      host = fp16.data();
      layer.dataType = HALF;
    } else {
      fp32.assign(count, 0.0F);
      host = fp32.data();
      layer.dataType = FLOAT;
    }
    layer.inferDims.numDims = 2U;
    layer.inferDims.d[0] = 300U;
    layer.inferDims.d[1] = 57U;
    layer.inferDims.numElements = count;
    host_array[0] = host;
    meta.unique_id = 2U;
    meta.num_output_layers = 1U;
    meta.output_layers_info = &layer;
    meta.out_buf_ptrs_host = host_array.data();
    meta.network_info.width = 640U;
    meta.network_info.height = 640U;
    meta.network_info.channels = 3U;
    meta.maintain_aspect_ratio = TRUE;
    meta.symmetric_padding = TRUE;
  }

  void set(std::size_t index, float value) {
    if (half) {
      fp16.at(index) = deepsafe::pose::float_to_half_bits(value);
    } else {
      fp32.at(index) = value;
    }
  }

  void one_pose() {
    set(0, 100.0F);
    set(1, 100.0F);
    set(2, 300.0F);
    set(3, 500.0F);
    set(4, 0.95F);
    set(5, 0.0F);
    for (std::size_t index = 0; index < 17U; ++index) {
      set(6U + index * 3U, 180.0F + static_cast<float>(index));
      set(7U + index * 3U, 180.0F + static_cast<float>(index * 5U));
      set(8U + index * 3U, 0.90F);
    }
  }

  bool half{false};
  std::vector<float> fp32{};
  std::vector<std::uint16_t> fp16{};
  void* host{nullptr};
  std::array<void*, 1> host_array{};
  NvDsInferLayerInfo layer{};
  NvDsInferTensorMeta meta{};
};

struct BufferFixture {
  BufferFixture() {
    buffer = gst_buffer_new();
    batch = nvds_create_batch_meta(12U);
    auto* gst_meta = gst_buffer_add_nvds_meta(
        buffer, batch, nullptr, nvds_batch_meta_copy_func,
        nvds_batch_meta_release_func);
    CHECK(gst_meta != nullptr);
    if (gst_meta != nullptr) {
      gst_meta->meta_type = NVDS_BATCH_GST_META;
    }
  }

  ~BufferFixture() {
    if (buffer != nullptr) {
      gst_buffer_unref(buffer);
    }
  }

  NvDsFrameMeta* add_frame(std::uint32_t source, std::uint64_t frame_num,
                           std::uint64_t pts) {
    auto* frame = nvds_acquire_frame_meta_from_pool(batch);
    CHECK(frame != nullptr);
    frame->source_id = source;
    frame->frame_num = static_cast<gint>(frame_num);
    frame->buf_pts = pts;
    frame->source_frame_width = 640U;
    frame->source_frame_height = 640U;
    frame->pipeline_width = 640U;
    frame->pipeline_height = 640U;
    nvds_add_frame_meta_to_batch(batch, frame);
    return frame;
  }

  NvDsObjectMeta* add_object(NvDsFrameMeta* frame, int gie, int class_id,
                             std::uint64_t object_id, float left, float top,
                             float width, float height, float confidence,
                             float tracker_confidence) {
    auto* object = nvds_acquire_obj_meta_from_pool(batch);
    CHECK(object != nullptr);
    object->unique_component_id = gie;
    object->class_id = class_id;
    object->object_id = object_id;
    object->rect_params.left = left;
    object->rect_params.top = top;
    object->rect_params.width = width;
    object->rect_params.height = height;
    object->confidence = confidence;
    object->tracker_confidence = tracker_confidence;
    nvds_add_obj_meta_to_frame(frame, object, nullptr);
    return object;
  }

  TensorFixture* add_pose_tensor(NvDsFrameMeta* frame, bool half) {
    auto fixture = std::make_unique<TensorFixture>(half);
    fixture->one_pose();
    auto* pointer = fixture.get();
    tensors.push_back(std::move(fixture));
    auto* user_meta = nvds_acquire_user_meta_from_pool(batch);
    CHECK(user_meta != nullptr);
    user_meta->base_meta.meta_type = NVDSINFER_TENSOR_OUTPUT_META;
    user_meta->user_meta_data = &pointer->meta;
    nvds_add_user_meta_to_frame(frame, user_meta);
    return pointer;
  }

  const DeepsafeFusionFrameMetaV1* output(NvDsFrameMeta* frame,
                                          NvDsUserMeta** owner = nullptr) {
    const auto meta_type = deepsafe_fusion_nvds_meta_type_v1();
    for (auto* item = frame->frame_user_meta_list; item != nullptr;
         item = item->next) {
      auto* user_meta = static_cast<NvDsUserMeta*>(item->data);
      if (user_meta != nullptr &&
          static_cast<std::uint32_t>(user_meta->base_meta.meta_type) ==
              meta_type) {
        if (owner != nullptr) {
          *owner = user_meta;
        }
        return deepsafe_fusion_frame_meta_view_v1(user_meta->user_meta_data);
      }
    }
    return nullptr;
  }

  GstBuffer* buffer{nullptr};
  NvDsBatchMeta* batch{nullptr};
  std::vector<std::unique_ptr<TensorFixture>> tensors{};
};

std::unique_ptr<DeepsafeFusionHandle, decltype(&deepsafe_fusion_destroy_v1)>
make_handle(const std::string& path, const std::string& sha256) {
  std::array<char, 512> error{};
  auto* value = deepsafe_fusion_create_v1(path.c_str(), sha256.c_str(),
                                           error.data(), error.size());
  if (value == nullptr) {
    std::cerr << "handle creation failed: " << error.data() << '\n';
    ++failures;
  }
  return {value, deepsafe_fusion_destroy_v1};
}

void add_person_and_helmet(BufferFixture& fixture, NvDsFrameMeta* frame) {
  fixture.add_object(frame, 1, 0, 77U, 100, 100, 200, 400, 0.95F, 0.90F);
  fixture.add_object(frame, 3, 0, UNTRACKED_OBJECT_ID,
                     150, 120, 100, 60, 0.90F, -0.1F);
}

void test_fp32_fp16_and_copy(const std::string& config,
                             const std::string& sha256) {
  auto handle = make_handle(config, sha256);
  for (const bool half : {false, true}) {
    BufferFixture fixture;
    auto* frame = fixture.add_frame(half ? 1U : 0U, 1U, 100U);
    add_person_and_helmet(fixture, frame);
    fixture.add_pose_tensor(frame, half);
    std::array<char, 512> error{};
    CHECK(deepsafe_fusion_process_gst_buffer_v1(
              handle.get(), fixture.buffer, error.data(), error.size()) == 0);
    NvDsUserMeta* owner = nullptr;
    const auto* output = fixture.output(frame, &owner);
    CHECK(output != nullptr);
    CHECK(output->abi_version == DEEPSAFE_FUSION_ABI_VERSION_V1);
    CHECK(output->person_count == 1U);
    CHECK(output->persons[0].track_id == 77U);
    CHECK(output->persons[0].pose_link_status ==
          DEEPSAFE_FUSION_LINK_MATCHED_V1);
    CHECK(output->persons[0].pose_keypoint_count == 17U);
    CHECK(output->persons[0].helmet.evidence ==
          DEEPSAFE_FUSION_EVIDENCE_PRESENT_V1);
    CHECK(output->persons[0].hi_vis.evidence ==
          DEEPSAFE_FUSION_EVIDENCE_UNKNOWN_V1);

    auto* copied_payload = deepsafe_fusion_nvds_meta_copy_v1(owner, nullptr);
    CHECK(copied_payload != nullptr);
    const auto* copied = deepsafe_fusion_frame_meta_view_v1(copied_payload);
    CHECK(copied != output);
    CHECK(copied->person_count == output->person_count);
    CHECK(copied->persons != output->persons);
    CHECK(copied->persons[0].track_id == output->persons[0].track_id);
    NvDsUserMeta copied_owner{};
    copied_owner.user_meta_data = copied_payload;
    deepsafe_fusion_nvds_meta_release_v1(&copied_owner, nullptr);
    CHECK(copied_owner.user_meta_data == nullptr);
  }
}

void test_batch12_partial_empty_stale_and_duplicate(
    const std::string& config, const std::string& sha256) {
  auto handle = make_handle(config, sha256);
  BufferFixture twelve;
  std::vector<NvDsFrameMeta*> frames;
  for (std::uint32_t source = 0U; source < 12U; ++source) {
    frames.push_back(twelve.add_frame(source, 1U, 100U));
  }
  std::array<char, 512> error{};
  CHECK(deepsafe_fusion_process_gst_buffer_v1(
            handle.get(), twelve.buffer, error.data(), error.size()) == 0);
  for (auto* frame : frames) {
    const auto* output = twelve.output(frame);
    CHECK(output != nullptr);
    CHECK(output->person_count == 0U);
    CHECK(output->frame_status == DEEPSAFE_FUSION_FRAME_OK_V1);
  }

  BufferFixture partial;
  auto* next = partial.add_frame(0U, 2U, 200U);
  add_person_and_helmet(partial, next);
  CHECK(deepsafe_fusion_process_gst_buffer_v1(
            handle.get(), partial.buffer, error.data(), error.size()) == 0);
  CHECK(partial.output(next)->person_count == 1U);

  BufferFixture stale;
  auto* stale_frame = stale.add_frame(0U, 3U, 200U);
  add_person_and_helmet(stale, stale_frame);
  CHECK(deepsafe_fusion_process_gst_buffer_v1(
            handle.get(), stale.buffer, error.data(), error.size()) == 0);
  const auto* stale_output = stale.output(stale_frame);
  CHECK(stale_output->frame_status == DEEPSAFE_FUSION_FRAME_STALE_V1);
  CHECK(stale_output->stale_pts == 1U);
  CHECK(stale_output->persons[0].helmet.evidence ==
        DEEPSAFE_FUSION_EVIDENCE_UNKNOWN_V1);

  BufferFixture duplicate;
  auto* duplicate_frame = duplicate.add_frame(0U, 3U, 300U);
  duplicate.add_pose_tensor(duplicate_frame, false);
  duplicate.add_pose_tensor(duplicate_frame, false);
  CHECK(deepsafe_fusion_process_gst_buffer_v1(
            handle.get(), duplicate.buffer, error.data(), error.size()) == 0);
  const auto* duplicate_output = duplicate.output(duplicate_frame);
  CHECK(duplicate_output->frame_status ==
        DEEPSAFE_FUSION_FRAME_REJECTED_INPUT_V1);
  CHECK(duplicate_output->duplicate_pose_tensor_meta == 1U);

  CHECK(deepsafe_fusion_process_gst_buffer_v1(
            handle.get(), duplicate.buffer, error.data(), error.size()) < 0);
}

void test_config_hash_and_environment_are_mandatory(
    const std::string& config, const std::string& sha256) {
  std::array<char, 512> error{};
  CHECK(deepsafe_fusion_create_v1(config.c_str(), std::string(64U, '0').c_str(),
                                  error.data(), error.size()) == nullptr);
  g_unsetenv("DEEPSAFE_FUSION_CONFIG");
  g_unsetenv("DEEPSAFE_FUSION_CONFIG_SHA256");
  CHECK(deepsafe_fusion_create_from_env_v1(error.data(), error.size()) == nullptr);
  g_setenv("DEEPSAFE_FUSION_CONFIG", config.c_str(), TRUE);
  g_setenv("DEEPSAFE_FUSION_CONFIG_SHA256", sha256.c_str(), TRUE);
  auto* handle = deepsafe_fusion_create_from_env_v1(error.data(), error.size());
  CHECK(handle != nullptr);
  deepsafe_fusion_destroy_v1(handle);
}

}  // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  if (argc != 3) {
    std::cerr << "usage: test_ds9_metadata CONFIG SHA256\n";
    return EXIT_FAILURE;
  }
  const std::string config(argv[1]);
  const std::string sha256(argv[2]);
  CHECK(deepsafe_fusion_abi_version_v1() == DEEPSAFE_FUSION_ABI_VERSION_V1);
  test_config_hash_and_environment_are_mandatory(config, sha256);
  test_fp32_fp16_and_copy(config, sha256);
  test_batch12_partial_empty_stale_and_duplicate(config, sha256);
  gst_deinit();
  if (failures != 0) {
    std::cerr << failures << " DS9 metadata checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "DS9 metadata checks passed\n";
  return EXIT_SUCCESS;
}
