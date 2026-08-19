#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define DEEPSAFE_FUSION_API __declspec(dllexport)
#else
#define DEEPSAFE_FUSION_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DEEPSAFE_FUSION_ABI_VERSION_V1 0x00010000U
#define DEEPSAFE_FUSION_COCO_KEYPOINTS_V1 17U

typedef struct DeepsafeFusionHandle DeepsafeFusionHandle;

typedef enum DeepsafeFusionFrameStatusV1 {
  DEEPSAFE_FUSION_FRAME_OK_V1 = 0,
  DEEPSAFE_FUSION_FRAME_STALE_V1 = 1,
  DEEPSAFE_FUSION_FRAME_REJECTED_INPUT_V1 = 2
} DeepsafeFusionFrameStatusV1;

typedef enum DeepsafeFusionLinkStatusV1 {
  DEEPSAFE_FUSION_LINK_MATCHED_V1 = 0,
  DEEPSAFE_FUSION_LINK_UNKNOWN_NO_OBSERVATION_V1 = 1,
  DEEPSAFE_FUSION_LINK_UNKNOWN_NO_PERSON_V1 = 2,
  DEEPSAFE_FUSION_LINK_UNKNOWN_AMBIGUOUS_V1 = 3,
  DEEPSAFE_FUSION_LINK_UNKNOWN_OCCLUDED_V1 = 4,
  DEEPSAFE_FUSION_LINK_UNKNOWN_DUPLICATE_V1 = 5,
  DEEPSAFE_FUSION_LINK_UNKNOWN_GEOMETRY_V1 = 6,
  DEEPSAFE_FUSION_LINK_UNKNOWN_TRACK_CONFLICT_V1 = 7,
  DEEPSAFE_FUSION_LINK_UNKNOWN_STALE_V1 = 8,
  DEEPSAFE_FUSION_LINK_UNKNOWN_INVALID_INPUT_V1 = 9
} DeepsafeFusionLinkStatusV1;

typedef enum DeepsafeFusionEvidenceV1 {
  DEEPSAFE_FUSION_EVIDENCE_PRESENT_V1 = 0,
  DEEPSAFE_FUSION_EVIDENCE_ABSENT_V1 = 1,
  DEEPSAFE_FUSION_EVIDENCE_UNKNOWN_V1 = 2
} DeepsafeFusionEvidenceV1;

typedef struct DeepsafeFusionRectV1 {
  float left;
  float top;
  float right;
  float bottom;
} DeepsafeFusionRectV1;

typedef struct DeepsafeFusionKeypointV1 {
  float x;
  float y;
  float confidence;
  uint8_t visible;
  uint8_t reserved[3];
} DeepsafeFusionKeypointV1;

typedef struct DeepsafeFusionEquipmentV1 {
  uint32_t link_status;
  uint32_t evidence;
  float confidence;
  uint64_t observation_id;
} DeepsafeFusionEquipmentV1;

typedef struct DeepsafeFusionPersonV1 {
  uint32_t struct_size;
  uint32_t pose_link_status;
  uint64_t track_id;
  DeepsafeFusionRectV1 bbox;
  uint8_t occluded;
  uint8_t reserved0[3];
  float pose_score;
  float pose_association_score;
  uint32_t pose_keypoint_count;
  uint32_t reserved1;
  DeepsafeFusionKeypointV1 keypoints[DEEPSAFE_FUSION_COCO_KEYPOINTS_V1];
  DeepsafeFusionEquipmentV1 helmet;
  DeepsafeFusionEquipmentV1 hi_vis;
} DeepsafeFusionPersonV1;

typedef struct DeepsafeFusionFrameMetaV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t frame_status;
  uint32_t source_id;
  uint64_t frame_num;
  uint64_t pts_ns;
  uint32_t person_count;
  uint32_t unassociated_pose_count;
  uint32_t ambiguous_pose_count;
  uint32_t unassociated_ppe_count;
  uint32_t ambiguous_ppe_count;
  uint8_t duplicate_pose_tensor_meta;
  uint8_t duplicate_fusion_output_meta;
  uint8_t invalid_pose_tensor_meta;
  uint8_t stale_pts;
  const DeepsafeFusionPersonV1* persons;
} DeepsafeFusionFrameMetaV1;

// Creation is fail-closed: path and lowercase SHA-256 are both mandatory.
DEEPSAFE_FUSION_API DeepsafeFusionHandle* deepsafe_fusion_create_v1(
    const char* config_path, const char* expected_sha256,
    char* error_buffer, size_t error_buffer_size);

// Reads DEEPSAFE_FUSION_CONFIG and DEEPSAFE_FUSION_CONFIG_SHA256.
DEEPSAFE_FUSION_API DeepsafeFusionHandle* deepsafe_fusion_create_from_env_v1(
    char* error_buffer, size_t error_buffer_size);

DEEPSAFE_FUSION_API void deepsafe_fusion_destroy_v1(
    DeepsafeFusionHandle* handle);

// gst_buffer is a GstBuffer*. Return 0 for processed/diagnostic output and a
// negative value for a fatal batch/config/metadata-pool failure.
DEEPSAFE_FUSION_API int deepsafe_fusion_process_gst_buffer_v1(
    DeepsafeFusionHandle* handle, void* gst_buffer,
    char* error_buffer, size_t error_buffer_size);

DEEPSAFE_FUSION_API uint32_t deepsafe_fusion_abi_version_v1(void);
DEEPSAFE_FUSION_API uint32_t deepsafe_fusion_nvds_meta_type_v1(void);

// user_meta_data is the opaque NvDsUserMeta::user_meta_data pointer.
DEEPSAFE_FUSION_API const DeepsafeFusionFrameMetaV1*
deepsafe_fusion_frame_meta_view_v1(const void* user_meta_data);

// Exported for NvDsUserMeta ownership tests and registered directly as the
// DeepStream copy/release callbacks by the adapter.
DEEPSAFE_FUSION_API void* deepsafe_fusion_nvds_meta_copy_v1(
    void* data, void* user_data);
DEEPSAFE_FUSION_API void deepsafe_fusion_nvds_meta_release_v1(
    void* data, void* user_data);

#ifdef __cplusplus
}
#endif
