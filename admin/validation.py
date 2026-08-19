"""Bounded, read-only projection of validation campaign artifacts.

The validation runners deliberately remain standalone command-line tools.  This
module only reads a small allow-list of their summary artifacts; it cannot start,
retry, or stop GPU work.
"""

from __future__ import annotations

import json
import hashlib
import errno
import math
import os
import re
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .ds91_status import load_deepstream91_static_status
from .ds91_native_status import load_deepstream91_native_status
from .ds91_engine_builder_r1c3_status import load_ds91_engine_builder_r1c3_status
from .ds91_preflight_r1_status import load_ds91_preflight_r1_status
from .ds91_preflight_r2_status import load_ds91_preflight_r2_status
from .driver595_r4_status import load_driver595_r4_status
from .driver595_r7_status import load_driver595_r7_status
from .model_gate_status import load_person_r14i_status, load_ppe_a32_status
from .pose_r13i_status import load_pose_r13i_status
from .ppe_construction_status import load_construction_ppe_status
from .ppe_safetyvision_status import load_safetyvision_challenger_status
from .gpu_lease_v5_status import load_gpu_lease_v5_status
from urllib.parse import quote

from validation.product_acceptance_policy import (
    APPROVED_POLICY_FINGERPRINT_SHA256,
    AcceptancePolicyError,
    load_approved_policy,
)
from validation.strict_json import loads as strict_json_loads


DEFAULT_VALIDATION_ROOT = Path("/workspace/validation-results")
DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024
HARD_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
HARD_MAX_ARTIFACT_OVERRIDE_BYTES = 32 * 1024 * 1024
MAX_LOAF_PLAN_ARTIFACT_BYTES = HARD_MAX_ARTIFACT_OVERRIDE_BYTES
MAX_PINNED_FILE_BYTES = 32 * 1024 * 1024 * 1024
MAX_SCHEMA_BYTES = 1024 * 1024
MAX_RLIVIT_RUN_DIRECTORIES = 256
MAX_RLIVIT_JOB_RECEIPT_BYTES = 512 * 1024
MAX_RLIVIT_TOTAL_JOB_RECEIPT_BYTES = 32 * 1024 * 1024
SITE_PLAN_SCHEMA = "site-distance-evaluation-plan-v1.schema.json"
SITE_EVALUATION_SCHEMA = "distance-validation-v1.schema.json"
CAMPAIGN_REPORT_SCHEMA = "validation-campaign-report-v1.schema.json"
PRODUCT_READINESS_SCHEMA = "product-readiness-v1.schema.json"
OBJECTIVE_COMPLETION_SCHEMA = "validation-objective-completion-v1.schema.json"
OBJECTIVE_COMPLETION_SCHEMA_VERSION = (
    "deepsafe.validation-objective-completion/v1"
)
OBJECTIVE_COMPLETION_CONTRACT_ID = "deepsafe-user-validation-objective-v1"
FINALIZATION_RECEIPT_SCHEMA = "deepsafe.validation-finalization-receipt/v1"
PRODUCT_FINALIZATION_V2_SCHEMA = "deepsafe.product-finalization-receipt/v2"
PRODUCT_FINALIZATION_V2_SCHEMA_FILE = "product-finalization-receipt-v2.schema.json"
FINALIZATION_CAMPAIGN_NAME = "deepstream9-12-camera-seven-day"
FINALIZATION_TARGET_SECONDS = 7 * 24 * 60 * 60
FINALIZATION_SEGMENT_COUNT = 28
FINALIZATION_INPUT_PATHS = (
    "validation/results/endurance/current/checkpoint.json",
    "validation/results/endurance/current/status.json",
    "validation/results/endurance/current/campaign-resolved.json",
    "validation/results/endurance/current/plan.json",
    "validation/report_campaign.py",
    "validation/objective_completion.py",
    "validation/product_readiness.py",
    "validation/finalize_validation.py",
    "validation/schemas/validation-campaign-report-v1.schema.json",
    "validation/schemas/validation-objective-completion-v1.schema.json",
    "validation/schemas/product-readiness-v1.schema.json",
)
FINALIZATION_OUTPUTS = (
    (
        "campaign_json",
        "validation/results/campaign-report/report.json",
        "application/json",
        "campaign_report_json",
    ),
    (
        "campaign_markdown",
        "validation/results/campaign-report/report.md",
        "text/markdown",
        "campaign_report_markdown",
    ),
    (
        "objective_json",
        "validation/results/objective-completion/current/report.json",
        "application/json",
        "objective_completion_json",
    ),
    (
        "objective_markdown",
        "validation/results/objective-completion/current/report.md",
        "text/markdown",
        "objective_completion_markdown",
    ),
    (
        "product_json",
        "validation/results/product-readiness/current/report.json",
        "application/json",
        "product_readiness_json",
    ),
    (
        "product_markdown",
        "validation/results/product-readiness/current/report.md",
        "text/markdown",
        "product_readiness_markdown",
    ),
)
OBJECTIVE_COMPLETION_GATE_TITLES = {
    "acceptance_safe_video_type_matrix": "10+ video tipi ve 640/960 matrisi",
    "ground_truth_two_profile_metrics": "Ground-truth 640/960 metrikleri",
    "hash_bound_gt_free_visual_inspection": "GT olmayan görsel hata incelemesi",
    "paired_640_960_comparison": "Eşlenmiş 640/960 karşılaştırması",
    "reproducible_raw_and_receipt_lineage": "Tekrarlanabilir kanıt zinciri",
    "seven_day_endurance": "Yedi günlük dayanıklılık kampanyası",
}
OBJECTIVE_COMPLETION_GATE_IDS = tuple(OBJECTIVE_COMPLETION_GATE_TITLES)
OBJECTIVE_COMPLETION_LIMITATION_IDS = {
    "rlivit_recall_is_low_observation_without_owner_quality_threshold",
    "top_view_ai_audit_contains_high_severity_undercoverage_finding",
    "loaf_artifacts_do_not_satisfy_calibrated_25m_detection",
}
PRODUCT_READINESS_GATE_COMPONENT_IDS = {
    "person_ds9_quality_and_capacity": [
        "campaign_report_contract",
        "real_ds9_ground_truth_metrics_640_960",
        "twelve_stream_640_960_five_minute_capacity",
        "approved_person_quality_acceptance",
    ],
    "deployment_calibrated_20_25m_exact_25m": [
        "deployment_camera_calibration_and_gt",
        "exact_25m_endpoint",
    ],
    "pose_ds9_gt_and_capacity": [
        "receipt_envelope_and_measurements",
        "owner_approved_acceptance_policy",
        "semantic_raw_replay",
        "model_weights",
        "onnx_export",
        "tensorrt_engines_640_960",
        "deepstream9_real_inference",
        "ground_truth_keypoint_metric",
        "performance_640",
        "performance_960",
    ],
    "ppe_ds9_gt_temporal_and_capacity": [
        "receipt_envelope_and_measurements",
        "owner_approved_acceptance_policy",
        "semantic_raw_replay",
        "helmet_and_hi_vis_model",
        "onnx_export",
        "tensorrt_engines_640_960",
        "deepstream9_real_inference",
        "helmet_ground_truth_metric",
        "hi_vis_ground_truth_metric",
        "track_temporal_metrics",
        "performance_640",
        "performance_960",
    ],
    "three_module_full_stack_capacity": [
        "receipt_envelope_and_measurements",
        "owner_approved_acceptance_policy",
        "semantic_raw_replay",
        "person_pose_ppe_enabled_together",
        "deepstream9_real_inference",
        "metadata_fusion_integrity",
        "runtime_health_no_xid_oom_fatal",
        "performance_640",
        "performance_960",
    ],
    "single_admin_visibility_and_readiness": [
        "admin_validation_source",
        "admin_ui_source",
        "admin_app_source",
        "module_readiness_source",
    ],
}
SITE_DISTANCE_V2_FINAL_RE = re.compile(
    r"evaluation-final-v2(?:-[0-9]{3,6})?\.json"
)
SITE_DISTANCE_V2_MAX_FINAL_CANDIDATES = 32
SITE_DISTANCE_V2_SCHEMA = "deepsafe.site-distance-evaluation-final/v2"
SITE_DISTANCE_V2_BOUNDARY_POLICY = (
    "lower_inclusive_upper_exclusive_except_24_25m_upper_inclusive"
)
SITE_DISTANCE_V2_BIN_IDS = (
    "20-21m",
    "21-22m",
    "22-23m",
    "23-24m",
    "24-25m",
)
SITE_DISTANCE_V2_MINIMUM_SOURCE_BYTES = 1024
SITE_DISTANCE_V2_MINIMUM_BIN_INSTANCES = 10
SITE_DISTANCE_V2_MINIMUM_BIN_EVENTS = 5
SITE_DISTANCE_V2_MINIMUM_UNAMBIGUOUS_EVENTS = 5
SITE_DISTANCE_V2_MINIMUM_ENDPOINT_EVENTS = 3
SITE_DISTANCE_V2_MINIMUM_EXACT_25_INSTANCES = 1
SITE_DISTANCE_V2_FIXTURE_RE = re.compile(
    r"(?:^|[-_.])(fixture|synthetic|placeholder|example|demo|test)(?:$|[-_.])",
    re.IGNORECASE,
)
PPE_VIDEO_SOURCE_REGISTRY_SHA256 = (
    "97cbf6017369e2783ef56084ee3aa5b3c45aa24a609910d6cc2705b1301c723c"
)
PPE_VIDEO_SOURCE_CANDIDATE_COUNT = 12
PPE_VIDEO_SOURCE_ELIGIBILITY_COUNTS = {
    "metric_distance_metrics": 0,
    "pose_metrics": 0,
    "quantitative_commercial_video_benchmark": 0,
    "temporal_event_metrics": 0,
    "track_metrics": 0,
}
PPE_VIDEO_SOURCE_CANDIDATE_GROUPS = {
    "written_license_contact_ids": [
        "al_azani_kfupm_ppe_cctv",
        "mobiusi_helmet_action",
    ],
    "qualitative_video_ids": [
        "foundation_pit_v2",
        "pixabay_construction_worker_348896",
        "mixkit_two_construction_workers_1436",
    ],
    "static_diagnostic_ids": [
        "tcrsf_sfchd",
        "ppe_cctv_topdown",
        "put_your_ppe_on",
    ],
    "ml_restricted_ids": [
        "pexels_construction_worker_roof_16393893",
        "pexels_construction_site_7448386",
    ],
}
PERSON_UPGRADE_PLAN_PIN = {
    "path": "models/person/upgrade-provenance-plan.json",
    "bytes": 12585,
    "sha256": "b9ae5c4bec3111ebbcf7dff5a4db8870ccb84c5e226ab6ea86c71bc79b444339",
}
PERSON_UPGRADE_PLAN_SCHEMA = "deepsafe.person-upgrade-provenance-plan/v1"
PERSON_UPGRADE_MANIFEST_PATH = (
    "data/derived/r-livit/person-finetune-v1/manifest.json"
)
PERSON_UPGRADE_TRAINING_PLAN_PATH = (
    "models/person/training-plans/yolo26s-r1.json"
)
PERSON_UPGRADE_RTDETR_PROVENANCE_PATH = (
    "models/person/candidates/rtdetrv4-s/provenance.json"
)
PERSON_UPGRADE_RTDETR_CHECKPOINT_PATH = (
    "models/person/candidates/rtdetrv4-s/RTv4-S-hgnet.pth"
)
PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN = {
    "path": (
        "validation/results/person/models/"
        "rtdetrv4-s-structural-load-r1.json"
    ),
    "bytes": 4264,
    "sha256": (
        "c5c36883453bd699c7930d7a8565c244316c0694f4b2d864eb7c01f85b755bf7"
    ),
    "receipt_sha256": (
        "498d1f7ccfb00f540d2ea44e4da7a097ffa867e149f130f08c2b6a45b8a76a06"
    ),
}
PERSON_UPGRADE_STRUCTURAL_SCHEMA_PIN = {
    "path": (
        "validation/schemas/"
        "person-checkpoint-structural-receipt-v1.schema.json"
    ),
    "bytes": 1066,
    "sha256": (
        "40ecab4c3dc2cf2ea09b15b580fc45f0cfb64551b69197db4bdc9d1688220cda"
    ),
}
PERSON_UPGRADE_STRUCTURAL_VALIDATOR_PIN = {
    "path": "validation/person_checkpoint_structural.py",
    "bytes": 16426,
    "sha256": (
        "fedc8bd57327b1c0276a4756ff3d8c302f86c5efad99b129d1e6ab7e609de38d"
    ),
}
PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN = {
    "path": (
        "validation/results/person/models/"
        "rtdetrv4-s-framework-profiles-r1.json"
    ),
    "bytes": 7348,
    "sha256": (
        "24a9c3025f04baca54dbec7c04e7ce6637d3d8ed50f0ce36d68af7fcfe9b6cbe"
    ),
    "receipt_sha256": (
        "8e44521abad4a8984859c61f4706b6fd37c614df829f2e13573bf3a17f9e1ea4"
    ),
}
PERSON_UPGRADE_FRAMEWORK_SCHEMA_PIN = {
    "path": (
        "validation/schemas/"
        "person-framework-profiles-receipt-v1.schema.json"
    ),
    "bytes": 881,
    "sha256": (
        "7d6359037dad3af052008e76b80fa4e9f5be637f4ea827cd03a210c513d8803a"
    ),
}
PERSON_UPGRADE_FRAMEWORK_VALIDATOR_PIN = {
    "path": "validation/person_framework_profiles.py",
    "bytes": 17070,
    "sha256": (
        "c6fa2e1d74395e53470d4216ddeb185b88638c0f82171c45c03f3b8f2856768c"
    ),
}
PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN = {
    "path": "models/person/candidates/rtdetrv4-s/export-plan-v1.json",
    "bytes": 3778,
    "sha256": (
        "8dc9e3a390eaf8da889a4db27136e93f4e64c9d0e0526bbf076840f07e0009fe"
    ),
    "fingerprint_sha256": (
        "5db543612d6ad0f2462697f59ebf2d4a9864de00007cb65f5b27d14522f0f011"
    ),
}
PERSON_UPGRADE_ONNX_EXPORTER_PIN = {
    "path": "models/person/export_rtdetrv4.py",
    "bytes": 20665,
    "sha256": (
        "ecb1503696b33a199cae66f8e1076d7551c8a68821306f82d20e8f8227ae5771"
    ),
}
PERSON_UPGRADE_ONNX_RECEIPT_SCHEMA_PIN = {
    "path": (
        "validation/schemas/"
        "person-rtdetrv4-onnx-export-receipt-v1.schema.json"
    ),
    "bytes": 1074,
    "sha256": (
        "47efa1f8fb0d4badf08b4d61f0f3395eecc5849cc1b605ea94d62997b9872ecc"
    ),
}
PERSON_UPGRADE_ONNX_PROFILE_PINS = {
    640: {
        "receipt": {
            "path": (
                "validation/results/person/models/"
                "rtdetrv4-s-onnx-640-r1.json"
            ),
            "bytes": 4939,
            "sha256": (
                "19c6baac7f841e2869501c79585a5bfcd10642cf6d766e5a690a5b5cc35391b3"
            ),
            "receipt_sha256": (
                "4a0f1550df7f0777a5d554b799ab29ffe6b7e68323d4e0aa1bd7e6eebc7324a1"
            ),
        },
        "onnx": {
            "path": (
                "models/person/candidates/rtdetrv4-s/onnx/640/"
                "rtdetrv4-s-640-bdynamic-opset18.onnx"
            ),
            "bytes": 41809642,
            "sha256": (
                "e86957041e221a0c93bbf57a028a2f533638e7a1ffadb3b2907906aadb166eaf"
            ),
        },
    },
    960: {
        "receipt": {
            "path": (
                "validation/results/person/models/"
                "rtdetrv4-s-onnx-960-r1.json"
            ),
            "bytes": 5020,
            "sha256": (
                "a35ad44aff56ea78d20998d78dd8094b3c806f94caa5e53ef401c5fc13dcaf13"
            ),
            "receipt_sha256": (
                "69c90420ee631f2d8a784c1fa042c56da19b49e49b5ca87c7366e195ec10713c"
            ),
        },
        "onnx": {
            "path": (
                "models/person/candidates/rtdetrv4-s/onnx/960/"
                "rtdetrv4-s-960-bdynamic-opset18.onnx"
            ),
            "bytes": 42531644,
            "sha256": (
                "15e9cc2479b5e31f90e0c236431ec30e263efe027bd6a20ac9e88d61e1e68f25"
            ),
        },
    },
}
PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN = {
    "path": (
        "models/person/candidates/rtdetrv4-s/"
        "real-image-parity-plan-v1.json"
    ),
    "bytes": 18377,
    "sha256": (
        "5d88d2caacbe634ded37e45c4bcbe63f453ee7dc628633c14f5b42800e956b2c"
    ),
    "fingerprint_sha256": (
        "1bf418f5c5c7e0ff7a81b6e4ad43fa82b2f439500a22f0c51f4182a3f357ea56"
    ),
}
PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN = {
    "path": (
        "validation/results/person/models/"
        "rtdetrv4-s-real-image-parity-r1.json"
    ),
    "bytes": 127094,
    "sha256": (
        "e9389b5bec3a60ebd1a8da5daa9f4896da797e8f7b7aa06aa4ec02cd2f631a05"
    ),
    "receipt_sha256": (
        "563d940aa0564961b7f3e79df50053ab894928ab3ba3cc8fc9c1c5ba3e8cd938"
    ),
}
PERSON_UPGRADE_REAL_IMAGE_PARITY_SCHEMA_PIN = {
    "path": (
        "validation/schemas/"
        "person-rtdetrv4-real-image-parity-receipt-v1.schema.json"
    ),
    "bytes": 1289,
    "sha256": (
        "f3e08fb0a9c8873b905cb74c93a3822809f1f051959d7cbaaba280d6823c1a66"
    ),
}
PERSON_UPGRADE_REAL_IMAGE_PARITY_VALIDATOR_PIN = {
    "path": "validation/person_rtdetrv4_real_image_parity.py",
    "bytes": 64697,
    "sha256": (
        "5ac2144a578dfe6e07b7ca75eef770b47169b2a0e22a7e68b17311cab1e6d237"
    ),
}
PERSON_UPGRADE_REAL_IMAGE_SOURCE_MANIFEST_PIN = {
    "path": "data/manifests/open-video-sources.json",
    "bytes": 32464,
    "sha256": (
        "0eefe99e2ea32f022133aeb956fa6361a64b1bc9e2f6db8104aad32c41dea515"
    ),
}
PERSON_UPGRADE_REAL_IMAGE_SOURCE_REVIEWS_PIN = {
    "path": "validation/open_video_review/source-frame-reviews-v1.jsonl",
    "bytes": 36903,
    "sha256": (
        "a7487e5db7d93d2bb01a7d5d059c3d70d721a4c3e88d0f8e0e88e7c6562ac5ae"
    ),
}
PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN = {
    "path": (
        "validation/results/person/models/"
        "rtdetrv4-s-onnx-batch12-r1.json"
    ),
    "bytes": 4291,
    "sha256": (
        "5d6297a483bb2aa28f911a1d3201972b8207ae2a121d1515f72af90020244750"
    ),
    "receipt_sha256": (
        "e14e5382bd2b66d60c37e9dfd0f0b2473db74757a90390f3aeb69348f9e6f499"
    ),
}
PERSON_UPGRADE_ONNX_BATCH12_SCHEMA_PIN = {
    "path": "validation/schemas/person-onnx-batch12-receipt-v1.schema.json",
    "bytes": 882,
    "sha256": (
        "f0699d1f08e083b63cc6c70d72ca134719b4135ac1a87191fd79f0243d86f95b"
    ),
}
PERSON_UPGRADE_ONNX_BATCH12_VALIDATOR_PIN = {
    "path": "validation/person_onnx_batch12.py",
    "bytes": 11477,
    "sha256": (
        "75743bbd3f78ad3373c7805b466aa2066d00a5e8c65f91880b14c79cd6c55fa5"
    ),
}
PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN = {
    "path": "models/runtime/rtdetrv4-parser-ds9-r1/build-receipt.json",
    "bytes": 4416,
    "sha256": (
        "0917d1d96b0757a3f7714b8adf8205a817e176c597b4c062a44a2a0fc402a405"
    ),
    "receipt_sha256": (
        "05fc33c32ede4f0090b38f2232c36936b1ac86de1b4296f21b260054e82e8e5c"
    ),
}
PERSON_UPGRADE_DS9_PARSER_ARTIFACT_PIN = {
    "path": (
        "models/runtime/rtdetrv4-parser-ds9-r1/"
        "libdeepsafe_rtdetrv4_parser.so"
    ),
    "bytes": 16480,
    "sha256": (
        "09b28e20a28799bd6a2c5382b4e54922454ec71dd63e9d4fe498bc6384d1c7dc"
    ),
}
PERSON_UPGRADE_DS9_PARSER_SOURCE_PINS = {
    "cmake": {
        "path": "models/person/postprocess/rtdetrv4_ds9/CMakeLists.txt",
        "bytes": 1493,
        "sha256": (
            "1341bd0f670832426f1c755c156c4c470a5e48cd5e77535e4994e0185a838923"
        ),
    },
    "exports_map": {
        "path": "models/person/postprocess/rtdetrv4_ds9/exports.map",
        "bytes": 136,
        "sha256": (
            "e577e620db55e325627965709bf7040135e052eed6c16ff8a57f22355c627c57"
        ),
    },
    "parser_source": {
        "path": (
            "models/person/postprocess/rtdetrv4_ds9/rtdetrv4_parser.cpp"
        ),
        "bytes": 6738,
        "sha256": (
            "3461b96f0ecd0391edd2f6375cf7be80403a47fcd93d4b81b89e1ca7a6c10bb5"
        ),
    },
    "test_source": {
        "path": (
            "models/person/postprocess/rtdetrv4_ds9/"
            "test_rtdetrv4_parser.cpp"
        ),
        "bytes": 6066,
        "sha256": (
            "27621b69e9630a6036f5f69ac5d090a53d26b30448e5f3dcdfb87c011d6c5c99"
        ),
    },
    "build_script": {
        "path": (
            "models/person/postprocess/rtdetrv4_ds9/build-runtime.sh"
        ),
        "bytes": 1440,
        "sha256": (
            "d53aee0fb90266a7da7c5bb7c14efddf804642735ce563fdd994ef69bdb99ce5"
        ),
    },
}
PERSON_UPGRADE_MAX_JSON_BYTES = 512 * 1024
PERSON_RTDETR_GPU_R10_PINS = {
    "plan": {
        "path": (
            "models/person/training-lanes/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/execution-plan-r10.json"
        ),
        "bytes": 68143,
        "sha256": (
            "b0e3db1eedba6a0b26cfbfb1a733b5ae09f274e69d379c4f81a5ad005b0166bc"
        ),
    },
    "build_receipt": {
        "path": (
            "validation/results/person/training/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/image-build-attempts/"
            "eval-device-r10-001/build-receipt.json"
        ),
        "bytes": 7052,
        "sha256": (
            "b46563aeec1323d4ef87aa6100d0426486b9394b166a1202ee20849940daa9ab"
        ),
    },
    "smoke_host_receipt": {
        "path": (
            "models/person/training-runs/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/smoke-one-step-006/"
            "host-receipt.json"
        ),
        "bytes": 8883,
        "sha256": (
            "bbb1db13d14f43157276184480e3ee59baac6a0c991d1eeb6406f477595eabcd"
        ),
    },
    "smoke_container_receipt": {
        "path": (
            "models/person/training-runs/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/smoke-one-step-006/"
            "container-receipt.json"
        ),
        "bytes": 42262,
        "sha256": (
            "47e324bd55e1c1e7034f6181451dfaad45ed7cd74b380dedb1dfb85e917f2c31"
        ),
    },
    "baseline_host_receipt": {
        "path": (
            "models/person/training-runs/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/baseline-eval-002/"
            "host-receipt.json"
        ),
        "bytes": 9241,
        "sha256": (
            "8f7ccc8884689d0762f49e3f5002877276268081f8b4f09cdeffc727d858f8b5"
        ),
    },
    "baseline_container_receipt": {
        "path": (
            "models/person/training-runs/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/baseline-eval-002/"
            "container-receipt.json"
        ),
        "bytes": 12556,
        "sha256": (
            "e208714161b4ff9d9d6d80ac7887e6fceba1bd3c5e9463ba761bdae04709ee0d"
        ),
    },
}
PERSON_RTDETR_GPU_R10_PLAN_FINGERPRINT = (
    "12eb66c8b52ca80083169ebcc43684e9f3da7a6caee9ba27cac5bac85b3d397b"
)
PERSON_RTDETR_GPU_R10_IMAGE_ID = (
    "sha256:1c1df86249fb721e1c7ccb869effc7c3b134c8737e64f1d547b460f98cd21a8b"
)
PERSON_RTDETR_DISTANCE_PROXY_R1_PINS = {
    "report": {
        "path": (
            "validation/results/person/distance-proxy/baseline-eval-002/"
            "distance-proxy-report.json"
        ),
        "bytes": 244749,
        "sha256": (
            "cafdbb87ff7f5996e897598120287aab3a7e44675ddb056cde0a3889f998289b"
        ),
    },
    "receipt": {
        "path": (
            "validation/results/person/distance-proxy/baseline-eval-002/"
            "receipt.json"
        ),
        "bytes": 3350,
        "sha256": (
            "19f6ba94ce981fdb5581b9137434c44dcd6729ba2b21633a62b2feb4dc745e52"
        ),
    },
}
PERSON_RTDETR_FULL_TRAINING_R10_PINS = {
    "host_receipt": {
        "path": (
            "models/person/training-runs/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/full-60e-001/"
            "host-receipt.json"
        ),
        "bytes": 10531,
        "sha256": (
            "7b7d090a7c9c6726b47e9076bf814635a3abf3b6c923726a74fdbbcd0d05e6fa"
        ),
    },
    "container_receipt": {
        "path": (
            "models/person/training-runs/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/full-60e-001/"
            "container-receipt.json"
        ),
        "bytes": 19957,
        "sha256": (
            "f6ae04db1d7e6c04df7eb596699617e2ad3037a03dfc14602283bec19cd59a5c"
        ),
    },
    "events": {
        "path": (
            "models/person/training-runs/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/full-60e-001/events.jsonl"
        ),
        "bytes": 153195,
        "sha256": (
            "cffa3742af1d90c15479054060e536e39d66fbab502ec160fc1daaf19a50b48e"
        ),
    },
    "best_checkpoint": {
        "path": (
            "models/person/training-runs/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/full-60e-001/"
            "checkpoints/best.pth"
        ),
        "bytes": 166647083,
        "sha256": (
            "2b19614b414e9c5c8f1df86f0c080b9742148ab2ae9114c0498b0dfdd0b43332"
        ),
    },
}
PERSON_RTDETR_FULL_TRAINING_R10_MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
PERSON_RTDETR_EXPORT_R11_PINS = {
    "plan": {
        "path": (
            "models/person/export-lanes/"
            "rtdetrv4-s-r-livit-person-r11/execution-plan-r11.json"
        ),
        "bytes": 13361,
        "sha256": (
            "9b9344f427a22c6ff79bfb692d7bbde95eb58dbd5b5fffefd418a06630332918"
        ),
    },
    "contract": {
        "path": (
            "models/person/export-lanes/"
            "rtdetrv4-s-r-livit-person-r11/export-contract-r11.json"
        ),
        "bytes": 12805,
        "sha256": (
            "72492433672e801faadbe95de4b128f354288c6299f8576fffbffdcb98234148"
        ),
    },
}
PERSON_RTDETR_EXPORT_R11_PLAN_FINGERPRINT = (
    "375d0e1d250ea7116972172743cda55dfbd52e0bef9d464b24501ee405a0aeaf"
)
PERSON_RTDETR_EXPORT_R11_CONTRACT_FINGERPRINT = (
    "baf37272b2f1d098d4735be30c2cd3c4a92e03921a6e25d3896a0a81a5d2d2f7"
)
PERSON_RTDETR_ONNX_R12_RECEIPT_PATHS = {
    640: (
        "validation/results/person/export/"
        "rtdetrv4-s-r-livit-person-r11/onnx-640/receipt.json"
    ),
    960: (
        "validation/results/person/export/"
        "rtdetrv4-s-r-livit-person-r11/onnx-960/receipt.json"
    ),
}
# A stage is projected only after its immutable receipt has an independently
# compiled file pin here.  A lone ONNX/recovery intent, or a receipt appearing
# at the expected name without such a pin, is deliberately not discovered.
PERSON_RTDETR_ONNX_R12_RECEIPT_PINS: dict[int, dict[str, Any]] = {
    640: {
        "path": PERSON_RTDETR_ONNX_R12_RECEIPT_PATHS[640],
        "bytes": 2777,
        "sha256": (
            "017a58d9e7a00ed7d4bfebe7cf0a662d5111bcaeee7a5e6d65bd36d6db53fa95"
        ),
        "fingerprint_sha256": (
            "7f14c69fffbae144d4c11daeb2a99382eb522e5ed6866659a56dd835f2bc960a"
        ),
    },
    960: {
        "path": PERSON_RTDETR_ONNX_R12_RECEIPT_PATHS[960],
        "bytes": 3096,
        "sha256": (
            "7bcdb82edf4fb724a464e47c83a875b4102f9573618f54aed6818fb23d29afea"
        ),
        "fingerprint_sha256": (
            "02160a78e497f1267343d66da08be034afcfa422963d65ed5666626ef6d101fb"
        ),
    },
}
PERSON_RTDETR_ONNX_R12_SCHEMA_PIN = {
    "path": (
        "validation/schemas/"
        "person-rtdetrv4-trained-export-evidence-r11.schema.json"
    ),
    "bytes": 13430,
    "sha256": (
        "97231eafb4398bc42f18bc847dbd1358de95f8c9ddf056911a54b23503daf51e"
    ),
}
PERSON_RTDETR_ONNX_R12_MAX_RECEIPT_BYTES = 128 * 1024
PERSON_RTDETR_ONNX_R12_MAX_ONNX_BYTES = 512 * 1024 * 1024
PERSON_RTDETR_THRESHOLD_R13B_PINS: dict[str, dict[str, Any]] = {
    "plan": {
        "path": (
            "models/person/export-lanes/rtdetrv4-s-r-livit-person-r11/"
            "threshold-calibration-plan-r13b.json"
        ),
        "bytes": 8700,
        "sha256": (
            "9bf6c72df2772fb7307468d4aa75684547f15efaa6341e3848e69da09bb41fff"
        ),
        "fingerprint_sha256": (
            "3ff013c44bdf7039741e92f96806bb5c76d5c52863c3c75165725ca565270412"
        ),
    },
    "executor": {
        "path": "validation/person_rtdetrv4_threshold_calibration_r13b.py",
        "bytes": 17021,
        "sha256": (
            "7eb04cbd2b9fab92efb36a98c014765e42ddf3214481e7dcfbb7511945bad6bb"
        ),
    },
    "sweep_schema": {
        "path": (
            "validation/schemas/"
            "person-rtdetrv4-threshold-sweep-r13b.schema.json"
        ),
        "bytes": 10111,
        "sha256": (
            "e321ebde968651363a329e8a522499d975a562b09ef308da48c1405eb5ee0b5d"
        ),
    },
    "sweep_640": {
        "path": (
            "validation/results/person/export/"
            "rtdetrv4-s-r-livit-person-r11/threshold-calibration/"
            "full-sweep-640-r13b.json"
        ),
        "bytes": 37854460,
        "sha256": (
            "62e94a49f8e84fe90bc44220a08ec8d8d2fc4b776732ba296e51deebd983a996"
        ),
        "fingerprint_sha256": (
            "695a1df4e591258831e4b95b4b2916a59fbf48304e555456cde27a597fe5c9a5"
        ),
    },
    "sweep_960": {
        "path": (
            "validation/results/person/export/"
            "rtdetrv4-s-r-livit-person-r11/threshold-calibration/"
            "full-sweep-960-r13b.json"
        ),
        "bytes": 37922847,
        "sha256": (
            "4594188aac104fd95839e84989f512aee7e21611232e5c67ae55a9f3b4b49fb9"
        ),
        "fingerprint_sha256": (
            "691f51b4af6a1d640830c06b7b6f8895de01e8d6e41da869ce361aa8a6ffefc4"
        ),
    },
    "final_receipt": {
        "path": (
            "validation/results/person/export/"
            "rtdetrv4-s-r-livit-person-r11/threshold-calibration/receipt.json"
        ),
        "bytes": 2949,
        "sha256": (
            "a9eb0ec5e95d375b81120446594b9dee5e4c61e9a00e4c5037d999c1c1558a94"
        ),
        "fingerprint_sha256": (
            "e4ca40e047d08cfd3422fb4bf800227df09db14f05756ad4bec09ec30bd9d157"
        ),
    },
}
PERSON_RTDETR_THRESHOLD_R13B_MAX_JSON_BYTES = 128 * 1024
PERSON_RTDETR_THRESHOLD_R13B_MAX_SWEEP_BYTES = 64 * 1024 * 1024
PPE_SEED_MANIFEST_PIN = {
    "path": "data/manifests/ppe-training-seed-sources.json",
    "bytes": 4155,
    "sha256": "82cc91fbddc545174de1ab72ad1825ef0959ccfada9c710171f099aec1812a7b",
}
PPE_SEED_SCHEMA_PINS = {
    "manifest": {
        "path": (
            "ppe_dataset/schemas/"
            "ppe-training-seed-sources-v1.schema.json"
        ),
        "bytes": 3722,
        "sha256": (
            "1f4e1311ad805e82b9c4158bd2e691450ff1f56427c499719d0ee381ada2695f"
        ),
    },
    "acquisition_receipt": {
        "path": (
            "ppe_dataset/schemas/"
            "ppe-seed-acquisition-receipt-v1.schema.json"
        ),
        "bytes": 4440,
        "sha256": (
            "a8db4a10c07e7e923c324d82c4fa4f6d058ed8d2b7b1ef26ce3ceb67f1b37b7f"
        ),
    },
    "quarantine_receipt": {
        "path": (
            "ppe_dataset/schemas/"
            "ppe-seed-quarantine-receipt-v1.schema.json"
        ),
        "bytes": 4839,
        "sha256": (
            "caa378995a78a7777043e20877a205a5057621e33f363b22f2309497aa020ff2"
        ),
    },
}
PPE_SEED_MANIFEST_SCHEMA = "deepsafe.ppe-training-seed-sources/v1"
PPE_SEED_REQUIRED_CLASSES = ("helmet", "no_helmet", "hi_vis", "no_hi_vis")
PPE_SEED_SOURCE_IDS = (
    "mendeley-ppe-v6-20250731",
    "mendeley-ppe-five-class-v1",
)
PPE_SEED_MAX_JSON_BYTES = 512 * 1024
PPE_SEED_RECEIPT_PINS = {
    "acquisition": {
        "path": (
            "validation/results/ppe/seeds/"
            "mendeley-ppe-v6-acquisition.json"
        ),
        "bytes": 2046,
        "sha256": (
            "1623d7f67409f9daad4e0e642db34fe13101c3da439c7c0e23b119c95dc498cb"
        ),
        "receipt_sha256": (
            "72aa6334f149fed6ddb2b62b46b543d6fc48e076e168a4bcbb4c13a6ae98c494"
        ),
    },
    "quarantine": {
        "path": (
            "validation/results/ppe/seeds/"
            "mendeley-ppe-v6-quarantine-r2.json"
        ),
        "bytes": 4246548,
        "sha256": (
            "99fef628ae3e8855cb373a6da84f0c9a7b7749e0783ebd0828133c094b16ca58"
        ),
        "receipt_sha256": (
            "d71ce547daf5cf316c48f0cf5bca6b5c475b51856f8db5a5b22b9f3e11722775"
        ),
    },
}
PPE_SEED_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
PPE_FIVE_CLASS_ADMIN_PINS = {
    "manifest": {
        "path": (
            "data/manifests/"
            "ppe-mendeley-five-class-v1-acquisition-r2.json"
        ),
        "bytes": 3187,
        "sha256": (
            "fc98a4731156dc49e5f3b0a3085885494d70b63a31861b553be3274b64351bab"
        ),
    },
    "authoritative_receipt": {
        "path": (
            "validation/results/ppe/quarantine/"
            "mendeley-ppe-five-class-v1-quarantine-r2.json"
        ),
        "bytes": 5456369,
        "sha256": (
            "c06e735749accdea2cae3cd08e9942816918f7f2418b8fa8c6898d5dfab71323"
        ),
        "receipt_sha256": (
            "2c845f047bc7983adb0f1f7f7a67831f973052e56c291ea1c543a582af326c9c"
        ),
    },
    "superseded_r1_receipt": {
        "path": (
            "validation/results/ppe/quarantine/"
            "mendeley-ppe-five-class-v1-quarantine-r1.json"
        ),
        "bytes": 5456581,
        "sha256": (
            "7d0d5369c71ceb5b0c0306349cbb8acb9cb55e155df8821e24c9b1f94d52f679"
        ),
        "receipt_sha256": (
            "f579875060f4acdf138d9a00a53b36eedd18801c13ce49de7ea3f4bd93658097"
        ),
    },
    "projection_receipt": {
        "path": (
            "validation/results/ppe/quarantine/"
            "mendeley-ppe-five-class-v1-admin-projection-r1.json"
        ),
        "bytes": 4013,
        "sha256": (
            "9ff4687c13d143d26a3e4d1eb8e6d6582f1ae5282db8dd91fb5a1febb6cc483f"
        ),
        "receipt_sha256": (
            "4bc432e1afb2f359c4b4f9faf8114fe9a155c1f3f138aeba919b2d6c39c8d87f"
        ),
    },
    "validator": {
        "path": "validation/ppe_five_class_admin_projection.py",
        "bytes": 22728,
        "sha256": (
            "ec30fdc89a500c5398e2dc9bbd9dec3ebc995fb7b281217c3a8e76db9e88c7eb"
        ),
    },
    "schema": {
        "path": (
            "validation/schemas/"
            "ppe-five-class-admin-projection-receipt-v1.schema.json"
        ),
        "bytes": 7987,
        "sha256": (
            "79a01bb11419bc2441fbe6c9d9d773119366d8f2dc348ca41529ba257b5c26d6"
        ),
    },
}
PPE_FIVE_CLASS_SOURCE_RECEIPT_MAX_BYTES = 6 * 1024 * 1024
PPE_FIVE_CLASS_COMPACT_MAX_BYTES = 64 * 1024
PPE_FIVE_CLASS_NORMALIZATION_R2_PINS = {
    "plan": {
        "path": (
            "data/manifests/"
            "ppe-mendeley-five-class-v1-normalization-group-split-r2.plan.json"
        ),
        "bytes": 3909,
        "sha256": (
            "35f7fa2b03aa8fb32c2a349144628f3e242f7ee912fad5339c783498030e349d"
        ),
    },
    "plan_schema": {
        "path": (
            "ppe_dataset/schemas/"
            "ppe-five-class-normalization-plan-v1.schema.json"
        ),
        "bytes": 7834,
        "sha256": (
            "7cb36b6665ee9493596e2095bef99260c41b1e8e72d22a5d772797e4d2c4b821"
        ),
    },
    "receipt": {
        "path": (
            "validation/results/ppe/normalization/"
            "mendeley-ppe-five-class-v1-normalization-group-split-dry-run-r2.json"
        ),
        "bytes": 607390,
        "sha256": (
            "54c54364785b2625afbc109c360ebb715fc03df7462ea420a4ec930ee0cfed62"
        ),
        "receipt_sha256": (
            "2391fe3c47f881da190e4dbc8801d83cf5a9f2d586b4eada5b74e6118e9ccd23"
        ),
    },
    "receipt_schema": {
        "path": (
            "validation/schemas/"
            "ppe-five-class-normalization-dry-run-receipt-v1.schema.json"
        ),
        "bytes": 19993,
        "sha256": (
            "2345ce837a7c8c3453c29b25310e5c8973e171fd2c27224f7769d58fdf9182ab"
        ),
    },
    "implementation": {
        "path": "ppe_dataset/five_class_normalization_r2.py",
        "bytes": 53755,
        "sha256": (
            "231e2e451ab84564ddf055c73d995011439b03f5c1a6df9b277ed105f754f008"
        ),
    },
}
PPE_FIVE_CLASS_NORMALIZATION_MAX_BYTES = 1024 * 1024
PPE_FIVE_CLASS_SEMANTIC_R4_PINS = {
    "plan": {
        "path": (
            "data/manifests/"
            "ppe-mendeley-five-class-v1-semantic-audit-r4.plan.json"
        ),
        "bytes": 4881,
        "sha256": (
            "ce9839151104c954c7414f7aae8d409a40ce90b79f0ea3627fc1350a84fe3b74"
        ),
    },
    "receipt": {
        "path": (
            "validation/results/ppe/semantic-audit/"
            "mendeley-ppe-five-class-v1-r4/receipt.json"
        ),
        "bytes": 10809,
        "sha256": (
            "298dc32bea3c101cfefd76513692202baa75d3cf796b3a3938f7409e7bcd5694"
        ),
        "receipt_sha256": (
            "42c1e5ef444c598cf8b80cefcb447c3f60e1ddd1d0467ee610dafaf9d5a038ff"
        ),
    },
    "receipt_schema": {
        "path": (
            "validation/schemas/"
            "ppe-five-class-semantic-audit-receipt-v1.schema.json"
        ),
        "bytes": 9248,
        "sha256": (
            "4c1d743e3501e8a4eb0aabffb7a9721a5cf4324f1bcaf107d250e2e60cbd1b8a"
        ),
    },
    "implementation": {
        "path": "ppe_dataset/five_class_semantic_audit_r4.py",
        "bytes": 55161,
        "sha256": (
            "fd2c3fdc2b7d372e60fea565b573821e19357ddb57f67a77f08468af36865a75"
        ),
    },
}
PPE_FOUR_CLASS_SEMANTIC_R5_PINS = {
    "contract": {
        "path": (
            "models/ppe/training-lanes/"
            "yolo11s-mendeley-four-class-remediated-r5/"
            "transform-contract-r5.json"
        ),
        "bytes": 7053,
        "sha256": (
            "2ad22266d82c56996f298d44d33c385d9c15c67e9f1b290377b4999668c47c9e"
        ),
    },
    "receipt": {
        "path": (
            "validation/results/ppe/semantic-remediation/"
            "mendeley-ppe-four-class-remediated-r5/receipt.json"
        ),
        "bytes": 3748,
        "sha256": (
            "8a2397315f070d3a6a945b778ca89e973f5af441e435eeb812c788e184a8058b"
        ),
        "receipt_sha256": (
            "880c54a687ca61dc111d8c634647b7d033c337ac718fa155b7e637b31eb0b887"
        ),
    },
    "contract_schema": {
        "path": (
            "validation/schemas/"
            "ppe-four-class-remediation-contract-v1.schema.json"
        ),
        "bytes": 13041,
        "sha256": (
            "29902959e6c4e1a5faf12f57c8c8f5f4dc96a6d697c13046236d6b52006dcf9f"
        ),
    },
    "receipt_schema": {
        "path": (
            "validation/schemas/"
            "ppe-four-class-remediation-receipt-v1.schema.json"
        ),
        "bytes": 8172,
        "sha256": (
            "e32fb7d15a43e60f36ae35c3da41d57b135ccf833c5c19338e6f239e166d94b2"
        ),
    },
    "implementation": {
        "path": "validation/ppe_semantic_remediation_r5.py",
        "bytes": 49930,
        "sha256": (
            "1eef7b9f2ef79bc063a803a440c4bb276e50576075084b7d09a2ac25449bea2e"
        ),
    },
}
PPE_FOUR_CLASS_SEMANTIC_R5_MAX_BYTES = 128 * 1024
PPE_HUMAN_QA_R6_PINS = {
    "plan": {
        "path": (
            "models/ppe/training-lanes/"
            "yolo11s-mendeley-four-class-remediated-r5/"
            "human-qa-plan-r6.json"
        ),
        "bytes": 5814,
        "sha256": (
            "7a122696d2b1adca5006cb556689fed2154d0e24df33ef73defa8e8355d091ae"
        ),
    },
    "receipt": {
        "path": (
            "validation/results/ppe/human-qa/"
            "mendeley-ppe-four-class-r6/receipt.json"
        ),
        "bytes": 4961,
        "sha256": (
            "f34218294ada32965206326c508dbab7613a1233fac66d2be63605a0c3eafc55"
        ),
        "receipt_sha256": (
            "87c62ea4ba3515ad549d30a32c9a628d68909f37ca3b372b06fddd44fa071325"
        ),
    },
    "samples": {
        "path": (
            "validation/results/ppe/human-qa/"
            "mendeley-ppe-four-class-r6/samples.jsonl"
        ),
        "bytes": 1401636,
        "sha256": (
            "bfff611a251e56f43916a218adbc99ef63d3f11d30bcc2f0d49dce71698e7942"
        ),
    },
    "artifact_manifest": {
        "path": (
            "validation/results/ppe/human-qa/"
            "mendeley-ppe-four-class-r6/artifact-manifest.jsonl"
        ),
        "bytes": 105808,
        "sha256": (
            "0c8af4c5e13ada15fbfb63de6938c5c32ac8f6f96a4e990621dfc26079c18a0b"
        ),
    },
    "payload_access": {
        "path": (
            "validation/results/ppe/human-qa/"
            "mendeley-ppe-four-class-r6/payload-access.jsonl"
        ),
        "bytes": 1793471,
        "sha256": (
            "5c9a240d5e5eaa10fa392b2f75d2eb11ade80b72a0aa55b2cb488ca394f8efeb"
        ),
    },
    "plan_schema": {
        "path": (
            "validation/schemas/"
            "ppe-human-qa-packet-plan-r6-v1.schema.json"
        ),
        "bytes": 7153,
        "sha256": (
            "28f148e3ea2b100a2e6040635bdf3199a1dcc4581d18fc8ba55ff4f0037c1ae5"
        ),
    },
    "receipt_schema": {
        "path": (
            "validation/schemas/"
            "ppe-human-qa-packet-receipt-r6-v1.schema.json"
        ),
        "bytes": 6702,
        "sha256": (
            "17a16498bdc18b5c057b5650515c81114a827a08637bfb5c4d6995f3dec083f3"
        ),
    },
    "sample_schema": {
        "path": (
            "validation/schemas/"
            "ppe-human-qa-sample-r6-v1.schema.json"
        ),
        "bytes": 5737,
        "sha256": (
            "aa62ec2caa8155d3a47b5c324be33b3a74ea8cb91f49890acf007f5e9cf1581d"
        ),
    },
    "adjudication_schema": {
        "path": (
            "validation/schemas/"
            "ppe-human-qa-adjudication-r6-v1.schema.json"
        ),
        "bytes": 4813,
        "sha256": (
            "2f0e8d8f4715dbb824b3e7e9fe8020fb1d440f937dee6076446fe61dfbaf6388"
        ),
    },
    "implementation": {
        "path": "validation/ppe_human_qa_r6.py",
        "bytes": 71478,
        "sha256": (
            "d881b64b3e8ca74384c8f239c7a3a7f0d24e6029fa77f6fecf9262a3150734b7"
        ),
    },
}
PPE_HUMAN_QA_R6_JSON_MAX_BYTES = 128 * 1024
PPE_HUMAN_QA_R6_STREAM_MAX_BYTES = 2 * 1024 * 1024
PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS = {
    "gate": {
        "path": (
            "validation/authorizations/"
            "ppe-yolo11s-semantic-launch-gate-r3.json"
        ),
        "bytes": 3966,
        "sha256": (
            "a95aa81bf70bdfdf960c44e3cc65390876d1d736796421cfbdef00a1ed9b5c47"
        ),
        "fingerprint_sha256": (
            "26680ed43b9ae6ffffa221a1e6bdf913347c45defdc220fe487a1555d7cce1c9"
        ),
    },
    "schema": {
        "path": (
            "validation/schemas/"
            "ppe-yolo11s-semantic-launch-gate-v1.schema.json"
        ),
        "bytes": 7376,
        "sha256": (
            "279e396f66d8166ae354ff656407682309eb982418c9bfdd849a5d772003610d"
        ),
    },
    "verifier": {
        "path": "validation/ppe_yolo11s_semantic_launch_gate_r3.py",
        "bytes": 11537,
        "sha256": (
            "f7159d87c2ef8f178263e3fd88dfa2a0d6c2700659be943276fb1ada85f8d560"
        ),
    },
}
PPE_FIVE_CLASS_SEMANTIC_MAX_BYTES = 128 * 1024
PPE_FIVE_CLASS_GATE_KEYS = (
    "embedded_rights_audit_complete",
    "camera_site_session_group_safe",
    "person_equipment_semantics_normalized",
    "published_independent_test_split_ready",
    "training_eligible",
    "training_complete",
    "export_complete",
    "deepstream9_evaluated",
    "ground_truth_quality_passed",
    "twelve_camera_640_passed",
    "twelve_camera_960_passed",
    "acceptance_passed",
    "production_ready",
)
PPE_LO_CPPED_SOURCE_ADMIN_PINS = {
    "manifest": {
        "path": "data/manifests/ppe-lo-cpped-source-quarantine-r1.json",
        "bytes": 17332,
        "sha256": (
            "abdfafcae701f32f502175f14e676db3e605d7646894b85be1131bbe0c4d34c9"
        ),
    },
    "current_receipt": {
        "path": (
            "validation/results/ppe/source-quarantine/"
            "lo-cpped-metadata-acquisition-r2.json"
        ),
        "bytes": 3989,
        "sha256": (
            "1bff485dba9261607236d7ca4f6ba8114d4059302025e9a07a6fbf1c2c0dd1c1"
        ),
    },
    "historical_receipt": {
        "path": (
            "validation/results/ppe/source-quarantine/"
            "lo-cpped-metadata-acquisition-r1.json"
        ),
        "bytes": 3989,
        "sha256": (
            "a8fb15c5839b3f1ddbf585dcdb309357594575efa9323a8683b60b8eceb50308"
        ),
    },
    "schema": {
        "path": (
            "validation/schemas/"
            "ppe-lo-cpped-source-quarantine-receipt-v1.schema.json"
        ),
        "bytes": 7985,
        "sha256": (
            "b3554fe66c474630a874358177c80ff2a02bd197cb40b88cf9270d53f8f2eed4"
        ),
    },
    "validator": {
        "path": "validation/ppe_lo_cpped_source_quarantine.py",
        "bytes": 32339,
        "sha256": (
            "6107e8b5885a386adc5965258f973cb6b0106926d58692dadd2be82b0593bd27"
        ),
    },
}
PPE_LO_CPPED_RECEIPT_SELF_SHA256 = {
    "current": (
        "93760142da146e139d2732f6598a7e804b6c43c3ce2b86739468c4e351009e5d"
    ),
    "historical": (
        "6b91c7e6e2b8f5b17944613bcc5d7d2c8f836516f96097a734c2f25ccc8dc48e"
    ),
}
PPE_LO_CPPED_HISTORICAL_VALIDATOR_PIN = {
    "path": "validation/ppe_lo_cpped_source_quarantine.py",
    "bytes": 32310,
    "sha256": (
        "c5a8b19e1efd9367ee5207a6c5c0cb8ea47748ce2996b5a9fa6392f626c6838f"
    ),
}
PPE_LO_CPPED_MAX_JSON_BYTES = 64 * 1024
GPU_REENTRY_R2_ADMIN_PINS = {
    "receipt": {
        "path": (
            "validation/results/gpu-reentry/r2-executions/"
            "gpu-reentry-r2-live-002/evidence.json"
        ),
        "bytes": 6748,
        "sha256": (
            "67eb1577c1eac1672eb01605735ac4681027319d975f0efc79d1ac88c018ce9e"
        ),
    },
    "plan": {
        "path": (
            "validation/results/gpu-reentry/r2-refresh-20260718/"
            "plan-entrypoint-fix.json"
        ),
        "bytes": 17090,
        "sha256": (
            "432d75dfe541e5ceb68696d7e951a246d87425d90aea4f5f415576385a1bce49"
        ),
    },
    "receipt_schema": {
        "path": "validation/schemas/gpu-reentry-evidence-v2.schema.json",
        "bytes": 7715,
        "sha256": (
            "714c110615294add8a9ab7b97e9eacf3411ec19182ec394e86460558301d44b0"
        ),
    },
    "plan_schema": {
        "path": (
            "validation/schemas/gpu-reentry-refresh-plan-v2.schema.json"
        ),
        "bytes": 14709,
        "sha256": (
            "207f4fa8d19ef0851d1b0588d65e2f961185833f187c605c97d0ff7b127ec10c"
        ),
    },
    "validator": {
        "path": "validation/gpu_reentry_refresh_v2.py",
        "bytes": 73558,
        "sha256": (
            "be2a513c68149c164b4ab8ed86a86c5b4017e22931cb0ac7e5fe414e37217df7"
        ),
    },
}
GPU_REENTRY_R2_MAX_JSON_BYTES = 128 * 1024
DS9_RUNTIME_QUALIFICATION_ADMIN_PINS = {
    "receipt": {
        "path": (
            "validation/results/ds9-runtime-compatibility/releases/"
            "sha256-ced1b59150dbfc040e3ff6afe8e749b2ad5f2c550934242bd7f43ee5bd898c46-"
            "reentry-edc37bb7-smoke-ef7605fa/receipt.json"
        ),
        "bytes": 287460,
        "sha256": (
            "fee39c6a0ade2f654c981e07a23c9ca7411709c9492e2dcb4cf8f2f61a070fd2"
        ),
    },
    "schema": {
        "path": "validation/schemas/ds9-runtime-compatibility-v1.schema.json",
        "bytes": 7169,
        "sha256": (
            "d18333b7637b2033fb3f6b2f658285d59ac4080dcf8d501278d1f9a1151ac189"
        ),
    },
    "validator": {
        "path": "validation/ds9_runtime_compatibility.py",
        "bytes": 83083,
        "sha256": (
            "5d04ecb95e0775b7638914659217f0d4bb34c50f9a73627012aa5a11f71580e6"
        ),
    },
    "runtime_controls": {
        "path": "deepstream/runtime-control-manifest.json",
        "bytes": 3691,
        "sha256": (
            "d3e56b3d8fdea084183f6f4be58cbbece9f46ae90fa3aab58fb478d1b830f66d"
        ),
    },
    "gpu_smoke_evidence": {
        "path": (
            "validation/results/ds9-runtime-compatibility/gpu-smoke/sessions/"
            "ef7605fa413423d93b6c3827247a34eb6b0da7d988e31839bf715a398c613bcc/"
            "gpu-smoke-evidence.json"
        ),
        "bytes": 1071251,
        "sha256": (
            "e4f62d0f21259fc5d04585de3973bb07b5c10eae8739a88528910e3b76de887b"
        ),
    },
    "parser_build_receipt": {
        "path": (
            "validation/results/ds9-runtime-compatibility/parser-bootstrap/"
            "117be9a7164f58a74adae2f4f4244796a6a630aa33d5fa868ce5b6abb6786c8a/"
            "pass-2-production/production-build-receipt.json"
        ),
        "bytes": 13904,
        "sha256": (
            "58ddee91379621ad6eb472e6750a6d8aa58b7d1407f0308e66a0dec999d7e5ec"
        ),
    },
}
DS9_RUNTIME_QUALIFICATION_MAX_BYTES = 2 * 1024 * 1024
FULL_STACK_BENCHMARK_ADMIN_PINS = {
    "plan": {
        "path": "validation/plans/deepstream-full-stack-benchmark-v1.json",
        "bytes": 18148,
        "sha256": (
            "7a68c69fdf881d17fb31508a3d4f4b39c5e498dc137490cecb01cac401036020"
        ),
    },
    "plan_schema": {
        "path": (
            "validation/schemas/"
            "deepstream-full-stack-benchmark-plan-v1.schema.json"
        ),
        "bytes": 17700,
        "sha256": (
            "76361f7622ecd3c9e5d7c844d3b218f095e6e939f1885492749e585d66b120d2"
        ),
    },
    "receipt_schema": {
        "path": (
            "validation/schemas/"
            "deepstream-full-stack-benchmark-receipt-v1.schema.json"
        ),
        "bytes": 13389,
        "sha256": (
            "637d2905c7b845a322c2ed826e2e558d171bedd1e4fa0332b906a1ca3660d471"
        ),
    },
    "validator": {
        "path": "validation/full_stack_benchmark_contract_v1.py",
        "bytes": 43569,
        "sha256": (
            "f9010e2ca53c35858aea61404e5e9cfeae4abcca560178312ce02a940c9cee2e"
        ),
    },
}
FULL_STACK_BENCHMARK_MAX_BYTES = 128 * 1024
FULL_STACK_BENCHMARK_PLAN_FINGERPRINT = (
    "5424a4e7f58f1c6b7855f96f0884083e324ab00f94da7bae81d17caab16fdfe5"
)
FULL_STACK_BENCHMARK_EXPECTED_IMAGE_ID = (
    "sha256:ced1b59150dbfc040e3ff6afe8e749b2ad5f2c550934242bd7f43ee5bd898c46"
)
FULL_STACK_BENCHMARK_EXPECTED_PARSER_SHA256 = (
    "2aa44a3395047ae371bee857476b1e78b438776c8a6b9643a055a16a0f15a7ae"
)
PPE_PROVENANCE_PLAN_PIN = {
    "path": "data/manifests/ppe-mendeley-v6-provenance-review-r2.plan.json",
    "bytes": 12111,
    "sha256": (
        "e7a9afbf4c5c9b78c0dcfdb4548d1ff2dccfabf0b36cda0de119eaa03817982c"
    ),
}
PPE_PROVENANCE_RECEIPT_PIN = {
    "path": (
        "validation/results/ppe/provenance/"
        "mendeley-ppe-v6-provenance-review-r2.json"
    ),
    "bytes": 21123,
    "sha256": (
        "09eb727102dd51b00be57ef96855222687df322cd8f20b2cf8ff611b62bede9f"
    ),
    "receipt_sha256": (
        "9358c9d6d67b302c0161c4de6d50389b8c683f8c005f8b0042757dec0ae195fa"
    ),
}
PPE_PROVENANCE_CODE_PIN = {
    "path": "ppe_dataset/provenance.py",
    "bytes": 38293,
    "sha256": (
        "ecd0ca9477e81f552bbee72471b737212972a8a317f0fc217548119cf0e2d58b"
    ),
}
PPE_PROVENANCE_SCHEMA_PINS = {
    "plan": {
        "path": (
            "ppe_dataset/schemas/"
            "ppe-seed-provenance-review-plan-v1.schema.json"
        ),
        "bytes": 11084,
        "sha256": (
            "84ebfafaf7ffc3ef8a3e04f9f200348327619792b39466e757e8cbbcebe02bc8"
        ),
    },
    "receipt": {
        "path": (
            "ppe_dataset/schemas/"
            "ppe-seed-provenance-review-receipt-v1.schema.json"
        ),
        "bytes": 15688,
        "sha256": (
            "02a4b060061f9b5f1eea0c27278e0e7c34ef2dc451d6cdf051daeb04c5d3afd8"
        ),
    },
}
PPE_NORMALIZATION_SUPERSEDED_R1_PINS = {
    "plan": {
        "path": "data/manifests/ppe-normalization-mendeley-v6.plan.json",
        "bytes": 4852,
        "sha256": (
            "41c9943dc0487a75f859a825f77da0e281b8699cb95b40604ac8c293e71df463"
        ),
    },
    "assessment": {
        "path": (
            "validation/results/ppe/normalization/"
            "mendeley-ppe-v6-normalization-assessment-r1.json"
        ),
        "bytes": 34452,
        "sha256": (
            "c182d78789d2cf927974803db23b233257b2d21c500c8ad3519f27f23f0361cb"
        ),
        "receipt_sha256": (
            "ae5b5473edbb3b8b76ffb934b09679d30d52760977982af8b74ad44ca56f3c1f"
        ),
    },
}
PPE_NORMALIZATION_PLAN_PIN = {
    "path": "data/manifests/ppe-normalization-mendeley-v6-r2.plan.json",
    "bytes": 4910,
    "sha256": (
        "a14e31b5c0aa08827ff935fff3a0bf2476495207ef2877ed6f00b13993450d5f"
    ),
}
PPE_NORMALIZATION_ASSESSMENT_PIN = {
    "path": (
        "validation/results/ppe/normalization/"
        "mendeley-ppe-v6-normalization-assessment-r2.json"
    ),
    "bytes": 37289,
    "sha256": (
        "033817d5782fc0ca3259eefc88fed5c4c32ba31e59bda5cda0a3ac4d72fb8327"
    ),
    "receipt_sha256": (
        "04af74793d48e9af477c85d8800170c71fc0d8b45100e79c19d94b3bd6acb704"
    ),
}
PPE_NORMALIZATION_SCHEMA_PINS = {
    "plan": {
        "path": "ppe_dataset/schemas/ppe-normalization-plan-v2.schema.json",
        "bytes": 7961,
        "sha256": (
            "5a6a7aa85b2e2d46b942dfcabd4d49103e77a8d405c1255979c37028309d00c3"
        ),
    },
    "assessment": {
        "path": (
            "ppe_dataset/schemas/"
            "ppe-normalization-assessment-receipt-v2.schema.json"
        ),
        "bytes": 4054,
        "sha256": (
            "d7922922611709f6a4d85d2faf38c82bf6be0cd5737464eca1da1a497f74afbd"
        ),
    },
    "canonical_dataset": {
        "path": "ppe_dataset/schemas/person-equipment-decisions-v2.schema.json",
        "bytes": 10109,
        "sha256": (
            "968e9b13060940fecede5b8771ca5271b3790a2efd4233545a021759b8822df8"
        ),
    },
}
PPE_NORMALIZATION_MAX_JSON_BYTES = 512 * 1024
PPE_SUPERSEDED_QUARANTINE_LINEAGE = {
    "path": "validation/results/ppe/seeds/mendeley-ppe-v6-quarantine.json",
    "sha256": (
        "d4f61854a301800ad7bfea0258c911bed23d3471714257d18fd90b6579d4e8bc"
    ),
    "receipt_sha256": (
        "8a17f97d52eb4272b8c6f6e83a3213c690ae131d26a1476e6a0d91d058615aff"
    ),
}
POSE_EXPORT_PROVENANCE_PIN = {
    "path": "models/pose/provenance-plan.json",
    "bytes": 3931,
    "sha256": "4d1fe6056b5b9e9f9920764c389e390f53101388676c96f93f60fe03b5d4fb4c",
}
POSE_EXPORT_PATHS = {
    "semantic_contract": "models/pose/semantic-contract-v1.json",
    "plan_640": "models/pose/export-plans/yolo26s-pose-640-r1.json",
    "plan_960": "models/pose/export-plans/yolo26s-pose-960-r1.json",
    "harness": "models/pose/export_pose.py",
    "wrapper": "models/pose/run_export.sh",
    "onnx_validator": "models/pose/validate_onnx.py",
}
POSE_GT_EVIDENCE_PINS = {
    "source_manifest": {
        "path": "data/manifests/pose-gt-evaluation-sources.json",
        "bytes": 15650,
        "sha256": (
            "639748b4d8770d31435721f5f35dab371a98623060609b403b8ac4d604dfcdc6"
        ),
    },
    "source_schema": {
        "path": "validation/schemas/pose-gt-evaluation-sources-v1.schema.json",
        "bytes": 43536,
        "sha256": (
            "9e7399833d8f87694f912dce3de806cc9773bdb0b5792a5297b22c06c4b7d734"
        ),
    },
    "source_validator": {
        "path": "validation/pose_gt_sources.py",
        "bytes": 23422,
        "sha256": (
            "7c28e3139493c73ca0a652ef0c4ebce7cd76b237170ea28f608b99cd1f1a810a"
        ),
    },
}
POSE_PCK_EVIDENCE_PINS = {
    "evaluator": {
        "path": "validation/pose_pck_evaluator.py",
        "bytes": 55501,
        "sha256": (
            "7d91c35f3b5866615776e46ae6dc6c46d8fd01554f3fc20942fa8dbcf5973354"
        ),
    },
    "ground_truth_schema": {
        "path": "validation/schemas/pose-pck-ground-truth-v1.schema.json",
        "bytes": 10051,
        "sha256": (
            "03d1ddf625b32ff4c1fcdb2c4ae1f86abe3ae49ddf0bf60aa8f1c11c13d67b67"
        ),
    },
    "predictions_schema": {
        "path": "validation/schemas/pose-pck-predictions-v1.schema.json",
        "bytes": 6655,
        "sha256": (
            "7bd9bb14b7725dd752c57ea9969a2242d5818a3c315997e87363c44121cd3e69"
        ),
    },
    "receipt_schema": {
        "path": "validation/schemas/pose-pck-evaluation-receipt-v1.schema.json",
        "bytes": 14102,
        "sha256": (
            "3272fc739423668261fa4dedb1ce611882805ba62a39b4d6fc83e07996623653"
        ),
    },
}
POSE_COCO17_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
POSE_PERMISSIVE_CHALLENGER_PATHS = {
    "plan": (
        "models/pose/challengers/mmpose-yoloxpose-s/"
        "provenance-plan-v1.json"
    ),
    "checkpoint": (
        "models/pose/candidates/mmpose-yoloxpose-s/"
        "yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth"
    ),
    "receipt": (
        "validation/results/pose/models/"
        "mmpose-yoloxpose-s-structural-r1.json"
    ),
    "validator": "validation/pose_mmpose_yoloxpose_structural.py",
    "schema": (
        "validation/schemas/"
        "pose-mmpose-yoloxpose-structural-receipt-v1.schema.json"
    ),
}
POSE_PERMISSIVE_CHALLENGER_RECEIPT_SHA256 = (
    "583270ea19bfab69da9cf7c4490502a209ef2228aaec6e8cb5d098d3341b2bd1"
)
POSE_MMPOSE_ONNX_PREFLIGHT_PINS = {
    "historical_r1": {
        "path": (
            "validation/results/pose/models/"
            "mmpose-yoloxpose-s-onnx-lane-r1.json"
        ),
        "bytes": 11684,
        "sha256": (
            "84ed90c532b5d87391fd3e11edd2a810866f2157c0cc8fda981d974964f71bc0"
        ),
    },
    "current_r2": {
        "path": (
            "validation/results/pose/models/"
            "mmpose-yoloxpose-s-onnx-lane-r2.json"
        ),
        "bytes": 11541,
        "sha256": (
            "62bbd3faed88d13df3de3673f5cc88a33d6f97397e7b76213c5d4c5bc657ed13"
        ),
    },
    "schema": {
        "path": (
            "validation/schemas/"
            "pose-mmpose-yoloxpose-onnx-lane-receipt-v1.schema.json"
        ),
        "bytes": 14562,
        "sha256": (
            "1dab30da481870ce48a4764b8127dafd9f25488180e1a40e99f8a10cd4c5153e"
        ),
    },
}
POSE_MMPOSE_ONNX_PREFLIGHT_SELF_SHA256 = {
    "historical_r1": (
        "bf9835d33cac2cdf7bc98e509633998fd321183a86483826055dd83e95623044"
    ),
    "current_r2": (
        "2fd5a445d5a51a4377def5d900515e2adf5a3773d160e2b8bc5e0da06e7686e0"
    ),
}
POSE_MMPOSE_ONNX_PREFLIGHT_BLOCKERS = {
    "historical_r1": (
        "mmdeploy_checkout_missing",
        "mmdeploy_distribution_missing_or_wrong",
        "compiled_mmcv_ops_missing",
    ),
    "current_r2": (
        "mmdeploy_distribution_missing_or_wrong",
        "compiled_mmcv_ops_missing",
    ),
}
POSE_MMPOSE_PERMISSION_PROBE_R9_PINS = {
    "plan": {
        "path": (
            "validation/results/pose/export-environment/"
            "mmpose-yoloxpose-s-permission-probe-plan-r9.json"
        ),
        "bytes": 13391,
        "sha256": (
            "0aafc1e63d4abc5b3686d4ee0a703bf115d872489c1628122a38e06c2689b68e"
        ),
    },
    "attempt_receipt": {
        "path": (
            "validation/results/pose/export-environment/child-image-attempts/"
            "child-v8-probe-r9-001/attempt-receipt.json"
        ),
        "bytes": 9021,
        "sha256": (
            "906d816e4cee1df91b501eceb7fcfbc85092671102d4778a91a04492c0ed5f0f"
        ),
    },
    "probe_receipt": {
        "path": (
            "validation/results/pose/export-environment/child-image-attempts/"
            "child-v8-probe-r9-001/probe/probe-receipt.json"
        ),
        "bytes": 7088,
        "sha256": (
            "4984b5a89f8810b72f9469f5b712e78b5cd104310a9494c56d0060f3fdfced6a"
        ),
    },
}
POSE_MMPOSE_PERMISSION_PROBE_R9_PLAN_PAYLOAD_SHA256 = (
    "e403ae81fc442e44c59929b072127b6fd5eb2ceb2e43f5cc7f3a4edbc04bd4d6"
)
POSE_MMPOSE_PERMISSION_PROBE_R9_ATTEMPT_PAYLOAD_SHA256 = (
    "29fb626d39d64bd4a546afaa3799cade0311752c37bfe775db781a5146a7c456"
)
POSE_MMPOSE_PERMISSION_PROBE_R9_PROBE_PAYLOAD_SHA256 = (
    "b817bfd3beeea3c6ba29d30f42460164654588f67ad289a70e1d5dc7f45a6f89"
)
POSE_MMPOSE_PERMISSION_PROBE_R9_IMAGE_ID = (
    "sha256:8ba836b80502277ce999ffb8b0c6a2c29368f09cb12cbe07abc10028e821915f"
)
POSE_MMPOSE_EXPORT_R10_FAILURE_PINS = {
    "plan": {
        "path": (
            "validation/results/pose/models/"
            "mmpose-yoloxpose-s-onnx-export-plan-r10.json"
        ),
        "bytes": 26659,
        "sha256": (
            "c1917bce08f884f547a32ac471a1de1fd272be919f141a8301425974ca460c9d"
        ),
    },
    "receipt": {
        "path": (
            "validation/results/pose/models/"
            "mmpose-yoloxpose-s-onnx-export-r10-runs/"
            "cpu-export-001.export-receipt.json"
        ),
        "bytes": 6192,
        "sha256": (
            "bfbf5bb854b83b929f8d62d654487ad9ed3931c93f413b147acd28b0fb864ae8"
        ),
    },
    "failure_log": {
        "path": (
            "validation/results/pose/models/"
            "mmpose-yoloxpose-s-onnx-export-r10-runs/"
            "cpu-export-001/docker-640.log"
        ),
        "bytes": 5875,
        "sha256": (
            "cca8e3e69bcd6843e76d00f2fef6b7855907dd23700749f6a06f172099b7c790"
        ),
    },
}
POSE_MMPOSE_EXPORT_R10_PLAN_FINGERPRINT = (
    "322383a81f14f223162c189e60503796e9d37413d5e3c95b905ad8d8c00a3a1e"
)
POSE_MMPOSE_EXPORT_R10_RECEIPT_FINGERPRINT = (
    "ca0a1e3ba0507051293c31f9d48714bb9827994c7cce3e4178323fdbdd94f39a"
)
POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PINS = {
    "plan": {
        "path": (
            "validation/results/pose/models/"
            "mmpose-yoloxpose-s-shape-diagnostic-plan-r11.json"
        ),
        "bytes": 17915,
        "sha256": (
            "10023d320c095ff3d7b4a8a1d5f9415e00717f0d669165808a10fb39a226875b"
        ),
    },
    "receipt": {
        "path": (
            "validation/results/pose/models/"
            "mmpose-yoloxpose-s-shape-diagnostic-r11-runs/"
            "cpu-shape-diag-001.diagnostic-receipt.json"
        ),
        "bytes": 4192,
        "sha256": (
            "c8af7366b5050753dcdc94917db07d75aeeb252bd854cf91b492da1c4389c20f"
        ),
    },
    "profile_receipt": {
        "path": (
            "validation/results/pose/models/"
            "mmpose-yoloxpose-s-shape-diagnostic-r11-runs/"
            "cpu-shape-diag-001/diagnostic/shape-diagnostic-receipt.json"
        ),
        "bytes": 47373,
        "sha256": (
            "96c5b3a2aecebc6e47c9998d2d179c8665350711cccfad4c95a9c9531b85acef"
        ),
    },
}
POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PLAN_FINGERPRINT = (
    "7a69baaf860c698d17c57ae5f73b0d122acdeeee3023b9570889d9251229552d"
)
POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_RECEIPT_FINGERPRINT = (
    "7fb9050c9f571942a96ae297966a16a651331c724266a05049d77f84a210b3e7"
)
POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PROFILE_FINGERPRINT = (
    "68ae724f33693d8e18218fcf53d78182c9942d5c94ab250143c438aee175e9e7"
)
POSE_MODEL_MAX_BYTES = 50 * 1024 * 1024
POSE_GT_SOURCE_FINGERPRINT = (
    "b4fb5c98286403c3b7422e432e0c58623c047c0bb0e442fbde4d853bc6a2f410"
)
POSE_MAX_JSON_BYTES = 512 * 1024
OPEN_VIDEO_EXPECTED_CANDIDATE_JOBS = 24
OPEN_VIDEO_EXPECTED_SOURCE_FRAMES = 21
OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS = 42
OPEN_VIDEO_EXPECTED_ASSETS = 168
OPEN_VIDEO_MINIMUM_DISTINCT_VIDEO_TYPES = 10


def _default_schema_root() -> Path:
    return Path(__file__).resolve().parents[1] / "validation/schemas"


def _default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    label: str
    candidates: tuple[str, ...]
    media_type: str = "application/json"
    schema_prefix: str | None = None
    raw_download_allowed: bool = True
    max_bytes_override: int | None = None


ARTIFACTS: dict[str, ArtifactSpec] = {
    "product_finalization_v2_receipt": ArtifactSpec(
        key="product_finalization_v2_receipt",
        label="Three-module product finalization commit v2",
        candidates=("product-finalization/v2/current/receipt.json",),
        schema_prefix="deepsafe.product-finalization-receipt/",
        raw_download_allowed=False,
        max_bytes_override=HARD_MAX_ARTIFACT_BYTES,
    ),
    "finalization_receipt": ArtifactSpec(
        key="finalization_receipt",
        label="Validation bundle commit receipt",
        candidates=("finalization/current/receipt.json",),
        schema_prefix="deepsafe.validation-finalization-receipt/",
        raw_download_allowed=False,
        max_bytes_override=HARD_MAX_ARTIFACT_BYTES,
    ),
    "objective_completion_json": ArtifactSpec(
        key="objective_completion_json",
        label="Validation objective completion ledger (JSON)",
        candidates=("objective-completion/current/report.json",),
        schema_prefix="deepsafe.validation-objective-completion/",
        raw_download_allowed=False,
        max_bytes_override=HARD_MAX_ARTIFACT_OVERRIDE_BYTES,
    ),
    "objective_completion_markdown": ArtifactSpec(
        key="objective_completion_markdown",
        label="Validation objective completion ledger (Markdown)",
        candidates=("objective-completion/current/report.md",),
        media_type="text/markdown; charset=utf-8",
        raw_download_allowed=False,
        max_bytes_override=8 * 1024 * 1024,
    ),
    "product_readiness_json": ArtifactSpec(
        key="product_readiness_json",
        label="Three-module product readiness (JSON)",
        candidates=("product-readiness/current/report.json",),
        schema_prefix="deepsafe.product-readiness/",
        raw_download_allowed=False,
        max_bytes_override=HARD_MAX_ARTIFACT_OVERRIDE_BYTES,
    ),
    "product_readiness_markdown": ArtifactSpec(
        key="product_readiness_markdown",
        label="Three-module product readiness (Markdown)",
        candidates=("product-readiness/current/report.md",),
        media_type="text/markdown; charset=utf-8",
        raw_download_allowed=False,
        max_bytes_override=8 * 1024 * 1024,
    ),
    "campaign_report_json": ArtifactSpec(
        key="campaign_report_json",
        label="Fail-closed campaign acceptance report (JSON)",
        candidates=("campaign-report/report.json",),
        schema_prefix="deepsafe.validation-campaign-report/",
        raw_download_allowed=False,
        max_bytes_override=HARD_MAX_ARTIFACT_OVERRIDE_BYTES,
    ),
    "campaign_report_markdown": ArtifactSpec(
        key="campaign_report_markdown",
        label="Fail-closed campaign acceptance report (Markdown)",
        candidates=("campaign-report/report.md",),
        media_type="text/markdown; charset=utf-8",
        raw_download_allowed=False,
        max_bytes_override=8 * 1024 * 1024,
    ),
    "scene_benchmark_summary": ArtifactSpec(
        key="scene_benchmark_summary",
        label="12-camera scene benchmark matrix",
        candidates=("scene-benchmark/matrix-summary.json",),
        schema_prefix="deepsafe.scene-benchmark-matrix/",
    ),
    "scene_benchmark_preflight": ArtifactSpec(
        key="scene_benchmark_preflight",
        label="Scene benchmark GPU preflight",
        candidates=("scene-benchmark/preflight.json",),
        raw_download_allowed=False,
    ),
    "caviar_batch_aggregate": ArtifactSpec(
        key="caviar_batch_aggregate",
        label="CAVIAR batch aggregate",
        candidates=("caviar/batch-aggregate.json",),
        schema_prefix="deepsafe.caviar-batch-aggregate/",
    ),
    "caviar_batch_manifest": ArtifactSpec(
        key="caviar_batch_manifest",
        label="CAVIAR batch manifest",
        candidates=("caviar/batch-manifest.json",),
        schema_prefix="deepsafe.caviar-batch-plan/",
        raw_download_allowed=False,
    ),
    "caviar_batch_report": ArtifactSpec(
        key="caviar_batch_report",
        label="CAVIAR compact report",
        candidates=("caviar/batch-aggregate.md",),
        media_type="text/markdown; charset=utf-8",
    ),
    "rlivit_public_status": ArtifactSpec(
        key="rlivit_public_status",
        label="R-LiViT DeepStream GT public status",
        candidates=("rlivit/current/status.json",),
        schema_prefix="deepsafe.rlivit-public-status/",
        raw_download_allowed=False,
    ),
    "rlivit_mp4_receipt": ArtifactSpec(
        key="rlivit_mp4_receipt",
        label="R-LiViT CPU MP4 materialization status",
        candidates=("rlivit/mp4-batch-receipt.json",),
        schema_prefix="deepsafe.rlivit-mp4-admin-receipt/",
        raw_download_allowed=False,
    ),
    "open_video_plan": ArtifactSpec(
        key="open_video_plan",
        label="Open-video review plan",
        candidates=(
            "open-video-review/campaign-plan.json",
            "open-video-review/dry-run-plan.json",
        ),
        schema_prefix="deepsafe.open-video-review-plan/",
        raw_download_allowed=False,
    ),
    "open_video_review": ArtifactSpec(
        key="open_video_review",
        label="Open-video qualitative review",
        candidates=("open-video-review/campaign-review.json",),
        schema_prefix="deepsafe.open-video-review/",
    ),
    "endurance_campaign_resolved": ArtifactSpec(
        key="endurance_campaign_resolved",
        label="Seven-day endurance resolved campaign",
        candidates=("endurance/current/campaign-resolved.json",),
        schema_prefix="deepsafe.endurance-campaign/",
        raw_download_allowed=False,
        max_bytes_override=HARD_MAX_ARTIFACT_BYTES,
    ),
    "endurance_checkpoint": ArtifactSpec(
        key="endurance_checkpoint",
        label="Seven-day endurance checkpoint",
        candidates=("endurance/current/checkpoint.json",),
        schema_prefix="deepsafe.endurance-checkpoint/",
        raw_download_allowed=False,
        max_bytes_override=HARD_MAX_ARTIFACT_OVERRIDE_BYTES,
    ),
    "endurance_status": ArtifactSpec(
        key="endurance_status",
        label="Seven-day endurance status",
        candidates=("endurance/current/status.json",),
        schema_prefix="deepsafe.endurance-status/",
        raw_download_allowed=False,
    ),
    "endurance_live": ArtifactSpec(
        key="endurance_live",
        label="Endurance live heartbeat",
        candidates=("endurance/current/live.json",),
        schema_prefix="deepsafe.endurance-live/",
        raw_download_allowed=False,
    ),
    "gpu_reentry_evidence": ArtifactSpec(
        key="gpu_reentry_evidence",
        label="GPU re-entry current evidence",
        candidates=("gpu-reentry/current/evidence.json",),
        schema_prefix="deepsafe.gpu-reentry-evidence/",
        raw_download_allowed=False,
    ),
    "gpu_reentry_verification": ArtifactSpec(
        key="gpu_reentry_verification",
        label="GPU re-entry current verification",
        candidates=("gpu-reentry/current/verification.json",),
        schema_prefix="deepsafe.gpu-reentry-verification/",
        raw_download_allowed=False,
    ),
    "loaf_batch_aggregate": ArtifactSpec(
        key="loaf_batch_aggregate",
        label="LOAF DeepStream batch aggregate",
        candidates=("loaf/val-20-25m/deepstream/batch-aggregate.json",),
        schema_prefix="deepsafe.loaf-deepstream-batch-aggregate/",
        raw_download_allowed=False,
    ),
    "loaf_batch_plan": ArtifactSpec(
        key="loaf_batch_plan",
        label="LOAF DeepStream batch plan",
        candidates=("loaf/val-20-25m/deepstream/dry-run-plan.json",),
        schema_prefix="deepsafe.loaf-deepstream-batch-plan/",
        raw_download_allowed=False,
        max_bytes_override=MAX_LOAF_PLAN_ARTIFACT_BYTES,
    ),
    "loaf_distance_bin_preparation": ArtifactSpec(
        key="loaf_distance_bin_preparation",
        label="LOAF distance-bin preparation",
        candidates=(
            "loaf/val-20-25m/distance-bins/preparation-manifest.json",
        ),
        schema_prefix="deepsafe.loaf-distance-bin-preparation/",
        raw_download_allowed=False,
    ),
    "loaf_distance_bin_evaluation_plan": ArtifactSpec(
        key="loaf_distance_bin_evaluation_plan",
        label="LOAF distance-bin evaluation plan",
        candidates=("loaf/val-20-25m/distance-bins/evaluation-plan.json",),
        schema_prefix="deepsafe.loaf-distance-bin-evaluation-plan/",
        raw_download_allowed=False,
    ),
    "loaf_distance_bin_aggregate": ArtifactSpec(
        key="loaf_distance_bin_aggregate",
        label="LOAF distance-bin aggregate",
        candidates=("loaf/val-20-25m/distance-bins/aggregate.json",),
        schema_prefix="deepsafe.loaf-distance-bin-aggregate/",
        raw_download_allowed=False,
    ),
    "site_distance_plan": ArtifactSpec(
        key="site_distance_plan",
        label="Deployment-site calibrated 20-25 m evaluation plan",
        candidates=("distance-25m/evaluation-plan.json",),
        schema_prefix="deepsafe.site-distance-evaluation-plan/",
        raw_download_allowed=False,
    ),
    "site_distance_evaluation": ArtifactSpec(
        key="site_distance_evaluation",
        label="Deployment-site calibrated 20-25 m final evaluation",
        candidates=("distance-25m/evaluation.json",),
        schema_prefix="deepsafe.distance-validation/",
        raw_download_allowed=False,
    ),
    "site_distance_evaluation_v2": ArtifactSpec(
        key="site_distance_evaluation_v2",
        label="Inclusive deployment-site 20-25 m final receipt v2",
        candidates=("distance-25m/evaluation-final-v2.json",),
        schema_prefix="deepsafe.site-distance-evaluation-final/v2",
        raw_download_allowed=False,
    ),
}


@dataclass(frozen=True)
class ArtifactRead:
    spec: ArtifactSpec
    state: str
    relative_path: str
    size_bytes: int | None = None
    content: bytes | None = None
    value: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return self.state == "ok"


@dataclass(frozen=True)
class ValidationArtifact:
    content: bytes
    media_type: str
    filename: str


class ValidationArtifactError(Exception):
    def __init__(self, state: str):
        super().__init__(state)
        self.state = state


@dataclass(frozen=True)
class WorkspacePinRead:
    """Result of one descriptor-bound, workspace-relative pin read."""

    state: str
    content: bytes | None = None

    @property
    def available(self) -> bool:
        return self.state == "ok"


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _schema_reference(root: dict[str, Any], reference: Any) -> dict[str, Any]:
    if not isinstance(reference, str) or not reference.startswith("#/"):
        raise ValueError("unsupported schema reference")
    current: Any = root
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            raise ValueError("unresolved schema reference")
        current = current[token]
    if not isinstance(current, dict):
        raise ValueError("schema reference does not resolve to an object")
    return current


def _schema_matches_node(
    value: Any, schema: Any, root: dict[str, Any]
) -> bool:
    try:
        _validate_schema_node(value, schema, root)
    except (TypeError, ValueError):
        return False
    return True


def _validate_schema_node(value: Any, schema: Any, root: dict[str, Any]) -> None:
    """Dependency-free Draft-2020-12 subset used by checked-in admin contracts."""

    if schema is True:
        return
    if schema is False or not isinstance(schema, dict):
        raise ValueError("invalid or rejecting schema node")
    if "$ref" in schema:
        _validate_schema_node(value, _schema_reference(root, schema["$ref"]), root)
    for component in schema.get("allOf", []):
        _validate_schema_node(value, component, root)
    if "oneOf" in schema:
        matches = sum(
            _schema_matches_node(value, component, root)
            for component in schema["oneOf"]
        )
        if matches != 1:
            raise ValueError("oneOf did not match exactly once")
    if "anyOf" in schema and not any(
        _schema_matches_node(value, component, root)
        for component in schema["anyOf"]
    ):
        raise ValueError("anyOf did not match")
    if "if" in schema:
        branch = "then" if _schema_matches_node(value, schema["if"], root) else "else"
        if branch in schema:
            _validate_schema_node(value, schema[branch], root)
    if "not" in schema and _schema_matches_node(value, schema["not"], root):
        raise ValueError("forbidden schema matched")
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ValueError("const mismatch")
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        raise ValueError("enum mismatch")
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        expected_types = [expected_type]
    elif isinstance(expected_type, list) and all(
        isinstance(item, str) for item in expected_type
    ):
        expected_types = expected_type
    elif expected_type is None:
        expected_types = []
    else:
        raise ValueError("invalid schema type")
    if expected_types and not any(
        _schema_type_matches(value, item) for item in expected_types
    ):
        raise ValueError("type mismatch")

    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or any(key not in value for key in required):
            raise ValueError("required property missing")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("invalid properties schema")
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema_node(value[key], child_schema, root)
        extras = set(value) - set(properties)
        additional = schema.get("additionalProperties", True)
        if additional is False and extras:
            raise ValueError("additional property is forbidden")
        if isinstance(additional, dict):
            for key in extras:
                _validate_schema_node(value[key], additional, root)
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            raise ValueError("too few properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            raise ValueError("too many properties")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError("too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError("too many items")
        if schema.get("uniqueItems") is True:
            for index, item in enumerate(value):
                if any(_json_equal(item, prior) for prior in value[:index]):
                    raise ValueError("array items are not unique")
        prefix = schema.get("prefixItems", [])
        if not isinstance(prefix, list):
            raise ValueError("invalid prefixItems")
        for index, child_schema in enumerate(prefix):
            if index < len(value):
                _validate_schema_node(value[index], child_schema, root)
        items = schema.get("items", True)
        if items is False and len(value) > len(prefix):
            raise ValueError("trailing array items are forbidden")
        if isinstance(items, dict):
            start = len(prefix) if prefix else 0
            for index in range(start, len(value)):
                _validate_schema_node(value[index], items, root)

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError("string is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError("string is too long")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError("string pattern mismatch")
        if schema.get("format") == "date-time":
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("invalid date-time") from exc
            if parsed.tzinfo is None:
                raise ValueError("date-time has no timezone")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("number is not finite")
        if "minimum" in schema and number < schema["minimum"]:
            raise ValueError("number is below minimum")
        if "exclusiveMinimum" in schema and number <= schema["exclusiveMinimum"]:
            raise ValueError("number is below exclusive minimum")
        if "maximum" in schema and number > schema["maximum"]:
            raise ValueError("number is above maximum")
        if "exclusiveMaximum" in schema and number >= schema["exclusiveMaximum"]:
            raise ValueError("number is above exclusive maximum")


def _max_artifact_bytes() -> int:
    raw = os.getenv(
        "DEEPSAFE_VALIDATION_MAX_ARTIFACT_BYTES",
        str(DEFAULT_MAX_ARTIFACT_BYTES),
    )
    try:
        requested = int(raw)
    except ValueError:
        requested = DEFAULT_MAX_ARTIFACT_BYTES
    return min(max(requested, 1), HARD_MAX_ARTIFACT_BYTES)


def _validation_root() -> Path:
    return Path(os.getenv("DEEPSAFE_VALIDATION_ROOT", str(DEFAULT_VALIDATION_ROOT)))


def _has_symlink_component(root: Path, candidate: Path) -> bool:
    """Reject leaf and intermediate symlinks below an already-resolved root."""

    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


class ArtifactReader:
    """Read fixed relative paths without allowing traversal or unbounded input."""

    def __init__(
        self,
        root: Path | None = None,
        max_bytes: int | None = None,
        *,
        workspace_root: Path | None = None,
        schema_root: Path | None = None,
    ):
        configured_root = root or _validation_root()
        self.root_error = False
        try:
            self.root = configured_root.resolve()
        except (OSError, RuntimeError):
            self.root = configured_root.absolute()
            self.root_error = True
        self.max_bytes = max_bytes if max_bytes is not None else _max_artifact_bytes()
        self.max_bytes = min(max(int(self.max_bytes), 1), HARD_MAX_ARTIFACT_BYTES)
        configured_workspace = workspace_root or Path(
            os.getenv(
                "DEEPSAFE_VALIDATION_WORKSPACE_ROOT",
                str(_default_workspace_root()),
            )
        )
        configured_schemas = schema_root or Path(
            os.getenv(
                "DEEPSAFE_VALIDATION_SCHEMA_ROOT",
                str(_default_schema_root()),
            )
        )
        self.workspace_root_error = False
        self.schema_root_error = False
        try:
            self.workspace_root = configured_workspace.resolve(strict=True)
            if not self.workspace_root.is_dir():
                self.workspace_root_error = True
        except (OSError, RuntimeError):
            self.workspace_root = configured_workspace.absolute()
            self.workspace_root_error = True
        try:
            self.schema_root = configured_schemas.resolve(strict=True)
            if not self.schema_root.is_dir():
                self.schema_root_error = True
        except (OSError, RuntimeError):
            self.schema_root = configured_schemas.absolute()
            self.schema_root_error = True
        self._cache: dict[str, ArtifactRead] = {}
        self._schema_cache: dict[str, dict[str, Any] | None] = {}

    def read(self, key: str) -> ArtifactRead:
        if key in self._cache:
            return self._cache[key]
        spec = ARTIFACTS.get(key)
        if spec is None:
            raise KeyError(key)
        result = self._read_spec(spec)
        self._cache[key] = result
        return result

    def _read_spec(self, spec: ArtifactSpec) -> ArtifactRead:
        if self.root_error:
            return ArtifactRead(spec, "invalid_root", spec.candidates[0])
        read_limit = self.max_bytes
        if spec.max_bytes_override is not None:
            read_limit = min(
                max(int(spec.max_bytes_override), 1),
                HARD_MAX_ARTIFACT_OVERRIDE_BYTES,
            )
        for relative in spec.candidates:
            requested = self.root / relative
            if _has_symlink_component(self.root, requested):
                return ArtifactRead(spec, "unsafe_path", relative)
            try:
                candidate = requested.resolve(strict=True)
            except FileNotFoundError:
                continue
            except (OSError, RuntimeError):
                return ArtifactRead(spec, "io_error", relative)

            try:
                candidate.relative_to(self.root)
            except ValueError:
                return ArtifactRead(spec, "unsafe_path", relative)

            try:
                info_before = candidate.lstat()
                if not stat.S_ISREG(info_before.st_mode):
                    return ArtifactRead(spec, "not_a_file", relative)
                if spec.key in {
                    "finalization_receipt",
                    "product_finalization_v2_receipt",
                } and info_before.st_nlink != 1:
                    return ArtifactRead(spec, "unsafe_path", relative)
                size = candidate.stat().st_size
                if size > read_limit:
                    return ArtifactRead(spec, "too_large", relative, size_bytes=size)
                with candidate.open("rb") as handle:
                    content = handle.read(read_limit + 1)
                if len(content) > read_limit:
                    return ArtifactRead(
                        spec, "too_large", relative, size_bytes=len(content)
                    )
            except OSError:
                return ArtifactRead(spec, "io_error", relative)

            if spec.media_type.startswith("application/json"):
                try:
                    value = (
                        strict_json_loads(content)
                        if spec.key
                        in {
                            "finalization_receipt",
                            "product_finalization_v2_receipt",
                            "objective_completion_json",
                            "product_readiness_json",
                            "endurance_campaign_resolved",
                            "endurance_checkpoint",
                            "endurance_status",
                        }
                        else json.loads(content.decode("utf-8"))
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                    return ArtifactRead(
                        spec, "invalid_json", relative, size_bytes=len(content)
                    )
                if not isinstance(value, dict):
                    return ArtifactRead(
                        spec, "invalid_shape", relative, size_bytes=len(content)
                    )
                if spec.schema_prefix is not None:
                    schema = value.get("schema_version")
                    if not isinstance(schema, str) or not schema.startswith(
                        spec.schema_prefix
                    ):
                        return ArtifactRead(
                            spec, "invalid_schema", relative, size_bytes=len(content)
                        )
                return ArtifactRead(
                    spec,
                    "ok",
                    relative,
                    size_bytes=len(content),
                    content=content,
                    value=value,
                )

            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                return ArtifactRead(
                    spec, "invalid_text", relative, size_bytes=len(content)
                )
            return ArtifactRead(
                spec,
                "ok",
                relative,
                size_bytes=len(content),
                content=content,
            )

        return ArtifactRead(spec, "missing", spec.candidates[0])

    def _schema(self, name: str) -> dict[str, Any] | None:
        if name in self._schema_cache:
            return self._schema_cache[name]
        allowed = {
            SITE_PLAN_SCHEMA,
            SITE_EVALUATION_SCHEMA,
            CAMPAIGN_REPORT_SCHEMA,
            PRODUCT_READINESS_SCHEMA,
            OBJECTIVE_COMPLETION_SCHEMA,
            PRODUCT_FINALIZATION_V2_SCHEMA_FILE,
        }
        if self.schema_root_error or name not in allowed:
            self._schema_cache[name] = None
            return None
        try:
            requested = self.schema_root / name
            if _has_symlink_component(self.schema_root, requested):
                raise ValueError("unsafe schema path")
            candidate = requested.resolve(strict=True)
            candidate.relative_to(self.schema_root)
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("unsafe schema path")
            size = candidate.stat().st_size
            if not 1 <= size <= MAX_SCHEMA_BYTES:
                raise ValueError("invalid schema size")
            content = candidate.read_bytes()
            if len(content) != size:
                raise ValueError("schema changed while reading")
            value = strict_json_loads(content)
            if not isinstance(value, dict):
                raise ValueError("schema root is not an object")
            if (
                name == OBJECTIVE_COMPLETION_SCHEMA
                and value.get("$schema")
                != "https://json-schema.org/draft/2020-12/schema"
            ):
                raise ValueError("objective schema is not Draft 2020-12")
        except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            value = None
        self._schema_cache[name] = value
        return value

    def validates_schema(self, value: Any, name: str) -> bool:
        schema = self._schema(name)
        if schema is None:
            return False
        try:
            _validate_schema_node(value, schema, schema)
        except (TypeError, ValueError, RecursionError):
            return False
        return True

    def resolve_workspace_path(self, value: Any) -> Path | None:
        if self.workspace_root_error or not isinstance(value, str) or not value:
            return None
        requested = Path(value)
        if not requested.is_absolute():
            requested = self.workspace_root / requested
        try:
            if _has_symlink_component(self.workspace_root, requested):
                return None
            resolved = requested.resolve(strict=True)
            resolved.relative_to(self.workspace_root)
            if not resolved.is_file():
                return None
        except (OSError, RuntimeError, ValueError):
            return None
        return resolved

    def verify_workspace_pin(self, pin: Any) -> Path | None:
        if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}:
            return None
        expected_size = pin.get("bytes")
        expected_sha = pin.get("sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not 1 <= expected_size <= MAX_PINNED_FILE_BYTES
            or not isinstance(expected_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        ):
            return None
        resolved = self.resolve_workspace_path(pin.get("path"))
        if resolved is None or any("loaf" in part.casefold() for part in resolved.parts):
            return None
        try:
            if resolved.stat().st_size != expected_size:
                return None
            digest = hashlib.sha256()
            observed_size = 0
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    observed_size += len(chunk)
                    if observed_size > expected_size:
                        return None
                    digest.update(chunk)
            if observed_size != expected_size or digest.hexdigest() != expected_sha:
                return None
        except OSError:
            return None
        return resolved

    def verify_finalization_input_pin(
        self, pin: Any, *, expected_path: str
    ) -> bool:
        """Verify one receipt input against its bounded live regular file."""

        if (
            not isinstance(pin, dict)
            or set(pin) != {"path", "size_bytes", "sha256"}
            or pin.get("path") != expected_path
        ):
            return False
        expected_size = pin.get("size_bytes")
        expected_sha = pin.get("sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not 1 <= expected_size <= HARD_MAX_ARTIFACT_OVERRIDE_BYTES
            or not isinstance(expected_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        ):
            return False

        results_prefix = "validation/results/"
        if expected_path.startswith(results_prefix):
            if self.root_error:
                return False
            root = self.root
            relative = Path(expected_path.removeprefix(results_prefix))
        else:
            if self.workspace_root_error:
                return False
            root = self.workspace_root
            relative = Path(*PurePosixPath(expected_path).parts)
        candidate = root / relative
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        if _has_symlink_component(root, candidate):
            return False

        descriptor: int | None = None
        try:
            before = candidate.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_size
            ):
                return False
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(candidate, flags)
            observed = os.fstat(descriptor)
            before_identity = (
                before.st_mode,
                before.st_dev,
                before.st_ino,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            observed_identity = (
                observed.st_mode,
                observed.st_dev,
                observed.st_ino,
                observed.st_nlink,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed_identity != before_identity
            ):
                return False

            digest = hashlib.sha256()
            observed_size = 0
            while True:
                remaining = expected_size + 1 - observed_size
                if remaining <= 0:
                    return False
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                observed_size += len(chunk)
                digest.update(chunk)
                if observed_size > expected_size:
                    return False
            after_fd = os.fstat(descriptor)
        except OSError:
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)

        try:
            after_path = candidate.lstat()
        except OSError:
            return False
        after_fd_identity = (
            after_fd.st_mode,
            after_fd.st_dev,
            after_fd.st_ino,
            after_fd.st_nlink,
            after_fd.st_size,
            after_fd.st_mtime_ns,
            after_fd.st_ctime_ns,
        )
        after_path_identity = (
            after_path.st_mode,
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_nlink,
            after_path.st_size,
            after_path.st_mtime_ns,
            after_path.st_ctime_ns,
        )
        return bool(
            observed_size == expected_size
            and digest.hexdigest() == expected_sha
            and after_fd_identity == before_identity
            and after_path_identity == before_identity
            and not _has_symlink_component(root, candidate)
        )


def _read_workspace_pin(
    reader: ArtifactReader,
    pin: Any,
    *,
    expected_path: str,
    maximum_bytes: int,
    collect: bool,
) -> WorkspacePinRead:
    """Verify one exact workspace pin using no-follow directory descriptors.

    The opened file descriptor, not a resolved name, is the object that is
    hashed and optionally parsed.  Intermediate symlinks therefore cannot
    redirect a trusted plan pin outside the read-only workspace boundary.
    """

    if reader.workspace_root_error:
        return WorkspacePinRead("invalid_workspace")
    if (
        not isinstance(pin, dict)
        or set(pin) != {"path", "bytes", "sha256"}
        or pin.get("path") != expected_path
    ):
        return WorkspacePinRead("pin_mismatch")
    expected_size = pin.get("bytes")
    expected_sha = pin.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or not 1 <= expected_size <= maximum_bytes
        or not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
    ):
        return WorkspacePinRead("pin_mismatch")
    relative = PurePosixPath(expected_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return WorkspacePinRead("unsafe_path")

    directory_descriptors: list[int] = []
    verification_descriptors: list[int] = []
    descriptor: int | None = None
    verification_descriptor: int | None = None
    flags_common = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        current = os.open(
            reader.workspace_root,
            flags_common
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        directory_descriptors.append(current)
        for part in relative.parts[:-1]:
            current = os.open(
                part,
                flags_common
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            directory_descriptors.append(current)
        descriptor = os.open(
            relative.parts[-1],
            flags_common | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
        info_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info_before.st_mode)
            or info_before.st_nlink != 1
        ):
            return WorkspacePinRead("unsafe_path")
        if info_before.st_size != expected_size:
            return WorkspacePinRead("pin_mismatch")

        digest = hashlib.sha256()
        observed_size = 0
        chunks: list[bytes] = []
        while True:
            remaining = expected_size + 1 - observed_size
            if remaining <= 0:
                return WorkspacePinRead("pin_mismatch")
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            observed_size += len(chunk)
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        info_after = os.fstat(descriptor)
        identity_before = (
            info_before.st_mode,
            info_before.st_dev,
            info_before.st_ino,
            info_before.st_nlink,
            info_before.st_size,
            info_before.st_mtime_ns,
            info_before.st_ctime_ns,
        )
        identity_after = (
            info_after.st_mode,
            info_after.st_dev,
            info_after.st_ino,
            info_after.st_nlink,
            info_after.st_size,
            info_after.st_mtime_ns,
            info_after.st_ctime_ns,
        )
        if (
            observed_size != expected_size
            or identity_after != identity_before
            or digest.hexdigest() != expected_sha
        ):
            return WorkspacePinRead("pin_mismatch")

        # Re-walk the live name from the workspace root after hashing.  The
        # first descriptor chain protects what was read; this fresh chain
        # additionally proves that the public projection still names that
        # exact inode after a concurrent rename or ancestor swap.
        try:
            verification_current = os.open(
                reader.workspace_root,
                flags_common
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            verification_descriptors.append(verification_current)
            for part in relative.parts[:-1]:
                verification_current = os.open(
                    part,
                    flags_common
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=verification_current,
                )
                verification_descriptors.append(verification_current)
            verification_descriptor = os.open(
                relative.parts[-1],
                flags_common | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=verification_current,
            )
            live_identity = os.fstat(verification_descriptor)
        except OSError:
            return WorkspacePinRead("changed")
        identity_live = (
            live_identity.st_mode,
            live_identity.st_dev,
            live_identity.st_ino,
            live_identity.st_nlink,
            live_identity.st_size,
            live_identity.st_mtime_ns,
            live_identity.st_ctime_ns,
        )
        if identity_live != identity_before:
            return WorkspacePinRead("changed")
        return WorkspacePinRead("ok", b"".join(chunks) if collect else None)
    except FileNotFoundError:
        return WorkspacePinRead("missing")
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            return WorkspacePinRead("unsafe_path")
        return WorkspacePinRead("io_error")
    finally:
        if verification_descriptor is not None:
            os.close(verification_descriptor)
        for verification_directory in reversed(verification_descriptors):
            os.close(verification_directory)
        if descriptor is not None:
            os.close(descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)


def _workspace_pin_json(
    reader: ArtifactReader,
    pin: Any,
    *,
    expected_path: str,
    maximum_bytes: int = PERSON_UPGRADE_MAX_JSON_BYTES,
) -> tuple[WorkspacePinRead, dict[str, Any] | None]:
    result = _read_workspace_pin(
        reader,
        pin,
        expected_path=expected_path,
        maximum_bytes=maximum_bytes,
        collect=True,
    )
    if not result.available or result.content is None:
        return result, None
    try:
        value = strict_json_loads(result.content)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return WorkspacePinRead("invalid_json"), None
    if not isinstance(value, dict):
        return WorkspacePinRead("invalid_shape"), None
    return result, value


def _text(value: Any, *, limit: int = 120) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:limit] if value else None


def _integer(value: Any, *, maximum: int = 100_000_000) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > maximum:
        return None
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return round(result, 6) if math.isfinite(result) else None


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _sizes(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return sorted(
        {
            item
            for item in value[:16]
            if isinstance(item, int) and not isinstance(item, bool) and 1 <= item <= 4096
        }
    )


def _enum(value: Any, allowed: set[str]) -> str | None:
    return value if isinstance(value, str) and value in allowed else None


def _identifiers(value: Any, *, maximum: int = 32) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    for item in value[:maximum]:
        if not isinstance(item, str) or not 1 <= len(item) <= 80:
            continue
        if set(item) <= allowed and item not in result:
            result.append(item)
    return result


def _identifier(value: Any, *, maximum: int = 80) -> str | None:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    return value if set(value) <= allowed else None


def _sha256(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    return value if set(value) <= set("0123456789abcdef") else None


def _identifier_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    for raw_key, raw_count in list(value.items())[:32]:
        if not isinstance(raw_key, str) or not 1 <= len(raw_key) <= 40:
            continue
        count = _integer(raw_count)
        if set(raw_key) <= allowed and count is not None:
            result[raw_key] = count
    return dict(sorted(result.items()))


def _timestamp(value: Any) -> str | None:
    rendered = _text(value)
    if rendered is None:
        return None
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError:
        return None
    return rendered if parsed.tzinfo is not None else None


def _counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for raw_key, raw_count in list(value.items())[:32]:
        key = _text(raw_key, limit=40)
        count = _integer(raw_count)
        if key is not None and count is not None:
            result[key] = count
    return dict(sorted(result.items()))


def _job_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: dict[str, int] = {}
    for item in value[:10_000]:
        if not isinstance(item, dict):
            continue
        state = _text(item.get("status") or item.get("state"), limit=40)
        if state is not None:
            counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def _progress(completed: int, total: int) -> dict[str, int | float | None]:
    completed = max(0, min(completed, total)) if total else 0
    return {
        "completed": completed,
        "total": total,
        "remaining": max(0, total - completed),
        "fraction": round(completed / total, 6) if total else None,
    }


def _campaign_state(
    *, available: bool, artifact_state: str, completed: int, total: int, counts: dict[str, int]
) -> str:
    if not available:
        return "not_started" if artifact_state == "missing" else "artifact_error"
    if total > 0 and completed >= total:
        return "complete"
    if any("fail" in key or "invalid" in key for key in counts):
        return "attention"
    if completed > 0 or counts.get("running", 0) > 0:
        return "in_progress"
    return "planned"


def _evidence(reader: ArtifactReader, *keys: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in keys:
        artifact = reader.read(key)
        item: dict[str, Any] = {
            "id": key,
            "label": artifact.spec.label,
            "path": artifact.relative_path,
            "available": artifact.available,
            "artifact_state": artifact.state,
            "raw_download_allowed": artifact.spec.raw_download_allowed,
        }
        if artifact.size_bytes is not None:
            item["size_bytes"] = artifact.size_bytes
        if artifact.available and artifact.spec.raw_download_allowed:
            item["href"] = f"/api/validation?artifact={quote(key, safe='')}"
        items.append(item)
    return items


def _scene_benchmark(reader: ArtifactReader) -> dict[str, Any]:
    artifact = reader.read("scene_benchmark_summary")
    value = artifact.value or {}
    raw_counts = _counts(value.get("status_counts"))
    total = _integer(value.get("expected_runs")) or 0
    report_artifact = reader.read("campaign_report_json")
    report_campaigns = (
        report_artifact.value.get("campaigns")
        if report_artifact.available
        and isinstance(report_artifact.value, dict)
        and isinstance(report_artifact.value.get("campaigns"), dict)
        else {}
    )
    report_scene = (
        report_campaigns.get("scene_benchmark")
        if isinstance(report_campaigns.get("scene_benchmark"), dict)
        else {}
    )
    acceptance_safe_completed = _integer(
        report_scene.get("acceptance_safe_complete_runs")
    )
    acceptance_projection_available = acceptance_safe_completed is not None
    completed = (
        acceptance_safe_completed
        if acceptance_projection_available
        else raw_counts.get("complete", 0)
    )
    counts = (
        {
            "acceptance_safe_complete": completed,
            "not_acceptance_safe": max(0, total - completed),
        }
        if acceptance_projection_available
        else raw_counts
    )
    return {
        "label": "12-kamera kişi benchmark matrisi",
        "available": artifact.available,
        "state": _campaign_state(
            available=artifact.available,
            artifact_state=artifact.state,
            completed=completed,
            total=total,
            counts=counts,
        ),
        "updated_at_utc": _text(value.get("generated_at_utc")),
        "progress": _progress(completed, total),
        "status_counts": counts,
        "historical_source_status_counts": raw_counts,
        "progress_basis": (
            "fail_closed_campaign_report_acceptance_safe_runs"
            if acceptance_projection_available
            else "raw_matrix_summary_status_counts"
        ),
        "scope": {
            "scene_types": _integer(value.get("selected_scenes")),
            "model_input_sizes": _sizes(value.get("selected_sizes")),
            "simulated_streams": _integer(value.get("streams")),
            "duration_seconds_per_run": _integer(
                value.get("duration_seconds_per_run")
            ),
            "warmup_seconds_per_run": _integer(
                value.get("warmup_seconds_per_run")
            ),
        },
        "evidence": _evidence(
            reader,
            "scene_benchmark_summary",
            "scene_benchmark_preflight",
            "campaign_report_json",
        ),
    }


def _campaign_report(reader: ArtifactReader) -> dict[str, Any]:
    """Project only the signed-off decision fields; never pass raw evidence through."""

    artifact = reader.read("campaign_report_json")
    value = artifact.value or {}
    claims_endurance_lineage = _campaign_report_claims_endurance_lineage(value)
    stale_lineage = bool(
        artifact.available
        and claims_endurance_lineage
        and not _campaign_report_lineage_matches(reader, value)
    )
    decision = value.get("decision") if isinstance(value.get("decision"), dict) else {}
    requirements = value.get("requirements") if isinstance(value.get("requirements"), list) else []
    valid_states = {"accepted", "preliminary", "blocked_by_hardware"}
    state = _text(decision.get("status"), limit=40)
    unsafe_unbound_acceptance = bool(
        artifact.available
        and not claims_endurance_lineage
        and (
            state == "accepted"
            or decision.get("accepted") is True
            or decision.get("final_claim_allowed") is True
        )
    )
    if not artifact.available:
        state = "not_started" if artifact.state == "missing" else "artifact_error"
    elif stale_lineage:
        state = "stale_lineage"
    elif unsafe_unbound_acceptance:
        state = "artifact_error"
    elif state not in valid_states:
        state = "artifact_error"
    total = len([item for item in requirements[:1_000] if isinstance(item, dict)])
    passed = len(
        [
            item
            for item in requirements[:1_000]
            if isinstance(item, dict) and item.get("state") == "pass"
        ]
    )
    counts = _counts(
        value.get("requirement_summary", {}).get("state_counts")
        if isinstance(value.get("requirement_summary"), dict)
        else {}
    )
    projected_evidence = _evidence(
        reader, "campaign_report_json", "campaign_report_markdown"
    )
    if stale_lineage:
        projected_evidence = _stale_lineage_evidence(projected_evidence)
    result = {
        "label": "Kampanya kabul kapısı",
        "available": bool(artifact.available and not stale_lineage),
        "artifact_state": "stale_lineage" if stale_lineage else artifact.state,
        "state": state or "artifact_error",
        "reason": (
            "stale_lineage"
            if stale_lineage
            else "unbound_acceptance_claim"
            if unsafe_unbound_acceptance
            else None
        ),
        "updated_at_utc": (
            None if stale_lineage else _text(value.get("generated_at_utc"))
        ),
        "accepted": (
            False
            if stale_lineage or unsafe_unbound_acceptance
            else _boolean(decision.get("accepted"))
        ),
        "final_claim_allowed": (
            False
            if stale_lineage or unsafe_unbound_acceptance
            else _boolean(decision.get("final_claim_allowed"))
        ),
        "progress": _progress(0, total) if stale_lineage else _progress(passed, total),
        "status_counts": {} if stale_lineage else counts,
        "evidence": projected_evidence,
    }
    campaigns = value.get("campaigns") if isinstance(value.get("campaigns"), dict) else {}
    loaf = campaigns.get("loaf_preparation") if isinstance(campaigns.get("loaf_preparation"), dict) else None
    if loaf is not None and not stale_lineage:
        splits = loaf.get("splits") if isinstance(loaf.get("splits"), dict) else {}
        preparation: dict[str, Any] = {
            "state": _text(loaf.get("state"), limit=40),
            "acceptance_effect": _text(loaf.get("acceptance_effect"), limit=40),
            "can_satisfy_calibrated_25m_detection": _boolean(
                loaf.get("can_satisfy_calibrated_25m_detection")
            ),
            "splits": {},
        }
        for split in ("val", "test_unseen"):
            raw = splits.get(split) if isinstance(splits.get(split), dict) else {}
            preparation["splits"][split] = {
                "state": _text(raw.get("state"), limit=40),
                "target_people": _integer(raw.get("target_people")),
                "frames": _integer(raw.get("frames")),
                "sequences": _integer(raw.get("sequences")),
                "media_status": _text(raw.get("media_status"), limit=40),
            }
        rights = (
            loaf.get("dataset_rights")
            if isinstance(loaf.get("dataset_rights"), dict)
            else None
        )
        if rights is not None:
            license_status = _text(rights.get("license_status"), limit=32)
            if license_status not in {"unverified", "verified", "restricted", "unknown"}:
                license_status = None
            preparation["dataset_rights"] = {
                "license_status": license_status,
                "internal_research_validation_only": _boolean(
                    rights.get("internal_research_validation_only")
                ),
                "model_training_allowed": _boolean(
                    rights.get("model_training_allowed")
                ),
                "redistribution_allowed": _boolean(
                    rights.get("redistribution_allowed")
                ),
                "written_rights_clearance_required": _boolean(
                    rights.get("written_rights_clearance_required")
                ),
                "guardrail_consistent": _boolean(
                    rights.get("guardrail_consistent")
                ),
            }
        result["loaf_preparation"] = preparation
    return result


def _person_detection_quality(reader: ArtifactReader) -> dict[str, Any]:
    """Project only bounded policy thresholds and live-evaluation outcomes."""

    artifact = reader.read("campaign_report_json")
    report = artifact.value or {}
    campaigns = report.get("campaigns") if isinstance(report.get("campaigns"), dict) else {}
    quality = (
        campaigns.get("person_detection_quality")
        if isinstance(campaigns.get("person_detection_quality"), dict)
        else None
    )
    report_schema_valid = bool(
        artifact.available
        and reader.validates_schema(report, CAMPAIGN_REPORT_SCHEMA)
    )
    requirements = report.get("requirements") if isinstance(report.get("requirements"), list) else []
    quality_requirements = [
        item
        for item in requirements[:1_000]
        if isinstance(item, dict) and item.get("id") == "person_detection_quality"
    ]
    caviar = campaigns.get("caviar_ground_truth") if isinstance(campaigns.get("caviar_ground_truth"), dict) else {}
    quality_outcome = caviar.get("quality_outcome") if isinstance(caviar.get("quality_outcome"), dict) else {}
    valid_states = {"passed", "failed", "draft_unapproved", "missing", "invalid"}
    raw_state = _text(quality.get("state"), limit=32) if quality is not None else None
    expected_requirement_state = (
        "pass"
        if raw_state == "passed"
        else ("invalid" if raw_state == "invalid" else "unproven")
    )
    evaluated_state = raw_state in {"passed", "failed"}
    decision = report.get("decision") if isinstance(report.get("decision"), dict) else {}
    failed_required = decision.get("failed_required_gates")
    quality_listed_as_failed = bool(
        isinstance(failed_required, list)
        and "person_detection_quality" in failed_required
    )
    structurally_valid = bool(
        report_schema_valid
        and quality is not None
        and raw_state in valid_states
        and _boolean(quality.get("accepted")) == (raw_state == "passed")
        and _boolean(quality.get("live_cpu_recomputed")) is evaluated_state
        and _boolean(quality.get("caviar_evidence_complete"))
        is _boolean(caviar.get("evidence_complete"))
        and isinstance(quality.get("policy"), dict)
        and isinstance(quality.get("evaluator"), dict)
        and _boolean(quality["evaluator"].get("invoked")) is True
        and _boolean(quality["evaluator"].get("prewritten_decision_used")) is False
        and len(quality_requirements) == 1
        and quality_requirements[0].get("required_for_acceptance") is True
        and quality_requirements[0].get("state") == expected_requirement_state
        and quality_listed_as_failed is (raw_state != "passed")
        and quality_outcome.get("state") == raw_state
        and _boolean(quality_outcome.get("accepted")) == (raw_state == "passed")
        and quality_outcome.get("separate_required_gate_id") == "person_detection_quality"
        and (
            not evaluated_state
            or (
                _boolean(quality.get("evidence_complete")) is True
                and quality["policy"].get("status") == "approved"
                and _boolean(
                    quality["policy"].get("approval_strictly_before_campaign")
                )
                is True
                and _boolean(
                    quality["evaluator"].get("gpu_or_docker_executed")
                )
                is False
            )
        )
    )
    if not artifact.available:
        state = "not_started" if artifact.state == "missing" else "artifact_error"
    elif not structurally_valid:
        state = "artifact_error"
    else:
        state = raw_state or "artifact_error"

    policy_raw = quality.get("policy") if structurally_valid else {}
    policy_status = _text(policy_raw.get("status"), limit=16)
    if policy_status not in {"draft", "approved", "missing", "invalid"}:
        policy_status = None
    contract_sha256 = _text(policy_raw.get("contract_sha256"), limit=64)
    if contract_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", contract_sha256) is None:
        contract_sha256 = None
    policy = {
        "policy_id": _text(policy_raw.get("policy_id"), limit=128),
        "status": policy_status,
        "contract_sha256": contract_sha256,
        "task": (
            "person_detection"
            if policy_raw.get("task") == "person_detection"
            else None
        ),
        "dataset": "CAVIAR" if policy_raw.get("dataset") == "CAVIAR" else None,
        "approved_at_utc": _text(policy_raw.get("approved_at_utc"), limit=64),
        "approval_strictly_before_campaign": _boolean(
            policy_raw.get("approval_strictly_before_campaign")
        ),
    }

    metrics: dict[str, dict[str, Any]] = {}
    raw_metrics = quality.get("metrics_by_profile") if structurally_valid else {}
    if isinstance(raw_metrics, dict):
        for profile in ("640", "960"):
            raw = raw_metrics.get(profile)
            if not isinstance(raw, dict):
                continue
            projected = {
                "sequences": _integer(raw.get("sequences")),
                "ground_truth": _integer(raw.get("ground_truth")),
                "tp": _integer(raw.get("tp")),
                "fp": _integer(raw.get("fp")),
                "fn": _integer(raw.get("fn")),
                "micro_precision": _number(raw.get("micro_precision")),
                "micro_recall": _number(raw.get("micro_recall")),
                "micro_f1": _number(raw.get("micro_f1")),
                "macro_ap50": _number(raw.get("macro_ap50")),
            }
            if all(value is not None for value in projected.values()):
                metrics[profile] = projected

    rules: list[dict[str, Any]] = []
    raw_rules = quality.get("rules") if structurally_valid else []
    if isinstance(raw_rules, list):
        for raw in raw_rules[:32]:
            if not isinstance(raw, dict):
                continue
            metric = _text(raw.get("metric"), limit=32)
            operator = _text(raw.get("operator"), limit=8)
            threshold = _number(raw.get("threshold"))
            status = _text(raw.get("status"), limit=20)
            scope = raw.get("scope") if isinstance(raw.get("scope"), dict) else {}
            if (
                metric not in {"micro_precision", "micro_recall", "micro_f1", "macro_ap50"}
                or operator not in {"gte", "gt", "lte", "lt", "eq"}
                or threshold is None
                or not 0 <= threshold <= 1
                or status not in {"pass", "fail", "not_evaluated"}
            ):
                continue
            if scope.get("kind") == "each_profile" and scope.get("profiles") == [640, 960]:
                compact_scope: dict[str, Any] = {"kind": "each_profile", "profiles": [640, 960]}
                selected = ("640", "960")
            elif scope.get("kind") == "profile" and scope.get("profile") in {640, 960}:
                compact_scope = {"kind": "profile", "profile": int(scope["profile"])}
                selected = (str(scope["profile"]),)
            else:
                continue
            values: dict[str, float] = {}
            raw_values = raw.get("profile_values")
            if isinstance(raw_values, dict):
                for profile in selected:
                    number = _number(raw_values.get(profile))
                    if number is not None and 0 <= number <= 1:
                        values[profile] = number
            rules.append(
                {
                    "rule_id": _text(raw.get("rule_id"), limit=128),
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "scope": compact_scope,
                    "profile_values": values,
                    "status": status,
                }
            )
    evaluated_rules = [rule for rule in rules if rule["status"] in {"pass", "fail"}]
    passed_rules = sum(rule["status"] == "pass" for rule in evaluated_rules)
    reasons = quality.get("reasons") if structurally_valid else []
    return {
        "label": "İnsan algılama doğruluk kapısı",
        "available": structurally_valid,
        "state": state,
        "updated_at_utc": _text(report.get("generated_at_utc")),
        "accepted": _boolean(quality.get("accepted")) if structurally_valid else False,
        "evidence_complete": _boolean(quality.get("evidence_complete")) if structurally_valid else False,
        "caviar_evidence_complete": _boolean(quality.get("caviar_evidence_complete")) if structurally_valid else False,
        "live_cpu_recomputed": _boolean(quality.get("live_cpu_recomputed")) if structurally_valid else False,
        "prewritten_decision_used": False,
        "policy": policy,
        "metrics_by_profile": metrics,
        "rules": rules,
        "progress": _progress(passed_rules, len(evaluated_rules)),
        "ap50_definition": (
            _text(quality.get("ap50_definition"), limit=128)
            if structurally_valid
            else None
        ),
        "reason_count": len(reasons[:64]) if isinstance(reasons, list) else 0,
        "evidence": _evidence(reader, "campaign_report_json"),
    }


def _ppe_video_source_registry(reader: ArtifactReader) -> dict[str, Any]:
    """Project the reporter's pathless PPE source plan, never raw registry data."""

    artifact = reader.read("campaign_report_json")
    report = artifact.value or {}
    campaigns = (
        report.get("campaigns")
        if isinstance(report.get("campaigns"), dict)
        else {}
    )
    section = (
        campaigns.get("ppe_video_source_registry")
        if isinstance(campaigns.get("ppe_video_source_registry"), dict)
        else None
    )
    report_schema_valid = bool(
        artifact.available
        and reader.validates_schema(report, CAMPAIGN_REPORT_SCHEMA)
    )
    raw_state = (
        _enum(
            section.get("state"),
            {"valid_metadata_only", "missing", "invalid"},
        )
        if section is not None
        else None
    )
    requirements = (
        report.get("requirements")
        if isinstance(report.get("requirements"), list)
        else []
    )
    matching_requirements = [
        item
        for item in requirements[:1_000]
        if isinstance(item, dict)
        and item.get("id") == "ppe_video_source_registry_integrity"
    ]
    expected_requirement_state = {
        "valid_metadata_only": "pass",
        "missing": "incomplete",
        "invalid": "invalid",
    }.get(raw_state)
    requirement_consistent = bool(
        len(matching_requirements) == 1
        and matching_requirements[0].get("required_for_acceptance") is False
        and matching_requirements[0].get("state")
        == expected_requirement_state
    )
    common_contract = bool(
        report_schema_valid
        and section is not None
        and raw_state is not None
        and requirement_consistent
        and section.get("evidence_kind")
        == "ppe_video_source_registry_metadata_only"
        and section.get("expected_candidate_count")
        == PPE_VIDEO_SOURCE_CANDIDATE_COUNT
        and section.get("quantitative_benchmark_ready") is False
        and section.get("ppe_model_ready") is False
        and section.get("acceptance_effect")
        == "planning_only_no_model_readiness"
        and section.get("evidence_ids") == ["ppe_video_source_registry"]
        and isinstance(section.get("acquisition"), dict)
        and section["acquisition"].get(
            "reporter_downloaded_or_decoded_media"
        )
        is False
        and section["acquisition"].get(
            "reporter_network_gpu_docker_or_inference_used"
        )
        is False
    )
    valid_registry_contract = bool(
        common_contract
        and raw_state == "valid_metadata_only"
        and section.get("registry_valid") is True
        and section.get("registry_sha256")
        == PPE_VIDEO_SOURCE_REGISTRY_SHA256
        and section.get("candidate_count")
        == PPE_VIDEO_SOURCE_CANDIDATE_COUNT
        and section.get("eligibility_counts")
        == PPE_VIDEO_SOURCE_ELIGIBILITY_COUNTS
        and section.get("primary_plan")
        == "user_owned_authorized_site_footage"
        and section.get("candidate_groups")
        == PPE_VIDEO_SOURCE_CANDIDATE_GROUPS
        and section["acquisition"].get("registry_metadata_only") is True
        and section["acquisition"].get(
            "media_or_annotations_downloaded"
        )
        is False
        and section.get("reasons") == []
    )
    withheld_registry_contract = bool(
        common_contract
        and raw_state in {"missing", "invalid"}
        and section.get("registry_valid") is False
        and section.get("candidate_count") is None
        and isinstance(section.get("eligibility_counts"), dict)
        and set(section["eligibility_counts"])
        == set(PPE_VIDEO_SOURCE_ELIGIBILITY_COUNTS)
        and all(
            value is None for value in section["eligibility_counts"].values()
        )
        and section.get("primary_plan") is None
        and isinstance(section.get("candidate_groups"), dict)
        and set(section["candidate_groups"])
        == set(PPE_VIDEO_SOURCE_CANDIDATE_GROUPS)
        and all(value == [] for value in section["candidate_groups"].values())
        and section["acquisition"].get("registry_metadata_only") is None
        and section["acquisition"].get(
            "media_or_annotations_downloaded"
        )
        is None
        and isinstance(section.get("reasons"), list)
        and bool(section["reasons"])
    )
    structurally_valid = valid_registry_contract or withheld_registry_contract

    if not artifact.available:
        state = (
            "not_started" if artifact.state == "missing" else "artifact_error"
        )
    elif not structurally_valid:
        state = "artifact_error"
    elif valid_registry_contract:
        state = "valid_metadata_only"
    elif raw_state == "missing":
        state = "not_started"
    else:
        state = "artifact_error"

    trusted = state == "valid_metadata_only"
    pathless_evidence = [
        {
            key: value
            for key, value in item.items()
            if key not in {"path", "href"}
        }
        for item in _evidence(reader, "campaign_report_json")
    ]
    return {
        "label": "PPE video kaynak kaydı (yalnız metadata)",
        "available": trusted,
        "state": state,
        "updated_at_utc": _text(report.get("generated_at_utc")),
        "registry_valid": trusted,
        "registry_sha256": (
            PPE_VIDEO_SOURCE_REGISTRY_SHA256 if trusted else None
        ),
        "progress": _progress(
            PPE_VIDEO_SOURCE_CANDIDATE_COUNT if trusted else 0,
            PPE_VIDEO_SOURCE_CANDIDATE_COUNT,
        ),
        "candidate_count": (
            PPE_VIDEO_SOURCE_CANDIDATE_COUNT if trusted else None
        ),
        "eligibility_counts": (
            dict(PPE_VIDEO_SOURCE_ELIGIBILITY_COUNTS) if trusted else {}
        ),
        "primary_plan": (
            "user_owned_authorized_site_footage" if trusted else None
        ),
        "candidate_groups": (
            {
                key: list(value)
                for key, value in PPE_VIDEO_SOURCE_CANDIDATE_GROUPS.items()
            }
            if trusted
            else {key: [] for key in PPE_VIDEO_SOURCE_CANDIDATE_GROUPS}
        ),
        "registry_metadata_only": True if trusted else None,
        "media_or_annotations_downloaded": False if trusted else None,
        "reporter_downloaded_or_decoded_media": False,
        "reporter_external_execution_used": False,
        "quantitative_benchmark_ready": False,
        "ppe_model_ready": False,
        "acceptance_effect": "planning_only_no_model_readiness",
        "raw_registry_download_available": False,
        "reason_count": (
            len(section["reasons"][:16])
            if structurally_valid and isinstance(section.get("reasons"), list)
            else 0
        ),
        "evidence": pathless_evidence,
    }


def _profile_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for profile in ("640", "960"):
        raw = value.get(profile)
        if not isinstance(raw, dict):
            continue
        micro = raw.get("micro") if isinstance(raw.get("micro"), dict) else {}
        macro = raw.get("macro") if isinstance(raw.get("macro"), dict) else {}
        result[profile] = {
            "complete_sequences": _integer(raw.get("complete_sequences")),
            "expected_sequences": _integer(raw.get("expected_sequences")),
            "precision": _number(micro.get("precision")),
            "recall": _number(micro.get("recall")),
            "f1": _number(micro.get("f1")),
            "ap_101_point_macro": _number(macro.get("ap_101_point")),
            "single_stream_offline_fps_macro": _number(
                macro.get("last_reported_average_fps")
            ),
        }
    return result


def _caviar(reader: ArtifactReader) -> dict[str, Any]:
    """Project only the schema-validated report's bounded nonce-session summary."""

    artifact = reader.read("campaign_report_json")
    report = artifact.value or {}
    report_schema_valid = bool(
        artifact.available
        and reader.validates_schema(report, CAMPAIGN_REPORT_SCHEMA)
    )
    campaigns = report.get("campaigns") if isinstance(report.get("campaigns"), dict) else {}
    value = (
        campaigns.get("caviar_ground_truth")
        if isinstance(campaigns.get("caviar_ground_truth"), dict)
        else {}
    )
    official = value.get("official_session") if isinstance(value.get("official_session"), dict) else {}
    session_state = _text(official.get("state"), limit=24)
    expected = _integer(value.get("expected_jobs"))
    validated = _integer(value.get("validated_jobs"))
    aggregate_jobs = _integer(value.get("aggregate_result_jobs"))
    candidate_count = _integer(official.get("candidate_count"))
    valid_count = _integer(official.get("valid_candidate_count"))
    conflict_count = _integer(official.get("conflicting_candidate_count"))
    metrics_withheld = _boolean(value.get("metrics_withheld"))
    structurally_valid = bool(
        report_schema_valid
        and expected == 16
        and validated is not None
        and 0 <= validated <= expected
        and aggregate_jobs is not None
        and 0 <= aggregate_jobs <= expected
        and session_state in {"not_authorized", "missing", "invalid", "conflict", "resolved"}
        and _boolean(official.get("authorized_policy")) is not None
        and candidate_count is not None
        and 0 <= candidate_count <= 256
        and valid_count in {0, 1}
        and conflict_count is not None
        and 0 <= conflict_count <= 256
        and _boolean(official.get("legacy_public_artifacts_ignored")) is not None
        and _boolean(official.get("private_identity_projected")) is False
        and metrics_withheld is (session_state != "resolved")
        and (
            metrics_withheld is False
            or (
                validated == 0
                and aggregate_jobs == 0
                and value.get("profiles") == {}
            )
        )
        and _boolean(value.get("accepted")) == (session_state == "resolved" and validated == expected)
        and _boolean(value.get("evidence_complete")) == _boolean(value.get("accepted"))
    )
    total = expected if structurally_valid and expected is not None else 0
    completed = validated if structurally_valid and validated is not None else 0
    pending = max(0, total - completed)
    counts = {"complete": completed, "pending": pending}
    if not artifact.available:
        state = "not_started" if artifact.state == "missing" else "artifact_error"
    elif not structurally_valid:
        state = "artifact_error"
    else:
        state = _campaign_state(
            available=True,
            artifact_state=artifact.state,
            completed=completed,
            total=total,
            counts=counts,
        )
    profiles: dict[str, dict[str, Any]] = {}
    raw_profiles = value.get("profiles") if structurally_valid else {}
    if isinstance(raw_profiles, dict):
        for profile in ("640", "960"):
            raw = raw_profiles.get(profile)
            if not isinstance(raw, dict):
                continue
            row = {
                "complete_sequences": _integer(raw.get("complete_sequences")),
                "expected_sequences": _integer(raw.get("expected_sequences")),
                "precision": _number(raw.get("precision")),
                "recall": _number(raw.get("recall")),
                "f1": _number(raw.get("f1")),
                "ap_101_point_macro": _number(raw.get("ap_101_point_macro")),
                "single_stream_offline_fps_macro": _number(
                    raw.get("single_stream_offline_fps_macro")
                ),
            }
            if (
                row["complete_sequences"] is not None
                and row["expected_sequences"] == 8
                and 0 <= row["complete_sequences"] <= 8
            ):
                profiles[profile] = row
    return {
        "label": "CAVIAR ground-truth doğruluk partisi",
        "available": bool(artifact.available and structurally_valid),
        "state": state,
        "updated_at_utc": _text(report.get("generated_at_utc"), limit=64),
        "progress": _progress(completed, total),
        "status_counts": counts if structurally_valid else {},
        "profiles": profiles,
        "official_session": {
            "state": session_state if structurally_valid else None,
            "candidate_count": candidate_count if structurally_valid else None,
            "valid_candidate_count": valid_count if structurally_valid else None,
            "conflicting_candidate_count": conflict_count if structurally_valid else None,
            "legacy_public_artifacts_ignored": (
                _boolean(official.get("legacy_public_artifacts_ignored"))
                if structurally_valid
                else None
            ),
        },
        "metric_context": {
            "ground_truth": True,
            "fps_is_single_stream_offline": True,
            "withheld_until_official_session": metrics_withheld,
        },
        "evidence": _evidence(reader, "campaign_report_json"),
    }


def _canonical_fingerprint_matches(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    fingerprint = _sha256(value.get("fingerprint_sha256"))
    if fingerprint is None:
        return False
    payload = {
        key: item for key, item in value.items() if key != "fingerprint_sha256"
    }
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, OverflowError):
        return False
    return hashlib.sha256(encoded).hexdigest() == fingerprint


def _rlivit_pathless_pin(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"size_bytes", "sha256"}:
        raise ValueError("invalid pathless R-LiViT pin")
    size = _integer(value.get("size_bytes"), maximum=MAX_PINNED_FILE_BYTES)
    sha256 = _sha256(value.get("sha256"))
    if size is None or size <= 0 or sha256 is None:
        raise ValueError("invalid pathless R-LiViT pin value")
    return {"size_bytes": size, "sha256": sha256}


def _rlivit_metric_row(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    base_fields = {
        "ground_truth",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "ap_101_point",
    }
    diagnostic_fields = {
        "evaluated_predictions",
        "ignored_predictions",
        "serialized_predictions_at_or_above_confidence",
        "ap_serialized_predictions",
        "ap_ignored_predictions",
    }
    if set(value) not in {
        frozenset(base_fields),
        frozenset(base_fields | diagnostic_fields),
    }:
        return None
    ground_truth = _integer(value.get("ground_truth"))
    tp = _integer(value.get("tp"))
    fp = _integer(value.get("fp"))
    fn = _integer(value.get("fn"))
    if (
        ground_truth is None
        or tp is None
        or fp is None
        or fn is None
        or tp + fn != ground_truth
    ):
        return None
    projected: dict[str, Any] = {
        "ground_truth": ground_truth,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    for key in ("precision", "recall", "f1", "ap_101_point"):
        raw = value.get(key)
        number = _number(raw)
        # Every published R-LiViT stratum has positive GT.  The evaluator
        # deliberately emits numeric zero (rather than null) when there are no
        # detections, so a completed result may never hide a metric as null.
        if number is None or not 0.0 <= number <= 1.0:
            return None
        projected[key] = number

    expected_precision = tp / (tp + fp) if tp + fp else (0.0 if ground_truth else None)
    expected_recall = tp / ground_truth if ground_truth else None
    if (
        projected["precision"] is not None
        and expected_precision is not None
        and abs(projected["precision"] - expected_precision) > 1e-5
    ):
        return None
    if (
        projected["recall"] is not None
        and expected_recall is not None
        and abs(projected["recall"] - expected_recall) > 1e-5
    ):
        return None
    if projected["f1"] is not None:
        precision = projected["precision"]
        recall = projected["recall"]
        expected_f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else (
                0.0
                if precision is not None and recall is not None
                else None
            )
        )
        if expected_f1 is None or abs(projected["f1"] - expected_f1) > 1e-5:
            return None
    if diagnostic_fields <= set(value):
        diagnostics = {
            key: _integer(value.get(key)) for key in diagnostic_fields
        }
        if (
            any(item is None or item < 0 for item in diagnostics.values())
            or diagnostics["evaluated_predictions"] != tp + fp
            or diagnostics["evaluated_predictions"]
            + diagnostics["ignored_predictions"]
            != diagnostics["serialized_predictions_at_or_above_confidence"]
            or diagnostics["serialized_predictions_at_or_above_confidence"]
            > diagnostics["ap_serialized_predictions"]
            or diagnostics["ap_ignored_predictions"]
            > diagnostics["ap_serialized_predictions"]
            or diagnostics["ap_ignored_predictions"]
            < diagnostics["ignored_predictions"]
        ):
            return None
        projected.update(diagnostics)
    return projected


def _rlivit_metric_partition(
    value: Any,
    expected_keys: set[str],
) -> dict[str, dict[str, Any]] | None:
    if not isinstance(value, dict) or set(value) != expected_keys:
        return None
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(expected_keys):
        metric = _rlivit_metric_row(value.get(key))
        if metric is None:
            return None
        result[key] = metric
    return result


def _rlivit_profile_metrics(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "overall",
        "daytime",
        "locations",
        "coco_area",
        "height_bands",
    }:
        return None
    overall = _rlivit_metric_row(value.get("overall"))
    daytime = _rlivit_metric_partition(value.get("daytime"), {"day", "night"})
    locations = _rlivit_metric_partition(
        value.get("locations"), {str(index) for index in range(8)}
    )
    coco_area = _rlivit_metric_partition(
        value.get("coco_area"), {"small", "medium", "large"}
    )
    height_bands = _rlivit_metric_partition(
        value.get("height_bands"), {"lt32", "32to95", "gte96"}
    )
    if any(
        item is None
        for item in (overall, daytime, locations, coco_area, height_bands)
    ):
        return None
    assert overall is not None
    assert daytime is not None
    assert locations is not None
    assert coco_area is not None
    assert height_bands is not None
    if overall["ground_truth"] != 4318:
        return None
    for partition in (daytime, locations, coco_area, height_bands):
        if sum(row["ground_truth"] for row in partition.values()) != 4318:
            return None
    # Day/night and location are disjoint frame partitions.  Unlike the
    # diagnostic area/height strata, their confusion counts must add back to
    # the exact overall replay.
    for partition in (daytime, locations):
        for field in ("tp", "fp", "fn"):
            if sum(row[field] for row in partition.values()) != overall[field]:
                return None
    return {
        "overall": overall,
        "daytime": daytime,
        "locations": locations,
        "coco_area": coco_area,
        "height_bands": height_bands,
    }


def _rlivit_full_pin(value: Any) -> dict[str, Any] | None:
    """Validate a private full pin and return only its non-path fields."""

    if not isinstance(value, dict):
        return None
    if set(value) == {"path", "size_bytes", "sha256"}:
        size = value.get("size_bytes")
    elif set(value) == {"path", "bytes", "sha256"}:
        size = value.get("bytes")
    else:
        return None
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not 1 <= len(raw_path) <= 512:
        return None
    pure = PurePosixPath(raw_path)
    if (
        pure.is_absolute()
        or pure.as_posix() != raw_path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        return None
    normalized = _rlivit_pathless_pin(
        {"size_bytes": size, "sha256": value.get("sha256")}
    )
    return normalized


def _rlivit_read_pinned_json(
    path: Path,
    *,
    containment_root: Path,
    pin: dict[str, Any],
    maximum_bytes: int,
) -> dict[str, Any] | None:
    """Read one exact, non-symlink JSON artifact through a pathless pin."""

    try:
        if path.is_symlink():
            return None
        resolved = path.resolve(strict=True)
        resolved.relative_to(containment_root)
        if not resolved.is_file():
            return None
        before = resolved.stat()
        if (
            before.st_size != pin["size_bytes"]
            or not 1 <= before.st_size <= maximum_bytes
        ):
            return None
        content = resolved.read_bytes()
        after = resolved.stat()
        if (
            len(content) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or hashlib.sha256(content).hexdigest() != pin["sha256"]
        ):
            return None
        value = json.loads(content.decode("utf-8"))
    except (
        OSError,
        RuntimeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None
    return value if isinstance(value, dict) else None


def _rlivit_find_final_run(
    reader: ArtifactReader,
    receipt_pin: dict[str, Any],
) -> tuple[Path, str, dict[str, Any]] | None:
    """Find the nonce-isolated final receipt solely through its exact pin."""

    runs = reader.root / "rlivit" / "runs"
    try:
        if runs.is_symlink():
            return None
        resolved_runs = runs.resolve(strict=True)
        resolved_runs.relative_to(reader.root)
        if not resolved_runs.is_dir():
            return None
        directories = list(resolved_runs.iterdir())
    except (OSError, RuntimeError, ValueError):
        return None
    if len(directories) > MAX_RLIVIT_RUN_DIRECTORIES:
        return None
    matches: list[tuple[Path, str, dict[str, Any]]] = []
    for directory in directories:
        if (
            directory.is_symlink()
            or re.fullmatch(r"[0-9a-f]{64}", directory.name) is None
        ):
            continue
        try:
            resolved_directory = directory.resolve(strict=True)
            resolved_directory.relative_to(resolved_runs)
            if not resolved_directory.is_dir():
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        candidate = resolved_directory / "batch-receipt.json"
        try:
            if not candidate.is_file() or candidate.stat().st_size != receipt_pin["size_bytes"]:
                continue
        except OSError:
            continue
        value = _rlivit_read_pinned_json(
            candidate,
            containment_root=resolved_runs,
            pin=receipt_pin,
            maximum_bytes=reader.max_bytes,
        )
        if value is not None:
            matches.append((resolved_directory, directory.name, value))
            if len(matches) > 1:
                return None
    return matches[0] if len(matches) == 1 else None


def _rlivit_job_receipt_map(value: Any) -> dict[str, dict[str, Any]] | None:
    if not isinstance(value, dict) or len(value) != 80:
        return None
    result: dict[str, dict[str, Any]] = {}
    sequences: dict[str, set[str]] = {}
    for job_id, raw_pin in value.items():
        match = re.fullmatch(r"rlivit:([0-9]{3}):(640|960)", job_id)
        pin = _rlivit_full_pin(raw_pin)
        if match is None or pin is None:
            return None
        sequence, profile = match.groups()
        sequences.setdefault(sequence, set()).add(profile)
        result[job_id] = pin
    if len(sequences) != 40 or any(profiles != {"640", "960"} for profiles in sequences.values()):
        return None
    return result


def _rlivit_complete_proof(
    reader: ArtifactReader,
    *,
    evidence_pins: dict[str, dict[str, Any] | None],
    profile_metrics: dict[str, dict[str, Any]],
    mp4: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verify live final artifacts; the public status self-hash is not authority."""

    proof = {
        "status_fingerprint_role": "self_hash_integrity_only",
        "mp4_receipt_status_pin_cross_bound": False,
        "final_receipt_live_pin_verified": False,
        "aggregate_live_pin_verified": False,
        "job_receipts_live_pins_verified": False,
        "aggregate_status_metrics_cross_bound": False,
        "complete_claim_verified": False,
    }
    mp4_pin = evidence_pins.get("mp4_batch_receipt")
    if (
        mp4 is None
        or mp4_pin is None
        or mp4.get("batch_receipt_pin") != mp4_pin
    ):
        return proof
    proof["mp4_receipt_status_pin_cross_bound"] = True

    receipt_pin = evidence_pins.get("batch_receipt")
    aggregate_pin = evidence_pins.get("batch_aggregate")
    if receipt_pin is None or aggregate_pin is None:
        return proof
    located = _rlivit_find_final_run(reader, receipt_pin)
    if located is None:
        return proof
    run_directory, nonce, receipt = located
    expected_run_relative = f"validation/results/rlivit/runs/{nonce}"
    expected_receipt_keys = {
        "schema_version",
        "status",
        "created_at_utc",
        "campaign_nonce",
        "plan",
        "session_claim",
        "execution_authorization",
        "mp4_batch_receipt",
        "runtime_policy",
        "ds9_compatibility_receipt",
        "aggregate",
        "state",
        "sequence_count",
        "profiles",
        "job_count",
        "job_receipts",
        "fingerprint_sha256",
    }
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("schema_version")
        != "deepsafe.rlivit-deepstream-batch-receipt/v1"
        or receipt.get("status")
        != "complete_80_jobs_independent_raw_replay"
        or _timestamp(receipt.get("created_at_utc")) is None
        or receipt.get("campaign_nonce") != nonce
        or receipt.get("sequence_count") != 40
        or receipt.get("profiles") != [640, 960]
        or receipt.get("job_count") != 80
        or not _canonical_fingerprint_matches(receipt)
    ):
        return proof
    authorization = receipt.get("execution_authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("schema_version")
        != "deepsafe.rlivit-execution-authorization/v1"
        or authorization.get("status") != "approved"
        or authorization.get("campaign_nonce") != nonce
        or authorization.get("authorized_results_root") != expected_run_relative
        or authorization.get("single_use") is not True
    ):
        return proof
    receipt_jobs = _rlivit_job_receipt_map(receipt.get("job_receipts"))
    if receipt_jobs is None:
        return proof
    proof["final_receipt_live_pin_verified"] = True

    for key, filename in (
        ("plan", "batch-plan.json"),
        ("aggregate", "batch-aggregate.json"),
    ):
        full_pin = receipt.get(key)
        pathless = _rlivit_full_pin(full_pin)
        if (
            pathless is None
            or full_pin.get("path") != f"{expected_run_relative}/{filename}"
        ):
            return proof
        if key == "aggregate" and pathless != aggregate_pin:
            return proof

    plan_pin = _rlivit_full_pin(receipt["plan"])
    if plan_pin is None:
        return proof
    plan = _rlivit_read_pinned_json(
        run_directory / "batch-plan.json",
        containment_root=run_directory,
        pin=plan_pin,
        maximum_bytes=reader.max_bytes,
    )
    aggregate = _rlivit_read_pinned_json(
        run_directory / "batch-aggregate.json",
        containment_root=run_directory,
        pin=aggregate_pin,
        maximum_bytes=reader.max_bytes,
    )
    if plan is None or aggregate is None:
        return proof
    campaign = plan.get("campaign") if isinstance(plan.get("campaign"), dict) else {}
    ground_truth = (
        plan.get("ground_truth_validation")
        if isinstance(plan.get("ground_truth_validation"), dict)
        else {}
    )
    runtime_policy = (
        campaign.get("runtime_policy")
        if isinstance(campaign.get("runtime_policy"), dict)
        else {}
    )
    if (
        plan.get("schema_version")
        != "deepsafe.rlivit-deepstream-batch-plan/v1"
        or plan.get("status") != "ready_for_authorized_execution"
        or plan.get("blockers") != []
        or not _canonical_fingerprint_matches(plan)
        or campaign.get("campaign_nonce") != nonce
        or campaign.get("results_root") != expected_run_relative
        or campaign.get("sequence_count") != 40
        or campaign.get("profiles") != [640, 960]
        or campaign.get("expected_jobs") != 80
        or campaign.get("expected_frames_per_job") != 12
        or campaign.get("expected_ground_truth_persons") != 4318
        or campaign.get("execution_authorization") != authorization
        or len(plan.get("jobs", [])) != 80
        or ground_truth.get("live_sources_verified") is not True
        or ground_truth.get("frame_count") != 480
        or ground_truth.get("person_count") != 4318
        or _rlivit_full_pin(campaign.get("source_campaign"))
        != evidence_pins.get("source_plan")
        or _rlivit_full_pin(campaign.get("mp4_batch_receipt")) != mp4_pin
        or _rlivit_full_pin(runtime_policy.get("artifact"))
        != evidence_pins.get("runtime_policy")
        or _rlivit_full_pin(campaign.get("ds9_compatibility_receipt"))
        != evidence_pins.get("ds9_compatibility_receipt")
        or receipt.get("mp4_batch_receipt") != campaign.get("mp4_batch_receipt")
        or receipt.get("runtime_policy") != campaign.get("runtime_policy")
        or receipt.get("ds9_compatibility_receipt")
        != campaign.get("ds9_compatibility_receipt")
    ):
        return proof

    expected_aggregate_keys = {
        "schema_version",
        "status",
        "created_at_utc",
        "plan_fingerprint_sha256",
        "matrix",
        "metrics_contract",
        "profiles",
        "job_receipts",
        "fingerprint_sha256",
    }
    if (
        set(aggregate) != expected_aggregate_keys
        or aggregate.get("schema_version")
        != "deepsafe.rlivit-batch-evaluation/v1"
        or aggregate.get("status") != "complete_independent_raw_replay"
        or _timestamp(aggregate.get("created_at_utc")) is None
        or aggregate.get("plan_fingerprint_sha256")
        != plan.get("fingerprint_sha256")
        or aggregate.get("matrix")
        != {
            "sequence_count": 40,
            "profiles": [640, 960],
            "job_count": 80,
            "frames_per_job": 12,
        }
        or aggregate.get("metrics_contract")
        != {"iou": 0.5, "confidence": 0.25, "ap": "AP101@IoU0.5"}
        or not _canonical_fingerprint_matches(aggregate)
    ):
        return proof
    aggregate_jobs = _rlivit_job_receipt_map(aggregate.get("job_receipts"))
    if (
        aggregate_jobs is None
        or aggregate_jobs != receipt_jobs
        or aggregate.get("job_receipts") != receipt.get("job_receipts")
    ):
        return proof
    proof["aggregate_live_pin_verified"] = True

    total_job_receipt_bytes = 0
    for job_id in sorted(receipt_jobs):
        match = re.fullmatch(r"rlivit:([0-9]{3}):(640|960)", job_id)
        if match is None:
            return proof
        sequence, profile = match.groups()
        full_pin = receipt["job_receipts"].get(job_id)
        expected_path = (
            f"{expected_run_relative}/jobs/{sequence}/{profile}/job-receipt.json"
        )
        pin = receipt_jobs[job_id]
        if full_pin.get("path") != expected_path:
            return proof
        if not 1 <= pin["size_bytes"] <= min(
            reader.max_bytes, MAX_RLIVIT_JOB_RECEIPT_BYTES
        ):
            return proof
        total_job_receipt_bytes += pin["size_bytes"]
        if total_job_receipt_bytes > MAX_RLIVIT_TOTAL_JOB_RECEIPT_BYTES:
            return proof
        relative = Path("jobs") / sequence / profile / "job-receipt.json"
        current = run_directory
        try:
            for component in relative.parts:
                current = current / component
                if current.is_symlink():
                    return proof
        except OSError:
            return proof
        job_receipt = _rlivit_read_pinned_json(
            run_directory / relative,
            containment_root=run_directory,
            pin=pin,
            maximum_bytes=min(reader.max_bytes, MAX_RLIVIT_JOB_RECEIPT_BYTES),
        )
        if (
            job_receipt is None
            or job_receipt.get("schema_version")
            != "deepsafe.rlivit-deepstream-job-receipt/v1"
            or job_receipt.get("status") != "complete_raw_replay_verified"
            or job_receipt.get("job_id") != job_id
            or job_receipt.get("sequence_id") != sequence
            or job_receipt.get("model_input") != int(profile)
            or not _canonical_fingerprint_matches(job_receipt)
        ):
            return proof
    proof["job_receipts_live_pins_verified"] = True

    raw_profiles = aggregate.get("profiles")
    if not isinstance(raw_profiles, dict) or set(raw_profiles) != {"640", "960"}:
        return proof
    for profile in ("640", "960"):
        raw_profile = raw_profiles[profile]
        if not isinstance(raw_profile, dict):
            return proof
        selected = {
            key: raw_profile.get(key)
            for key in ("overall", "daytime", "locations", "coco_area", "height_bands")
        }
        aggregate_metrics = _rlivit_profile_metrics(selected)
        if aggregate_metrics is None or aggregate_metrics != profile_metrics.get(profile):
            return proof
    proof["aggregate_status_metrics_cross_bound"] = True
    proof["complete_claim_verified"] = True
    return proof


def _rlivit_mp4_contract(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    sequences = value.get("sequences") if isinstance(value.get("sequences"), dict) else {}
    gpu_jobs = value.get("gpu_jobs") if isinstance(value.get("gpu_jobs"), dict) else {}
    frames = value.get("frames") if isinstance(value.get("frames"), dict) else {}
    video = value.get("video") if isinstance(value.get("video"), dict) else {}
    coverage = value.get("coverage") if isinstance(value.get("coverage"), dict) else {}
    daytime_counts = (
        coverage.get("daytime_sequence_counts")
        if isinstance(coverage.get("daytime_sequence_counts"), dict)
        else {}
    )
    location_counts = (
        coverage.get("location_sequence_counts")
        if isinstance(coverage.get("location_sequence_counts"), dict)
        else {}
    )
    quality = (
        value.get("source_quality")
        if isinstance(value.get("source_quality"), dict)
        else {}
    )
    thresholds = (
        quality.get("thresholds")
        if isinstance(quality.get("thresholds"), dict)
        else {}
    )
    minima = (
        quality.get("minimum_observed")
        if isinstance(quality.get("minimum_observed"), dict)
        else {}
    )
    minimum_values: dict[str, float] = {}
    try:
        for key in ("psnr_y_db", "psnr_average_db", "ssim_y", "ssim_all"):
            raw = minima[key]
            if raw == "inf":
                minimum_values[key] = float("inf")
            elif isinstance(raw, str):
                minimum_values[key] = float(raw)
            else:
                raise ValueError("R-LiViT MP4 quality minimum is not text")
            if math.isnan(minimum_values[key]) or minimum_values[key] < 0:
                raise ValueError("R-LiViT MP4 quality minimum is invalid")
    except (KeyError, TypeError, ValueError, OverflowError):
        minimum_values = {}
    maximum_temperature = _integer(
        value.get("maximum_cpu_platform_temperature_millidegrees_celsius"),
        maximum=150_000,
    )
    thermal_policy_id = value.get("thermal_policy_id", "legacy_strict")
    temperature_contract_valid = bool(
        maximum_temperature is not None
        and (
            (
                thermal_policy_id == "legacy_strict"
                and 0 <= maximum_temperature < 85_000
            )
            or (
                thermal_policy_id == "workstation_managed"
                and 0 <= maximum_temperature <= 150_000
            )
        )
    )
    try:
        batch_receipt_pin = _rlivit_pathless_pin(value.get("batch_receipt_pin"))
    except ValueError:
        return None
    base_fields = {
            "schema_version",
            "status",
            "dataset_id",
            "sequences",
            "gpu_jobs",
            "frames",
            "coverage",
            "video",
            "source_quality",
            "maximum_cpu_platform_temperature_millidegrees_celsius",
            "gpu_executed",
            "docker_executed",
            "inference_executed",
            "batch_fingerprint_sha256",
            "batch_receipt_pin",
            "fingerprint_sha256",
    }
    if (
        set(value)
        not in {
            frozenset(base_fields),
            frozenset(base_fields | {"thermal_policy_id"}),
        }
        or value.get("schema_version") != "deepsafe.rlivit-mp4-admin-receipt/v2"
        or value.get("status") != "complete_verified_cpu_only"
        or value.get("dataset_id") != "R-LiViT_RGB-T_v1.0"
        or sequences != {"complete": 40, "expected": 40}
        or gpu_jobs.get("blocked") != 80
        or gpu_jobs.get("expected") != 80
        or gpu_jobs.get("status")
        != "blocked_pending_mp4_gpu_reentry_and_model_binding"
        or frames != {"per_video": 12, "total": 480}
        or video.get("codec_name") != "h264"
        or video.get("profile") != "High"
        or video.get("pixel_format") != "yuv420p"
        or video.get("width") != 1280
        or video.get("height") != 720
        or video.get("fps") != "5/4"
        or video.get("duration_seconds") != "9.600000"
        or set(daytime_counts) != {"day", "night"}
        or any(_integer(item) is None for item in daytime_counts.values())
        or sum(daytime_counts.values()) != 40
        or set(location_counts) != {str(index) for index in range(8)}
        or any(_integer(item) is None for item in location_counts.values())
        or sum(location_counts.values()) != 40
        or coverage.get("distinct_locations") != 8
        or quality.get("status")
        != "all_frames_pass_fixed_psnr_ssim_floors"
        or quality.get("software_decode_and_filter_only") is not True
        or thresholds
        != {
            "comparison_pixel_format": "yuv420p",
            "minimum_frame_psnr_y_db": 45.0,
            "minimum_frame_psnr_average_db": 45.0,
            "minimum_frame_ssim_y": 0.99,
            "minimum_frame_ssim_all": 0.99,
            "software_decode_and_filter_only": True,
        }
        or set(minimum_values)
        != {"psnr_y_db", "psnr_average_db", "ssim_y", "ssim_all"}
        or minimum_values.get("psnr_y_db", -1) < 45.0
        or minimum_values.get("psnr_average_db", -1) < 45.0
        or minimum_values.get("ssim_y", -1) < 0.99
        or minimum_values.get("ssim_all", -1) < 0.99
        or minimum_values.get("ssim_y", 2) > 1.0
        or minimum_values.get("ssim_all", 2) > 1.0
        or not temperature_contract_valid
        or value.get("gpu_executed") is not False
        or value.get("docker_executed") is not False
        or value.get("inference_executed") is not False
        or _sha256(value.get("batch_fingerprint_sha256")) is None
        or batch_receipt_pin is None
        or not _canonical_fingerprint_matches(value)
    ):
        return None
    return {
        "status": "complete_verified_cpu_only",
        "complete_sequences": 40,
        "expected_sequences": 40,
        "frames": 480,
        "codec": "H.264 High / yuv420p",
        "fps": "5/4",
        "minimum_psnr_db": (
            "inf"
            if math.isinf(
                min(
                    minimum_values["psnr_y_db"],
                    minimum_values["psnr_average_db"],
                )
            )
            else round(
                min(
                    minimum_values["psnr_y_db"],
                    minimum_values["psnr_average_db"],
                ),
                6,
            )
        ),
        "minimum_ssim": min(
            minimum_values["ssim_y"], minimum_values["ssim_all"]
        ),
        "maximum_cpu_platform_temperature_c": round(
            maximum_temperature / 1000, 3
        ),
        "thermal_policy_id": thermal_policy_id,
        "gpu_executed": False,
        "batch_receipt_pin": batch_receipt_pin,
        "cross_bound_to_campaign": False,
    }


def _rlivit(reader: ArtifactReader) -> dict[str, Any]:
    status_artifact = reader.read("rlivit_public_status")
    mp4_artifact = reader.read("rlivit_mp4_receipt")
    value = status_artifact.value or {}
    mp4 = (
        _rlivit_mp4_contract(mp4_artifact.value)
        if mp4_artifact.available
        else None
    )
    structurally_valid = False
    profiles: dict[str, Any] = {}
    profile_metrics: dict[str, dict[str, Any]] = {}
    evidence_readiness = {
        "source_plan": False,
        "mp4_batch_receipt": False,
        "runtime_policy": False,
        "ds9_compatibility_receipt": False,
        "batch_aggregate": False,
        "batch_receipt": False,
    }
    evidence_pins: dict[str, dict[str, Any] | None] = {
        key: None for key in evidence_readiness
    }
    proof_binding = _rlivit_complete_proof(
        reader,
        evidence_pins=evidence_pins,
        profile_metrics=profile_metrics,
        mp4=None,
    )
    status_self_hash_valid = bool(
        status_artifact.available and _canonical_fingerprint_matches(value)
    )
    blocker_codes: list[str] = []
    phase: str | None = None
    progress_value: dict[str, Any] = {}

    try:
        matrix = value.get("matrix") if isinstance(value.get("matrix"), dict) else {}
        ground_truth = value.get("ground_truth") if isinstance(value.get("ground_truth"), dict) else {}
        progress_raw = value.get("progress") if isinstance(value.get("progress"), dict) else {}
        profile_raw = value.get("profiles") if isinstance(value.get("profiles"), dict) else {}
        evidence_raw = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
        phase = _enum(
            value.get("status"),
            {"blocked", "awaiting_execution", "running", "failed", "complete"},
        )
        expected_matrix = {
            "sequence_count": 40,
            "profiles": [640, 960],
            "job_count": 80,
            "frames_per_job": 12,
            "ground_truth_frames_per_profile": 480,
        }
        expected_gt = {
            "status": "valid",
            "sequence_count": 40,
            "frame_count": 480,
            "person_count": 4318,
            "daytime": ["day", "night"],
            "locations": [str(index) for index in range(8)],
            "live_sources_verified": ground_truth.get("live_sources_verified"),
        }
        if (
            set(value)
            != {
                "schema_version",
                "status",
                "updated_at_utc",
                "dataset_id",
                "gpu_docker_inference_executed",
                "matrix",
                "ground_truth",
                "progress",
                "profiles",
                "blocker_codes",
                "evidence",
                "fingerprint_sha256",
            }
            or value.get("schema_version") != "deepsafe.rlivit-public-status/v1"
            or value.get("dataset_id") != "R-LiViT_RGB-T_v1.0"
            or phase is None
            or _timestamp(value.get("updated_at_utc")) is None
            or matrix != expected_matrix
            or ground_truth != expected_gt
            or not isinstance(ground_truth.get("live_sources_verified"), bool)
            or set(progress_raw)
            != {
                "planned_jobs",
                "launched_jobs",
                "gpu_process_started_jobs",
                "completed_jobs",
                "failed_jobs",
                "remaining_jobs",
            }
            or set(profile_raw) != {"640", "960"}
            or set(evidence_raw) != set(evidence_readiness)
            or not _canonical_fingerprint_matches(value)
        ):
            raise ValueError("R-LiViT public status contract differs")

        planned = _integer(progress_raw.get("planned_jobs"))
        launched = _integer(progress_raw.get("launched_jobs"))
        gpu_started = _integer(progress_raw.get("gpu_process_started_jobs"))
        completed = _integer(progress_raw.get("completed_jobs"))
        failed = _integer(progress_raw.get("failed_jobs"))
        remaining = _integer(progress_raw.get("remaining_jobs"))
        if (
            planned != 80
            or launched is None
            or gpu_started is None
            or completed is None
            or failed not in {0, 1}
            or remaining != 80 - completed
            or not 0 <= completed <= gpu_started <= launched <= 80
            or launched < completed + failed
            or value.get("gpu_docker_inference_executed") is not (gpu_started > 0)
        ):
            raise ValueError("R-LiViT progress is inconsistent")

        blocker_codes = _identifiers(value.get("blocker_codes"), maximum=16)
        if blocker_codes != value.get("blocker_codes"):
            raise ValueError("R-LiViT blocker codes are invalid")
        for key in evidence_readiness:
            pin = _rlivit_pathless_pin(evidence_raw.get(key))
            evidence_pins[key] = pin
            evidence_readiness[key] = pin is not None

        completed_by_profile = 0
        for profile in ("640", "960"):
            raw = profile_raw[profile]
            if not isinstance(raw, dict) or set(raw) != {
                "status",
                "planned_jobs",
                "completed_jobs",
                "metrics",
            }:
                raise ValueError("R-LiViT profile contract differs")
            profile_state = _enum(
                raw.get("status"),
                {
                    "blocked",
                    "pending",
                    "running",
                    "jobs_complete_aggregate_pending",
                    "complete",
                },
            )
            profile_completed = _integer(raw.get("completed_jobs"))
            if (
                raw.get("planned_jobs") != 40
                or profile_state is None
                or profile_completed is None
                or profile_completed > 40
            ):
                raise ValueError("R-LiViT profile progress differs")
            metrics = (
                _rlivit_profile_metrics(raw.get("metrics"))
                if raw.get("metrics") is not None
                else None
            )
            if (profile_state == "complete") is not (metrics is not None):
                raise ValueError("R-LiViT metrics/profile state differs")
            if (
                (profile_state == "pending" and profile_completed != 0)
                or (
                    profile_state == "running"
                    and not 0 <= profile_completed < 40
                )
                or (
                    profile_state == "jobs_complete_aggregate_pending"
                    and profile_completed != 40
                )
                or (profile_state == "complete" and profile_completed != 40)
            ):
                raise ValueError("R-LiViT profile phase/count differs")
            profiles[profile] = {
                "status": profile_state,
                "planned_jobs": 40,
                "completed_jobs": profile_completed,
                "overall": metrics["overall"] if metrics is not None else None,
            }
            if metrics is not None:
                profile_metrics[profile] = metrics
            completed_by_profile += profile_completed
        if completed_by_profile != completed:
            raise ValueError("R-LiViT profile totals differ")

        proof_binding = _rlivit_complete_proof(
            reader,
            evidence_pins=evidence_pins,
            profile_metrics=profile_metrics,
            mp4=mp4,
        )
        prerequisite_evidence_ready = all(
            evidence_readiness[key]
            for key in (
                "source_plan",
                "mp4_batch_receipt",
                "runtime_policy",
                "ds9_compatibility_receipt",
            )
        )
        final_evidence_ready = all(
            evidence_readiness[key]
            for key in ("batch_aggregate", "batch_receipt")
        )
        final_evidence_absent = not any(
            evidence_readiness[key]
            for key in ("batch_aggregate", "batch_receipt")
        )
        live_sources_verified = ground_truth.get("live_sources_verified") is True

        if phase == "blocked":
            valid_phase = bool(
                blocker_codes
                and launched == gpu_started == completed == failed == 0
                and all(item["status"] == "blocked" for item in profiles.values())
            )
        elif phase == "awaiting_execution":
            valid_phase = bool(
                not blocker_codes
                and launched == gpu_started == completed == failed == 0
                and all(item["status"] == "pending" for item in profiles.values())
                and live_sources_verified
                and prerequisite_evidence_ready
                and final_evidence_absent
            )
        elif phase == "running":
            valid_phase = bool(
                not blocker_codes
                and completed < 80
                and failed == 0
                and live_sources_verified
                and prerequisite_evidence_ready
                and final_evidence_absent
                and all(
                    item["status"]
                    in {"pending", "running", "jobs_complete_aggregate_pending"}
                    for item in profiles.values()
                )
            )
        elif phase == "failed":
            valid_phase = bool(
                not blocker_codes
                and completed < 80
                and failed == 1
                and live_sources_verified
                and prerequisite_evidence_ready
                and final_evidence_absent
                and all(
                    item["status"]
                    in {"pending", "running", "jobs_complete_aggregate_pending"}
                    for item in profiles.values()
                )
            )
        else:
            valid_phase = bool(
                not blocker_codes
                and launched == gpu_started == completed == 80
                and failed == remaining == 0
                and live_sources_verified
                and all(
                    item["status"] == "complete"
                    and item["completed_jobs"] == 40
                    and item["overall"] is not None
                    for item in profiles.values()
                )
                and prerequisite_evidence_ready
                and final_evidence_ready
                and proof_binding["job_receipts_live_pins_verified"] is True
                and proof_binding["complete_claim_verified"] is True
            )
        if not valid_phase:
            raise ValueError("R-LiViT phase semantics differ")
        progress_value = _progress(completed, 80)
        structurally_valid = True
    except (KeyError, TypeError, ValueError):
        structurally_valid = False
        profiles = {}
        profile_metrics = {}
        blocker_codes = []
        evidence_readiness = {key: False for key in evidence_readiness}
        evidence_pins = {key: None for key in evidence_pins}
        proof_binding = _rlivit_complete_proof(
            reader,
            evidence_pins=evidence_pins,
            profile_metrics=profile_metrics,
            mp4=None,
        )

    artifact_error = bool(
        status_artifact.state not in {"ok", "missing"}
        or mp4_artifact.state not in {"ok", "missing"}
        or (status_artifact.available and not structurally_valid)
        or (mp4_artifact.available and mp4 is None)
    )
    if artifact_error:
        state = "artifact_error"
    elif not status_artifact.available:
        state = "not_started"
    else:
        state = {
            "blocked": "blocked",
            "awaiting_execution": "planned",
            "running": "running",
            "failed": "failed",
            "complete": "complete",
        }.get(phase, "artifact_error")
    return {
        "label": "R-LiViT yüksek/oblik kamera person GT",
        "available": status_artifact.available or mp4_artifact.available,
        "state": state,
        "updated_at_utc": _timestamp(value.get("updated_at_utc")) if structurally_valid else None,
        "progress": progress_value if structurally_valid else _progress(0, 0),
        "scope": {
            "split": "test",
            "sequences": 40,
            "frames": 480,
            "targets": 4318,
            "jobs": 80,
            "model_input_sizes": [640, 960],
            "camera_geometry": "fixed_high_oblique_road_surveillance",
            "day_night": True,
            "locations": 8,
        },
        "ground_truth": {
            "valid": bool(
                structurally_valid
                and value.get("ground_truth", {}).get("live_sources_verified")
            ),
            "live_sources_verified": (
                value.get("ground_truth", {}).get("live_sources_verified")
                if structurally_valid
                else False
            ),
            "person_boxes_per_profile": 4318,
        },
        "accelerated_execution_occurred": (
            value.get("gpu_docker_inference_executed")
            if structurally_valid
            else None
        ),
        "blocker_codes": blocker_codes,
        "input_materialization": (
            {
                **mp4,
                "cross_bound_to_campaign": proof_binding[
                    "mp4_receipt_status_pin_cross_bound"
                ],
            }
            if mp4 is not None
            else {"status": "not_ready", "cross_bound_to_campaign": False}
        ),
        "evidence_readiness": evidence_readiness,
        "proof_binding": {
            **proof_binding,
            "status_self_hash_valid": status_self_hash_valid,
        },
        "profiles": profiles,
        "metric_context": {
            "ground_truth": True,
            "dataset_id": "R-LiViT_RGB-T_v1.0",
            "metric": "AP101@IoU0.5",
            "iou_threshold": 0.5,
            "confidence_threshold_for_precision_recall_f1": 0.25,
            "coco_map": False,
            "withheld_until_complete": not (
                structurally_valid and phase == "complete"
            ),
        },
        "evidence": _evidence(
            reader, "rlivit_public_status", "rlivit_mp4_receipt"
        ),
    }


def _open_video_review(reader: ArtifactReader) -> dict[str, Any]:
    """Project three GT-free evidence levels from the schema-validated report.

    Raw renderer job states are intentionally not an acceptance source: a
    rendered frame is only an automatic candidate, and a withheld frame is
    neither AI-audited nor human-reviewed evidence.
    """

    report_artifact = reader.read("campaign_report_json")
    report = report_artifact.value or {}
    report_schema_valid = bool(
        report_artifact.available
        and reader.validates_schema(report, CAMPAIGN_REPORT_SCHEMA)
    )
    campaigns = (
        report.get("campaigns")
        if report_schema_valid and isinstance(report.get("campaigns"), dict)
        else {}
    )
    section = (
        campaigns.get("open_video_manual_review")
        if isinstance(campaigns.get("open_video_manual_review"), dict)
        else None
    )
    automatic = (
        section.get("automatic_candidate_generation")
        if isinstance(section, dict)
        and isinstance(section.get("automatic_candidate_generation"), dict)
        else {}
    )
    ai_audit = (
        section.get("ai_qualitative_visual_audit")
        if isinstance(section, dict)
        and isinstance(section.get("ai_qualitative_visual_audit"), dict)
        else {}
    )
    human_qa = (
        section.get("human_terminal_qa")
        if isinstance(section, dict)
        and isinstance(section.get("human_terminal_qa"), dict)
        else {}
    )
    paired = (
        section.get("paired_profile_comparison")
        if isinstance(section, dict)
        and isinstance(section.get("paired_profile_comparison"), dict)
        else {}
    )
    assets = (
        automatic.get("candidate_assets")
        if isinstance(automatic.get("candidate_assets"), dict)
        else {}
    )
    coverage = (
        ai_audit.get("coverage")
        if isinstance(ai_audit.get("coverage"), dict)
        else {}
    )

    rendered_scenes = automatic.get("rendered_scene_ids", [])
    withheld_scenes = automatic.get("withheld_scene_ids", [])
    mixed_scenes = automatic.get("mixed_scene_ids", [])
    reviewed_scenes = ai_audit.get("reviewed_scene_ids", [])
    requirements = report.get("requirements") if isinstance(report.get("requirements"), list) else []
    open_requirements = [
        item
        for item in requirements[:1_000]
        if isinstance(item, dict)
        and item.get("id") == "open_video_ai_qualitative_audit"
    ]
    automatic_accepted = automatic.get("accepted") is True
    ai_accepted = ai_audit.get("accepted") is True
    human_accepted = human_qa.get("accepted") is True
    candidate_records = _integer(section.get("candidate_job_records")) if section else None
    validated_jobs = _integer(automatic.get("validated_jobs"))
    rendered_jobs = _integer(automatic.get("rendered_jobs"))
    withheld_jobs = _integer(automatic.get("withheld_jobs"))
    ai_source_frames = _integer(ai_audit.get("source_frame_count"))
    ai_profile_decisions = _integer(ai_audit.get("profile_decision_count"))
    human_terminal = _integer(human_qa.get("terminal_decisions"))
    human_pending = _integer(human_qa.get("pending_decisions"))
    pair_rows = paired.get("pairs", []) if isinstance(paired.get("pairs"), list) else []
    paired_scenes = [
        row.get("scene_id")
        for row in pair_rows
        if isinstance(row, dict) and isinstance(row.get("scene_id"), str)
    ]
    paired_source_ids = [
        source_id
        for row in pair_rows
        if isinstance(row, dict) and isinstance(row.get("source_review_ids"), list)
        for source_id in row["source_review_ids"]
    ]
    paired_decision_ids = [
        decision_id
        for row in pair_rows
        if isinstance(row, dict) and isinstance(row.get("decision_ids"), list)
        for decision_id in row["decision_ids"]
    ]

    structurally_valid = bool(
        report_schema_valid
        and section is not None
        and section.get("evidence_kind")
        == "tiered_gt_free_open_video_evidence"
        and section.get("ground_truth") is False
        and section.get("expected_candidate_jobs")
        == OPEN_VIDEO_EXPECTED_CANDIDATE_JOBS
        and candidate_records is not None
        and validated_jobs is not None
        and section.get("validated_candidate_jobs") == validated_jobs
        and rendered_jobs is not None
        and withheld_jobs is not None
        and rendered_jobs + withheld_jobs == candidate_records
        and automatic.get("expected_jobs")
        == OPEN_VIDEO_EXPECTED_CANDIDATE_JOBS
        and assets.get("expected_decision_count")
        == OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS
        and assets.get("expected_asset_count") == OPEN_VIDEO_EXPECTED_ASSETS
        and automatic.get("not_a_visual_review") is True
        and isinstance(rendered_scenes, list)
        and isinstance(withheld_scenes, list)
        and isinstance(mixed_scenes, list)
        and not set(rendered_scenes).intersection(withheld_scenes)
        and not mixed_scenes
        and ai_audit.get("evidence_level")
        == "hash_bound_ai_visual_audit_not_human_qa"
        and ai_audit.get("expected_source_frame_count")
        == OPEN_VIDEO_EXPECTED_SOURCE_FRAMES
        and ai_audit.get("expected_profile_decision_count")
        == OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS
        and ai_audit.get("minimum_distinct_video_types")
        == OPEN_VIDEO_MINIMUM_DISTINCT_VIDEO_TYPES
        and ai_source_frames is not None
        and ai_profile_decisions is not None
        and isinstance(reviewed_scenes, list)
        and ai_audit.get("distinct_reviewed_video_types")
        == len(reviewed_scenes)
        and not set(reviewed_scenes).intersection(withheld_scenes)
        and set(reviewed_scenes).issubset(set(rendered_scenes))
        and ai_audit.get("human_review_claimed") is False
        and ai_audit.get("ground_truth_claimed") is False
        and human_qa.get("evidence_level") == "terminal_human_qa"
        and human_qa.get("expected_decisions")
        == OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS
        and human_terminal is not None
        and human_pending is not None
        and human_terminal + human_pending
        == OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS
        and human_qa.get("required_for_ai_audit_acceptance") is False
        and paired.get("paired_source_frame_count") == ai_source_frames
        and paired.get("paired_scene_count")
        == len(pair_rows)
        and len(paired_scenes) == len(pair_rows)
        and set(paired_scenes) == set(reviewed_scenes)
        and len(paired_source_ids) == ai_source_frames
        and len(set(paired_source_ids)) == len(paired_source_ids)
        and len(paired_decision_ids) == ai_profile_decisions
        and len(set(paired_decision_ids)) == len(paired_decision_ids)
        and ai_profile_decisions == 2 * ai_source_frames
        and all(
            isinstance(row, dict)
            and isinstance(row.get("source_review_ids"), list)
            and isinstance(row.get("decision_ids"), list)
            and len(row["decision_ids"]) == 2 * len(row["source_review_ids"])
            for row in pair_rows
        )
        and section.get("accepted") is ai_accepted
        and len(open_requirements) == 1
        and open_requirements[0].get("required_for_acceptance") is True
        and open_requirements[0].get("state")
        == ("pass" if ai_accepted else "incomplete")
        and (not automatic_accepted or (
            candidate_records == OPEN_VIDEO_EXPECTED_CANDIDATE_JOBS
            and validated_jobs == OPEN_VIDEO_EXPECTED_CANDIDATE_JOBS
            and assets.get("decision_count")
            == OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS
            and assets.get("asset_count") == OPEN_VIDEO_EXPECTED_ASSETS
            and assets.get("live_hash_verified_assets")
            == OPEN_VIDEO_EXPECTED_ASSETS
            and assets.get("index_contract_valid") is True
        ))
        and (not ai_accepted or (
            automatic_accepted
            and ai_audit.get("status") == "complete"
            and ai_source_frames == OPEN_VIDEO_EXPECTED_SOURCE_FRAMES
            and ai_profile_decisions == OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS
            and _integer(ai_audit.get("distinct_reviewed_video_types")) is not None
            and ai_audit["distinct_reviewed_video_types"]
            >= OPEN_VIDEO_MINIMUM_DISTINCT_VIDEO_TYPES
            and all(
                (_integer(coverage.get(key)) or 0) > 0
                for key in (
                    "medium_close_source_frames",
                    "overhead_top_view_source_frames",
                    "high_oblique_source_frames",
                )
            )
            and coverage.get("required_coverage_proven") is True
            and ai_audit.get("input_hash_binding_valid") is True
            and ai_audit.get("exact_id_profile_count_binding_valid") is True
            and ai_audit.get("withheld_scenes_excluded") is True
        ))
        and human_accepted
        == (
            human_qa.get("state") == "complete"
            and human_qa.get("artifact_contract_valid") is True
            and human_terminal == OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS
            and human_pending == 0
        )
    )

    if not report_artifact.available:
        state = "not_started" if report_artifact.state == "missing" else "artifact_error"
    elif not structurally_valid:
        state = "artifact_error"
    elif ai_accepted:
        state = "complete"
    elif automatic_accepted or (ai_profile_decisions or 0) > 0:
        state = "attention"
    else:
        state = "planned"

    trusted_validated_jobs = validated_jobs if structurally_valid else 0
    trusted_ai_source_frames = ai_source_frames if structurally_valid else 0
    trusted_ai_profile_decisions = (
        ai_profile_decisions if structurally_valid else 0
    )
    trusted_human_terminal = human_terminal if structurally_valid else 0

    automatic_projection = {
        "state": (
            state
            if not structurally_valid
            else (
                "complete"
                if automatic_accepted
                else (
                    "attention"
                    if (trusted_validated_jobs or 0) > 0
                    else "planned"
                )
            )
        ),
        "accepted": automatic_accepted if structurally_valid else False,
        "progress": _progress(
            trusted_validated_jobs or 0, OPEN_VIDEO_EXPECTED_CANDIDATE_JOBS
        ),
        "rendered_jobs": rendered_jobs if structurally_valid else None,
        "withheld_jobs": withheld_jobs if structurally_valid else None,
        "candidate_assets": {
            "decisions": _progress(
                (
                    _integer(assets.get("decision_count"))
                    if structurally_valid
                    else 0
                )
                or 0,
                OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS,
            ),
            "assets": _progress(
                (
                    _integer(assets.get("live_hash_verified_assets"))
                    if structurally_valid
                    else 0
                )
                or 0,
                OPEN_VIDEO_EXPECTED_ASSETS,
            ),
            "index_contract_valid": (
                _boolean(assets.get("index_contract_valid"))
                if structurally_valid
                else False
            ),
        },
        "not_a_visual_review": True,
    }
    ai_projection = {
        "state": ai_audit.get("status") if structurally_valid else state,
        "accepted": ai_accepted if structurally_valid else False,
        "source_frames": _progress(
            trusted_ai_source_frames or 0, OPEN_VIDEO_EXPECTED_SOURCE_FRAMES
        ),
        "profile_decisions": _progress(
            trusted_ai_profile_decisions or 0,
            OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS,
        ),
        "distinct_video_types": (
            _integer(ai_audit.get("distinct_reviewed_video_types"))
            if structurally_valid
            else None
        ),
        "minimum_distinct_video_types": OPEN_VIDEO_MINIMUM_DISTINCT_VIDEO_TYPES,
        "coverage": {
            key: (_integer(coverage.get(key)) if structurally_valid else None)
            for key in (
                "medium_close_source_frames",
                "overhead_top_view_source_frames",
                "high_oblique_source_frames",
            )
        },
        "input_hash_binding_valid": (
            _boolean(ai_audit.get("input_hash_binding_valid"))
            if structurally_valid
            else False
        ),
        "exact_id_profile_count_binding_valid": (
            _boolean(ai_audit.get("exact_id_profile_count_binding_valid"))
            if structurally_valid
            else False
        ),
        "withheld_scenes_excluded": (
            _boolean(ai_audit.get("withheld_scenes_excluded"))
            if structurally_valid
            else False
        ),
        "human_review_claimed": False,
        "ground_truth_claimed": False,
    }
    human_projection = {
        "state": human_qa.get("state") if structurally_valid else state,
        "accepted": human_accepted if structurally_valid else False,
        "progress": _progress(
            trusted_human_terminal or 0, OPEN_VIDEO_EXPECTED_PROFILE_DECISIONS
        ),
        "pending_decisions": human_pending if structurally_valid else None,
        "artifact_contract_valid": (
            _boolean(human_qa.get("artifact_contract_valid"))
            if structurally_valid
            else False
        ),
        "required_for_ai_audit_acceptance": False,
    }
    return {
        "label": "Açık video hash-bağlı AI nitel audit",
        "available": structurally_valid,
        "state": state,
        "updated_at_utc": (
            _timestamp(report.get("generated_at_utc"))
            if structurally_valid
            else None
        ),
        "progress_label": "Hash-bağlı AI audit",
        "progress": ai_projection["profile_decisions"],
        "automatic_candidate_generation": automatic_projection,
        "ai_qualitative_visual_audit": ai_projection,
        "human_terminal_qa": human_projection,
        "paired_profile_comparison": {
            "paired_scene_count": (
                _integer(paired.get("paired_scene_count"))
                if structurally_valid
                else None
            ),
            "paired_source_frame_count": (
                _integer(paired.get("paired_source_frame_count"))
                if structurally_valid
                else None
            ),
        },
        "scope": {
            "scene_types": (
                _integer(ai_audit.get("distinct_reviewed_video_types"))
                if structurally_valid
                else None
            ),
            "model_input_sizes": [640, 960],
            "jobs": OPEN_VIDEO_EXPECTED_CANDIDATE_JOBS,
        },
        "metric_context": {
            "ground_truth": False,
            "qualitative_review_only": True,
            "accuracy_metrics_forbidden": True,
            "automatic_candidates_are_not_visual_review": True,
            "ai_audit_is_not_human_qa": True,
        },
        "evidence": _evidence(
            reader,
            "campaign_report_json",
            "open_video_plan",
            "open_video_review",
        ),
    }


_ENDURANCE_FLOOR_BINDING_SCHEMA = (
    "deepsafe.endurance-throughput-floor-binding/v1"
)
_ENDURANCE_FLOOR_ARTIFACT_SCHEMA = "deepsafe.endurance-throughput-floor/v1"
_ENDURANCE_FLOOR_EVIDENCE_SCHEMA = (
    "deepsafe.endurance-throughput-floor-evidence/v1"
)
_ENDURANCE_POWER_LIMIT_FIELDS = [
    "power_requested_limit_w",
    "power_current_limit_w",
    "power_default_limit_w",
]
_ENDURANCE_DIAGNOSTIC_SLOWDOWN_FLAGS = [
    "clock_event_sw_thermal_slowdown",
    "clock_event_hw_slowdown",
    "clock_event_hw_thermal_slowdown",
    "clock_event_hw_power_brake_slowdown",
]
_ENDURANCE_SCENE_POWER_SAFETY_FIELDS = {
    "operating_policy_mode",
    "hardware_protection_owner",
    "static_signal_action",
    "power_limit_drop_tolerance_w",
    "slowdown_consecutive_samples",
    "preflight_samples",
    "preflight_sample_interval_seconds",
    "power_limit_fields",
    "diagnostic_slowdown_flags",
    "abort_slowdown_flags",
    "required_telemetry_failure_action",
}


def _positive_fps(value: Any) -> float | None:
    result = _number(value)
    if result is None or not 0 < result <= 1_000_000:
        return None
    return result


def _private_pin_shape(value: Any, *, size_key: str) -> bool:
    """Validate a private pin without ever returning its path."""

    if not isinstance(value, dict) or set(value) != {"path", size_key, "sha256"}:
        return False
    path = value.get("path")
    size = value.get(size_key)
    return bool(
        isinstance(path, str)
        and 1 <= len(path) <= 4_096
        and not any(ord(character) < 32 for character in path)
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 1 <= size <= MAX_PINNED_FILE_BYTES
        and _sha256(value.get("sha256")) is not None
    )


def _safe_gpu_name(value: Any) -> str | None:
    name = _text(value, limit=100)
    if name is None or re.fullmatch(r"[A-Za-z0-9 ._()+-]+", name) is None:
        return None
    return name


def _safe_driver_version(value: Any) -> str | None:
    driver = _text(value, limit=40)
    if driver is None or re.fullmatch(r"[A-Za-z0-9._-]+", driver) is None:
        return None
    return driver


def _endurance_scene_power_safety_is_exact(value: Any) -> bool:
    """Validate the exact mode-aware policy emitted with a frozen FPS floor."""

    if (
        not isinstance(value, dict)
        or set(value) != _ENDURANCE_SCENE_POWER_SAFETY_FIELDS
    ):
        return False
    mode = value.get("operating_policy_mode")
    if mode not in {"workstation_managed", "legacy_strict"}:
        return False
    tolerance = _number(value.get("power_limit_drop_tolerance_w"))
    slowdown_samples = _integer(
        value.get("slowdown_consecutive_samples"), maximum=10_000
    )
    preflight_samples = _integer(value.get("preflight_samples"), maximum=10_000)
    preflight_interval = _number(value.get("preflight_sample_interval_seconds"))
    if (
        tolerance is None
        or not 0 < tolerance <= 10_000
        or slowdown_samples is None
        or slowdown_samples < 1
        or preflight_samples is None
        or preflight_samples < 1
        or preflight_interval is None
        or not 0 < preflight_interval <= 10_000
    ):
        return False
    strict = mode == "legacy_strict"
    expected = {
        "operating_policy_mode": mode,
        "hardware_protection_owner": "workstation_bios_ec_nvidia_driver",
        "static_signal_action": (
            "safety_abort" if strict else "record_measurement_quality_diagnostic"
        ),
        "power_limit_drop_tolerance_w": value["power_limit_drop_tolerance_w"],
        "slowdown_consecutive_samples": slowdown_samples,
        "preflight_samples": preflight_samples,
        "preflight_sample_interval_seconds": value[
            "preflight_sample_interval_seconds"
        ],
        "power_limit_fields": list(_ENDURANCE_POWER_LIMIT_FIELDS),
        "diagnostic_slowdown_flags": list(
            _ENDURANCE_DIAGNOSTIC_SLOWDOWN_FLAGS
        ),
        "abort_slowdown_flags": (
            list(_ENDURANCE_DIAGNOSTIC_SLOWDOWN_FLAGS) if strict else []
        ),
        "required_telemetry_failure_action": "safety_abort",
    }
    return value == expected


def _synthetic_floor_binding_is_exact(value: dict[str, Any]) -> bool:
    return bool(
        value.get("artifact_schema") is None
        and value.get("artifact_fingerprint") is None
        and value.get("artifact_pin") is None
        and value.get("source_runtime_identity") is None
        and value.get("profiles")
        == {
            "640": {
                "aggregate_fps_floor": None,
                "per_stream_fps_floor": None,
            },
            "960": {
                "aggregate_fps_floor": None,
                "per_stream_fps_floor": None,
            },
        }
        and value.get("verification")
        == {
            "status": "not_required_for_synthetic_dry_run",
            "live_rederived": False,
            "verified_safe_runs": 0,
        }
        and value.get("source_inputs")
        == {"summary": None, "scene_manifest": None}
    )


def _throughput_floor_projection(value: Any) -> dict[str, Any]:
    """Project only the public root of a verified frozen-floor binding."""

    unavailable = {
        "status": "missing",
        "acceptance_safe": False,
        "artifact_fingerprint": None,
        "artifact_fingerprint_short": None,
        "verified_safe_runs": 0,
        "live_rederived": False,
        "profiles": {},
        "source_runtime": {"gpu_name": None, "driver_version": None},
    }
    if value is None:
        return unavailable
    expected_fields = {
        "schema_version",
        "status",
        "artifact_schema",
        "artifact_fingerprint",
        "artifact_pin",
        "profiles",
        "verification",
        "source_runtime_identity",
        "source_inputs",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("schema_version") != _ENDURANCE_FLOOR_BINDING_SCHEMA
    ):
        return {**unavailable, "status": "invalid"}
    if value.get("status") == "synthetic_pending":
        return {
            **unavailable,
            "status": (
                "synthetic_pending"
                if _synthetic_floor_binding_is_exact(value)
                else "invalid"
            ),
        }
    fingerprint = _sha256(value.get("artifact_fingerprint"))
    verification = value.get("verification")
    profiles = value.get("profiles")
    runtime = value.get("source_runtime_identity")
    inputs = value.get("source_inputs")
    if (
        value.get("status") != "verified"
        or value.get("artifact_schema") != _ENDURANCE_FLOOR_ARTIFACT_SCHEMA
        or fingerprint is None
        or not _private_pin_shape(value.get("artifact_pin"), size_key="size_bytes")
        or verification
        != {
            "status": "verified",
            "live_rederived": True,
            "verified_safe_runs": 24,
        }
        or not isinstance(profiles, dict)
        or set(profiles) != {"640", "960"}
        or not isinstance(runtime, dict)
        or set(runtime)
        != {
            "image",
            "image_id",
            "gpu_index",
            "gpu_identity",
            "power_profile",
            "max_temperature_c",
            "power_safety_policy",
        }
        or not isinstance(inputs, dict)
        or set(inputs) != {"summary", "scene_manifest"}
        or not all(
            _private_pin_shape(inputs.get(key), size_key="bytes")
            for key in ("summary", "scene_manifest")
        )
    ):
        return {**unavailable, "status": "invalid"}

    projected_profiles: dict[str, dict[str, float]] = {}
    for profile in ("640", "960"):
        item = profiles.get(profile)
        if not isinstance(item, dict) or set(item) != {
            "aggregate_fps_floor",
            "per_stream_fps_floor",
        }:
            return {**unavailable, "status": "invalid"}
        aggregate = _positive_fps(item.get("aggregate_fps_floor"))
        per_stream = _positive_fps(item.get("per_stream_fps_floor"))
        if aggregate is None or per_stream is None or per_stream > aggregate:
            return {**unavailable, "status": "invalid"}
        projected_profiles[profile] = {
            "aggregate_fps_floor": aggregate,
            "per_stream_fps_floor": per_stream,
        }

    gpu = runtime.get("gpu_identity")
    gpu_index = runtime.get("gpu_index")
    temperature = _number(runtime.get("max_temperature_c"))
    power_safety = runtime.get("power_safety_policy")
    if (
        not isinstance(runtime.get("image"), str)
        or not 1 <= len(runtime["image"]) <= 512
        or not isinstance(runtime.get("image_id"), str)
        or not 1 <= len(runtime["image_id"]) <= 512
        or isinstance(gpu_index, bool)
        or not isinstance(gpu_index, int)
        or not 0 <= gpu_index <= 63
        or not isinstance(gpu, dict)
        or set(gpu)
        != {
            "index",
            "uuid",
            "name",
            "driver_version",
            "memory.total",
            "pci.bus_id",
        }
        or str(gpu.get("index")) != str(gpu_index)
        or not isinstance(gpu.get("uuid"), str)
        or not 1 <= len(gpu["uuid"]) <= 160
        or not isinstance(gpu.get("memory.total"), str)
        or not 1 <= len(gpu["memory.total"]) <= 80
        or not isinstance(gpu.get("pci.bus_id"), str)
        or not 1 <= len(gpu["pci.bus_id"]) <= 80
        or runtime.get("power_profile")
        != {"available": True, "value": "performance"}
        or temperature is None
        or not 0 < temperature <= 150
        or not _endurance_scene_power_safety_is_exact(power_safety)
    ):
        return {**unavailable, "status": "invalid"}
    gpu_name = _safe_gpu_name(gpu.get("name"))
    driver_version = _safe_driver_version(gpu.get("driver_version"))
    if gpu_name is None or driver_version is None:
        return {**unavailable, "status": "invalid"}
    return {
        "status": "verified",
        "acceptance_safe": True,
        "artifact_fingerprint": fingerprint,
        "artifact_fingerprint_short": f"{fingerprint[:12]}…{fingerprint[-8:]}",
        "verified_safe_runs": 24,
        "live_rederived": True,
        "profiles": projected_profiles,
        "source_runtime": {
            "gpu_name": gpu_name,
            "driver_version": driver_version,
        },
    }


def _heartbeat_floor_evaluation(
    value: dict[str, Any], floor: dict[str, Any]
) -> dict[str, Any]:
    default_status = "pending" if value.get("state") == "running" else "not_reported"
    unavailable = {"status": default_status, "acceptance_safe": False}
    if floor.get("status") != "verified":
        return {"status": "blocked_by_floor_binding", "acceptance_safe": False}
    raw = value.get("throughput_floor_evaluation")
    if raw is None:
        return unavailable
    expected_fields = {
        "schema_version",
        "status",
        "artifact_fingerprint",
        "profile",
        "aggregate_fps_floor",
        "per_stream_fps_floor",
        "observed_aggregate_fps_p05",
    }
    status = _enum(
        raw.get("status") if isinstance(raw, dict) else None,
        {
            "passed",
            "failed_below_floor",
            "failed_missing_measurement",
            "interrupted_not_evaluated",
        },
    )
    profile = raw.get("profile") if isinstance(raw, dict) else None
    profile_key = str(profile)
    expected_profile = floor.get("profiles", {}).get(profile_key)
    if (
        not isinstance(raw, dict)
        or set(raw) != expected_fields
        or raw.get("schema_version") != _ENDURANCE_FLOOR_EVIDENCE_SCHEMA
        or status is None
        or profile not in (640, 960)
        or raw.get("artifact_fingerprint") != floor.get("artifact_fingerprint")
        or not isinstance(expected_profile, dict)
        or _positive_fps(raw.get("aggregate_fps_floor"))
        != expected_profile.get("aggregate_fps_floor")
        or _positive_fps(raw.get("per_stream_fps_floor"))
        != expected_profile.get("per_stream_fps_floor")
    ):
        return {"status": "invalid", "acceptance_safe": False}
    live_verification = value.get("throughput_floor_live_verification")
    if live_verification != {
        "status": "verified",
        "artifact_fingerprint": floor.get("artifact_fingerprint"),
    }:
        return {"status": "integrity_failed", "acceptance_safe": False}
    observed = raw.get("observed_aggregate_fps_p05")
    observed_fps = _positive_fps(observed)
    aggregate_floor = expected_profile["aggregate_fps_floor"]
    if status == "passed" and (
        observed_fps is None or observed_fps < aggregate_floor
    ):
        return {"status": "invalid", "acceptance_safe": False}
    if status == "failed_below_floor" and (
        observed_fps is None or observed_fps >= aggregate_floor
    ):
        return {"status": "invalid", "acceptance_safe": False}
    if status in {"failed_missing_measurement", "interrupted_not_evaluated"}:
        if observed is not None:
            return {"status": "invalid", "acceptance_safe": False}
        observed_fps = None
    return {
        "status": status,
        "acceptance_safe": status == "passed",
        "profile": profile,
        "aggregate_fps_floor": aggregate_floor,
        "per_stream_fps_floor": expected_profile["per_stream_fps_floor"],
        "observed_aggregate_fps_p05": observed_fps,
    }


def _heartbeat_projection(
    value: dict[str, Any], throughput_floor: dict[str, Any]
) -> dict[str, Any]:
    raw_updated_at = value.get("updated_at_utc")
    updated_at = _timestamp(raw_updated_at)
    age: float | None = None
    invalid_timestamp = raw_updated_at is not None and updated_at is None
    if updated_at is not None:
        try:
            updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if updated.tzinfo is None:
                raise ValueError("timezone required")
            age = round(
                max(0.0, (datetime.now(timezone.utc) - updated).total_seconds()), 3
            )
        except ValueError:
            invalid_timestamp = True
    stale_seconds = 90.0
    try:
        stale_seconds = max(
            1.0, float(os.getenv("DEEPSAFE_ENDURANCE_STALE_SECONDS", "90"))
        )
    except ValueError:
        pass
    state = _enum(
        value.get("state"), {"running", "healthy", "failed", "interrupted"}
    )
    profile = value.get("profile")
    return {
        "state": state,
        "updated_at_utc": updated_at,
        "age_seconds": age,
        "stale": bool(state == "running" and age is not None and age > stale_seconds),
        "timestamp_invalid": invalid_timestamp,
        "profile": profile if profile in (640, 960) else None,
        "throughput_floor_evaluation": _heartbeat_floor_evaluation(
            value, throughput_floor
        ),
    }


def _endurance(reader: ArtifactReader) -> dict[str, Any]:
    artifact = reader.read("endurance_status")
    live_artifact = reader.read("endurance_live")
    value = artifact.value or {}
    status_schema_valid = (
        artifact.available
        and value.get("schema_version") == "deepsafe.endurance-status/v1"
    )
    dry_run = _boolean(value.get("dry_run"))
    target = _integer(value.get("target_validated_seconds")) or 0
    validated = _integer(value.get("validated_seconds")) or 0
    segments = value.get("segments") if isinstance(value.get("segments"), dict) else {}
    profiles = (
        value.get("profiles_validated_seconds")
        if isinstance(value.get("profiles_validated_seconds"), dict)
        else {}
    )
    throughput_floor = _throughput_floor_projection(value.get("throughput_floor"))
    if artifact.available and (
        not status_schema_valid
        or dry_run is None
        or (dry_run is True and throughput_floor.get("status") == "verified")
        or (
            dry_run is False
            and throughput_floor.get("status") == "synthetic_pending"
        )
    ):
        throughput_floor = {
            **_throughput_floor_projection(None),
            "status": "invalid",
        }
    state = _enum(
        value.get("state"),
        {"planned", "running", "paused", "paused_health_gate", "complete"},
    )
    if not artifact.available:
        state = "not_started" if artifact.state == "missing" else "artifact_error"
    elif not status_schema_valid:
        state = "artifact_error"
    elif state is None:
        state = "artifact_error"
    elif throughput_floor.get("status") != "verified":
        state = "attention"
    return {
        "label": "7 günlük dayanıklılık kampanyası",
        "available": artifact.available,
        "state": state or "unknown",
        "updated_at_utc": _timestamp(value.get("updated_at_utc")),
        "dry_run": dry_run,
        "progress": _progress(validated, target),
        "validated_seconds_by_profile": {
            profile: seconds
            for profile in ("640", "960")
            if (seconds := _integer(profiles.get(profile))) is not None
        },
        "segments": {
            "total": _integer(segments.get("total")),
            "status_counts": _counts(segments.get("status_counts")),
        },
        "health": {
            "unexpected_restarts": _integer(value.get("unexpected_restarts")),
            "orphan_recoveries": _integer(value.get("orphan_recoveries")),
            "health_gate_count": (
                len(value.get("campaign_health_gates", [])[:1_000])
                if isinstance(value.get("campaign_health_gates"), list)
                else None
            ),
        },
        "throughput_floor": throughput_floor,
        "heartbeat": (
            _heartbeat_projection(live_artifact.value or {}, throughput_floor)
            if live_artifact.available
            and (live_artifact.value or {}).get("schema_version")
            == "deepsafe.endurance-live/v1"
            else {
                "state": "unavailable",
                "artifact_state": (
                    live_artifact.state
                    if not live_artifact.available
                    else "invalid_schema_contract"
                ),
                "throughput_floor_evaluation": {
                    "status": "not_reported",
                    "acceptance_safe": False,
                },
            }
        ),
        "evidence": [
            {key: value for key, value in item.items() if key != "path"}
            for item in _evidence(reader, "endurance_status", "endurance_live")
        ],
    }


def _unavailable_state(*artifacts: ArtifactRead) -> str | None:
    if any(artifact.available for artifact in artifacts):
        return None
    if all(artifact.state == "missing" for artifact in artifacts):
        return "not_started"
    return "artifact_error"


def _gpu_verification_contract(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") != "deepsafe.gpu-reentry-verification/v1":
        return None
    status = _enum(value.get("status"), {"blocked", "ready_for_operator_review"})
    all_present = _boolean(value.get("all_required_evidence_present"))
    authorized = _boolean(value.get("sustained_load_authorized"))
    raw_failed = value.get("failed_gate_ids")
    gates = value.get("gates")
    if (
        status is None
        or all_present is None
        or authorized is not False
        or not isinstance(raw_failed, list)
        or not isinstance(gates, list)
        or not gates
        or len(gates) > 128
    ):
        return None
    failed = _identifiers(raw_failed, maximum=128)
    if len(failed) != len(raw_failed) or len(failed) != len(set(failed)):
        return None
    required_ids: list[str] = []
    computed_failed: list[str] = []
    passed = 0
    for gate in gates:
        if not isinstance(gate, dict):
            return None
        gate_id = _identifier(gate.get("id"))
        required = _boolean(gate.get("required"))
        gate_passed = _boolean(gate.get("passed"))
        if gate_id is None or required is None or gate_passed is None:
            return None
        if gate_id in required_ids:
            return None
        if required:
            required_ids.append(gate_id)
            if gate_passed:
                passed += 1
            else:
                computed_failed.append(gate_id)
    expected_status = "blocked" if computed_failed else "ready_for_operator_review"
    if (
        failed != computed_failed
        or all_present is not (not computed_failed)
        or status != expected_status
    ):
        return None
    return {
        "status": status,
        "all_required_evidence_present": all_present,
        "sustained_load_authorized": authorized,
        "failed_gate_ids": failed,
        "passed": passed,
        "total": len(required_ids),
    }


def _gpu_reentry_r2_schema_replay(
    value: dict[str, Any] | None,
    schema: dict[str, Any] | None,
    *,
    expected_id: str,
) -> bool:
    if value is None or schema is None:
        return False
    if (
        schema.get("$schema")
        != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("$id") != expected_id
    ):
        return False
    try:
        _validate_schema_node(value, schema, schema)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def _gpu_reentry_r2_linked_pins_verified(
    reader: ArtifactReader,
    pins: list[Any],
) -> bool:
    unique: dict[str, dict[str, Any]] = {}
    for raw in pins:
        pin = _person_pin_core(raw)
        if pin is None or not isinstance(pin.get("path"), str):
            return False
        # The successful run deliberately emits one empty stderr log.  Its
        # exact descriptor remains sealed inside the exact-pinned receipt;
        # the bounded workspace reader intentionally accepts non-empty files
        # only, so no semantic claim depends on re-opening that empty log.
        if pin.get("bytes") == 0:
            if pin.get("sha256") != hashlib.sha256(b"").hexdigest():
                return False
            continue
        unique[pin["path"]] = pin
    for path, pin in unique.items():
        result = _read_workspace_pin(
            reader,
            pin,
            expected_path=path,
            maximum_bytes=MAX_PINNED_FILE_BYTES,
            collect=False,
        )
        if not result.available:
            return False
    return True


def _gpu_reentry_r2_semantics_valid(
    receipt: dict[str, Any], plan: dict[str, Any]
) -> bool:
    expected_claims = {
        "physical_verification_performed": False,
        "deepstream_smoke_executed": False,
        "endurance_executed": False,
        "training_executed": False,
        "quality_claimed": False,
        "sustained_load_authorized": False,
    }
    expected_gates = {
        "fresh_live_v1_reentry": (True, "live_reentry"),
        "lease_acquire_release_complete": (True, "live_reentry"),
        "official_image_reinspected": (True, "live_reentry"),
        "minimal_cuda_tensor_sync": (True, "live_reentry"),
        "deepstream9_engine_smoke": (False, "deepstream_qualification"),
        "seven_day_endurance": (False, "project_completion"),
    }
    try:
        gates = receipt["gates"]
        gate_projection = {
            gate["id"]: (gate["passed"], gate["required_for"])
            for gate in gates
        }
        prior = plan["prior_failed_live_smoke"]
        prior_claims = prior["claims"]
        docker = prior["docker"]
        smoke = receipt["minimal_cuda_smoke"]
        smoke_output = smoke["output"]
        gpu = receipt["gpu"]
        plan_pin = _person_pin_core(receipt["plan"]["artifact"])
        source_pins = plan["source_pins"]
        return bool(
            receipt.get("schema_version")
            == "deepsafe.gpu-reentry-evidence/v2"
            and receipt.get("status")
            == "live_reentry_smoke_passed_deepstream_and_endurance_pending"
            and receipt.get("run_id") == "gpu-reentry-r2-live-002"
            and receipt.get("receipt_id")
            == "gpu-reentry-r2-gpu-reentry-r2-live-002"
            and _canonical_fingerprint_matches(receipt)
            and receipt.get("claims") == expected_claims
            and len(gates) == len(expected_gates)
            and gate_projection == expected_gates
            and plan_pin == GPU_REENTRY_R2_ADMIN_PINS["plan"]
            and receipt["plan"].get("fingerprint_sha256")
            == plan.get("fingerprint_sha256")
            and plan.get("schema_version")
            == "deepsafe.gpu-reentry-refresh-plan/v2"
            and plan.get("plan_id")
            == "gpu-reentry-lease-bound-r2-20260718"
            and plan.get("status") == "planned_live_gpu_smoke_required"
            and _canonical_fingerprint_matches(plan)
            and _person_pin_core(source_pins.get("validator"))
            == GPU_REENTRY_R2_ADMIN_PINS["validator"]
            and _person_pin_core(source_pins.get("plan_schema"))
            == GPU_REENTRY_R2_ADMIN_PINS["plan_schema"]
            and _person_pin_core(source_pins.get("evidence_schema"))
            == GPU_REENTRY_R2_ADMIN_PINS["receipt_schema"]
            and gpu
            == {
                "index": 0,
                "uuid": "GPU-8cbaba1c-2629-a732-f528-66f459089ef6",
                "name": "NVIDIA RTX A5000 Laptop GPU",
                "driver_version": "590.48.01",
                "memory_total_mib": 16384,
            }
            and receipt.get("legacy_v1_reentry")
            == {
                "evidence": {
                    "path": (
                        "validation/results/gpu-reentry/r2-executions/"
                        "gpu-reentry-r2-live-002/reentry-v1.json"
                    ),
                    "bytes": 21526,
                    "sha256": (
                        "c2615cfa18122d0e04116bc3522c946be3ead0a3b2da4fe63308d5b89e36f77a"
                    ),
                },
                "status": "ready_for_operator_review",
                "failed_gate_ids": [],
                "sustained_load_authorized": False,
            }
            and smoke.get("status") == "passed"
            and smoke_output.get("cuda_available") is True
            and smoke_output.get("device_count") == 1
            and smoke_output.get("device_name") == gpu["name"]
            and smoke_output.get("tensor_device") == "cuda:0"
            and smoke_output.get("tensor_finite") is True
            and smoke_output.get("compute_capability") == [8, 6]
            and receipt.get("gpu_lease", {}).get("lifecycle_complete") is True
            and receipt.get("gpu_lease", {}).get("owner_kind")
            == "legacy_validation"
            and receipt.get("training_execution_image", {}).get(
                "image_reference"
            )
            == "deepsafe-rtdetrv4-person:r5"
            and receipt.get("training_execution_image", {}).get(
                "resolved_image_id"
            )
            == (
                "sha256:74fa3c2ede1fb2ccfce93228a5d7d814"
                "c556ce61078a5f5aa17c69618aa41b9e"
            )
            and prior.get("run_id") == "gpu-reentry-r2-live-001"
            and prior.get("status")
            == "failed_docker_entrypoint_interception"
            and prior.get("preserved_no_overwrite") is True
            and prior.get("gpu_result_valid") is False
            and prior.get("v1_reentry_status")
            == "ready_for_operator_review"
            and prior_claims
            == {
                "deepstream_smoke_executed": False,
                "endurance_executed": False,
                "quality_claimed": False,
                "training_executed": False,
            }
            and docker
            == {
                "exit_status": 2,
                "previous_entrypoint_override_present": False,
                "root_cause": (
                    "the derived training image ENTRYPOINT "
                    "container_runner.py intercepted the intended python -c "
                    "child command"
                ),
                "stderr_marker": "usage: container_runner.py",
                "corrective_control": (
                    "explicit --entrypoint=python before the image ID"
                ),
            }
            and prior.get("lease", {}).get("lifecycle_complete") is True
        )
    except (KeyError, TypeError, ValueError):
        return False


def _gpu_reentry_r2(reader: ArtifactReader) -> dict[str, Any] | None:
    integrity = {
        "current_receipt_exact_pin_verified": False,
        "current_receipt_schema_exact_pin_verified": False,
        "current_receipt_schema_replay_verified": False,
        "current_receipt_fingerprint_replayed": False,
        "current_plan_exact_pin_verified": False,
        "current_plan_schema_exact_pin_verified": False,
        "current_plan_schema_replay_verified": False,
        "current_plan_fingerprint_replayed": False,
        "validator_exact_pin_verified": False,
        "current_linked_artifact_pins_verified": False,
        "failed_live_001_artifact_pins_verified": False,
        "current_and_historical_semantics_verified": False,
    }
    receipt_pin = GPU_REENTRY_R2_ADMIN_PINS["receipt"]
    receipt_read, receipt = _workspace_pin_json(
        reader,
        receipt_pin,
        expected_path=receipt_pin["path"],
        maximum_bytes=GPU_REENTRY_R2_MAX_JSON_BYTES,
    )
    plan_pin = GPU_REENTRY_R2_ADMIN_PINS["plan"]
    plan_read, plan = _workspace_pin_json(
        reader,
        plan_pin,
        expected_path=plan_pin["path"],
        maximum_bytes=GPU_REENTRY_R2_MAX_JSON_BYTES,
    )
    if receipt_read.state == "missing" and plan_read.state == "missing":
        return None
    integrity["current_receipt_exact_pin_verified"] = receipt_read.available
    integrity["current_plan_exact_pin_verified"] = plan_read.available

    receipt_schema_pin = GPU_REENTRY_R2_ADMIN_PINS["receipt_schema"]
    receipt_schema_read, receipt_schema = _workspace_pin_json(
        reader,
        receipt_schema_pin,
        expected_path=receipt_schema_pin["path"],
        maximum_bytes=GPU_REENTRY_R2_MAX_JSON_BYTES,
    )
    integrity["current_receipt_schema_exact_pin_verified"] = (
        receipt_schema_read.available
    )
    plan_schema_pin = GPU_REENTRY_R2_ADMIN_PINS["plan_schema"]
    plan_schema_read, plan_schema = _workspace_pin_json(
        reader,
        plan_schema_pin,
        expected_path=plan_schema_pin["path"],
        maximum_bytes=GPU_REENTRY_R2_MAX_JSON_BYTES,
    )
    integrity["current_plan_schema_exact_pin_verified"] = (
        plan_schema_read.available
    )
    validator_pin = GPU_REENTRY_R2_ADMIN_PINS["validator"]
    validator_read = _read_workspace_pin(
        reader,
        validator_pin,
        expected_path=validator_pin["path"],
        maximum_bytes=GPU_REENTRY_R2_MAX_JSON_BYTES,
        collect=False,
    )
    integrity["validator_exact_pin_verified"] = validator_read.available

    reads = {
        "current_receipt": receipt_read,
        "current_plan": plan_read,
        "current_receipt_schema": receipt_schema_read,
        "current_plan_schema": plan_schema_read,
        "validator": validator_read,
    }
    if any(not result.available for result in reads.values()):
        key, result = next(
            (key, result)
            for key, result in reads.items()
            if not result.available
        )
        return {
            "label": "GPU yeniden giriş R2",
            "available": False,
            "state": "artifact_error",
            "reason": f"{key}_{result.state}",
            "progress": _progress(0, 6),
            "sustained_load_authorized": False,
            "pending_gate_ids": [
                "deepstream9_engine_smoke",
                "seven_day_endurance",
            ],
            "integrity": integrity,
            "current_run": {},
            "historical_failed_run": {},
            "caveats": [
                "R2 current veya tarihsel exact-pin zinciri doğrulanamadı; GPU başarı iddiası kapalıdır."
            ],
            "evidence": [],
        }
    if (
        receipt is None
        or plan is None
        or receipt_schema is None
        or plan_schema is None
    ):
        return {
            "label": "GPU yeniden giriş R2",
            "available": False,
            "state": "artifact_error",
            "reason": "r2_json_contract_invalid",
            "progress": _progress(0, 6),
            "sustained_load_authorized": False,
            "pending_gate_ids": [],
            "integrity": integrity,
            "current_run": {},
            "historical_failed_run": {},
            "caveats": ["R2 JSON sözleşmesi doğrulanamadı."],
            "evidence": [],
        }

    integrity["current_receipt_schema_replay_verified"] = (
        _gpu_reentry_r2_schema_replay(
            receipt,
            receipt_schema,
            expected_id=(
                "https://deepsafe.local/schemas/"
                "gpu-reentry-evidence-v2.schema.json"
            ),
        )
    )
    integrity["current_plan_schema_replay_verified"] = (
        _gpu_reentry_r2_schema_replay(
            plan,
            plan_schema,
            expected_id=(
                "https://deepsafe.local/schemas/"
                "gpu-reentry-refresh-plan-v2.schema.json"
            ),
        )
    )
    integrity["current_receipt_fingerprint_replayed"] = (
        _canonical_fingerprint_matches(receipt)
    )
    integrity["current_plan_fingerprint_replayed"] = (
        _canonical_fingerprint_matches(plan)
    )

    current_pins: list[Any] = list(receipt.get("artifacts", []))
    lease = receipt.get("gpu_lease", {})
    current_pins.extend(
        [
            lease.get("acquire_receipt"),
            lease.get("release_receipt"),
            *lease.get("renew_receipts", []),
            lease.get("contract_projection", {}).get("artifact"),
            receipt.get("legacy_v1_reentry", {}).get("evidence"),
            receipt.get("minimal_cuda_smoke", {}).get("raw_output"),
            receipt.get("training_execution_image", {}).get("build_receipt"),
        ]
    )
    integrity["current_linked_artifact_pins_verified"] = (
        _gpu_reentry_r2_linked_pins_verified(reader, current_pins)
    )
    prior = plan.get("prior_failed_live_smoke", {})
    historical_pins: list[Any] = [prior.get("plan", {}).get("artifact")]
    historical_pins.extend(prior.get("artifacts", {}).values())
    historical_pins.extend(
        [
            prior.get("lease", {}).get("acquire_receipt"),
            prior.get("lease", {}).get("release_receipt"),
        ]
    )
    integrity["failed_live_001_artifact_pins_verified"] = (
        _gpu_reentry_r2_linked_pins_verified(reader, historical_pins)
    )
    integrity["current_and_historical_semantics_verified"] = (
        _gpu_reentry_r2_semantics_valid(receipt, plan)
    )
    if not all(integrity.values()):
        return {
            "label": "GPU yeniden giriş R2",
            "available": False,
            "state": "artifact_error",
            "reason": "r2_exact_pin_or_replay_contract_invalid",
            "progress": _progress(0, 6),
            "sustained_load_authorized": False,
            "pending_gate_ids": [],
            "integrity": integrity,
            "current_run": {},
            "historical_failed_run": {},
            "caveats": [
                "R2 current başarı veya live-001 tarihçesi bağımsız replay denetimini geçmedi."
            ],
            "evidence": [],
        }

    gpu = receipt["gpu"]
    smoke_output = receipt["minimal_cuda_smoke"]["output"]
    return {
        "label": "GPU yeniden giriş R2",
        "available": True,
        "state": receipt["status"],
        "reason": "deepstream9_engine_smoke_and_seven_day_endurance_pending",
        "updated_at_utc": _timestamp(receipt.get("created_at_utc")),
        "progress": _progress(4, 6),
        "status_counts": {"passed": 4, "pending": 2},
        "all_required_evidence_present": True,
        "sustained_load_authorized": False,
        "failed_gate_ids": [],
        "pending_gate_ids": [
            "deepstream9_engine_smoke",
            "seven_day_endurance",
        ],
        "scope": {"gpu_index": gpu["index"], "idle_samples": 3},
        "collection_policy": {
            "read_only": True,
            "benchmark_started": False,
            "gpu_stress_performed": False,
            "settings_changed": False,
            "sudo_executed": False,
        },
        "current_run": {
            "run_id": receipt["run_id"],
            "status": "minimal_cuda_smoke_passed",
            "gpu_name": gpu["name"],
            "driver_version": gpu["driver_version"],
            "memory_total_mib": gpu["memory_total_mib"],
            "cuda": smoke_output["cuda"],
            "torch": smoke_output["torch"],
            "compute_capability": smoke_output["compute_capability"],
            "tensor_sync_passed": True,
            "lease_lifecycle_complete": True,
            "deepstream_smoke_executed": False,
            "endurance_executed": False,
        },
        "historical_failed_run": {
            "run_id": "gpu-reentry-r2-live-001",
            "status": "failed_docker_entrypoint_interception",
            "exit_status": 2,
            "gpu_result_valid": False,
            "preserved_no_overwrite": True,
            "root_cause": "derived_image_entrypoint_intercepted_child_command",
            "corrective_control": "explicit_python_entrypoint",
            "lease_lifecycle_complete": True,
        },
        "integrity": integrity,
        "caveats": [
            "live-002 exact-pinli lease ve küçük CUDA tensor senkron smoke'unu geçti; bu DeepStream motor veya model kalite kanıtı değildir.",
            "DeepStream 9 motor smoke'u ve 7 günlük dayanıklılık bu makbuzda çalıştırılmadı ve açık kalır.",
            "live-001 Docker ENTRYPOINT yakalamasıyla exit 2 oldu; GPU başarı kanıtı değildir ve değiştirilemez tarihçe olarak korunur.",
        ],
        "evidence": [],
    }


def _gpu_reentry_v1(reader: ArtifactReader) -> dict[str, Any]:
    evidence_artifact = reader.read("gpu_reentry_evidence")
    verification_artifact = reader.read("gpu_reentry_verification")
    evidence = evidence_artifact.value or {}
    verification = verification_artifact.value or {}
    external_contract = _gpu_verification_contract(verification)
    embedded_contract = _gpu_verification_contract(evidence.get("verification"))
    contracts_match = bool(
        external_contract is not None
        and embedded_contract is not None
        and external_contract == embedded_contract
        and evidence.get("status") == external_contract["status"]
    )
    contract = external_contract if contracts_match else None
    passed = contract["passed"] if contract is not None else 0
    total = contract["total"] if contract is not None else 0
    unavailable = _unavailable_state(evidence_artifact, verification_artifact)
    if any(
        artifact.state not in {"ok", "missing"}
        for artifact in (evidence_artifact, verification_artifact)
    ):
        state = "artifact_error"
    elif unavailable is not None:
        state = unavailable
    elif verification_artifact.available and not evidence_artifact.available:
        state = "artifact_error"
    elif verification_artifact.available:
        state = contract["status"] if contract is not None else "artifact_error"
    else:
        state = (
            "collected_not_verified"
            if embedded_contract is not None
            and evidence.get("status") == embedded_contract["status"]
            else "artifact_error"
        )
    collection = (
        evidence.get("collection_policy")
        if isinstance(evidence.get("collection_policy"), dict)
        else {}
    )
    idle = (
        evidence.get("idle_gpu_telemetry")
        if isinstance(evidence.get("idle_gpu_telemetry"), dict)
        else {}
    )
    samples = idle.get("samples") if isinstance(idle.get("samples"), list) else []
    return {
        "label": "GPU yeniden giriş güvenlik kapısı",
        "available": evidence_artifact.available or verification_artifact.available,
        "state": state,
        "updated_at_utc": _timestamp(
            verification.get("verified_at_utc") or evidence.get("collected_at_utc")
        ),
        "progress": _progress(passed, total),
        "status_counts": (
            {"passed": passed, "failed": max(0, total - passed)}
            if verification_artifact.available
            else {}
        ),
        "all_required_evidence_present": (
            contract["all_required_evidence_present"] if contract is not None else None
        ),
        "sustained_load_authorized": (
            contract["sustained_load_authorized"] if contract is not None else None
        ),
        "failed_gate_ids": contract["failed_gate_ids"] if contract is not None else [],
        "scope": {
            "gpu_index": _integer(evidence.get("gpu_index"), maximum=128),
            "idle_samples": min(len(samples), 1_000),
        },
        "collection_policy": {
            key: _boolean(collection.get(key))
            for key in (
                "read_only",
                "benchmark_started",
                "gpu_stress_performed",
                "settings_changed",
                "sudo_executed",
            )
        },
        "evidence": _evidence(
            reader, "gpu_reentry_evidence", "gpu_reentry_verification"
        ),
    }


def _ds9_runtime_receipt_semantics_valid(
    receipt: dict[str, Any], parser_receipt: dict[str, Any]
) -> bool:
    """Replay the narrow DS9 qualification claim without exposing lineage.

    The JSON schema constrains shape.  This independent replay binds the exact
    image, controller, runtime-control manifest, GPU smoke evidence and parser
    build receipt that were approved for this one qualification window.
    """

    expected_image_id = (
        "sha256:ced1b59150dbfc040e3ff6afe8e749b2ad5f2c550934242bd7f43ee5bd898c46"
    )
    expected_checks = {
        "cpu_cuda_parser_parity_640": "pass",
        "cpu_cuda_parser_parity_960": "pass",
        "cuda_parser_kernel_launch_sm86": "pass",
        "deepstream_640_engine_deserialize_no_fallback": "pass",
        "deepstream_960_engine_deserialize_no_fallback": "pass",
    }
    try:
        image = receipt["image"]
        runtime_controls = receipt["runtime_controls"]
        artifacts = runtime_controls["artifacts"]
        static_probe = receipt["static_probe"]
        static_evidence = static_probe["evidence"]
        facts = static_evidence["facts"]
        gpu_smoke = receipt["gpu_smoke"]
        parser_image = parser_receipt["image"]
        return bool(
            receipt.get("schema_version")
            == "deepsafe.ds9-runtime-compatibility-receipt/v1"
            and receipt.get("status") == "production_ready"
            and receipt.get("production_ready") is True
            and receipt.get("created_at_utc") == "2026-07-18T01:21:47Z"
            and receipt.get("expires_at_utc") == "2026-07-19T01:21:47Z"
            and receipt.get("requested_image") == "deepsafe-deepstream:9.0"
            and image.get("requested_image") == receipt["requested_image"]
            and image.get("resolved_image_id") == expected_image_id
            and image.get("architecture") == "amd64"
            and image.get("os") == "linux"
            and image.get("repo_digests")
            == [
                "deepsafe-deepstream@"
                "sha256:ced1b59150dbfc040e3ff6afe8e749b2ad5f2c550934242bd7f43ee5bd898c46"
            ]
            and runtime_controls.get("pin")
            == DS9_RUNTIME_QUALIFICATION_ADMIN_PINS["runtime_controls"]
            and artifacts.get("ds9_runtime_compatibility")
            == DS9_RUNTIME_QUALIFICATION_ADMIN_PINS["validator"]
            and artifacts.get("ds9_runtime_compatibility_schema")
            == DS9_RUNTIME_QUALIFICATION_ADMIN_PINS["schema"]
            and len(artifacts) == 18
            and static_probe.get("status") == "pass"
            and static_evidence.get("schema_version")
            == "deepsafe.ds9-static-container-probe/v1"
            and static_evidence.get("status") == "pass"
            and facts.get("deepstream_app_version") == "9.0.0"
            and facts.get("deepstream_sdk_version") == "9.0.0"
            and facts.get("cuda_version") == "13.1"
            and facts.get("tensorrt_version") == "10.14.1.48"
            and facts.get("abi_compile_passed") is True
            and facts.get("sm86_cubin_present") is True
            and facts.get("compute86_only_ptx_set") is True
            and facts.get("forward_compatible_ptx_present") is True
            and facts.get("dlsym")
            == {
                "NvDsInferParseYoloCuda": True,
                "NvDsInferYoloCudaEngineGet": True,
            }
            and facts.get("ldd_missing") == []
            and facts.get("file_errors") == []
            and gpu_smoke.get("status") == "pass"
            and gpu_smoke.get("checks") == expected_checks
            and gpu_smoke.get("evidence")
            == DS9_RUNTIME_QUALIFICATION_ADMIN_PINS["gpu_smoke_evidence"]
            and parser_receipt.get("schema_version")
            == "deepsafe.ds9-parser-production-build-receipt/v1"
            and parser_receipt.get("status") == "candidate_image_built"
            and parser_receipt.get("pass_number") == 2
            and parser_receipt.get("two_pass_complete") is True
            and parser_receipt.get("image_tag") == "deepsafe-deepstream:9.0"
            and parser_receipt.get("resolved_image_id") == expected_image_id
            and parser_receipt.get("environment")
            == {"DOCKER_BUILDKIT": "1", "gpu_requested": False}
            and parser_receipt.get("inputs", {}).get("runtime_controller")
            == DS9_RUNTIME_QUALIFICATION_ADMIN_PINS["validator"]
            and parser_receipt.get("inputs", {}).get(
                "runtime_control_manifest"
            )
            == DS9_RUNTIME_QUALIFICATION_ADMIN_PINS["runtime_controls"]
            and parser_image.get("id") == expected_image_id
            and parser_image.get("architecture") == "amd64"
            and parser_image.get("os") == "linux"
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ds9_runtime_qualification_unavailable(
    reason: str, *, integrity: dict[str, bool]
) -> dict[str, Any]:
    return {
        "label": "DeepStream 9 çalışma zamanı yeterliliği",
        "available": False,
        "current": False,
        "state": "artifact_error",
        "reason": reason,
        "runtime_qualification_passed": False,
        "does_not_imply_product_readiness": True,
        "product_production_ready": False,
        "training_complete": False,
        "final_test_complete": False,
        "checks": {"passed": 0, "total": 5},
        "integrity": integrity,
        "caveats": [
            "DS9 yeterlilik zinciri canlı olarak doğrulanamadı; üst GPU kapısı kapalıdır."
        ],
        "evidence": [],
    }


def _ds9_runtime_qualification(
    reader: ArtifactReader, *, now: datetime | None = None
) -> dict[str, Any]:
    """Read-only currentness projection for the exact DS9 qualification."""

    integrity = {
        "receipt_exact_pin_verified": False,
        "schema_exact_pin_verified": False,
        "validator_exact_pin_verified": False,
        "runtime_controls_exact_pin_verified": False,
        "gpu_smoke_evidence_exact_pin_verified": False,
        "parser_build_receipt_exact_pin_verified": False,
        "receipt_schema_replay_verified": False,
        "receipt_semantics_replayed": False,
    }
    receipt_pin = DS9_RUNTIME_QUALIFICATION_ADMIN_PINS["receipt"]
    receipt_read, receipt = _workspace_pin_json(
        reader,
        receipt_pin,
        expected_path=receipt_pin["path"],
        maximum_bytes=DS9_RUNTIME_QUALIFICATION_MAX_BYTES,
    )
    integrity["receipt_exact_pin_verified"] = receipt_read.available

    schema_pin = DS9_RUNTIME_QUALIFICATION_ADMIN_PINS["schema"]
    schema_read, schema = _workspace_pin_json(
        reader,
        schema_pin,
        expected_path=schema_pin["path"],
        maximum_bytes=DS9_RUNTIME_QUALIFICATION_MAX_BYTES,
    )
    integrity["schema_exact_pin_verified"] = schema_read.available

    parser_pin = DS9_RUNTIME_QUALIFICATION_ADMIN_PINS["parser_build_receipt"]
    parser_read, parser_receipt = _workspace_pin_json(
        reader,
        parser_pin,
        expected_path=parser_pin["path"],
        maximum_bytes=DS9_RUNTIME_QUALIFICATION_MAX_BYTES,
    )
    integrity["parser_build_receipt_exact_pin_verified"] = (
        parser_read.available
    )

    stream_keys = (
        "validator",
        "runtime_controls",
        "gpu_smoke_evidence",
    )
    stream_reads: dict[str, WorkspacePinRead] = {}
    for key in stream_keys:
        pin = DS9_RUNTIME_QUALIFICATION_ADMIN_PINS[key]
        result = _read_workspace_pin(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=DS9_RUNTIME_QUALIFICATION_MAX_BYTES,
            collect=False,
        )
        stream_reads[key] = result
        integrity[f"{key}_exact_pin_verified"] = result.available

    reads = {
        "receipt": receipt_read,
        "schema": schema_read,
        "parser_build_receipt": parser_read,
        **stream_reads,
    }
    if any(not result.available for result in reads.values()):
        key, result = next(
            (key, result)
            for key, result in reads.items()
            if not result.available
        )
        return _ds9_runtime_qualification_unavailable(
            f"{key}_{result.state}", integrity=integrity
        )
    if receipt is None or schema is None or parser_receipt is None:
        return _ds9_runtime_qualification_unavailable(
            "json_contract_invalid", integrity=integrity
        )

    schema_identity_valid = bool(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "ds9-runtime-compatibility-v1.schema.json"
        )
    )
    try:
        _validate_schema_node(receipt, schema, schema)
    except (TypeError, ValueError, RecursionError):
        schema_replay_valid = False
    else:
        schema_replay_valid = schema_identity_valid
    integrity["receipt_schema_replay_verified"] = schema_replay_valid
    integrity["receipt_semantics_replayed"] = (
        _ds9_runtime_receipt_semantics_valid(receipt, parser_receipt)
    )
    if not all(integrity.values()):
        return _ds9_runtime_qualification_unavailable(
            "exact_pin_schema_or_semantic_replay_invalid",
            integrity=integrity,
        )

    try:
        created = datetime.fromisoformat(
            receipt["created_at_utc"].replace("Z", "+00:00")
        )
        expires = datetime.fromisoformat(
            receipt["expires_at_utc"].replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError):
        return _ds9_runtime_qualification_unavailable(
            "qualification_window_invalid", integrity=integrity
        )
    observed_now = now or datetime.now(timezone.utc)
    if observed_now.tzinfo is None:
        observed_now = observed_now.replace(tzinfo=timezone.utc)
    observed_now = observed_now.astimezone(timezone.utc)
    current = bool(created <= observed_now < expires)
    if observed_now < created:
        state = "pending_qualification_window"
        reason = "qualification_not_yet_current"
    elif observed_now >= expires:
        state = "stale_pending_requalification"
        reason = "qualification_expired"
    else:
        state = "runtime_qualified_current"
        reason = "seven_day_endurance_pending"

    return {
        "label": "DeepStream 9 çalışma zamanı yeterliliği",
        "available": True,
        "current": current,
        "state": state,
        "reason": reason,
        "created_at_utc": receipt["created_at_utc"],
        "expires_at_utc": receipt["expires_at_utc"],
        "runtime_qualification_passed": current,
        "does_not_imply_product_readiness": True,
        "product_production_ready": False,
        "training_complete": False,
        "final_test_complete": False,
        "image_binding": "exact_pinned_deepstream_9_image",
        "runtime": {
            "deepstream": "9.0.0",
            "cuda": "13.1",
            "tensorrt": "10.14.1.48",
            "architecture": "sm_86",
        },
        "checks": {
            "passed": 5,
            "total": 5,
            "cpu_cuda_parser_parity_640": True,
            "cpu_cuda_parser_parity_960": True,
            "cuda_parser_kernel_launch_sm86": True,
            "deepstream_640_engine_deserialize_no_fallback": True,
            "deepstream_960_engine_deserialize_no_fallback": True,
        },
        "integrity": integrity,
        "caveats": [
            "Bu kanıt yalnız exact-pinli DS9 çalışma zamanı, parser ve 640/960 motor smoke yeterliliğidir; model kalitesi veya ürün kabulü değildir.",
            (
                "Yeterlilik zaman penceresi dolduğunda panel kapıyı otomatik "
                "olarak stale/pending durumuna indirir."
            ),
        ],
        "evidence": [],
    }


def _gpu_reentry(
    reader: ArtifactReader, *, now: datetime | None = None
) -> dict[str, Any]:
    current = _gpu_reentry_r2(reader)
    projected = current if current is not None else _gpu_reentry_v1(reader)
    qualification = _ds9_runtime_qualification(reader, now=now)
    projected["deepstream_runtime_qualification"] = qualification
    if projected.get("available") is True and qualification.get("current") is True:
        projected["state"] = "deepstream_runtime_qualified_endurance_pending"
        projected["reason"] = "seven_day_endurance_pending"
        projected["progress"] = _progress(5, 6)
        projected["status_counts"] = {"passed": 5, "pending": 1}
        projected["pending_gate_ids"] = ["seven_day_endurance"]
        projected.setdefault("caveats", []).append(
            "Ayrı exact-pinli DS9 yeterlilik zinciri üst çalışma zamanı kapısını geçti; GPU R2 receipt içindeki tarihsel deepstream bayrağı değiştirilmedi."
        )
    return projected


def _loaf_batch_profiles(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for profile in ("640", "960"):
        raw = value.get(profile)
        if not isinstance(raw, dict):
            continue
        overall = raw.get("overall") if isinstance(raw.get("overall"), dict) else {}
        if not overall:
            continue
        ground_truth = _integer(overall.get("ground_truth"))
        tp = _integer(overall.get("tp"))
        fp = _integer(overall.get("fp"))
        fn = _integer(overall.get("fn"))
        precision = _number(overall.get("precision"))
        recall = _number(overall.get("recall"))
        f1 = _number(overall.get("f1"))
        ap = _number(overall.get("ap_101_point"))
        if (
            ground_truth is None
            or ground_truth <= 0
            or tp is None
            or fp is None
            or fn is None
            or tp + fn != ground_truth
            or precision is None
            or recall is None
            or f1 is None
            or ap is None
            or not all(0.0 <= number <= 1.0 for number in (precision, recall, f1, ap))
        ):
            continue
        result[profile] = {
            "ground_truth": ground_truth,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap_101_point": ap,
        }
    return result


def _loaf_batch_plan_contract(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    campaign = value.get("campaign") if isinstance(value.get("campaign"), dict) else {}
    source = (
        value.get("source_contract")
        if isinstance(value.get("source_contract"), dict)
        else {}
    )
    jobs = value.get("jobs")
    fingerprint = _sha256(value.get("plan_fingerprint"))
    sizes = campaign.get("model_input_sizes")
    distance = (
        campaign.get("distance_band_m")
        if isinstance(campaign.get("distance_band_m"), dict)
        else {}
    )
    status = _enum(
        value.get("status"),
        {"planned", "running", "execution-finished", "failed", "safety-abort"},
    )
    gpu_execution_requested = _boolean(value.get("gpu_execution_requested"))
    if (
        value.get("schema_version") != "deepsafe.loaf-deepstream-batch-plan/v1"
        or status is None
        or gpu_execution_requested is None
        or (status == "planned" and gpu_execution_requested is not False)
        or (status != "planned" and gpu_execution_requested is not True)
        or fingerprint is None
        or campaign.get("split") != "val"
        or _integer(campaign.get("sequence_count")) != 8
        or _integer(campaign.get("frame_count")) != 2948
        or _integer(campaign.get("expected_jobs")) != 16
        or sizes != [640, 960]
        or _number(distance.get("minimum_inclusive")) != 20.0
        or _number(distance.get("maximum_exclusive")) != 25.0
        or _integer(source.get("sequence_count")) != 8
        or _integer(source.get("frame_count")) != 2948
        or not isinstance(jobs, list)
        or len(jobs) != 16
    ):
        return None
    return {
        "fingerprint": fingerprint,
        "sequence_count": 8,
        "frame_count": 2948,
        "expected_jobs": 16,
        "model_input_sizes": [640, 960],
    }


def _loaf_deepstream(reader: ArtifactReader) -> dict[str, Any]:
    aggregate_artifact = reader.read("loaf_batch_aggregate")
    plan_artifact = reader.read("loaf_batch_plan")
    aggregate = aggregate_artifact.value or {}
    plan = plan_artifact.value or {}
    completeness = (
        aggregate.get("completeness")
        if isinstance(aggregate.get("completeness"), dict)
        else {}
    )
    campaign = plan.get("campaign") if isinstance(plan.get("campaign"), dict) else {}
    source = (
        plan.get("source_contract")
        if isinstance(plan.get("source_contract"), dict)
        else {}
    )
    aggregate_total = _integer(completeness.get("expected_jobs"))
    aggregate_completed = _integer(completeness.get("complete_jobs"))
    plan_total = _integer(campaign.get("expected_jobs"))
    total = aggregate_total if aggregate_total is not None else plan_total or 0
    completed = aggregate_completed if aggregate_completed is not None else 0
    counts = _identifier_counts(completeness.get("state_counts"))
    aggregation_status = _enum(
        aggregate.get("aggregation_status"), {"complete", "withheld_incomplete"}
    )
    plan_contract = _loaf_batch_plan_contract(plan) if plan_artifact.available else None
    aggregate_fingerprint = _sha256(aggregate.get("plan_fingerprint"))
    profiles = _loaf_batch_profiles(aggregate.get("profiles"))
    profiles_complete = bool(
        set(profiles) == {"640", "960"}
        and all(
            all(metric is not None for metric in profile.values())
            for profile in profiles.values()
        )
    )
    pending = _integer(completeness.get("pending_jobs"))
    is_complete = _boolean(completeness.get("is_complete"))
    results = aggregate.get("results")
    aggregate_contract_valid = bool(
        aggregate_artifact.available
        and aggregate.get("schema_version")
        == "deepsafe.loaf-deepstream-batch-aggregate/v1"
        and aggregate_fingerprint is not None
        and aggregate_total == 16
        and aggregate_completed is not None
        and 0 <= aggregate_completed <= aggregate_total
        and pending == total - completed
        and isinstance(is_complete, bool)
        and isinstance(results, list)
        and all(isinstance(row, dict) for row in results)
        and sum(counts.values()) == 16
        and counts.get("complete", 0) == completed
        and (
            (
                aggregation_status == "complete"
                and completed == total
                and is_complete is True
                and len(results) == 16
                and profiles_complete
            )
            or (
                aggregation_status == "withheld_incomplete"
                and completed < total
                and is_complete is False
                and not results
                and not profiles
            )
        )
    )
    contracts_match = bool(
        aggregate_contract_valid
        and plan_contract is not None
        and aggregate_fingerprint == plan_contract["fingerprint"]
    )
    unavailable = _unavailable_state(aggregate_artifact, plan_artifact)
    if aggregate_artifact.state not in {"ok", "missing"}:
        state = "artifact_error"
    elif plan_artifact.state not in {"ok", "missing"}:
        state = "artifact_error"
    elif unavailable is not None:
        state = unavailable
    elif aggregate_artifact.available and not plan_artifact.available:
        state = "artifact_error"
    elif plan_artifact.available and plan_contract is None:
        state = "artifact_error"
    elif aggregate_artifact.available and not contracts_match:
        state = "artifact_error"
    elif aggregation_status == "complete" and total and completed >= total:
        state = "complete"
    elif any(
        counts.get(name, 0) > 0
        for name in ("failed", "conflict", "incomplete", "reentry_blocked")
    ):
        state = "attention"
    elif completed:
        state = "in_progress"
    else:
        state = "planned"
    distance = (
        campaign.get("distance_band_m")
        if isinstance(campaign.get("distance_band_m"), dict)
        else {}
    )
    geometry = _enum(
        source.get("metric_geometry"), {"axis_aligned_envelope_of_rotated_box"}
    )
    return {
        "label": "LOAF 20–25 m DeepStream 9 batch",
        "available": aggregate_artifact.available or plan_artifact.available,
        "state": state,
        "updated_at_utc": _timestamp(
            aggregate.get("generated_at") or plan.get("created_at")
        ),
        "progress": _progress(completed, total),
        "status_counts": counts,
        "aggregation_status": aggregation_status,
        "gpu_execution_requested": _boolean(plan.get("gpu_execution_requested")),
        "scope": {
            "split": _enum(campaign.get("split"), {"val"}),
            "distance_m": {
                "minimum_inclusive": _number(distance.get("minimum_inclusive")),
                "maximum_exclusive": _number(distance.get("maximum_exclusive")),
            },
            "sequences": _integer(campaign.get("sequence_count")),
            "frames": _integer(campaign.get("frame_count")),
            "model_input_sizes": [
                size for size in _sizes(campaign.get("model_input_sizes")) if size in (640, 960)
            ],
            "jobs": total or None,
            "metric_geometry": geometry,
        },
        "profiles": profiles if contracts_match else {},
        "metric_context": {
            "ground_truth": True,
            "metric": "AP101@IoU0.5",
            "coco_map": False,
            "withheld_until_complete": aggregation_status != "complete",
        },
        "evidence": _evidence(reader, "loaf_batch_aggregate", "loaf_batch_plan"),
    }


def _loaf_distance_results(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, dict):
        return None
    completeness = (
        value.get("completeness")
        if isinstance(value.get("completeness"), dict)
        else {}
    )
    metric = value.get("metric") if isinstance(value.get("metric"), dict) else {}
    rows = value.get("rows")
    allowed_bins = ("20-21m", "21-22m", "22-23m", "23-24m", "24-25m")
    expected_pairs = {
        (profile, bin_id) for profile in (640, 960) for bin_id in allowed_bins
    }
    if (
        value.get("schema_version") != "deepsafe.loaf-distance-bin-aggregate/v1"
        or value.get("status") != "complete"
        or value.get("split") != "val"
        or value.get("test_unseen_opened") is not False
        or metric.get("name") != "AP101@IoU0.5"
        or metric.get("explicitly_not") != "not COCO mAP@[.50:.95]"
        or _integer(completeness.get("expected_evaluations")) != 10
        or _integer(completeness.get("complete_evaluations")) != 10
        or completeness.get("is_complete") is not True
        or not isinstance(rows, list)
        or len(rows) != 10
    ):
        return None
    projected: list[dict[str, Any]] = []
    observed: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            return None
        profile = row.get("model_input")
        bin_id = row.get("distance_bin_id")
        pair = (profile, bin_id)
        ground_truth = _integer(row.get("ground_truth"))
        tp = _integer(row.get("tp"))
        fp = _integer(row.get("fp"))
        fn = _integer(row.get("fn"))
        ignored = _integer(row.get("ignored_predictions"))
        precision = _number(row.get("precision"))
        recall = _number(row.get("recall"))
        f1 = _number(row.get("f1"))
        ap = _number(row.get("ap101_iou_0_5"))
        if (
            pair not in expected_pairs
            or pair in observed
            or ground_truth is None
            or ground_truth <= 0
            or tp is None
            or fp is None
            or fn is None
            or ignored is None
            or tp + fn != ground_truth
            or precision is None
            or recall is None
            or f1 is None
            or ap is None
            or not all(0.0 <= number <= 1.0 for number in (precision, recall, f1, ap))
        ):
            return None
        observed.add(pair)
        projected.append(
            {
                "model_input": profile,
                "distance_bin_id": bin_id,
                "ground_truth": ground_truth,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ignored_predictions": ignored,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap101_iou_0_5": ap,
            }
        )
    if observed != expected_pairs:
        return None
    return sorted(projected, key=lambda row: (row["model_input"], row["distance_bin_id"]))


def _loaf_distance_bins(reader: ArtifactReader) -> dict[str, Any]:
    preparation_artifact = reader.read("loaf_distance_bin_preparation")
    plan_artifact = reader.read("loaf_distance_bin_evaluation_plan")
    aggregate_artifact = reader.read("loaf_distance_bin_aggregate")
    preparation = preparation_artifact.value or {}
    plan = plan_artifact.value or {}
    aggregate = aggregate_artifact.value or {}
    unavailable = _unavailable_state(
        preparation_artifact, plan_artifact, aggregate_artifact
    )
    allowed_states = {
        "prepared_not_evaluated",
        "waiting_for_predictions",
        "ready_for_cpu_evaluation",
    }
    state: str | None = None
    if preparation_artifact.state not in {"ok", "missing"}:
        state = "artifact_error"
    elif plan_artifact.state not in {"ok", "missing"}:
        state = "artifact_error"
    elif aggregate_artifact.state not in {"ok", "missing"}:
        state = "artifact_error"
    elif unavailable is not None:
        state = unavailable
    elif plan_artifact.available and not preparation_artifact.available:
        state = "artifact_error"
    expected_profiles = [
        size for size in _sizes(plan.get("expected_profiles")) if size in (640, 960)
    ]
    profile_counts = {
        key: count
        for key, count in _job_counts(plan.get("profiles")).items()
        if key
        in {
            "ready_for_cpu_evaluation",
            "waiting_for_complete_profile_merge",
        }
    }
    ready_profiles = profile_counts.get("ready_for_cpu_evaluation", 0)
    allowed_bins = ("20-21m", "21-22m", "22-23m", "23-24m", "24-25m")
    bin_targets: dict[str, int] = {}
    raw_bins = preparation.get("distance_bins")
    if isinstance(raw_bins, list):
        for item in raw_bins[: len(allowed_bins)]:
            if not isinstance(item, dict) or item.get("bin_id") not in allowed_bins:
                continue
            count = _integer(item.get("target_annotation_count"))
            if count is not None:
                bin_targets[item["bin_id"]] = count
    expected_bins = [
        item
        for item in plan.get("expected_distance_bins", [])[: len(allowed_bins)]
        if item in allowed_bins
    ] if isinstance(plan.get("expected_distance_bins"), list) else []
    metric = plan.get("metric") if isinstance(plan.get("metric"), dict) else {}
    base_targets = _integer(preparation.get("base_target_count"))
    partition_targets = _integer(preparation.get("partition_target_count"))
    preparation_contract_valid = bool(
        preparation_artifact.available
        and preparation.get("schema_version")
        == "deepsafe.loaf-distance-bin-preparation/v1"
        and preparation.get("status") == "prepared_not_evaluated"
        and preparation.get("split") == "val"
        and preparation.get("test_unseen_opened") is False
        and preparation.get("gpu_or_inference_executed") is False
        and _integer(preparation.get("sequence_count")) == 8
        and _integer(preparation.get("frame_count")) == 2948
        and base_targets == 7539
        and partition_targets == 7539
        and list(bin_targets) == list(allowed_bins)
        and sum(bin_targets.values()) == 7539
        and preparation.get("metric_geometry")
        == "axis_aligned_envelope_of_rotated_box"
    )
    raw_profiles = plan.get("profiles")
    observed_profile_inputs = (
        {
            item.get("model_input")
            for item in raw_profiles
            if isinstance(item, dict)
            and item.get("status")
            in {
                "ready_for_cpu_evaluation",
                "waiting_for_complete_profile_merge",
            }
        }
        if isinstance(raw_profiles, list)
        else set()
    )
    plan_contract_valid = bool(
        plan_artifact.available
        and plan.get("schema_version")
        == "deepsafe.loaf-distance-bin-evaluation-plan/v1"
        and _enum(plan.get("status"), allowed_states) is not None
        and plan.get("split") == "val"
        and plan.get("dry_run") is True
        and plan.get("gpu_or_inference_executed") is False
        and plan.get("test_unseen_opened") is False
        and expected_profiles == [640, 960]
        and expected_bins == list(allowed_bins)
        and _integer(plan.get("expected_evaluations")) == 10
        and metric.get("name") == "AP101@IoU0.5"
        and metric.get("explicitly_not") == "not COCO mAP@[.50:.95]"
        and isinstance(raw_profiles, list)
        and len(raw_profiles) == 2
        and observed_profile_inputs == {640, 960}
    )
    projected_results = (
        _loaf_distance_results(aggregate) if aggregate_artifact.available else None
    )
    if state is None:
        if not preparation_contract_valid or not plan_contract_valid:
            state = "artifact_error"
        elif aggregate_artifact.available:
            state = "complete" if projected_results is not None else "artifact_error"
        else:
            state = _enum(plan.get("status"), allowed_states) or "artifact_error"
    complete_evaluations = 10 if state == "complete" else 0
    reported_ready_profiles = len(expected_profiles) if state == "complete" else ready_profiles
    return {
        "label": "LOAF 1 metrelik mesafe dilimleri",
        "available": (
            preparation_artifact.available
            or plan_artifact.available
            or aggregate_artifact.available
        ),
        "state": state,
        "progress": _progress(complete_evaluations, 10),
        "status_counts": profile_counts,
        "profiles_ready": reported_ready_profiles,
        "scope": {
            "split": _enum(plan.get("split") or preparation.get("split"), {"val"}),
            "sequences": _integer(preparation.get("sequence_count")),
            "frames": _integer(preparation.get("frame_count")),
            "targets": base_targets,
            "model_input_sizes": expected_profiles,
            "distance_bins": expected_bins,
            "expected_evaluations": _integer(plan.get("expected_evaluations")),
            "metric_geometry": _enum(
                preparation.get("metric_geometry"),
                {"axis_aligned_envelope_of_rotated_box"},
            ),
        },
        "bin_target_counts": dict(sorted(bin_targets.items())),
        "partition_complete": bool(
            base_targets is not None
            and base_targets > 0
            and base_targets == partition_targets
        ),
        "metric_context": {
            "ground_truth": True,
            "metric": _enum(metric.get("name"), {"AP101@IoU0.5"}),
            "coco_map": False,
            "per_bin_only": True,
        },
        "results": projected_results if state == "complete" else [],
        "safety": {
            "dry_run": _boolean(plan.get("dry_run")),
            "gpu_or_inference_executed": _boolean(
                plan.get("gpu_or_inference_executed")
            ),
            "test_unseen_opened": _boolean(plan.get("test_unseen_opened")),
        },
        "evidence": _evidence(
            reader,
            "loaf_distance_bin_preparation",
            "loaf_distance_bin_evaluation_plan",
            "loaf_distance_bin_aggregate",
        ),
    }


def _site_distance_plan_contract(
    reader: ArtifactReader, value: Any
) -> tuple[bool, str | None, int]:
    if not isinstance(value, dict) or not reader.validates_schema(value, SITE_PLAN_SCHEMA):
        return False, None, 0
    status = _enum(
        value.get("status"), {"waiting_for_inputs", "ready_for_cpu_evaluation"}
    )
    inputs = value.get("inputs") if isinstance(value.get("inputs"), dict) else {}
    expected_inputs = {
        "calibration",
        "ground_truth",
        "acceptance",
        "profile_640",
        "profile_960",
    }
    present_values: list[bool] = []
    if set(inputs) == expected_inputs:
        for key in sorted(expected_inputs):
            item = inputs.get(key)
            if not isinstance(item, dict) or not isinstance(item.get("present"), bool):
                present_values = []
                break
            present_values.append(item["present"])
    ready = sum(present_values)
    status_consistent = bool(
        len(present_values) == 5
        and (
            (status == "ready_for_cpu_evaluation" and ready == 5)
            or (status == "waiting_for_inputs" and ready < 5)
        )
    )
    valid = status_consistent
    return valid, status, ready


def _site_distance_profile(value: Any, profile: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    metrics = value.get("metrics") if isinstance(value.get("metrics"), dict) else {}
    ground_truth = _integer(value.get("ground_truth_instances"))
    frames = _integer(value.get("frame_records"))
    tp = _integer(metrics.get("tp"))
    fp = _integer(metrics.get("fp"))
    fn = _integer(metrics.get("fn"))
    ignored = _integer(metrics.get("ignored_predictions"))
    outside = _integer(metrics.get("predictions_excluded_outside_calibrated_band"))
    precision = _number(metrics.get("precision"))
    recall = _number(metrics.get("recall"))
    f1 = _number(metrics.get("f1"))
    ap = _number(metrics.get("ap_101_point"))
    model_id = _identifier(value.get("model_id"), maximum=80)
    if (
        value.get("status") != "complete"
        or value.get("model_input") != profile
        or ground_truth is None
        or ground_truth < 1
        or frames is None
        or frames < 1
        or tp is None
        or fp is None
        or fn is None
        or ignored is None
        or outside is None
        or tp + fn != ground_truth
        or precision is None
        or recall is None
        or f1 is None
        or ap is None
        or model_id is None
        or not all(0.0 <= item <= 1.0 for item in (precision, recall, f1, ap))
    ):
        return None
    expected_precision = tp / (tp + fp) if tp + fp else 0.0
    expected_recall = tp / ground_truth
    expected_f1 = (
        2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    if not all(
        math.isclose(observed, expected, abs_tol=1e-6)
        for observed, expected in (
            (precision, expected_precision),
            (recall, expected_recall),
            (f1, expected_f1),
        )
    ):
        return None
    return {
        "model_input": profile,
        "model_id": model_id,
        "frame_records": frames,
        "ground_truth_instances": ground_truth,
        "metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap_101_point": ap,
            "ignored_predictions": ignored,
            "predictions_excluded_outside_calibrated_band": outside,
        },
    }


def _site_distance_named_pins(
    evaluation: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], str]]:
    profiles = evaluation["profiles"]
    return {
        "distance_25m_implementation": (
            evaluation["implementation"],
            "text/x-python",
        ),
        "distance_25m_calibration": (
            evaluation["calibration"]["artifact"],
            "application/json",
        ),
        "distance_25m_calibration_verification": (
            evaluation["calibration"]["verification_document"],
            "text/markdown",
        ),
        "distance_25m_ground_truth": (
            evaluation["ground_truth"]["artifact"],
            "application/json",
        ),
        "distance_25m_source_asset": (
            evaluation["ground_truth"]["source_asset"],
            "application/octet-stream",
        ),
        "distance_25m_annotation_document": (
            evaluation["ground_truth"]["annotation_document"],
            "text/markdown",
        ),
        "distance_25m_acceptance": (
            evaluation["acceptance"]["artifact"],
            "application/json",
        ),
        "distance_25m_acceptance_approval": (
            evaluation["acceptance"]["approval_document"],
            "text/markdown",
        ),
        "distance_25m_profile_640_manifest": (
            profiles["640"]["completion_manifest"],
            "application/json",
        ),
        "distance_25m_profile_640_predictions": (
            profiles["640"]["predictions"],
            "application/x-ndjson",
        ),
        "distance_25m_profile_960_manifest": (
            profiles["960"]["completion_manifest"],
            "application/json",
        ),
        "distance_25m_profile_960_predictions": (
            profiles["960"]["predictions"],
            "application/x-ndjson",
        ),
    }


def _pin_identity(pin: Any) -> tuple[Any, Any, Any] | None:
    if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}:
        return None
    return pin.get("path"), pin.get("bytes"), pin.get("sha256")


def _site_input_fingerprint(
    pins: list[dict[str, Any]], config: dict[str, Any]
) -> str:
    unique = {
        (pin["path"], pin["sha256"]): {
            "path": pin["path"],
            "bytes": pin["bytes"],
            "sha256": pin["sha256"],
        }
        for pin in pins
    }
    canonical = {
        "artifacts": [unique[key] for key in sorted(unique)],
        "config": config,
    }
    content = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(content).hexdigest()


def _site_distance_live_pins(
    reader: ArtifactReader, evaluation: dict[str, Any]
) -> dict[str, tuple[dict[str, Any], str, Path]] | None:
    try:
        named = _site_distance_named_pins(evaluation)
        integrity_pins = evaluation["integrity"]["input_artifacts"]
        evaluation_config = evaluation["evaluation_config"]
    except (KeyError, TypeError):
        return None
    named_input_pins = [
        pin for name, (pin, _) in named.items() if name != "distance_25m_implementation"
    ]
    named_identities = {_pin_identity(pin) for pin in named_input_pins}
    integrity_identities = [_pin_identity(pin) for pin in integrity_pins]
    if (
        None in named_identities
        or any(identity is None for identity in integrity_identities)
        or len(integrity_identities) != len(set(integrity_identities))
        or set(integrity_identities) != named_identities
        or evaluation["integrity"].get("input_fingerprint_sha256")
        != _site_input_fingerprint(integrity_pins, evaluation_config)
    ):
        return None
    verified: dict[str, tuple[dict[str, Any], str, Path]] = {}
    for evidence_id, (pin, media_type) in named.items():
        path = reader.verify_workspace_pin(pin)
        if path is None:
            return None
        verified[evidence_id] = (pin, media_type, path)
    expected_implementation = reader.resolve_workspace_path(
        "validation/site_distance_evaluation.py"
    )
    if (
        expected_implementation is None
        or verified["distance_25m_implementation"][2] != expected_implementation
    ):
        return None
    return verified


def _report_evidence_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    rows = report.get("evidence")
    if not isinstance(rows, list):
        return None
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            return None
        evidence_id = row["id"]
        if evidence_id in result:
            return None
        result[evidence_id] = row
    return result


_ENDURANCE_CHECKPOINT_STATUS_IDENTITY_FIELDS = (
    "state",
    "dry_run",
    "campaign_name",
    "config_fingerprint",
    "static_input_fingerprint",
    "throughput_floor",
    "power_safety_policy",
    "updated_at_utc",
    "started_at_utc",
    "finished_at_utc",
    "target_validated_seconds",
    "validated_seconds",
    "active",
    "unexpected_restarts",
    "orphan_recoveries",
    "campaign_health_gates",
)
_ENDURANCE_REPORT_EVIDENCE_IDS = {
    "endurance_campaign_resolved",
    "endurance_checkpoint",
    "endurance_status",
}


def _campaign_report_claims_endurance_lineage(report: dict[str, Any]) -> bool:
    """Identify reports that claim any live endurance fact or evidence pin."""

    campaigns = report.get("campaigns")
    if isinstance(campaigns, dict) and "endurance" in campaigns:
        return True
    requirements = report.get("requirements")
    if isinstance(requirements, list) and any(
        isinstance(item, dict) and item.get("id") == "seven_day_endurance"
        for item in requirements
    ):
        return True
    evidence = report.get("evidence")
    return bool(
        isinstance(evidence, list)
        and any(
            isinstance(item, dict)
            and item.get("id") in _ENDURANCE_REPORT_EVIDENCE_IDS
            for item in evidence
        )
    )


def _artifact_evidence_pin_matches(
    artifact: ArtifactRead,
    row: Any,
    *,
    campaign_evidence: bool = False,
) -> bool:
    """Match a report evidence row to the exact live allow-listed artifact."""

    if (
        not artifact.available
        or artifact.content is None
        or artifact.value is None
        or not isinstance(row, dict)
        or row.get("path")
        != f"validation/results/{artifact.relative_path}"
        or row.get("size_bytes") != len(artifact.content)
        or row.get("sha256") != hashlib.sha256(artifact.content).hexdigest()
    ):
        return False
    if not campaign_evidence:
        return True
    return bool(
        row.get("state") == "ok"
        and row.get("media_type") == "application/json"
        and row.get("schema_version") == artifact.value.get("schema_version")
    )


def _live_endurance_lineage(reader: ArtifactReader) -> dict[str, Any] | None:
    """Return one internally coherent live endurance identity, or fail closed."""

    resolved_artifact = reader.read("endurance_campaign_resolved")
    checkpoint_artifact = reader.read("endurance_checkpoint")
    status_artifact = reader.read("endurance_status")
    artifacts = {
        "endurance_campaign_resolved": resolved_artifact,
        "endurance_checkpoint": checkpoint_artifact,
        "endurance_status": status_artifact,
    }
    if any(
        not artifact.available or artifact.value is None
        for artifact in artifacts.values()
    ):
        return None

    resolved = resolved_artifact.value or {}
    checkpoint = checkpoint_artifact.value or {}
    status = status_artifact.value or {}
    if (
        resolved.get("schema_version") != "deepsafe.endurance-campaign/v1"
        or checkpoint.get("schema_version")
        != "deepsafe.endurance-checkpoint/v1"
        or status.get("schema_version") != "deepsafe.endurance-status/v1"
    ):
        return None

    config_fingerprint = _sha256(resolved.get("config_fingerprint"))
    static_input_fingerprint = _sha256(
        resolved.get("static_input_fingerprint")
    )
    campaign_name = _text(resolved.get("name"), limit=128)
    target_seconds = _integer(resolved.get("duration_seconds"))
    if (
        config_fingerprint is None
        or static_input_fingerprint is None
        or campaign_name is None
        or target_seconds is None
        or target_seconds <= 0
        or _boolean(checkpoint.get("dry_run")) is None
        or _integer(checkpoint.get("validated_seconds")) is None
        or _enum(
            checkpoint.get("state"),
            {"planned", "running", "paused", "paused_health_gate", "complete"},
        )
        is None
    ):
        return None

    if any(
        document.get("config_fingerprint") != config_fingerprint
        or document.get("static_input_fingerprint")
        != static_input_fingerprint
        or document.get("campaign_name") != campaign_name
        or document.get("target_validated_seconds") != target_seconds
        for document in (checkpoint, status)
    ):
        return None
    if any(
        not _json_equal(checkpoint.get(field), status.get(field))
        for field in _ENDURANCE_CHECKPOINT_STATUS_IDENTITY_FIELDS
    ):
        return None
    if (
        not _json_equal(
            resolved.get("throughput_floor"), checkpoint.get("throughput_floor")
        )
        or not _json_equal(
            resolved.get("power_safety"), checkpoint.get("power_safety_policy")
        )
        or not _json_equal(resolved.get("input_pins"), checkpoint.get("input_pins"))
    ):
        return None

    return {
        "campaign_name": campaign_name,
        "config_fingerprint": config_fingerprint,
        "static_input_fingerprint": static_input_fingerprint,
        "target_validated_seconds": target_seconds,
        "artifacts": artifacts,
    }


def _campaign_report_lineage_matches(
    reader: ArtifactReader,
    report: dict[str, Any],
    *,
    live: dict[str, Any] | None = None,
) -> bool:
    live = live or _live_endurance_lineage(reader)
    evidence = _report_evidence_by_id(report)
    if live is None or evidence is None:
        return False
    artifacts = live["artifacts"]
    return all(
        _artifact_evidence_pin_matches(
            artifacts[evidence_id],
            evidence.get(evidence_id),
            campaign_evidence=True,
        )
        for evidence_id in (
            "endurance_campaign_resolved",
            "endurance_checkpoint",
            "endurance_status",
        )
    )


def _objective_report_lineage_matches(
    reader: ArtifactReader, report: dict[str, Any]
) -> bool:
    live = _live_endurance_lineage(reader)
    evidence = _report_evidence_by_id(report)
    campaign_artifact = reader.read("campaign_report_json")
    if (
        live is None
        or evidence is None
        or not campaign_artifact.available
        or campaign_artifact.value is None
        or not _campaign_report_lineage_matches(
            reader, campaign_artifact.value, live=live
        )
    ):
        return False
    artifacts = live["artifacts"]
    return bool(
        _artifact_evidence_pin_matches(
            campaign_artifact, evidence.get("campaign_report")
        )
        and _artifact_evidence_pin_matches(
            artifacts["endurance_checkpoint"],
            evidence.get("endurance_checkpoint"),
        )
        and _artifact_evidence_pin_matches(
            artifacts["endurance_status"], evidence.get("endurance_status")
        )
    )


def _stale_lineage_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prevent stale report files from looking like usable evidence."""

    return [
        {
            **{key: value for key, value in item.items() if key != "href"},
            "available": False,
            "artifact_state": "stale_lineage",
        }
        for item in items
    ]


def _site_distance_report_proves(
    reader: ArtifactReader,
    evaluation_artifact: ArtifactRead,
    evaluation: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    live_pins: dict[str, tuple[dict[str, Any], str, Path]],
) -> bool:
    report_artifact = reader.read("campaign_report_json")
    if (
        not report_artifact.available
        or report_artifact.value is None
        or evaluation_artifact.content is None
        or not reader.validates_schema(
            report_artifact.value, CAMPAIGN_REPORT_SCHEMA
        )
    ):
        return False
    report = report_artifact.value
    campaigns = report.get("campaigns") if isinstance(report.get("campaigns"), dict) else {}
    distance = campaigns.get("distance_25m") if isinstance(campaigns.get("distance_25m"), dict) else {}
    contract = distance.get("contract") if isinstance(distance.get("contract"), dict) else {}
    report_profiles = distance.get("profiles") if isinstance(distance.get("profiles"), dict) else {}
    evidence_by_id = _report_evidence_by_id(report)
    if evidence_by_id is None:
        return False
    evidence = evidence_by_id.get("distance_25m_evaluation")
    evaluation_sha = hashlib.sha256(evaluation_artifact.content).hexdigest()
    evidence_bound = bool(
        isinstance(evidence, dict)
        and evidence.get("state") == "ok"
        and evidence.get("media_type") == "application/json"
        and evidence.get("sha256") == evaluation_sha
        and evidence.get("size_bytes") == len(evaluation_artifact.content)
    )
    expected_evidence_ids = {"distance_25m_evaluation", *live_pins}
    pin_evidence_bound = bool(
        set(distance.get("evidence_ids", [])) == expected_evidence_ids
    )
    if pin_evidence_bound:
        for evidence_id, (pin, media_type, resolved) in live_pins.items():
            row = evidence_by_id.get(evidence_id)
            report_path = (
                reader.resolve_workspace_path(row.get("path"))
                if isinstance(row, dict)
                else None
            )
            if not (
                isinstance(row, dict)
                and row.get("state") == "ok"
                and row.get("media_type") == media_type
                and row.get("size_bytes") == pin["bytes"]
                and row.get("sha256") == pin["sha256"]
                and report_path == resolved
            ):
                pin_evidence_bound = False
                break
    report_contract = bool(
        distance.get("evidence_kind") == "calibrated_distance_ground_truth"
        and distance.get("target_maximum_distance_m") == 25
        and distance.get("state") == "proven"
        and distance.get("accepted") is True
        and distance.get("schema_contract_valid") is True
        and distance.get("pin_matrix_valid") is True
        and distance.get("independent_cpu_recomputation_valid") is True
        and contract.get("required_schema") == "deepsafe.distance-validation/v1"
        and contract.get("required_bin_m") == [20, 25]
        and contract.get("required_profiles") == [640, 960]
        and contract.get("requires_verified_calibration") is True
        and contract.get("requires_documented_passing_acceptance_criterion") is True
        and set(report_profiles) == {"640", "960"}
        and distance.get("criterion") == evaluation["acceptance"]["criterion"]
    )
    if not evidence_bound or not pin_evidence_bound or not report_contract:
        return False
    for profile in ("640", "960"):
        expected = profiles[profile]
        observed = report_profiles.get(profile)
        if not isinstance(observed, dict):
            return False
        observed_metrics = (
            observed.get("metrics")
            if isinstance(observed.get("metrics"), dict)
            else {}
        )
        projected = {
            "ground_truth_instances": _integer(observed.get("ground_truth_instances")),
            "frame_records": _integer(observed.get("frame_records")),
            "model_id": _identifier(observed.get("model_id"), maximum=80),
            "metrics": {
                key: (
                    _integer(observed_metrics.get(key))
                    if key
                    in {
                        "tp",
                        "fp",
                        "fn",
                        "ignored_predictions",
                        "predictions_excluded_outside_calibrated_band",
                    }
                    else _number(observed_metrics.get(key))
                )
                for key in expected["metrics"]
            },
        }
        comparable_expected = {
            "ground_truth_instances": expected["ground_truth_instances"],
            "frame_records": expected["frame_records"],
            "model_id": expected["model_id"],
            "metrics": expected["metrics"],
        }
        if projected != comparable_expected:
            return False
    return True


def _site_distance_v2_selection(
    reader: ArtifactReader,
) -> tuple[ArtifactRead, int, list[str]]:
    """Select exactly one immutable-looking v2 final without following links."""

    spec = ARTIFACTS["site_distance_evaluation_v2"]
    if reader.root_error:
        return ArtifactRead(spec, "invalid_root", spec.candidates[0]), 0, [
            "invalid_root"
        ]
    directory = reader.root / "distance-25m"
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return reader.read("site_distance_evaluation_v2"), 0, []
    except OSError:
        return ArtifactRead(spec, "io_error", spec.candidates[0]), 0, [
            "result_directory_unreadable"
        ]
    if directory.is_symlink() or not directory.is_dir():
        return ArtifactRead(spec, "unsafe_path", spec.candidates[0]), 0, [
            "result_directory_unsafe"
        ]
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return ArtifactRead(spec, "io_error", spec.candidates[0]), 0, [
            "result_directory_unreadable"
        ]
    candidates: list[str] = []
    unsafe = False
    for entry in entries:
        if SITE_DISTANCE_V2_FINAL_RE.fullmatch(entry.name) is None:
            continue
        candidates.append(entry.name)
        try:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                unsafe = True
        except OSError:
            unsafe = True
    candidates.sort()
    count = min(len(candidates), SITE_DISTANCE_V2_MAX_FINAL_CANDIDATES + 1)
    if len(candidates) > SITE_DISTANCE_V2_MAX_FINAL_CANDIDATES:
        artifact = ArtifactRead(
            spec,
            "invalid_shape",
            f"distance-25m/{candidates[0]}",
        )
        reader._cache[spec.key] = artifact
        return artifact, count, ["final_candidate_limit_exceeded"]
    if len(candidates) != 1:
        if not candidates:
            return reader.read("site_distance_evaluation_v2"), 0, []
        artifact = ArtifactRead(
            spec,
            "invalid_shape",
            f"distance-25m/{candidates[0]}",
        )
        reader._cache[spec.key] = artifact
        return artifact, count, ["multiple_final_candidates"]
    relative = f"distance-25m/{candidates[0]}"
    selected_spec = ArtifactSpec(
        key=spec.key,
        label=spec.label,
        candidates=(relative,),
        schema_prefix=spec.schema_prefix,
        raw_download_allowed=False,
    )
    if unsafe:
        artifact = ArtifactRead(selected_spec, "unsafe_path", relative)
        reader._cache[spec.key] = artifact
        return artifact, 1, ["final_candidate_unsafe"]
    artifact = reader._read_spec(selected_spec)
    reader._cache[spec.key] = artifact
    return artifact, 1, []


def _site_distance_v2_input_paths(
    reader: ArtifactReader, payload: dict[str, Any]
) -> dict[str, Path] | None:
    integrity = payload.get("integrity")
    ledger = integrity.get("input_ledger") if isinstance(integrity, dict) else None
    if not isinstance(ledger, list):
        return None
    resolved: dict[str, Path] = {}
    for row in ledger:
        if not isinstance(row, dict) or set(row) != {"name", "pin"}:
            return None
        name = row.get("name")
        if not isinstance(name, str) or name in resolved:
            return None
        path = reader.verify_workspace_pin(row.get("pin"))
        if path is None:
            return None
        resolved[name] = path
    return resolved


def _site_distance_v2_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _site_distance_v2_fixture_marker(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and SITE_DISTANCE_V2_FIXTURE_RE.search(value.strip()) is not None
    )


def _site_distance_v2_production_contract(
    reader: ArtifactReader, payload: dict[str, Any]
) -> tuple[bool, dict[str, Any], dict[str, dict[str, Any]]]:
    projection = {
        "workspace_scoped_inputs": False,
        "fixture_markers_absent": False,
        "minimum_source_bytes": SITE_DISTANCE_V2_MINIMUM_SOURCE_BYTES,
        "minimum_per_bin_instances": SITE_DISTANCE_V2_MINIMUM_BIN_INSTANCES,
        "minimum_per_bin_independent_events": SITE_DISTANCE_V2_MINIMUM_BIN_EVENTS,
        "minimum_per_bin_unambiguous_events": (
            SITE_DISTANCE_V2_MINIMUM_UNAMBIGUOUS_EVENTS
        ),
        "minimum_endpoint_independent_events": (
            SITE_DISTANCE_V2_MINIMUM_ENDPOINT_EVENTS
        ),
        "minimum_exact_25m_instances": SITE_DISTANCE_V2_MINIMUM_EXACT_25_INSTANCES,
    }
    paths = _site_distance_v2_input_paths(reader, payload)
    if paths is None:
        return False, projection, {}
    required = {
        "acceptance",
        "camera_configuration",
        "ground_truth",
        "media_frame_ledger",
        "profile_640_manifest",
        "profile_960_manifest",
    }
    if not required <= set(paths):
        return False, projection, {}
    allowed = (
        reader.workspace_root / "validation/inputs/distance-25m",
        reader.workspace_root / "validation/results/distance-25m",
    )
    operational = {
        "acceptance",
        "calibration",
        "camera_configuration",
        "ground_truth",
        "media_frame_ledger",
        "preflight_receipt",
        "profile_640_manifest",
        "profile_640_predictions",
        "profile_960_manifest",
        "profile_960_predictions",
        "profile_pair_receipt",
    }
    for name in operational:
        path = paths.get(name)
        if path is None or not any(path == root or root in path.parents for root in allowed):
            return False, projection, {}
        try:
            relative = path.relative_to(reader.workspace_root)
        except ValueError:
            return False, projection, {}
        if any(_site_distance_v2_fixture_marker(part) for part in relative.parts):
            return False, projection, {}
    projection["workspace_scoped_inputs"] = True
    documents = {name: _site_distance_v2_json(paths[name]) for name in required}
    if any(value is None for value in documents.values()):
        return False, projection, {}
    typed = {name: value for name, value in documents.items() if isinstance(value, dict)}
    camera = typed["camera_configuration"]
    media = typed["media_frame_ledger"]
    ground_truth = typed["ground_truth"]
    identifiers = [
        camera.get("site_id"),
        camera.get("camera_id"),
        camera.get("camera", {}).get("manufacturer")
        if isinstance(camera.get("camera"), dict)
        else None,
        camera.get("camera", {}).get("model")
        if isinstance(camera.get("camera"), dict)
        else None,
        camera.get("camera", {}).get("serial_number")
        if isinstance(camera.get("camera"), dict)
        else None,
        media.get("dataset_id"),
        media.get("sequence_id"),
        ground_truth.get("dataset_id"),
        ground_truth.get("sequence_id"),
    ]
    if any(_site_distance_v2_fixture_marker(value) for value in identifiers):
        return False, projection, typed
    source_pin = media.get("source_asset")
    source_path = reader.verify_workspace_pin(source_pin)
    source_bytes = _integer(source_pin.get("bytes")) if isinstance(source_pin, dict) else None
    source_roots = (
        reader.workspace_root / "validation/inputs/distance-25m",
        reader.workspace_root / "data",
    )
    if (
        source_path is None
        or source_bytes is None
        or source_bytes < SITE_DISTANCE_V2_MINIMUM_SOURCE_BYTES
        or not any(source_path == root or root in source_path.parents for root in source_roots)
        or _site_distance_v2_fixture_marker(source_path.name)
    ):
        return False, projection, typed
    quota = payload.get("quota_recomputation")
    rows = quota.get("bins") if isinstance(quota, dict) else None
    endpoint = quota.get("endpoint") if isinstance(quota, dict) else None
    quota_valid = bool(
        isinstance(quota, dict)
        and quota.get("status") == "pass"
        and quota.get("boundary_policy") == SITE_DISTANCE_V2_BOUNDARY_POLICY
        and quota.get("matches_preflight_receipt") is True
        and isinstance(rows, list)
        and [row.get("bin_id") for row in rows if isinstance(row, dict)]
        == list(SITE_DISTANCE_V2_BIN_IDS)
        and all(
            isinstance(row, dict)
            and row.get("status") == "pass"
            and (_integer(row.get("instances")) or 0)
            >= SITE_DISTANCE_V2_MINIMUM_BIN_INSTANCES
            and (_integer(row.get("independent_events")) or 0)
            >= SITE_DISTANCE_V2_MINIMUM_BIN_EVENTS
            and (_integer(row.get("unambiguous_independent_events")) or 0)
            >= SITE_DISTANCE_V2_MINIMUM_UNAMBIGUOUS_EVENTS
            for row in rows
        )
        and isinstance(endpoint, dict)
        and endpoint.get("status") == "pass"
        and endpoint.get("maximum_nominal_distance_inclusive_m") == 25
        and (_integer(endpoint.get("independent_events")) or 0)
        >= SITE_DISTANCE_V2_MINIMUM_ENDPOINT_EVENTS
        and (_integer(quota.get("exact_25m_instances")) or 0)
        >= SITE_DISTANCE_V2_MINIMUM_EXACT_25_INSTANCES
    )
    if not quota_valid:
        return False, projection, typed
    projection["fixture_markers_absent"] = True
    return True, projection, typed


def _site_distance_v2_metrics(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    integers = {
        key: _integer(value.get(key))
        for key in ("ground_truth", "tp", "fp", "fn", "ignored_predictions")
    }
    numbers = {
        key: _number(value.get(key))
        for key in ("precision", "recall", "f1", "ap_101_point")
    }
    if (
        any(item is None for item in integers.values())
        or any(item is None or not 0 <= item <= 1 for item in numbers.values())
        or integers["tp"] + integers["fn"] != integers["ground_truth"]
    ):
        return None
    return {**integers, **numbers}


def _site_distance_v2_projection(
    payload: dict[str, Any], documents: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    raw_profiles = payload.get("profiles")
    quota = payload.get("quota_recomputation")
    media = documents.get("media_frame_ledger", {})
    frames = media.get("frames") if isinstance(media.get("frames"), list) else []
    if not isinstance(raw_profiles, dict) or set(raw_profiles) != {"640", "960"}:
        return None
    profiles: dict[str, Any] = {}
    for profile in (640, 960):
        raw = raw_profiles.get(str(profile))
        manifest = documents.get(f"profile_{profile}_manifest", {})
        overall = _site_distance_v2_metrics(raw.get("overall") if isinstance(raw, dict) else None)
        frame_records = _integer(
            manifest.get("frame_contract", {}).get("serialized_frames")
            if isinstance(manifest.get("frame_contract"), dict)
            else None
        )
        invariants = manifest.get("cross_profile_invariants")
        model_id = _identifier(
            invariants.get("base_model_id") if isinstance(invariants, dict) else None,
            maximum=80,
        )
        prediction_filter = raw.get("prediction_filter") if isinstance(raw, dict) else None
        outside = _integer(
            prediction_filter.get("excluded_outside_inclusive_20_25m")
            if isinstance(prediction_filter, dict)
            else None
        )
        if (
            not isinstance(raw, dict)
            or raw.get("profile") != profile
            or raw.get("status") != "pass"
            or overall is None
            or frame_records is None
            or frame_records != len(frames)
            or model_id is None
            or outside is None
        ):
            return None
        profiles[str(profile)] = {
            "model_input": profile,
            "model_id": model_id,
            "frame_records": frame_records,
            "ground_truth_instances": overall["ground_truth"],
            "metrics": {
                **{key: overall[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1", "ap_101_point", "ignored_predictions")},
                "predictions_excluded_outside_calibrated_band": outside,
            },
        }
    rows = quota.get("bins") if isinstance(quota, dict) else None
    if not isinstance(rows, list):
        return None
    coverage = {row.get("bin_id"): row for row in rows if isinstance(row, dict)}
    bins: list[dict[str, Any]] = []
    for bin_id in SITE_DISTANCE_V2_BIN_IDS:
        cover = coverage.get(bin_id)
        if not isinstance(cover, dict):
            return None
        per_profile: dict[str, Any] = {}
        for profile in (640, 960):
            raw_rows = raw_profiles[str(profile)].get("bins")
            matches = [
                row for row in raw_rows
                if isinstance(raw_rows, list)
                and isinstance(row, dict)
                and row.get("bin_id") == bin_id
            ] if isinstance(raw_rows, list) else []
            if len(matches) != 1:
                return None
            metrics = _site_distance_v2_metrics(matches[0].get("metrics"))
            if metrics is None:
                return None
            per_profile[str(profile)] = metrics
        row = {
            "bin_id": bin_id,
            "instances": _integer(cover.get("instances")),
            "independent_events": _integer(cover.get("independent_events")),
            "unambiguous_independent_events": _integer(
                cover.get("unambiguous_independent_events")
            ),
            "ambiguous_instances": _integer(cover.get("ambiguous_instances")),
            "status": _enum(cover.get("status"), {"pass"}),
            "profiles": per_profile,
        }
        if any(row[key] is None for key in (
            "instances", "independent_events", "unambiguous_independent_events",
            "ambiguous_instances", "status"
        )):
            return None
        bins.append(row)
    return profiles, bins


def _site_distance_v2_report_projection(
    value: Any, profile: int
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        return None
    projected_metrics = {
        key: (
            _integer(metrics.get(key))
            if key in {"tp", "fp", "fn", "ignored_predictions", "predictions_excluded_outside_calibrated_band"}
            else _number(metrics.get(key))
        )
        for key in (
            "tp", "fp", "fn", "precision", "recall", "f1", "ap_101_point",
            "ignored_predictions", "predictions_excluded_outside_calibrated_band",
        )
    }
    result = {
        "model_input": profile,
        "model_id": _identifier(value.get("model_id"), maximum=80),
        "frame_records": _integer(value.get("frame_records")),
        "ground_truth_instances": _integer(value.get("ground_truth_instances")),
        "metrics": projected_metrics,
    }
    if any(item is None for item in (
        result["model_id"], result["frame_records"], result["ground_truth_instances"],
        *projected_metrics.values(),
    )):
        return None
    return result


def _site_distance_v2_report_proves(
    reader: ArtifactReader,
    artifact: ArtifactRead,
    profiles: dict[str, Any],
    bins: list[dict[str, Any]],
    exact_25m_instances: int,
) -> bool:
    report_artifact = reader.read("campaign_report_json")
    if (
        not artifact.available
        or artifact.content is None
        or not report_artifact.available
        or report_artifact.value is None
        or not reader.validates_schema(report_artifact.value, CAMPAIGN_REPORT_SCHEMA)
    ):
        return False
    report = report_artifact.value
    campaigns = report.get("campaigns") if isinstance(report.get("campaigns"), dict) else {}
    distance = campaigns.get("distance_25m") if isinstance(campaigns.get("distance_25m"), dict) else {}
    selection = distance.get("selection") if isinstance(distance.get("selection"), dict) else {}
    contract = distance.get("contract") if isinstance(distance.get("contract"), dict) else {}
    production = (
        distance.get("production_evidence_contract")
        if isinstance(distance.get("production_evidence_contract"), dict)
        else {}
    )
    report_profiles = distance.get("profiles") if isinstance(distance.get("profiles"), dict) else {}
    projected_report_profiles = {
        key: _site_distance_v2_report_projection(value, int(key))
        for key, value in report_profiles.items()
        if key in {"640", "960"}
    }
    evidence = _report_evidence_by_id(report)
    evidence_row = evidence.get("distance_25m_evaluation_v2_final") if evidence else None
    if not (
        isinstance(evidence_row, dict)
        and evidence_row.get("state") == "ok"
        and evidence_row.get("media_type") == "application/json"
        and evidence_row.get("sha256") == hashlib.sha256(artifact.content).hexdigest()
        and evidence_row.get("size_bytes") == len(artifact.content)
        and distance.get("evidence_kind") == "calibrated_distance_ground_truth"
        and distance.get("evidence_version") == "inclusive_v2"
        and distance.get("state") == "proven"
        and distance.get("accepted") is True
        and distance.get("schema_contract_valid") is True
        and distance.get("pin_matrix_valid") is True
        and distance.get("independent_cpu_recomputation_valid") is True
        and distance.get("production_evidence_contract_valid") is True
        and selection == {
            "state": "inclusive_v2",
            "legacy_v1_present": False,
            "inclusive_v2_final_candidates": 1,
            "conflict": False,
        }
        and contract.get("required_schema") == SITE_DISTANCE_V2_SCHEMA
        and contract.get("boundary_policy") == SITE_DISTANCE_V2_BOUNDARY_POLICY
        and contract.get("required_profiles") == [640, 960]
        and contract.get("requires_production_evidence_contract") is True
        and production.get("workspace_scoped_inputs") is True
        and production.get("fixture_markers_absent") is True
        and production.get("minimum_source_bytes")
        == SITE_DISTANCE_V2_MINIMUM_SOURCE_BYTES
        and production.get("minimum_per_bin_instances")
        == SITE_DISTANCE_V2_MINIMUM_BIN_INSTANCES
        and production.get("minimum_per_bin_independent_events")
        == SITE_DISTANCE_V2_MINIMUM_BIN_EVENTS
        and production.get("minimum_per_bin_unambiguous_events")
        == SITE_DISTANCE_V2_MINIMUM_UNAMBIGUOUS_EVENTS
        and production.get("minimum_endpoint_independent_events")
        == SITE_DISTANCE_V2_MINIMUM_ENDPOINT_EVENTS
        and production.get("minimum_exact_25m_instances")
        == SITE_DISTANCE_V2_MINIMUM_EXACT_25_INSTANCES
        and set(distance.get("evidence_ids", []))
        == {"distance_25m_evaluation_v2_final"}
        and projected_report_profiles == profiles
        and distance.get("distance_bins") == bins
        and _integer(distance.get("exact_25m_instances")) == exact_25m_instances
    ):
        return False
    return True


def _site_distance_25m_v2(
    reader: ArtifactReader,
    plan_artifact: ArtifactRead,
    plan: dict[str, Any],
    plan_valid: bool,
    ready_inputs: int,
    legacy_artifact: ArtifactRead,
    artifact: ArtifactRead,
    candidate_count: int,
    selection_errors: list[str],
) -> dict[str, Any]:
    reasons = list(selection_errors)
    legacy_present = legacy_artifact.state != "missing"
    if legacy_present:
        reasons.append("legacy_v1_conflict")
    replay_valid = False
    production_valid = False
    production = {
        "workspace_scoped_inputs": False,
        "fixture_markers_absent": False,
        "minimum_source_bytes": SITE_DISTANCE_V2_MINIMUM_SOURCE_BYTES,
        "minimum_per_bin_instances": SITE_DISTANCE_V2_MINIMUM_BIN_INSTANCES,
        "minimum_per_bin_independent_events": SITE_DISTANCE_V2_MINIMUM_BIN_EVENTS,
        "minimum_per_bin_unambiguous_events": SITE_DISTANCE_V2_MINIMUM_UNAMBIGUOUS_EVENTS,
        "minimum_endpoint_independent_events": SITE_DISTANCE_V2_MINIMUM_ENDPOINT_EVENTS,
        "minimum_exact_25m_instances": SITE_DISTANCE_V2_MINIMUM_EXACT_25_INSTANCES,
    }
    profiles: dict[str, Any] = {}
    bins: list[dict[str, Any]] = []
    exact_25m_instances: int | None = None
    value = artifact.value or {}
    if artifact.available and not reasons and candidate_count == 1:
        try:
            from validation.site_distance_evaluator_v2 import verify_evaluation_receipt

            receipt_path = (reader.root / artifact.relative_path).resolve(strict=True)
            verified = verify_evaluation_receipt(receipt_path)
            replay_valid = verified == value
        except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError):
            reasons.append("live_semantic_replay_failed")
        if replay_valid:
            production_valid, production, documents = (
                _site_distance_v2_production_contract(reader, value)
            )
            if not production_valid:
                reasons.append("production_evidence_contract_failed")
            projection = _site_distance_v2_projection(value, documents)
            if projection is None:
                reasons.append("public_metric_projection_failed")
            else:
                profiles, bins = projection
                exact_25m_instances = _integer(
                    value.get("quota_recomputation", {}).get("exact_25m_instances")
                    if isinstance(value.get("quota_recomputation"), dict)
                    else None
                )
    elif not artifact.available:
        reasons.append("final_receipt_unavailable")
    final_contract = bool(
        value.get("schema_version") == SITE_DISTANCE_V2_SCHEMA
        and value.get("status") == "complete"
        and value.get("acceptance_status") == "pass"
        and isinstance(value.get("paired_acceptance"), dict)
        and value["paired_acceptance"].get("status") == "pass"
        and value["paired_acceptance"].get("required_profiles") == [640, 960]
        and value["paired_acceptance"].get("both_profiles_required_to_pass") is True
        and isinstance(value.get("scope"), dict)
        and value["scope"].get("minimum_inclusive_m") == 20
        and value["scope"].get("maximum_inclusive_m") == 25
        and value["scope"].get("boundary_policy") == SITE_DISTANCE_V2_BOUNDARY_POLICY
        and isinstance(value.get("quota_recomputation"), dict)
        and value["quota_recomputation"].get("status") == "pass"
        and value["quota_recomputation"].get("matches_preflight_receipt") is True
    )
    if artifact.available and not final_contract:
        reasons.append("final_acceptance_contract_failed")
    report_bound = bool(
        replay_valid
        and production_valid
        and exact_25m_instances is not None
        and _site_distance_v2_report_proves(
            reader, artifact, profiles, bins, exact_25m_instances
        )
    )
    reader._cache.pop("site_distance_evaluation_v2", None)
    artifact_after, candidate_count_after, selection_errors_after = (
        _site_distance_v2_selection(reader)
    )
    selection_stable = bool(
        candidate_count_after == candidate_count
        and selection_errors_after == selection_errors
        and artifact_after.relative_path == artifact.relative_path
        and artifact_after.state == artifact.state
        and artifact_after.content == artifact.content
    )
    try:
        legacy_present_after = os.path.lexists(
            reader.root / "distance-25m/evaluation.json"
        )
    except OSError:
        legacy_present_after = True
    if not selection_stable or legacy_present_after != legacy_present:
        reasons.append("final_selection_changed_during_replay")
    proven = bool(
        not reasons
        and candidate_count == 1
        and not legacy_present
        and replay_valid
        and production_valid
        and final_contract
        and report_bound
        and selection_stable
    )
    if proven:
        state = "proven"
    elif candidate_count or legacy_present or artifact.state != "missing":
        state = "unproven"
    elif plan_artifact.available and plan_valid:
        state = "waiting_for_inputs"
    elif plan_artifact.state == "missing":
        state = "not_started"
    else:
        state = "artifact_error"
    return {
        "label": "Saha kalibrasyonlu inclusive 20–25 m insan algılama",
        "available": plan_artifact.available or artifact.available or legacy_present,
        "state": state,
        "updated_at_utc": None,
        "accepted": proven,
        "proven": proven,
        "final_claim_allowed": proven,
        "evidence_version": "inclusive_v2",
        "final_evaluation_present": artifact.available,
        "selection": {
            "legacy_v1_present": legacy_present,
            "inclusive_v2_final_candidates": candidate_count,
            "conflict": legacy_present or candidate_count != 1,
        },
        "progress": _progress(2 if proven else 0, 2),
        "input_readiness": {
            "ready": ready_inputs if plan_valid else 0,
            "required": 5,
            "plan_contract_valid": plan_valid,
        },
        "scope": {
            "model_input_sizes": [640, 960],
            "distance_m": {"minimum_inclusive": 20, "maximum_inclusive": 25},
            "boundary": SITE_DISTANCE_V2_BOUNDARY_POLICY,
            "requires_verified_deployment_camera_calibration": True,
        },
        "calibration_verified": proven,
        "ground_truth_instances_20_25m": (
            profiles.get("640", {}).get("ground_truth_instances") if proven else None
        ),
        "exact_25m_instances": exact_25m_instances if proven else None,
        "criterion_id": "owner_approved_paired_rules_v2" if proven else None,
        "profiles": profiles if proven else {},
        "distance_bins": bins if proven else [],
        "production_evidence_contract": production,
        "metric_context": {
            "ground_truth": proven,
            "site_calibrated": proven,
            "loaf_can_substitute": False,
            "metric": "AP101@IoU0.5" if proven else None,
            "inclusive_exact_25m": proven,
        },
        "safety": {
            "plan_dry_run": _boolean(plan.get("dry_run")),
            "plan_external_execution": _boolean(plan.get("gpu_or_docker_executed")),
            "cpu_evaluator_external_execution": False if replay_valid else None,
        },
        "verification": {
            "live_semantic_replay": replay_valid,
            "production_evidence_contract": production_valid,
            "campaign_report_hash_binding": report_bound,
            "failure_codes": sorted(set(reasons))[:16],
        },
        "evidence": _evidence(
            reader,
            "site_distance_plan",
            "site_distance_evaluation",
            "site_distance_evaluation_v2",
        ),
    }


def _site_distance_25m(reader: ArtifactReader) -> dict[str, Any]:
    plan_artifact = reader.read("site_distance_plan")
    evaluation_artifact = reader.read("site_distance_evaluation")
    plan = plan_artifact.value or {}
    evaluation = evaluation_artifact.value or {}
    plan_valid, plan_status, ready_inputs = _site_distance_plan_contract(reader, plan)
    v2_artifact, v2_candidate_count, v2_selection_errors = (
        _site_distance_v2_selection(reader)
    )
    if v2_candidate_count or v2_selection_errors:
        return _site_distance_25m_v2(
            reader,
            plan_artifact,
            plan,
            plan_valid,
            ready_inputs,
            evaluation_artifact,
            v2_artifact,
            v2_candidate_count,
            v2_selection_errors,
        )
    evaluation_schema_valid = bool(
        evaluation_artifact.available
        and reader.validates_schema(evaluation, SITE_EVALUATION_SCHEMA)
    )

    raw_profiles = (
        evaluation.get("profiles")
        if evaluation_schema_valid and isinstance(evaluation.get("profiles"), dict)
        else {}
    )
    profiles: dict[str, dict[str, Any]] = {}
    for profile in (640, 960):
        projected = _site_distance_profile(raw_profiles.get(str(profile)), profile)
        if projected is not None:
            profiles[str(profile)] = projected
    calibration = evaluation.get("calibration") if isinstance(evaluation.get("calibration"), dict) else {}
    ground_truth = evaluation.get("ground_truth") if isinstance(evaluation.get("ground_truth"), dict) else {}
    acceptance = evaluation.get("acceptance") if isinstance(evaluation.get("acceptance"), dict) else {}
    integrity = evaluation.get("integrity") if isinstance(evaluation.get("integrity"), dict) else {}
    loaf = evaluation.get("loaf_evidence") if isinstance(evaluation.get("loaf_evidence"), dict) else {}
    config = evaluation.get("evaluation_config") if isinstance(evaluation.get("evaluation_config"), dict) else {}
    rules = acceptance.get("rules") if isinstance(acceptance.get("rules"), list) else []
    evaluation_contract_valid = bool(
        evaluation_schema_valid
        and evaluation.get("schema_version") == "deepsafe.distance-validation/v1"
        and evaluation.get("status") == "complete"
        and evaluation.get("evidence_kind")
        == "deployment_site_calibrated_ground_plane_person_detection"
        and evaluation.get("distance_unit") == "m"
        and evaluation.get("distance_bin_m") == [20, 25]
        and evaluation.get("boundary") == "lower_inclusive_upper_exclusive"
        and calibration.get("status") == "verified"
        and calibration.get("model") == "planar_homography_image_to_ground"
        and ground_truth.get("status") == "complete"
        and ground_truth.get("evidence_kind")
        == "deployment_site_calibrated_person_ground_truth"
        and _integer(ground_truth.get("ground_truth_instances_20_25m")) is not None
        and set(profiles) == {"640", "960"}
        and profiles["640"]["ground_truth_instances"]
        == ground_truth.get("ground_truth_instances_20_25m")
        and profiles["960"]["ground_truth_instances"]
        == ground_truth.get("ground_truth_instances_20_25m")
        and config.get("distance_point") == "bbox_bottom_center_ground_contact"
        and config.get("metric_geometry") == "axis_aligned_bbox_iou"
        and acceptance.get("status") == "pass"
        and _identifier(acceptance.get("criterion_id"), maximum=80) is not None
        and rules
        and all(isinstance(rule, dict) and rule.get("status") == "pass" for rule in rules)
        and integrity.get("frame_set_status")
        == "exact_for_ground_truth_and_both_profiles"
        and integrity.get("profile_pair_status") == "complete_640_and_960"
        and loaf
        == {
            "used": False,
            "role": "auxiliary_only_not_deployment_site_calibration",
            "can_substitute_for_site_calibration": False,
        }
        and evaluation.get("gpu_or_docker_executed_by_evaluator") is False
    )
    live_pins = (
        _site_distance_live_pins(reader, evaluation)
        if evaluation_contract_valid
        else None
    )
    proven = bool(
        evaluation_contract_valid
        and live_pins is not None
        and _site_distance_report_proves(
            reader,
            evaluation_artifact,
            evaluation,
            profiles,
            live_pins,
        )
    )
    if proven:
        state = "proven"
    elif evaluation_artifact.available:
        state = "unproven"
    elif evaluation_artifact.state != "missing":
        state = "artifact_error"
    elif plan_artifact.available and plan_valid:
        state = plan_status or "artifact_error"
    elif plan_artifact.state == "missing":
        state = "not_started"
    else:
        state = "artifact_error"
    awaiting_v2 = evaluation_artifact.state == "missing"
    return {
        "label": "Saha kalibrasyonlu 20–25 m insan algılama",
        "available": plan_artifact.available or evaluation_artifact.available,
        "state": state,
        "updated_at_utc": _timestamp(evaluation.get("generated_at")) if proven else None,
        "accepted": proven,
        "proven": proven,
        "final_claim_allowed": proven,
        "evidence_version": "legacy_v1" if evaluation_artifact.available else "none",
        "final_evaluation_present": evaluation_artifact.available,
        "selection": {
            "legacy_v1_present": evaluation_artifact.state != "missing",
            "inclusive_v2_final_candidates": 0,
            "conflict": False,
        },
        "progress": _progress(2 if proven else 0, 2),
        "input_readiness": {
            "ready": ready_inputs if plan_valid else 0,
            "required": 5,
            "plan_contract_valid": plan_valid,
        },
        "scope": {
            "model_input_sizes": [640, 960],
            "distance_m": (
                {"minimum_inclusive": 20, "maximum_inclusive": 25}
                if awaiting_v2
                else {"minimum_inclusive": 20, "maximum_exclusive": 25}
            ),
            "boundary": (
                SITE_DISTANCE_V2_BOUNDARY_POLICY
                if awaiting_v2
                else "lower_inclusive_upper_exclusive"
            ),
            "requires_verified_deployment_camera_calibration": True,
        },
        "calibration_verified": proven,
        "ground_truth_instances_20_25m": (
            _integer(ground_truth.get("ground_truth_instances_20_25m")) if proven else None
        ),
        "exact_25m_instances": None,
        "criterion_id": (
            _identifier(acceptance.get("criterion_id"), maximum=80) if proven else None
        ),
        "profiles": profiles if proven else {},
        "distance_bins": [],
        "production_evidence_contract": {
            "workspace_scoped_inputs": proven,
            "fixture_markers_absent": proven,
            "minimum_source_bytes": (
                SITE_DISTANCE_V2_MINIMUM_SOURCE_BYTES if awaiting_v2 else None
            ),
            "minimum_per_bin_instances": (
                SITE_DISTANCE_V2_MINIMUM_BIN_INSTANCES if awaiting_v2 else None
            ),
            "minimum_per_bin_independent_events": (
                SITE_DISTANCE_V2_MINIMUM_BIN_EVENTS if awaiting_v2 else None
            ),
            "minimum_per_bin_unambiguous_events": (
                SITE_DISTANCE_V2_MINIMUM_UNAMBIGUOUS_EVENTS
                if awaiting_v2
                else None
            ),
            "minimum_endpoint_independent_events": (
                SITE_DISTANCE_V2_MINIMUM_ENDPOINT_EVENTS if awaiting_v2 else None
            ),
            "minimum_exact_25m_instances": (
                SITE_DISTANCE_V2_MINIMUM_EXACT_25_INSTANCES
                if awaiting_v2
                else None
            ),
        },
        "metric_context": {
            "ground_truth": proven,
            "site_calibrated": proven,
            "loaf_can_substitute": False,
            "metric": "AP101@IoU0.5" if proven else None,
            "inclusive_exact_25m": awaiting_v2,
        },
        "safety": {
            "plan_dry_run": _boolean(plan.get("dry_run")),
            "plan_external_execution": _boolean(
                plan.get("gpu_or_docker_executed")
            ),
            "cpu_evaluator_external_execution": (
                _boolean(evaluation.get("gpu_or_docker_executed_by_evaluator"))
                if evaluation_artifact.available
                else None
            ),
        },
        "verification": {
            "live_semantic_replay": proven,
            "production_evidence_contract": proven,
            "campaign_report_hash_binding": proven,
            "failure_codes": (
                []
                if proven
                else ["awaiting_inclusive_v2_final"]
                if awaiting_v2
                else ["legacy_v1_unproven"]
            ),
        },
        "evidence": _evidence(
            reader,
            "site_distance_plan",
            "site_distance_evaluation",
            "site_distance_evaluation_v2",
        ),
    }


def _finalization_projection(
    *, available: bool, state: str, committed: bool, reason: str | None
) -> dict[str, Any]:
    return {
        "label": "Doğrulama bundle commit",
        "available": available,
        "state": state,
        "committed": committed,
        "reason": reason,
        "read_only": True,
        "raw_download_allowed": False,
        "verified_output_count": 0,
        "output_count": len(FINALIZATION_OUTPUTS),
    }


def _canonical_sha256(value: Any) -> str | None:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _receipt_pin_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        return False
    path = value.get("path")
    size = value.get("size_bytes")
    return bool(
        isinstance(path, str)
        and 1 <= len(path) <= 512
        and not PurePosixPath(path).is_absolute()
        and ".." not in PurePosixPath(path).parts
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 1 <= size <= MAX_PINNED_FILE_BYTES
        and _sha256(value.get("sha256")) is not None
    )


def _completion_identity_valid(receipt: dict[str, Any]) -> bool:
    identity = receipt.get("completion_identity")
    inputs = receipt.get("inputs")
    if not isinstance(identity, dict) or set(identity) != {
        "campaign_name",
        "config_fingerprint",
        "static_input_fingerprint",
        "started_at_utc",
        "updated_at_utc",
        "finished_at_utc",
        "target_validated_seconds",
        "validated_seconds",
        "total_attempt_receipt_count",
    }:
        return False
    if (
        not isinstance(inputs, list)
        or not inputs
        or not all(_receipt_pin_valid(item) for item in inputs)
        or len({item["path"] for item in inputs}) != len(inputs)
    ):
        return False
    attempts = identity.get("total_attempt_receipt_count")
    identity_payload = {
        "completion_identity": identity,
        "inputs": inputs,
    }
    if (
        identity.get("campaign_name") != FINALIZATION_CAMPAIGN_NAME
        or _sha256(identity.get("config_fingerprint")) is None
        or _sha256(identity.get("static_input_fingerprint")) is None
        or identity.get("target_validated_seconds") != FINALIZATION_TARGET_SECONDS
        or identity.get("validated_seconds") != FINALIZATION_TARGET_SECONDS
        or isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < FINALIZATION_SEGMENT_COUNT
        or receipt.get("finalization_identity_sha256")
        != _canonical_sha256(identity_payload)
    ):
        return False
    try:
        started = datetime.fromisoformat(identity["started_at_utc"].replace("Z", "+00:00"))
        finished = datetime.fromisoformat(identity["finished_at_utc"].replace("Z", "+00:00"))
        updated = datetime.fromisoformat(identity["updated_at_utc"].replace("Z", "+00:00"))
        completed = datetime.fromisoformat(receipt["completed_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return bool(
        all(item.tzinfo is not None for item in (started, finished, updated, completed))
        and started <= finished <= updated <= completed
    )


def _finalization_input_contract_valid(receipt: dict[str, Any]) -> bool:
    inputs = receipt.get("inputs")
    return bool(
        isinstance(inputs, list)
        and tuple(
            item.get("path") if isinstance(item, dict) else None
            for item in inputs
        )
        == FINALIZATION_INPUT_PATHS
    )


def _finalization_attempt_receipt_total(checkpoint: dict[str, Any]) -> int | None:
    segments = checkpoint.get("segments")
    if not isinstance(segments, list) or len(segments) != FINALIZATION_SEGMENT_COUNT:
        return None
    total = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            return None
        segment_id = f"segment-{index:03d}-{640 if index % 2 == 0 else 960}"
        attempts = segment.get("attempts")
        receipts = segment.get("attempt_receipts")
        if (
            segment.get("segment_id") != segment_id
            or not isinstance(attempts, list)
            or not isinstance(receipts, list)
            or len(attempts) != len(receipts)
            or len(receipts) < 1
        ):
            return None
        for attempt_index, pin in enumerate(receipts, start=1):
            expected_path = (
                "validation/results/endurance/current/segments/"
                f"{segment_id}/attempt-{attempt_index:02d}/attempt-receipt.json"
            )
            if not _receipt_pin_valid(pin) or pin.get("path") != expected_path:
                return None
        total += len(receipts)
    return total


def _finalization_inputs_match_live(
    reader: ArtifactReader, receipt: dict[str, Any]
) -> bool:
    inputs = receipt.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(FINALIZATION_INPUT_PATHS):
        return False
    if any(
        not reader.verify_finalization_input_pin(pin, expected_path=expected_path)
        for pin, expected_path in zip(inputs, FINALIZATION_INPUT_PATHS, strict=True)
    ):
        return False

    live = _live_endurance_lineage(reader)
    identity = receipt.get("completion_identity")
    if live is None or not isinstance(identity, dict):
        return False
    checkpoint = live["artifacts"]["endurance_checkpoint"].value
    status = live["artifacts"]["endurance_status"].value
    if not isinstance(checkpoint, dict) or not isinstance(status, dict):
        return False
    progress = status.get("progress_fraction")
    if (
        any(
            document.get("state") != "complete"
            or document.get("dry_run") is not False
            or "active" not in document
            or document.get("active") is not None
            or document.get("target_validated_seconds")
            != FINALIZATION_TARGET_SECONDS
            or document.get("validated_seconds") != FINALIZATION_TARGET_SECONDS
            or document.get("campaign_health_gates") != []
            for document in (checkpoint, status)
        )
        or status.get("available") is not True
        or isinstance(progress, bool)
        or not isinstance(progress, (int, float))
        or not math.isfinite(float(progress))
        or float(progress) != 1.0
    ):
        return False

    identity_fields = (
        "campaign_name",
        "config_fingerprint",
        "static_input_fingerprint",
        "started_at_utc",
        "updated_at_utc",
        "finished_at_utc",
        "target_validated_seconds",
        "validated_seconds",
    )
    if any(
        checkpoint.get(field) != identity.get(field)
        or status.get(field) != identity.get(field)
        for field in identity_fields
    ):
        return False
    return (
        _finalization_attempt_receipt_total(checkpoint)
        == identity.get("total_attempt_receipt_count")
    )


def _finalization_receipt_shape_valid(receipt: dict[str, Any]) -> bool:
    if set(receipt) != {
        "schema_version",
        "state",
        "completed_at_utc",
        "finalization_identity_sha256",
        "completion_identity",
        "inputs",
        "outputs",
        "semantics",
        "generator_commands",
        "lock_contract",
        "fingerprint_sha256",
    }:
        return False
    inputs = receipt.get("inputs")
    commands = receipt.get("generator_commands")
    return bool(
        receipt.get("schema_version") == FINALIZATION_RECEIPT_SCHEMA
        and receipt.get("state") == "complete"
        and _timestamp(receipt.get("completed_at_utc")) is not None
        and _sha256(receipt.get("finalization_identity_sha256")) is not None
        and isinstance(inputs, list)
        and len(inputs) >= 1
        and all(_receipt_pin_valid(item) for item in inputs)
        and len({item["path"] for item in inputs}) == len(inputs)
        and isinstance(commands, list)
        and len(commands) == 3
        and all(
            isinstance(command, list)
            and command
            and len(command) <= 32
            and all(isinstance(part, str) and 1 <= len(part) <= 1024 for part in command)
            for command in commands
        )
        and receipt.get("lock_contract")
        == {
            "exclusive_nonblocking_finalizer_lock": True,
            "exclusive_nonblocking_supervisor_lock": True,
            "supervisor_lock_held_through_receipt_commit": True,
            "receipt_committed_last": True,
        }
        and _completion_identity_valid(receipt)
    )


def _finalization_semantics_valid(
    receipt: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> bool:
    semantics = receipt.get("semantics")
    if not isinstance(semantics, dict) or set(semantics) != {
        "campaign_endurance_accepted",
        "objective_evidence_complete",
        "objective_passed_gate_count",
        "product_snapshot_valid",
        "product_status",
        "product_ready_required",
        "campaign_overall_accepted",
        "objective_fingerprint_sha256",
        "product_fingerprint_sha256",
    }:
        return False
    campaign = reports.get("campaign_json", {})
    objective_report = reports.get("objective_json", {})
    product_report = reports.get("product_json", {})
    campaign_decision = campaign.get("decision")
    endurance = (
        campaign.get("campaigns", {}).get("endurance")
        if isinstance(campaign.get("campaigns"), dict)
        else None
    )
    objective = objective_report.get("objective")
    product_decision = product_report.get("decision")
    endurance_requirement = next(
        (
            item
            for item in campaign.get("requirements", [])
            if isinstance(item, dict) and item.get("id") == "seven_day_endurance"
        ),
        None,
    )
    product_status = semantics.get("product_status")
    expected_product_ready = product_status == "ready"
    return bool(
        semantics.get("campaign_endurance_accepted") is True
        and semantics.get("objective_evidence_complete") is True
        and semantics.get("objective_passed_gate_count") == 6
        and semantics.get("product_snapshot_valid") is True
        and product_status in {"ready", "not_ready"}
        and semantics.get("product_ready_required") is False
        and isinstance(semantics.get("campaign_overall_accepted"), bool)
        and _sha256(semantics.get("objective_fingerprint_sha256")) is not None
        and _sha256(semantics.get("product_fingerprint_sha256")) is not None
        and isinstance(campaign_decision, dict)
        and campaign_decision.get("accepted")
        is semantics.get("campaign_overall_accepted")
        and isinstance(endurance, dict)
        and endurance.get("accepted") is True
        and endurance.get("evidence_complete") is True
        and endurance.get("target_validated_seconds") == FINALIZATION_TARGET_SECONDS
        and endurance.get("reported_validated_seconds") == FINALIZATION_TARGET_SECONDS
        and endurance.get("expected_segments") == FINALIZATION_SEGMENT_COUNT
        and endurance.get("healthy_checkpoint_segments") == FINALIZATION_SEGMENT_COUNT
        and endurance.get("verified_attempt_receipts")
        == receipt["completion_identity"]["total_attempt_receipt_count"]
        and isinstance(endurance_requirement, dict)
        and endurance_requirement.get("state") == "pass"
        and isinstance(objective, dict)
        and objective.get("state") == "complete"
        and objective.get("evidence_complete") is True
        and objective.get("passed_gate_count") == 6
        and _canonical_fingerprint_matches(objective_report)
        and objective_report.get("fingerprint_sha256")
        == semantics.get("objective_fingerprint_sha256")
        and isinstance(product_decision, dict)
        and product_decision.get("status") == product_status
        and product_decision.get("ready") is expected_product_ready
        and product_decision.get("final_claim_allowed") is expected_product_ready
        and _canonical_fingerprint_matches(product_report)
        and product_report.get("fingerprint_sha256")
        == semantics.get("product_fingerprint_sha256")
    )


def _finalization_bundle(reader: ArtifactReader) -> dict[str, Any]:
    receipt_artifact = reader.read("finalization_receipt")
    if receipt_artifact.state == "missing":
        return _finalization_projection(
            available=False,
            state="pending",
            committed=False,
            reason="receipt_missing",
        )
    if not receipt_artifact.available or not isinstance(receipt_artifact.value, dict):
        return _finalization_projection(
            available=True,
            state="invalid",
            committed=False,
            reason=f"receipt_{receipt_artifact.state}",
        )
    receipt = receipt_artifact.value
    if not _finalization_receipt_shape_valid(receipt):
        return _finalization_projection(
            available=True,
            state="invalid",
            committed=False,
            reason="receipt_schema_invalid",
        )
    if not _canonical_fingerprint_matches(receipt):
        return _finalization_projection(
            available=True,
            state="invalid",
            committed=False,
            reason="receipt_fingerprint_invalid",
        )
    if not _finalization_input_contract_valid(receipt):
        return _finalization_projection(
            available=True,
            state="invalid",
            committed=False,
            reason="input_pin_contract_invalid",
        )
    if not _finalization_inputs_match_live(reader, receipt):
        return _finalization_projection(
            available=True,
            state="stale_lineage",
            committed=False,
            reason="stale_lineage",
        )
    pins = receipt.get("outputs")
    if not isinstance(pins, list) or len(pins) != len(FINALIZATION_OUTPUTS):
        return _finalization_projection(
            available=True,
            state="invalid",
            committed=False,
            reason="output_pin_contract_invalid",
        )
    reports: dict[str, dict[str, Any]] = {}
    for pin, (artifact_id, expected_path, media_type, artifact_key) in zip(
        pins, FINALIZATION_OUTPUTS, strict=True
    ):
        if (
            not isinstance(pin, dict)
            or set(pin) != {"id", "path", "size_bytes", "sha256", "media_type"}
            or pin.get("id") != artifact_id
            or pin.get("path") != expected_path
            or pin.get("media_type") != media_type
            or not _receipt_pin_valid(
                {key: pin.get(key) for key in ("path", "size_bytes", "sha256")}
            )
        ):
            return _finalization_projection(
                available=True,
                state="invalid",
                committed=False,
                reason="output_pin_contract_invalid",
            )
        artifact = reader.read(artifact_key)
        if (
            not artifact.available
            or artifact.content is None
            or f"validation/results/{artifact.relative_path}" != expected_path
            or len(artifact.content) != pin["size_bytes"]
            or hashlib.sha256(artifact.content).hexdigest() != pin["sha256"]
        ):
            return _finalization_projection(
                available=True,
                state="invalid",
                committed=False,
                reason="output_mismatch",
            )
        if media_type == "application/json":
            if not isinstance(artifact.value, dict):
                return _finalization_projection(
                    available=True,
                    state="invalid",
                    committed=False,
                    reason="output_semantics_invalid",
                )
            reports[artifact_id] = artifact.value
    if not _finalization_semantics_valid(receipt, reports):
        return _finalization_projection(
            available=True,
            state="invalid",
            committed=False,
            reason="output_semantics_invalid",
        )
    result = _finalization_projection(
        available=True,
        state="complete",
        committed=True,
        reason=None,
    )
    result.update(
        {
            "verified_output_count": len(FINALIZATION_OUTPUTS),
            "completed_at_utc": receipt.get("completed_at_utc"),
            "product_status": receipt["semantics"]["product_status"],
            "campaign_overall_accepted": receipt["semantics"][
                "campaign_overall_accepted"
            ],
        }
    )
    return result


def _product_v2_report_semantics_valid(
    reader: ArtifactReader,
    receipt: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> bool:
    semantics = receipt.get("semantics")
    objective_report = reports.get("objective_json", {})
    product_report = reports.get("product_json", {})
    objective = objective_report.get("objective")
    product_decision = product_report.get("decision")
    product_summary = product_report.get("summary")
    product_gates = product_report.get("required_gates")
    return bool(
        isinstance(semantics, dict)
        and semantics.get("objective_evidence_complete") is True
        and semantics.get("objective_passed_gate_count") == 6
        and semantics.get("product_status") == "ready"
        and semantics.get("product_ready_required") is True
        and semantics.get("all_six_product_gates_passed") is True
        and semantics.get("three_modules_enabled_together") is True
        and semantics.get("profiles") == [640, 960]
        and semantics.get("simulated_streams") == 12
        and semantics.get("minimum_elapsed_ms_per_profile") == 300000
        and semantics.get("human_visual_qa_bound") is True
        and semantics.get("physical_execution_proof_role")
        == "machine_local_hash_bound_evidence_not_external_attestation"
        and isinstance(objective, dict)
        and objective.get("state") == "complete"
        and objective.get("evidence_complete") is True
        and objective.get("passed_gate_count") == 6
        and _canonical_fingerprint_matches(objective_report)
        and reader.validates_schema(objective_report, OBJECTIVE_COMPLETION_SCHEMA)
        and objective_report.get("fingerprint_sha256")
        == semantics.get("objective_fingerprint_sha256")
        and isinstance(product_decision, dict)
        and product_decision.get("status") == "ready"
        and product_decision.get("ready") is True
        and product_decision.get("final_claim_allowed") is True
        and product_decision.get("failed_required_gate_ids") == []
        and isinstance(product_summary, dict)
        and product_summary.get("required_gate_count") == 6
        and product_summary.get("passed_required_gate_count") == 6
        and product_summary.get("remaining_required_gate_count") == 0
        and isinstance(product_gates, list)
        and len(product_gates) == 6
        and all(
            isinstance(gate, dict)
            and gate.get("state") == "pass"
            and gate.get("passed") is True
            for gate in product_gates
        )
        and _canonical_fingerprint_matches(product_report)
        and reader.validates_schema(product_report, PRODUCT_READINESS_SCHEMA)
        and product_report.get("fingerprint_sha256")
        == semantics.get("product_fingerprint_sha256")
    )


def _product_finalization_v2(reader: ArtifactReader) -> dict[str, Any]:
    """Bounded, read-only projection of the separate product commit marker."""

    artifact = reader.read("product_finalization_v2_receipt")
    base = {
        "label": "Üç modül ürün acceptance commit v2",
        "available": artifact.available,
        "state": "pending",
        "committed": False,
        "reason": "receipt_missing",
        "read_only": True,
        "execution_actions_available": False,
        "raw_download_allowed": False,
        "required_profiles": [640, 960],
        "required_modules": ["person", "pose", "ppe"],
        "required_streams": 12,
        "minimum_elapsed_ms_per_profile": 300000,
        "verified_input_count": 0,
        "committed_input_count": 0,
        "verified_output_count": 0,
        "output_count": 4,
        "product_status": None,
        "caveats": [
            "Bu v2 marker eski v1 receipt'i değiştirmez veya üzerine yazmaz.",
            "Eksik fiziksel koşu ya da insan QA kanıtı ready durumuna yükseltilemez.",
            "Admin yalnız sabit manifest/receipt/rapor pinlerini canlı doğrular; büyük model ve medya pinlerinin tam denetimi CPU-only CLI inspector'dadır.",
            "Hash zinciri harici kimlik veya donanım attestation'ı değildir.",
        ],
    }
    if artifact.state == "missing":
        return base
    if not artifact.available or not isinstance(artifact.value, dict):
        return {
            **base,
            "available": True,
            "state": "invalid",
            "reason": f"receipt_{artifact.state}",
        }
    receipt = artifact.value
    if not (
        receipt.get("schema_version") == PRODUCT_FINALIZATION_V2_SCHEMA
        and _canonical_fingerprint_matches(receipt)
        and reader.validates_schema(receipt, PRODUCT_FINALIZATION_V2_SCHEMA_FILE)
    ):
        return {
            **base,
            "available": True,
            "state": "invalid",
            "reason": "receipt_schema_invalid",
        }
    try:
        from validation.product_finalization_v2_contract import (
            FIXED_INPUT_PATH_STRINGS,
        )

        expected_fixed_paths = FIXED_INPUT_PATH_STRINGS
    except (AttributeError, ImportError, TypeError):
        return {
            **base,
            "available": True,
            "state": "invalid",
            "reason": "verifier_contract_unavailable",
        }
    inputs = receipt.get("inputs")
    lineage = receipt.get("lineage")
    if not (
        isinstance(inputs, list)
        and 30 <= len(inputs) <= 256
        and all(_receipt_pin_valid(pin) for pin in inputs)
        and [pin["path"] for pin in inputs] == sorted(pin["path"] for pin in inputs)
        and len({pin["path"] for pin in inputs}) == len(inputs)
        and isinstance(lineage, dict)
        and lineage.get("input_pin_count") == len(inputs)
        and receipt.get("lineage_identity_sha256")
        == _canonical_sha256({"lineage": lineage, "inputs": inputs})
    ):
        return {
            **base,
            "available": True,
            "state": "invalid",
            "reason": "input_pin_contract_invalid",
        }
    pins_by_path = {pin["path"]: pin for pin in inputs}
    if any(path not in pins_by_path for path in expected_fixed_paths):
        return {
            **base,
            "available": True,
            "state": "invalid",
            "reason": "fixed_input_pin_missing",
        }
    verified_fixed = 0
    for path in expected_fixed_paths:
        if not reader.verify_finalization_input_pin(
            pins_by_path[path], expected_path=path
        ):
            return {
                **base,
                "available": True,
                "state": "stale_lineage",
                "reason": "stale_lineage",
                "committed_input_count": len(inputs),
                "verified_input_count": verified_fixed,
            }
        verified_fixed += 1

    expected_outputs = (
        (
            "objective_json",
            "validation/results/objective-completion/current/report.json",
            "application/json",
            "objective_completion_json",
        ),
        (
            "objective_markdown",
            "validation/results/objective-completion/current/report.md",
            "text/markdown",
            "objective_completion_markdown",
        ),
        (
            "product_json",
            "validation/results/product-readiness/current/report.json",
            "application/json",
            "product_readiness_json",
        ),
        (
            "product_markdown",
            "validation/results/product-readiness/current/report.md",
            "text/markdown",
            "product_readiness_markdown",
        ),
    )
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected_outputs):
        return {
            **base,
            "available": True,
            "state": "invalid",
            "reason": "output_pin_contract_invalid",
            "committed_input_count": len(inputs),
            "verified_input_count": verified_fixed,
        }
    reports: dict[str, dict[str, Any]] = {}
    for pin, (artifact_id, expected_path, media_type, artifact_key) in zip(
        outputs, expected_outputs, strict=True
    ):
        core_pin = (
            {key: pin.get(key) for key in ("path", "size_bytes", "sha256")}
            if isinstance(pin, dict)
            else None
        )
        if not (
            isinstance(pin, dict)
            and set(pin) == {"id", "path", "size_bytes", "sha256", "media_type"}
            and pin.get("id") == artifact_id
            and pin.get("path") == expected_path
            and pin.get("media_type") == media_type
            and _receipt_pin_valid(core_pin)
        ):
            return {
                **base,
                "available": True,
                "state": "invalid",
                "reason": "output_pin_contract_invalid",
                "committed_input_count": len(inputs),
                "verified_input_count": verified_fixed,
            }
        output_artifact = reader.read(artifact_key)
        if not (
            output_artifact.available
            and output_artifact.content is not None
            and f"validation/results/{output_artifact.relative_path}" == expected_path
            and len(output_artifact.content) == pin["size_bytes"]
            and hashlib.sha256(output_artifact.content).hexdigest() == pin["sha256"]
        ):
            return {
                **base,
                "available": True,
                "state": "stale_lineage",
                "reason": "stale_lineage",
                "committed_input_count": len(inputs),
                "verified_input_count": verified_fixed,
            }
        if media_type == "application/json":
            if not isinstance(output_artifact.value, dict):
                return {
                    **base,
                    "available": True,
                    "state": "invalid",
                    "reason": "output_semantics_invalid",
                }
            reports[artifact_id] = output_artifact.value
    if not _product_v2_report_semantics_valid(reader, receipt, reports):
        return {
            **base,
            "available": True,
            "state": "invalid",
            "reason": "output_semantics_invalid",
            "committed_input_count": len(inputs),
            "verified_input_count": verified_fixed,
            "verified_output_count": len(outputs),
        }
    return {
        **base,
        "available": True,
        "state": "complete",
        "committed": True,
        "reason": None,
        "completed_at_utc": _timestamp(receipt.get("completed_at_utc")),
        "lineage_identity_sha256": _sha256(
            receipt.get("lineage_identity_sha256")
        ),
        "verified_input_count": verified_fixed,
        "committed_input_count": len(inputs),
        "verified_output_count": len(outputs),
        "product_status": "ready",
    }


def _objective_completion(reader: ArtifactReader) -> dict[str, Any]:
    """Project the user validation objective ledger as a bounded, separate truth."""

    artifact = reader.read("objective_completion_json")
    value = artifact.value or {}
    objective = (
        value.get("objective")
        if isinstance(value.get("objective"), dict)
        else {}
    )
    quality = (
        value.get("quality_context")
        if isinstance(value.get("quality_context"), dict)
        else {}
    )
    raw_gates = (
        objective.get("gates")
        if isinstance(objective.get("gates"), list)
        else []
    )
    raw_evidence = (
        value.get("evidence") if isinstance(value.get("evidence"), list) else []
    )

    contract_valid = bool(
        artifact.available
        and value.get("schema_version") == OBJECTIVE_COMPLETION_SCHEMA_VERSION
        and value.get("contract_id") == OBJECTIVE_COMPLETION_CONTRACT_ID
        and _canonical_fingerprint_matches(value)
        and reader.validates_schema(value, OBJECTIVE_COMPLETION_SCHEMA)
        and len(raw_gates) == len(OBJECTIVE_COMPLETION_GATE_IDS)
    )

    evidence_ids: set[str] = set()
    if contract_valid:
        for record in raw_evidence:
            if not isinstance(record, dict):
                contract_valid = False
                break
            evidence_id = _identifier(record.get("id"), maximum=128)
            if evidence_id is None or evidence_id in evidence_ids:
                contract_valid = False
                break
            evidence_ids.add(evidence_id)

    gates: list[dict[str, Any]] = []
    if contract_valid:
        for expected_id, raw in zip(
            OBJECTIVE_COMPLETION_GATE_IDS, raw_gates, strict=True
        ):
            if not isinstance(raw, dict) or raw.get("id") != expected_id:
                contract_valid = False
                break
            gate_state = _enum(raw.get("state"), {"pass", "pending", "invalid"})
            passed = _boolean(raw.get("passed"))
            raw_gate_evidence = raw.get("evidence_ids")
            raw_reasons = raw.get("reasons")
            if (
                gate_state is None
                or passed is None
                or passed != (gate_state == "pass")
                or not isinstance(raw_gate_evidence, list)
                or not isinstance(raw_reasons, list)
                or any(item not in evidence_ids for item in raw_gate_evidence)
            ):
                contract_valid = False
                break
            reasons = _identifiers(raw_reasons, maximum=16)
            if len(reasons) != len(raw_reasons):
                contract_valid = False
                break
            gates.append(
                {
                    "id": expected_id,
                    "title": OBJECTIVE_COMPLETION_GATE_TITLES[expected_id],
                    "state": gate_state,
                    "passed": passed,
                    "evidence_count": len(raw_gate_evidence),
                    "reasons": reasons,
                }
            )

    passed_count = sum(gate["passed"] is True for gate in gates)
    expected_state = (
        "complete"
        if passed_count == len(OBJECTIVE_COMPLETION_GATE_IDS)
        else "invalid"
        if any(gate["state"] == "invalid" for gate in gates)
        else "incomplete"
    )
    evidence_complete = _boolean(objective.get("evidence_complete"))
    observed_limitations = quality.get("observed_limitations")
    limitation_ids = (
        observed_limitations
        if isinstance(observed_limitations, list)
        and all(
            isinstance(item, str)
            and item in OBJECTIVE_COMPLETION_LIMITATION_IDS
            for item in observed_limitations
        )
        else None
    )
    semantic_valid = bool(
        contract_valid
        and len(gates) == len(OBJECTIVE_COMPLETION_GATE_IDS)
        and objective.get("required_gate_count")
        == len(OBJECTIVE_COMPLETION_GATE_IDS)
        and objective.get("passed_gate_count") == passed_count
        and objective.get("state") == expected_state
        and evidence_complete is (expected_state == "complete")
        and objective.get("does_not_imply_product_readiness") is True
        and quality.get("acceptance_effect_on_objective_completion") == "none"
        and quality.get("product_readiness_decision")
        == "out_of_scope_separate_truth"
        and quality.get("person_quality_decision")
        == "not_made_by_this_ledger"
        and quality.get("calibrated_25m_decision")
        == "not_made_by_this_ledger"
        and quality.get("pose_decision") == "not_made_by_this_ledger"
        and quality.get("ppe_decision") == "not_made_by_this_ledger"
        and quality.get("rlivit_quality_threshold_applied") is False
        and limitation_ids is not None
    )
    stale_lineage = bool(
        semantic_valid and not _objective_report_lineage_matches(reader, value)
    )
    projection_valid = bool(semantic_valid and not stale_lineage)

    if not artifact.available:
        state = "not_started" if artifact.state == "missing" else "artifact_error"
    elif not semantic_valid:
        state = "artifact_error"
    elif stale_lineage:
        state = "stale_lineage"
    else:
        state = expected_state

    caveats = [
        "Kanıt tamamlanması, üç modül ürün hazırlığı anlamına gelmez.",
        "Kalite, kalibre 25 m, pose ve PPE kararları ayrı kabul kapılarıdır.",
    ]
    limitation_labels = {
        "rlivit_recall_is_low_observation_without_owner_quality_threshold": (
            "R-LiViT recall gözlemi için sahip-onaylı kalite eşiği uygulanmadı."
        ),
        "top_view_ai_audit_contains_high_severity_undercoverage_finding": (
            "Üst açı AI incelemesinde yüksek önem dereceli eksik kapsama bulgusu var."
        ),
        "loaf_artifacts_do_not_satisfy_calibrated_25m_detection": (
            "LOAF kanıtı kalibre 25 m kabulünün yerine geçmez."
        ),
    }
    if projection_valid:
        caveats.extend(
            limitation_labels[item]
            for item in limitation_ids or []
            if item in limitation_labels
        )

    projected_evidence = _evidence(
        reader,
        "objective_completion_json",
        "objective_completion_markdown",
    )
    if stale_lineage:
        projected_evidence = _stale_lineage_evidence(projected_evidence)

    return {
        "label": "Yedi günlük doğrulama hedefi",
        "available": bool(artifact.available and not stale_lineage),
        "artifact_state": "stale_lineage" if stale_lineage else artifact.state,
        "state": state,
        "reason": "stale_lineage" if stale_lineage else None,
        "evidence_complete": evidence_complete if projection_valid else False,
        "does_not_imply_product_readiness": True,
        "product_readiness_decision": "separate_truth",
        "read_only": True,
        "execution_actions_available": False,
        "progress_label": "Zorunlu kanıt kapısı",
        "progress": _progress(
            passed_count if projection_valid else 0,
            len(OBJECTIVE_COMPLETION_GATE_IDS),
        ),
        "required_gates": gates if projection_valid else [],
        "caveats": caveats,
        "evidence": projected_evidence,
    }


def _product_readiness(reader: ArtifactReader) -> dict[str, Any]:
    """Project the independent product contract without exposing raw evidence."""

    artifact = reader.read("product_readiness_json")
    value = artifact.value or {}
    decision = value.get("decision") if isinstance(value.get("decision"), dict) else {}
    summary = value.get("summary") if isinstance(value.get("summary"), dict) else {}
    raw_gates = value.get("required_gates") if isinstance(value.get("required_gates"), list) else []
    expected_gate_ids = list(PRODUCT_READINESS_GATE_COMPONENT_IDS)
    gates: list[dict[str, Any]] = []
    contract_valid = bool(
        artifact.available
        and value.get("schema_version") == "deepsafe.product-readiness/v1"
        and _canonical_fingerprint_matches(value)
        and reader.validates_schema(value, PRODUCT_READINESS_SCHEMA)
        and len(raw_gates) == len(expected_gate_ids)
    )
    if contract_valid:
        for expected_id, raw in zip(expected_gate_ids, raw_gates, strict=True):
            if not isinstance(raw, dict) or raw.get("id") != expected_id:
                contract_valid = False
                break
            gate_state = _enum(
                raw.get("state"), {"pass", "unproven", "missing", "invalid", "fail"}
            )
            passed = _boolean(raw.get("passed"))
            checks = raw.get("component_checks")
            if (
                gate_state is None
                or passed is None
                or (passed and gate_state != "pass")
                or (not passed and gate_state == "pass")
                or not isinstance(checks, list)
            ):
                contract_valid = False
                break
            component_states: list[tuple[bool, str]] = []
            component_ids: list[str] = []
            for check in checks:
                if not isinstance(check, dict):
                    contract_valid = False
                    break
                check_passed = _boolean(check.get("passed"))
                check_state = _enum(
                    check.get("state"),
                    {"pass", "unproven", "missing", "invalid", "fail"},
                )
                if (
                    check_passed is None
                    or check_state is None
                    or (check_passed and check_state != "pass")
                    or (not check_passed and check_state == "pass")
                ):
                    contract_valid = False
                    break
                check_id = _identifier(check.get("id"), maximum=100)
                if check_id is None:
                    contract_valid = False
                    break
                component_ids.append(check_id)
                component_states.append((check_passed, check_state))
            if (
                not contract_valid
                or not component_states
                or component_ids != PRODUCT_READINESS_GATE_COMPONENT_IDS[expected_id]
                or passed != all(item[0] for item in component_states)
            ):
                contract_valid = False
                break
            missing_components = [
                check_id
                for check in checks[:32]
                if isinstance(check, dict)
                and check.get("passed") is False
                and (check_id := _identifier(check.get("id"), maximum=100)) is not None
            ]
            gates.append(
                {
                    "id": expected_id,
                    "title": _text(raw.get("title"), limit=180) or expected_id,
                    "state": gate_state,
                    "passed": passed,
                    "missing_components": missing_components,
                }
            )
    passed_count = sum(item["passed"] is True for item in gates)
    ready = _boolean(decision.get("ready"))
    decision_state = _enum(decision.get("status"), {"ready", "not_ready"})
    summary_total = _integer(summary.get("required_gate_count"))
    summary_passed = _integer(summary.get("passed_required_gate_count"))
    expected_failed_gate_ids = [item["id"] for item in gates if not item["passed"]]
    expected_state_counts: dict[str, int] = {}
    for item in gates:
        expected_state_counts[item["state"]] = (
            expected_state_counts.get(item["state"], 0) + 1
        )
    semantic_valid = bool(
        contract_valid
        and len(gates) == 6
        and summary_total == 6
        and summary_passed == passed_count
        and _integer(summary.get("remaining_required_gate_count")) == 6 - passed_count
        and ready is not None
        and decision_state == ("ready" if ready else "not_ready")
        and _boolean(decision.get("final_claim_allowed")) is ready
        and ready == all(item["passed"] for item in gates)
        and decision.get("failed_required_gate_ids") == expected_failed_gate_ids
        and summary.get("state_counts") == dict(sorted(expected_state_counts.items()))
    )
    optional: list[dict[str, Any]] = []
    raw_optional = value.get("optional_hardening")
    if semantic_valid and isinstance(raw_optional, list):
        for raw in raw_optional[:16]:
            if not isinstance(raw, dict):
                continue
            item_id = _identifier(raw.get("id"), maximum=100)
            state = _enum(raw.get("state"), {"complete", "pending", "missing", "invalid"})
            if item_id is not None and state is not None:
                optional.append(
                    {
                        "id": item_id,
                        "title": _text(raw.get("title"), limit=180) or item_id,
                        "state": state,
                        "acceptance_effect": "none",
                    }
                )
    stage = value.get("person_validation_stage") if isinstance(value.get("person_validation_stage"), dict) else {}
    if not artifact.available:
        state = "not_started" if artifact.state == "missing" else "artifact_error"
    elif not semantic_valid:
        state = "artifact_error"
    else:
        state = decision_state or "artifact_error"
    return {
        "label": "Üç modül ürün hazırlığı",
        "available": artifact.available,
        "state": state,
        "ready": ready if semantic_valid else False,
        "final_claim_allowed": (
            _boolean(decision.get("final_claim_allowed")) if semantic_valid else False
        ),
        "read_only": True,
        "execution_actions_available": False,
        "progress": _progress(passed_count if semantic_valid else 0, 6),
        "required_gates": gates if semantic_valid else [],
        "optional_hardening": optional if semantic_valid else [],
        "person_validation_stage": {
            "role": _enum(
                stage.get("role"),
                {"person_validation_stage_not_three_module_product_acceptance"},
            ),
            "campaign_decision_status": _enum(
                stage.get("campaign_decision_status"),
                {"accepted", "preliminary", "blocked_by_hardware"},
            ),
            "product_acceptance_equivalent": False,
        },
        "evidence": _evidence(
            reader, "product_readiness_json", "product_readiness_markdown"
        ),
    }


def _person_upgrade_unavailable(
    reason: str,
    *,
    integrity: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "label": "Kişi modeli yükseltme hazırlığı",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "license": {
            "decision": None,
            "selected": False,
            "download_and_training_authorized": False,
        },
        "selection": {},
        "preparation": {
            "training_data_prepared": False,
            "frozen_training_plan_verified": False,
            "permissive_checkpoint_acquired": False,
        },
        "dataset": {},
        "training_plan": {},
        "permissive_challenger": {},
        "gates": {
            "model_selected": False,
            "license_selected": False,
            "training_complete": False,
            "export_complete": False,
            "parity_complete": False,
            "acceptance_passed": False,
            "production_ready": False,
        },
        "integrity": integrity or {},
        "caveats": [
            "Hazırlık zinciri doğrulanamadı; veri, model ve kabul hazırlığı kapalıdır.",
        ],
        "evidence": [],
    }


def _person_pin_core(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "path": value.get("path"),
        "bytes": value.get("bytes"),
        "sha256": value.get("sha256"),
    }


def _self_fingerprint_matches(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    recorded = value.get("fingerprint_sha256")
    unsigned = dict(value)
    unsigned.pop("fingerprint_sha256", None)
    return bool(
        isinstance(recorded, str)
        and re.fullmatch(r"[0-9a-f]{64}", recorded) is not None
        and _canonical_sha256(unsigned) == recorded
    )


def _external_receipt_self_hash_matches(value: Any, *, expected: str) -> bool:
    if not isinstance(value, dict):
        return False
    unsigned = dict(value)
    observed = unsigned.pop("receipt_sha256", None)
    return bool(
        observed == expected
        and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
        and _canonical_sha256(unsigned) == expected
    )


def _person_structural_schema_contract_valid(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    properties = schema.get("properties")
    required = schema.get("required")
    return bool(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id")
        == "deepsafe.person-checkpoint-structural-receipt/v1"
        and schema.get("type") == "object"
        and schema.get("additionalProperties") is False
        and isinstance(properties, dict)
        and properties.get("schema_version", {}).get("const")
        == "deepsafe.person-checkpoint-structural-receipt/v1"
        and properties.get("status", {}).get("const")
        == "verified_cpu_strict_load_not_exported_not_evaluated"
        and properties.get("candidate_id", {}).get("const") == "rtdetrv4-s"
        and isinstance(required, list)
        and set(required) == set(properties)
    )


def _person_structural_receipt_semantics_valid(
    receipt: Any,
    *,
    checkpoint_pin: dict[str, Any],
) -> bool:
    if not isinstance(receipt, dict):
        return False
    inputs = receipt.get("inputs")
    execution = receipt.get("execution")
    checkpoint_structure = receipt.get("checkpoint_structure")
    architecture = receipt.get("architecture")
    conclusions = receipt.get("conclusions")
    if not all(
        isinstance(item, dict)
        for item in (
            inputs,
            execution,
            checkpoint_structure,
            architecture,
            conclusions,
        )
    ):
        return False
    assert isinstance(inputs, dict)
    assert isinstance(execution, dict)
    assert isinstance(checkpoint_structure, dict)
    assert isinstance(architecture, dict)
    assert isinstance(conclusions, dict)
    model_state = checkpoint_structure.get("model")
    ema_state = checkpoint_structure.get("ema_module")
    strict_loads = architecture.get("strict_loads")
    if not all(
        isinstance(item, dict)
        for item in (model_state, ema_state, strict_loads)
    ):
        return False
    expected_execution = {
        "cuda_visible_devices": "",
        "export_executed": False,
        "forward_pass_executed": False,
        "gpu_touched": False,
        "inference_executed": False,
        "map_location": "cpu",
        "network_download_calls": 0,
        "runtime": "cpu_only",
        "torch_build_cuda": None,
        "torch_cuda_available": False,
        "training_executed": False,
        "weights_only": True,
    }
    expected_conclusions = {
        "checkpoint_integrity_verified": True,
        "deepstream9_parity_passed": False,
        "framework_forward_parity_passed": False,
        "onnx_exported": False,
        "onnx_parity_passed": False,
        "person_quality_passed": False,
        "production_ready": False,
        "structural_load_verified": True,
        "tensorrt_built": False,
        "tensorrt_parity_passed": False,
    }
    return bool(
        receipt.get("schema_version")
        == "deepsafe.person-checkpoint-structural-receipt/v1"
        and receipt.get("status")
        == "verified_cpu_strict_load_not_exported_not_evaluated"
        and receipt.get("candidate_id") == "rtdetrv4-s"
        and isinstance(receipt.get("created_at"), str)
        and _person_pin_core(inputs.get("checkpoint")) == checkpoint_pin
        and _person_pin_core(inputs.get("schema"))
        == PERSON_UPGRADE_STRUCTURAL_SCHEMA_PIN
        and _person_pin_core(inputs.get("validator"))
        == PERSON_UPGRADE_STRUCTURAL_VALIDATOR_PIN
        and inputs.get("official_config")
        == {
            "bytes": 948,
            "path": (
                "third_party/RT-DETRv4/configs/rtv4/"
                "rtv4_hgnetv2_s_coco.yml"
            ),
            "sha256": (
                "45cf2abdc91e2a83b2d759b7c49526880d12a70ee44c8cdd8674dd604985bbe0"
            ),
        }
        and inputs.get("requirements")
        == {
            "bytes": 89,
            "path": "third_party/RT-DETRv4/requirements.txt",
            "sha256": (
                "83f707d7ec22ff66361d87839c237059e61a7576f7d329dbd2672b3f40ccfb1a"
            ),
        }
        and inputs.get("license")
        == {
            "bytes": 11357,
            "path": "third_party/RT-DETRv4/LICENSE",
            "sha256": (
                "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
            ),
        }
        and inputs.get("upstream")
        == {
            "commit": "55fefaaed7efe2a5f72d0a18fd4e05965e35c292",
            "git_tree": "284f2568c163aa4ffdaa9b61631a8082868bf132",
            "repository": "https://github.com/RT-DETRs/RT-DETRv4",
        }
        and execution == expected_execution
        and checkpoint_structure.get("root_type") == "OrderedDict"
        and checkpoint_structure.get("model_and_ema_keys_identical") is True
        and checkpoint_structure.get("ema_updates") == 443520
        and model_state == ema_state
        and model_state.get("tensor_count") == 796
        and model_state.get("tensor_value_count") == 10589534
        and architecture.get("class") == "RTv4"
        and architecture.get("parameter_count") == 10519253
        and architecture.get("state_tensor_count") == 796
        and architecture.get("parameter_device_types") == ["cpu"]
        and architecture.get(
            "hgnetv2_pretrained_disabled_before_instantiation"
        )
        is True
        and architecture.get("teacher_model_instantiated") is False
        and strict_loads.get("model")
        == {
            "missing_key_count": 0,
            "strict": True,
            "unexpected_key_count": 0,
        }
        and strict_loads.get("ema.module")
        == {
            "missing_key_count": 0,
            "strict": True,
            "unexpected_key_count": 0,
        }
        and conclusions == expected_conclusions
    )


def _person_closed_receipt_schema_valid(
    schema: Any,
    *,
    schema_id: str,
    schema_version: str,
) -> bool:
    if not isinstance(schema, dict):
        return False
    properties = schema.get("properties")
    required = schema.get("required")
    return bool(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id") == schema_id
        and schema.get("type") == "object"
        and schema.get("additionalProperties") is False
        and isinstance(properties, dict)
        and properties.get("schema_version", {}).get("const")
        == schema_version
        and isinstance(required, list)
        and set(required) == set(properties)
        and "receipt_sha256" in required
    )


def _person_framework_receipt_semantics_valid(
    receipt: Any,
    *,
    checkpoint_pin: dict[str, Any],
) -> bool:
    if not isinstance(receipt, dict):
        return False
    inputs = receipt.get("inputs")
    execution = receipt.get("execution")
    contract = receipt.get("profile_contract")
    conclusions = receipt.get("conclusions")
    if not all(
        isinstance(item, dict)
        for item in (inputs, execution, contract, conclusions)
    ):
        return False
    assert isinstance(inputs, dict)
    assert isinstance(execution, dict)
    assert isinstance(contract, dict)
    assert isinstance(conclusions, dict)
    profiles = contract.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 2:
        return False
    profile_by_size = {
        item.get("profile"): item
        for item in profiles
        if isinstance(item, dict)
    }
    if set(profile_by_size) != {640, 960}:
        return False
    expected_shapes = {
        640: {
            "pos": [1, 400, 256],
            "anchors": [1, 8400, 4],
            "mask": [1, 8400, 1],
            "regenerated": [],
        },
        960: {
            "pos": [1, 900, 256],
            "anchors": [1, 18900, 4],
            "mask": [1, 18900, 1],
            "regenerated": [
                "encoder.pos_embed2",
                "decoder.anchors",
                "decoder.valid_mask",
            ],
        },
    }
    profiles_valid = True
    learned_hash: str | None = None
    for profile, expected in expected_shapes.items():
        row = profile_by_size[profile]
        tensors = row.get("spatial_tensors")
        outputs = row.get("outputs")
        if not isinstance(tensors, dict) or not isinstance(outputs, dict):
            profiles_valid = False
            continue
        current_hash = row.get("learned_parameter_sha256_after")
        profiles_valid = bool(
            profiles_valid
            and row.get("checkpoint_source") == "ema.module"
            and row.get("model_instantiated_at_publisher_spatial_size") == 640
            and row.get("spatial_profile_isolated") is True
            and row.get("learned_parameters_unchanged") is True
            and row.get("learned_parameter_sha256_before") == current_hash
            and isinstance(current_hash, str)
            and re.fullmatch(r"[0-9a-f]{64}", current_hash) is not None
            and row.get("regenerated_nonlearned_tensor_allowlist")
            == expected["regenerated"]
            and row.get("strict_load_before_profile_isolation")
            == {
                "missing_key_count": 0,
                "strict": True,
                "unexpected_key_count": 0,
            }
            and tensors.get("encoder.pos_embed2", {}).get("shape")
            == expected["pos"]
            and tensors.get("decoder.anchors", {}).get("shape")
            == expected["anchors"]
            and tensors.get("decoder.valid_mask", {}).get("shape")
            == expected["mask"]
            and outputs.get("labels")
            == {"all_finite": True, "dtype": "int64", "shape": [1, 300]}
            and outputs.get("boxes")
            == {
                "all_finite": True,
                "dtype": "float32",
                "shape": [1, 300, 4],
            }
            and outputs.get("scores")
            == {
                "all_finite": True,
                "dtype": "float32",
                "shape": [1, 300],
            }
        )
        if learned_hash is not None and current_hash != learned_hash:
            profiles_valid = False
        learned_hash = current_hash
    expected_conclusions = {
        "capacity_passed": False,
        "deepstream9_parity_passed": False,
        "onnx_exported": False,
        "onnx_parity_passed": False,
        "person_quality_passed": False,
        "production_ready": False,
        "structural_receipt_verified": True,
        "synthetic_framework_io_640_verified": True,
        "synthetic_framework_io_960_verified": True,
        "tensorrt_built": False,
        "tensorrt_parity_passed": False,
    }
    return bool(
        receipt.get("schema_version")
        == "deepsafe.person-framework-profiles-receipt/v1"
        and receipt.get("status")
        == "synthetic_framework_io_verified_not_exported_not_evaluated"
        and receipt.get("candidate_id") == "rtdetrv4-s"
        and isinstance(receipt.get("created_at"), str)
        and _person_pin_core(inputs.get("checkpoint")) == checkpoint_pin
        and inputs.get("structural_receipt")
        == PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN
        and inputs.get("schema") == PERSON_UPGRADE_FRAMEWORK_SCHEMA_PIN
        and inputs.get("validator") == PERSON_UPGRADE_FRAMEWORK_VALIDATOR_PIN
        and execution
        == {
            "deepstream9_executed": False,
            "gpu_touched": False,
            "network_download_calls": 0,
            "onnx_export_executed": False,
            "real_image_inference_executed": False,
            "runtime": "cpu_only",
            "synthetic_forward_executed": True,
            "tensorrt_executed": False,
            "training_executed": False,
            "weights_only": True,
        }
        and contract.get("batch_axis_export_intent")
        == "dynamic_min1_opt12_max12_not_yet_exported"
        and contract.get("separate_fixed_spatial_exports_required") is True
        and contract.get("spatial_axes_dynamic") is False
        and profiles_valid
        and conclusions == expected_conclusions
    )


def _person_onnx_export_plan_semantics_valid(
    plan: Any,
    *,
    checkpoint_pin: dict[str, Any],
) -> bool:
    if not isinstance(plan, dict):
        return False
    inputs = plan.get("inputs")
    profiles = plan.get("profiles")
    if not isinstance(inputs, dict) or not isinstance(profiles, dict):
        return False
    return bool(
        plan.get("schema_version")
        == "deepsafe.rtdetrv4-onnx-export-plan/v1"
        and plan.get("status") == "planned_cpu_export_not_executed"
        and plan.get("candidate_id") == "rtdetrv4-s"
        and plan.get("checkpoint_source") == "ema.module"
        and _self_fingerprint_matches(plan)
        and plan.get("fingerprint_sha256")
        == PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN["fingerprint_sha256"]
        and _person_pin_core(inputs.get("checkpoint")) == checkpoint_pin
        and _person_pin_core(inputs.get("structural_receipt"))
        == _person_pin_core(PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN)
        and _person_pin_core(inputs.get("framework_profiles_receipt"))
        == _person_pin_core(PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN)
        and inputs.get("exporter") == PERSON_UPGRADE_ONNX_EXPORTER_PIN
        and inputs.get("receipt_schema")
        == PERSON_UPGRADE_ONNX_RECEIPT_SCHEMA_PIN
        and plan.get("source_receipt_fingerprints")
        == {
            "framework_profiles": PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN[
                "receipt_sha256"
            ],
            "structural": PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN[
                "receipt_sha256"
            ],
        }
        and plan.get("opset") == 18
        and plan.get("spatial_axes_dynamic") is False
        and plan.get("batch_axis_dynamic") is True
        and plan.get("batch_profile") == {"min": 1, "opt": 12, "max": 12}
        and plan.get("synthetic_parity")
        == {
            "batches": [1, 2],
            "input": "seeded_cpu_prng_not_quality_evidence",
            "seed": 20260717,
            "labels_exact": True,
            "boxes_max_abs_lte": 0.01,
            "scores_max_abs_lte": 0.0001,
            "not_quality_evidence": True,
        }
        and profiles.get("640", {}).get("spatial") == [640, 640]
        and profiles.get("640", {}).get("onnx_path")
        == PERSON_UPGRADE_ONNX_PROFILE_PINS[640]["onnx"]["path"]
        and profiles.get("640", {}).get("receipt_path")
        == PERSON_UPGRADE_ONNX_PROFILE_PINS[640]["receipt"]["path"]
        and profiles.get("960", {}).get("spatial") == [960, 960]
        and profiles.get("960", {}).get("onnx_path")
        == PERSON_UPGRADE_ONNX_PROFILE_PINS[960]["onnx"]["path"]
        and profiles.get("960", {}).get("receipt_path")
        == PERSON_UPGRADE_ONNX_PROFILE_PINS[960]["receipt"]["path"]
        and isinstance(plan.get("acceptance"), dict)
        and len(plan["acceptance"]) == 8
        and all(value is False for value in plan["acceptance"].values())
    )


def _person_onnx_receipt_semantics_valid(
    receipt: Any,
    *,
    profile: int,
    checkpoint_pin: dict[str, Any],
) -> bool:
    if not isinstance(receipt, dict) or profile not in {640, 960}:
        return False
    inputs = receipt.get("inputs")
    execution = receipt.get("execution")
    isolation = receipt.get("profile_isolation")
    export = receipt.get("export")
    parity = receipt.get("synthetic_onnx_parity")
    acceptance = receipt.get("acceptance")
    if not all(
        isinstance(item, dict)
        for item in (inputs, execution, isolation, export, parity, acceptance)
    ):
        return False
    assert isinstance(inputs, dict)
    assert isinstance(execution, dict)
    assert isinstance(isolation, dict)
    assert isinstance(export, dict)
    assert isinstance(parity, dict)
    assert isinstance(acceptance, dict)
    cases = parity.get("cases")
    cases_valid = bool(
        isinstance(cases, list)
        and [item.get("batch") for item in cases if isinstance(item, dict)]
        == [1, 2]
        and all(
            isinstance(item, dict)
            and item.get("labels_exact") is True
            and item.get("all_finite") is True
            and item.get("passed") is True
            and isinstance(item.get("boxes_max_abs"), (int, float))
            and not isinstance(item.get("boxes_max_abs"), bool)
            and item["boxes_max_abs"] <= 0.01
            and isinstance(item.get("scores_max_abs"), (int, float))
            and not isinstance(item.get("scores_max_abs"), bool)
            and item["scores_max_abs"] <= 0.0001
            for item in cases
        )
    )
    expected_regenerated = (
        []
        if profile == 640
        else ["encoder.pos_embed2", "decoder.anchors", "decoder.valid_mask"]
    )
    return bool(
        receipt.get("schema_version")
        == "deepsafe.rtdetrv4-onnx-export-receipt/v1"
        and receipt.get("status")
        == "exported_synthetic_parity_passed_not_evaluated"
        and receipt.get("candidate_id") == "rtdetrv4-s"
        and receipt.get("profile") == profile
        and isinstance(receipt.get("created_at"), str)
        and _person_pin_core(inputs.get("checkpoint")) == checkpoint_pin
        and inputs.get("plan")
        == _person_pin_core(PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN)
        and inputs.get("exporter") == PERSON_UPGRADE_ONNX_EXPORTER_PIN
        and inputs.get("receipt_schema")
        == PERSON_UPGRADE_ONNX_RECEIPT_SCHEMA_PIN
        and execution
        == {
            "deepstream9_executed": False,
            "gpu_touched": False,
            "network_download_calls": 0,
            "real_image_inference_executed": False,
            "runtime": "cpu_only",
            "tensorrt_executed": False,
            "training_executed": False,
            "weights_only": True,
        }
        and isolation.get("checkpoint_source") == "ema.module"
        and isolation.get("learned_parameters_unchanged") is True
        and isolation.get("learned_parameter_sha256_before")
        == isolation.get("learned_parameter_sha256_after")
        and isolation.get("strict_load") is True
        and isolation.get("missing_key_count") == 0
        and isolation.get("unexpected_key_count") == 0
        and isolation.get("regenerated_nonlearned_tensor_allowlist")
        == expected_regenerated
        and export.get("opset") == 18
        and export.get("spatial_axes_dynamic") is False
        and export.get("batch_axis_dynamic") is True
        and export.get("export_dummy_batch") == 2
        and export.get("tensorrt_profile_intent")
        == {"min": 1, "opt": 12, "max": 12}
        and export.get("onnx")
        == PERSON_UPGRADE_ONNX_PROFILE_PINS[profile]["onnx"]
        and export.get("metadata", {}).get("checker_full_passed") is True
        and export.get("metadata", {}).get("external_data_files") == []
        and export.get("metadata", {}).get("inputs", {}).get("images")
        == ["batch", 3, profile, profile]
        and export.get("metadata", {}).get("inputs", {}).get(
            "orig_target_sizes"
        )
        == ["batch", 2]
        and parity.get("kind")
        == "synthetic_seeded_prng_input_not_quality_evidence"
        and parity.get("seed") == 20260717
        and parity.get("thresholds")
        == {
            "boxes_max_abs_lte": 0.01,
            "labels_exact": True,
            "scores_max_abs_lte": 0.0001,
        }
        and parity.get("passed") is True
        and cases_valid
        and len(acceptance) == 8
        and all(value is False for value in acceptance.values())
    )


def _person_real_image_parity_plan_semantics_valid(
    plan: Any,
    *,
    checkpoint_pin: dict[str, Any],
) -> bool:
    if not isinstance(plan, dict):
        return False
    inputs = plan.get("inputs")
    selections = plan.get("selections")
    if not isinstance(inputs, dict) or not isinstance(selections, list):
        return False
    case_ids = [
        item.get("case_id") for item in selections if isinstance(item, dict)
    ]
    scene_ids = [
        item.get("scene_id") for item in selections if isinstance(item, dict)
    ]
    video_types = [
        item.get("primary_video_type")
        for item in selections
        if isinstance(item, dict)
    ]
    required_types = {
        "security_camera_like",
        "true_top_view",
        "elevated_view",
        "medium_close",
        "distant_workers",
    }
    selections_valid = bool(
        len(selections) == 11
        and len(case_ids) == len(scene_ids) == len(video_types) == 11
        and len(set(case_ids)) == len(set(scene_ids)) == len(set(video_types))
        == 11
        and required_types.issubset(set(video_types))
        and all(
            isinstance(item, dict)
            and item.get("segment_role")
            in {"person_visible", "partial_body_only"}
            and isinstance(item.get("source"), dict)
            and set(item["source"]) == {"path", "bytes", "sha256"}
            and isinstance(item["source"].get("bytes"), int)
            and item["source"]["bytes"] > 0
            and isinstance(item["source"].get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["source"]["sha256"])
            is not None
            and isinstance(item.get("license"), dict)
            and isinstance(item["license"].get("spdx"), str)
            and isinstance(item["license"].get("url"), str)
            and isinstance(item["license"].get("attribution"), str)
            and isinstance(item.get("frame"), dict)
            and isinstance(item["frame"].get("index"), int)
            and item["frame"]["index"] >= 0
            and isinstance(item["frame"].get("timestamp_seconds"), (int, float))
            and isinstance(item["frame"].get("fps_fraction"), str)
            and isinstance(item.get("review_flags"), dict)
            for item in selections
        )
        and any(item["review_flags"].get("medium_close") is True for item in selections)
        and any(item["review_flags"].get("high_oblique") is True for item in selections)
        and any(item["review_flags"].get("top_view") is True for item in selections)
    )
    return bool(
        plan.get("schema_version")
        == "deepsafe.rtdetrv4-real-image-parity-plan/v1"
        and plan.get("status")
        == "planned_cpu_real_image_parity_not_quality_not_performance"
        and plan.get("candidate_id") == "rtdetrv4-s"
        and _self_fingerprint_matches(plan)
        and plan.get("fingerprint_sha256")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN["fingerprint_sha256"]
        and _person_pin_core(inputs.get("checkpoint")) == checkpoint_pin
        and inputs.get("official_config")
        == {
            "path": (
                "third_party/RT-DETRv4/configs/rtv4/"
                "rtv4_hgnetv2_s_coco.yml"
            ),
            "bytes": 948,
            "sha256": (
                "45cf2abdc91e2a83b2d759b7c49526880d12a70ee44c8cdd8674dd604985bbe0"
            ),
        }
        and inputs.get("export_plan")
        == _person_pin_core(PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN)
        and inputs.get("source_manifest")
        == PERSON_UPGRADE_REAL_IMAGE_SOURCE_MANIFEST_PIN
        and inputs.get("source_frame_reviews")
        == PERSON_UPGRADE_REAL_IMAGE_SOURCE_REVIEWS_PIN
        and inputs.get("validator")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_VALIDATOR_PIN
        and inputs.get("receipt_schema")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_SCHEMA_PIN
        and inputs.get("onnx_profiles")
        == {
            str(profile): {
                "export_receipt": PERSON_UPGRADE_ONNX_PROFILE_PINS[profile][
                    "receipt"
                ],
                "onnx": PERSON_UPGRADE_ONNX_PROFILE_PINS[profile]["onnx"],
            }
            for profile in (640, 960)
        }
        and plan.get("profiles") == [640, 960]
        and plan.get("batches")
        == {
            "batch1": "all_selected_frames_individually",
            "batch2_case_ids": [
                "fr_paris_snow_umbrellas.f000116",
                "ee_swimmers_overhead.f000292",
            ],
        }
        and plan.get("tolerances")
        == {
            "labels": "class_preserving_bijection",
            "boxes_max_abs_lte": 0.01,
            "scores_max_abs_lte": 0.0001,
        }
        and plan.get("matching", {}).get("tolerance_relaxation") is False
        and plan.get("matching", {}).get(
            "topk_tie_diagnostics_override_acceptance"
        )
        is False
        and plan.get("matching", {}).get(
            "failed_receipt_required_when_any_strict_case_fails"
        )
        is True
        and plan.get("claim_boundary")
        == {
            "real_image_framework_onnx_parity": (
                "may_pass_only_if_every_case_passes"
            ),
            "quality": False,
            "distance_25m": False,
            "latency_or_fps": False,
            "capacity": False,
            "tensorrt": False,
            "deepstream9": False,
            "production_ready": False,
        }
        and selections_valid
    )


def _person_real_image_parity_receipt_semantics_valid(
    receipt: Any,
    *,
    plan: dict[str, Any],
    checkpoint_pin: dict[str, Any],
) -> bool:
    if not isinstance(receipt, dict):
        return False
    inputs = receipt.get("inputs")
    frames = receipt.get("frames")
    profiles = receipt.get("profiles")
    source = receipt.get("source_verification")
    execution = receipt.get("execution")
    outcome = receipt.get("outcome")
    acceptance = receipt.get("acceptance")
    if not all(
        isinstance(item, dict)
        for item in (inputs, source, execution, outcome, acceptance)
    ) or not isinstance(frames, list) or not isinstance(profiles, list):
        return False
    case_ids = [item["case_id"] for item in plan["selections"]]
    frame_types = [
        item["primary_video_type"] for item in plan["selections"]
    ]
    frames_valid = bool(
        [item.get("case_id") for item in frames if isinstance(item, dict)]
        == case_ids
        and all(
            isinstance(frame, dict)
            and all(
                frame.get(key) == selection.get(key)
                for key in (
                    "case_id",
                    "asset_id",
                    "scene_id",
                    "primary_video_type",
                    "source",
                    "license",
                    "frame",
                    "review_flags",
                )
            )
            and set(frame.get("profile_inputs", {})) == {"640", "960"}
            and all(
                all(
                    isinstance(
                        frame["profile_inputs"][str(profile)].get(name), str
                    )
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        frame["profile_inputs"][str(profile)][name],
                    )
                    is not None
                    for name in (
                        "rgb_uint8_hwc_sha256",
                        "input_float32_nchw_sha256",
                        "orig_target_sizes_int64_sha256",
                    )
                )
                for profile in (640, 960)
            )
            and isinstance(frame.get("extraction"), dict)
            and frame["extraction"].get("decoded_index")
            == selection["frame"]["index"]
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(frame["extraction"].get("rgb_sha256", "")),
            )
            is not None
            for frame, selection in zip(frames, plan["selections"])
        )
    )
    if [item.get("profile") for item in profiles if isinstance(item, dict)] != [
        640,
        960,
    ]:
        return False
    derived_failures: list[tuple[int, int, str]] = []
    derived_batch_passed = {1: True, 2: True}
    profiles_valid = True
    for profile_row in profiles:
        profile = profile_row["profile"]
        framework = profile_row.get("framework")
        debug_view = profile_row.get("onnx_raw_debug_view")
        if not isinstance(framework, dict) or not isinstance(debug_view, dict):
            return False
        expected_allowlist = (
            []
            if profile == 640
            else [
                "encoder.pos_embed2",
                "decoder.anchors",
                "decoder.valid_mask",
            ]
        )
        profiles_valid = bool(
            profiles_valid
            and framework.get("checkpoint_source") == "ema.module"
            and framework.get("strict_load") is True
            and framework.get("missing_key_count") == 0
            and framework.get("unexpected_key_count") == 0
            and framework.get("learned_parameters_unchanged") is True
            and framework.get("regenerated_nonlearned_tensor_allowlist")
            == expected_allowlist
            and profile_row.get("onnx")
            == PERSON_UPGRADE_ONNX_PROFILE_PINS[profile]["onnx"]
            and debug_view.get("base_onnx_graph_node_count")
            == debug_view.get("derived_graph_node_count")
            and debug_view.get("only_graph_outputs_appended") is True
            and debug_view.get("base_final_outputs_bit_exact_to_debug_view")
            is True
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(debug_view.get("derived_in_memory_model_sha256", "")),
            )
            is not None
        )
        profile_passed = True
        for batch_number, batch_key, expected_ids in (
            (1, "batch1", case_ids),
            (2, "batch2", plan["batches"]["batch2_case_ids"]),
        ):
            batch = profile_row.get(batch_key)
            if not isinstance(batch, dict):
                return False
            cases = batch.get("cases")
            if not isinstance(cases, list) or [
                item.get("case_id") for item in cases if isinstance(item, dict)
            ] != expected_ids:
                return False
            batch_passed = True
            for result in cases:
                if not isinstance(result, dict):
                    return False
                expected_passed = bool(
                    result.get("labels_histogram_exact") is True
                    and result.get("matched_pair_count") == 300
                    and isinstance(result.get("boxes_max_abs"), (int, float))
                    and not isinstance(result.get("boxes_max_abs"), bool)
                    and result["boxes_max_abs"] <= 0.01
                    and isinstance(result.get("scores_max_abs"), (int, float))
                    and not isinstance(result.get("scores_max_abs"), bool)
                    and result["scores_max_abs"] <= 0.0001
                    and result.get("all_finite") is True
                )
                raw = result.get("raw_diagnostics")
                if (
                    result.get("matching_semantics")
                    != (
                        "class_preserving_perfect_bipartite_matching_on_"
                        "tolerance_bounded_edges"
                    )
                    or result.get("passed") is not expected_passed
                    or not isinstance(raw, dict)
                    or raw.get("topk", {}).get(
                        "tie_band_is_diagnostic_not_an_acceptance_override"
                    )
                    is not True
                ):
                    return False
                for raw_key in ("framework_raw", "onnx_raw"):
                    raw_hashes = raw.get(raw_key)
                    if not isinstance(raw_hashes, dict) or not all(
                        re.fullmatch(r"[0-9a-f]{64}", str(value))
                        is not None
                        for value in raw_hashes.values()
                    ):
                        return False
                batch_passed = batch_passed and expected_passed
                if not expected_passed:
                    derived_failures.append(
                        (profile, batch_number, result["case_id"])
                    )
            if batch.get("passed") is not batch_passed:
                return False
            derived_batch_passed[batch_number] = bool(
                derived_batch_passed[batch_number] and batch_passed
            )
            profile_passed = profile_passed and batch_passed
        if profile_row.get("passed") is not profile_passed:
            return False
    derived_all_passed = not derived_failures
    return bool(
        receipt.get("schema_version")
        == "deepsafe.person-rtdetrv4-real-image-parity-receipt/v1"
        and receipt.get("status")
        == "real_image_framework_onnx_parity_failed_not_quality_not_performance"
        and receipt.get("candidate_id") == "rtdetrv4-s"
        and isinstance(receipt.get("created_at"), str)
        and inputs.get("plan")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN
        and _person_pin_core(inputs.get("checkpoint")) == checkpoint_pin
        and inputs.get("official_config") == plan["inputs"]["official_config"]
        and inputs.get("source_manifest")
        == PERSON_UPGRADE_REAL_IMAGE_SOURCE_MANIFEST_PIN
        and inputs.get("source_frame_reviews")
        == PERSON_UPGRADE_REAL_IMAGE_SOURCE_REVIEWS_PIN
        and inputs.get("onnx_profiles") == plan["inputs"]["onnx_profiles"]
        and inputs.get("validator")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_VALIDATOR_PIN
        and inputs.get("receipt_schema")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_SCHEMA_PIN
        and receipt.get("implementation")
        == {
            "runtime_versions": plan["runtime"]["versions"],
            "frame_extraction": plan["frame_extraction"],
            "preprocessing": plan["preprocessing"],
            "matching": plan["matching"],
            "tolerances": plan["tolerances"],
        }
        and frames_valid
        and source
        == {
            "manifest_asset_sha_license_bindings_verified": True,
            "source_frame_review_bindings_verified": True,
            "exact_frame_index_timestamp_rationals_verified": True,
            "selected_frame_count": 11,
            "unique_scene_count": 11,
            "unique_primary_video_type_count": 11,
            "primary_video_types": frame_types,
            "medium_close_present": True,
            "high_oblique_present": True,
            "top_view_present": True,
            "ground_truth_used": False,
        }
        and profiles_valid
        and derived_all_passed is False
        and len(derived_failures) == 4
        and outcome.get("all_selected_cases_and_batches_passed") is False
        and outcome.get("failure_count") == len(derived_failures)
        and isinstance(outcome.get("failures"), list)
        and len(outcome["failures"]) == len(derived_failures)
        and [
            (item.get("profile"), item.get("batch"), item.get("case_id"))
            for item in outcome["failures"]
            if isinstance(item, dict)
        ]
        == derived_failures
        and outcome.get("tolerances_relaxed") is False
        and outcome.get("topk_tie_diagnostics_override_acceptance") is False
        and execution
        == {
            "runtime": "cpu_only",
            "gpu_touched": False,
            "cuda_build_present": False,
            "network_download_calls": 0,
            "weights_only_checkpoint_load": True,
            "framework_network_forward_calls": 24,
            "onnxruntime_base_forward_calls": 24,
            "onnxruntime_debug_view_forward_calls": 24,
            "batch1_executed_for_every_frame_and_profile": True,
            "batch2_executed_for_every_profile": True,
            "timing_collected": False,
            "latency_or_fps_measured": False,
            "training_executed": False,
            "tensorrt_executed": False,
            "deepstream9_executed": False,
        }
        and acceptance
        == {
            "source_manifest_asset_sha_license_bindings_passed": True,
            "exact_real_pixel_inputs_pinned": True,
            "batch1_framework_onnx_parity_passed": derived_batch_passed[1],
            "batch2_framework_onnx_parity_passed": derived_batch_passed[2],
            "real_image_framework_onnx_parity_passed": False,
            "independent_ground_truth_quality_passed": False,
            "exact_25m_passed": False,
            "latency_or_fps_passed": False,
            "twelve_camera_capacity_passed": False,
            "tensorrt_parity_passed": False,
            "deepstream9_parity_passed": False,
            "three_module_full_stack_passed": False,
            "production_ready": False,
        }
    )


def _person_ds9_parser_receipt_semantics_valid(receipt: Any) -> bool:
    if not isinstance(receipt, dict):
        return False
    inputs = receipt.get("inputs")
    build_runtime = receipt.get("build_runtime")
    tests = receipt.get("tests")
    artifact = receipt.get("artifact")
    capabilities = receipt.get("capabilities")
    readiness = receipt.get("readiness")
    if not all(
        isinstance(item, dict)
        for item in (
            inputs,
            build_runtime,
            tests,
            artifact,
            capabilities,
            readiness,
        )
    ):
        return False
    assert isinstance(inputs, dict)
    expected_inputs = {
        **PERSON_UPGRADE_DS9_PARSER_SOURCE_PINS,
        "export_plan": PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN,
        "onnx_640_receipt": PERSON_UPGRADE_ONNX_PROFILE_PINS[640][
            "receipt"
        ],
        "onnx_960_receipt": PERSON_UPGRADE_ONNX_PROFILE_PINS[960][
            "receipt"
        ],
    }
    return bool(
        receipt.get("schema_version")
        == "deepsafe.rtdetrv4-ds9-parser-build-receipt/v1"
        and receipt.get("status")
        == "built_cpu_contract_tested_not_gpu_integrated"
        and receipt.get("candidate_id") == "rtdetrv4-s"
        and isinstance(receipt.get("created_at"), str)
        and inputs == expected_inputs
        and build_runtime
        == {
            "image_reference": (
                "deepsafe-deepstream@sha256:"
                "96aedaba7ebb8d50359a7f73db251d46a81fd23e42c7c7ae215542795f88d663"
            ),
            "image_id": (
                "sha256:"
                "96aedaba7ebb8d50359a7f73db251d46a81fd23e42c7c7ae215542795f88d663"
            ),
            "network": "none",
            "container_runtime": "runc",
            "gpu_exposed": False,
            "deepstream": "9.0.0",
            "cuda_headers": "13.1",
            "tensorrt_headers": "10.14.1.48",
            "gstreamer": "1.24.2",
            "compiler": "g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0",
            "cmake": "3.19.6",
            "build_type": "Release",
            "parallel_jobs": 2,
        }
        and tests
        == {
            "ctest_total": 1,
            "ctest_passed": 1,
            "ctest_failed": 0,
            "float32_int64_parse_passed": True,
            "float16_int32_parse_passed": True,
            "orig_target_sizes_batch12_initializer_passed": True,
            "missing_layer_rejected": True,
            "wrong_class_count_rejected": True,
            "nonfinite_threshold_rejected": True,
            "elf_undefined_symbol_count": 0,
        }
        and receipt.get("abi")
        == {
            "version_node": "DEEPSAFE_RTDETRV4_PARSER_1.0",
            "symbols": [
                (
                    "NvDsInferInitializeInputLayers@@"
                    "DEEPSAFE_RTDETRV4_PARSER_1.0"
                ),
                (
                    "NvDsInferParseCustomRTDETRv4Person@@"
                    "DEEPSAFE_RTDETRV4_PARSER_1.0"
                ),
            ],
        }
        and artifact
        == {
            **PERSON_UPGRADE_DS9_PARSER_ARTIFACT_PIN,
            "mode": "0440",
            "elf": "ELF64_x86_64_shared_object",
        }
        and capabilities
        == {
            "person_only_coco_class_zero": True,
            "queries": 300,
            "outputs": ["labels", "boxes", "scores"],
            "labels_dtypes": ["INT64", "INT32"],
            "boxes_scores_dtypes": ["FP32", "FP16"],
            "boxes_coordinate_space": "network_pixels_xyxy",
            "non_image_input": "orig_target_sizes_height_width",
            "max_batch_contract": 12,
        }
        and readiness
        == {
            "parser_cpu_contract_ready": True,
            "onnx_profiles_present": True,
            "tensorrt_engines_built": False,
            "gpu_integration_validated": False,
            "deepstream9_real_inference_validated": False,
            "real_image_parity_passed": False,
            "quality_passed": False,
            "capacity_passed": False,
            "production_ready": False,
        }
    )


def _person_onnx_batch12_receipt_semantics_valid(receipt: Any) -> bool:
    if not isinstance(receipt, dict):
        return False
    observations = receipt.get("observations")
    if not isinstance(observations, list) or len(observations) != 2:
        return False
    by_profile = {
        item.get("profile"): item
        for item in observations
        if isinstance(item, dict)
    }
    if set(by_profile) != {640, 960}:
        return False
    observations_valid = True
    for profile in (640, 960):
        row = by_profile[profile]
        input_row = row.get("input")
        outputs = row.get("outputs")
        if not isinstance(input_row, dict) or not isinstance(outputs, dict):
            observations_valid = False
            continue
        observations_valid = bool(
            observations_valid
            and input_row
            == {
                "batch": 12,
                "images_dtype": "float32",
                "images_shape": [12, 3, profile, profile],
                "kind": "seeded_prng_not_real_image",
                "orig_target_sizes_dtype": "int64",
                "orig_target_sizes_shape": [12, 2],
                "seed": 20260717,
            }
            and outputs.get("labels")
            == {
                "all_finite": True,
                "dtype": "int64",
                "shape": [12, 300],
            }
            and outputs.get("boxes")
            == {
                "all_finite": True,
                "dtype": "float32",
                "shape": [12, 300, 4],
            }
            and outputs.get("scores")
            == {
                "all_finite": True,
                "dtype": "float32",
                "shape": [12, 300],
            }
        )
    return bool(
        receipt.get("schema_version")
        == "deepsafe.person-onnx-batch12-receipt/v1"
        and receipt.get("status")
        == "batch12_shape_finite_verified_not_performance_not_evaluated"
        and receipt.get("candidate_id") == "rtdetrv4-s"
        and isinstance(receipt.get("created_at"), str)
        and receipt.get("inputs")
        == {
            str(profile): {
                "export_receipt": PERSON_UPGRADE_ONNX_PROFILE_PINS[profile][
                    "receipt"
                ],
                "onnx": PERSON_UPGRADE_ONNX_PROFILE_PINS[profile]["onnx"],
            }
            for profile in (640, 960)
        }
        and receipt.get("implementation")
        == {
            "schema": PERSON_UPGRADE_ONNX_BATCH12_SCHEMA_PIN,
            "validator": PERSON_UPGRADE_ONNX_BATCH12_VALIDATOR_PIN,
        }
        and receipt.get("execution")
        == {
            "framework_parity_claimed": False,
            "gpu_touched": False,
            "latency_or_fps_claimed": False,
            "provider": "CPUExecutionProvider",
            "real_image_inference_executed": False,
            "training_executed": False,
        }
        and observations_valid
        and receipt.get("gates")
        == {
            "capacity_passed": False,
            "deepstream9_batch12_verified": False,
            "onnx_640_batch12_shape_finite_verified": True,
            "onnx_960_batch12_shape_finite_verified": True,
            "production_ready": False,
            "quality_passed": False,
            "real_image_parity_passed": False,
            "tensorrt_batch12_verified": False,
        }
    )


def _person_rtdetr_gpu_r10_unavailable(
    reason: str, *, integrity: dict[str, bool]
) -> dict[str, Any]:
    return {
        "evidence_version": "r10",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "full_training_complete": False,
        "export_plan_ready": False,
        "export_complete": False,
        "onnx_export_complete": False,
        "deepstream9_complete": False,
        "production_ready": False,
        "distance_proxy_r1": {
            "available": False,
            "state": "artifact_error",
            "metric_distance_established": False,
            "twenty_to_twenty_five_m_status": (
                "blocked_missing_per_camera_metric_calibration"
            ),
        },
        "integrity": integrity,
        "caveats": [
            "R10 exact-pin zinciri doğrulanamadı; GPU smoke, baseline veya kalite iddiası yayınlanmadı.",
        ],
    }


def _person_distance_proxy_r1_unavailable(
    reason: str, *, integrity: dict[str, bool]
) -> dict[str, Any]:
    return {
        "evidence_version": "r1",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "metric_distance_established": False,
        "twenty_to_twenty_five_m_status": (
            "blocked_missing_per_camera_metric_calibration"
        ),
        "integrity": integrity,
    }


def _person_distance_proxy_r1(reader: ArtifactReader) -> dict[str, Any]:
    integrity = {
        "report_exact_pin_verified": False,
        "receipt_exact_pin_verified": False,
        "report_fingerprint_replayed": False,
        "receipt_fingerprint_replayed": False,
        "report_receipt_binding_verified": False,
        "baseline_r10_lineage_verified": False,
        "proxy_only_semantics_verified": False,
    }
    values: dict[str, dict[str, Any]] = {}
    for key, pin in PERSON_RTDETR_DISTANCE_PROXY_R1_PINS.items():
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=str(pin["path"]),
            maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        )
        integrity[f"{key}_exact_pin_verified"] = result.available
        if value is None:
            return _person_distance_proxy_r1_unavailable(
                f"distance_proxy_{key}_{result.state}", integrity=integrity
            )
        values[key] = value

    report = values["report"]
    receipt = values["receipt"]
    integrity["report_fingerprint_replayed"] = _self_fingerprint_matches(
        report
    )
    integrity["receipt_fingerprint_replayed"] = _self_fingerprint_matches(
        receipt
    )
    integrity["report_receipt_binding_verified"] = bool(
        receipt.get("report") == PERSON_RTDETR_DISTANCE_PROXY_R1_PINS["report"]
    )
    lineage = receipt.get("lineage", {})
    input_pins = lineage.get("input_pins", {}) if isinstance(lineage, dict) else {}
    integrity["baseline_r10_lineage_verified"] = bool(
        isinstance(input_pins, dict)
        and lineage.get("baseline_run_id") == "baseline-eval-002"
        and lineage.get("plan_id")
        == "rtdetrv4-s-r-livit-person-r1-gpu-v1"
        and lineage.get("plan_fingerprint_sha256")
        == PERSON_RTDETR_GPU_R10_PLAN_FINGERPRINT
        and lineage.get("resolved_image_id") == PERSON_RTDETR_GPU_R10_IMAGE_ID
        and input_pins.get("build_receipt")
        == PERSON_RTDETR_GPU_R10_PINS["build_receipt"]
        and input_pins.get("host_receipt")
        == PERSON_RTDETR_GPU_R10_PINS["baseline_host_receipt"]
        and input_pins.get("container_receipt")
        == PERSON_RTDETR_GPU_R10_PINS["baseline_container_receipt"]
    )
    interpretation = report.get("interpretation", {})
    calibration = report.get("metric_20_25m_calibration_requirements", {})
    inventory = report.get("inventory", {})
    thresholds = report.get("overall_threshold_metrics", {})
    height_layers = report.get("gt_bbox_size_layers", {}).get(
        "height_px", {}
    )
    rows_025 = height_layers.get("0.25", []) if isinstance(height_layers, dict) else []
    height_by_id = {
        row.get("layer_id"): row
        for row in rows_025
        if isinstance(row, dict) and isinstance(row.get("layer_id"), str)
    }
    operating_025 = thresholds.get("0.25", {}) if isinstance(thresholds, dict) else {}
    tiny = height_by_id.get("le_16_px", {})
    large = height_by_id.get("gt_96_px", {})
    integrity["proxy_only_semantics_verified"] = bool(
        report.get("schema_version")
        == "deepsafe.person-baseline-distance-proxy-report/v1"
        and report.get("report_id")
        == (
            "rtdetrv4-s-r-livit-person-r1-gpu-v1-baseline-eval-002-"
            "distance-proxy-v1"
        )
        and report.get("status")
        == "passed_proxy_only_metric_distance_not_established"
        and receipt.get("schema_version")
        == "deepsafe.person-baseline-distance-proxy-receipt/v1"
        and receipt.get("status") == "passed"
        and receipt.get("receipt_id")
        == (
            "rtdetrv4-s-r-livit-person-r1-gpu-v1-baseline-eval-002-"
            "distance-proxy-v1-receipt"
        )
        and receipt.get("execution")
        == {
            "cpu_only_serialized_detection_replay": True,
            "docker_used": False,
            "gpu_used": False,
            "network_used": False,
            "official_test_opened": False,
            "source_image_payloads_opened": 0,
            "test_unseen_opened": False,
        }
        and inventory.get("images") == 384
        and inventory.get("ground_truth_persons") == 3256
        and inventory.get("source_sequences") == 32
        and inventory.get("capture_groups") == 26
        and interpretation.get(
            "bbox_pixel_size_is_apparent_scale_proxy_only"
        )
        is True
        and interpretation.get("pixel_size_is_metric_distance") is False
        and interpretation.get("detection_at_20m_established") is False
        and interpretation.get("detection_at_25m_established") is False
        and calibration.get("proxy_can_establish_20m_or_25m") is False
        and calibration.get("metric_distance_acceptance_status")
        == "blocked_missing_per_camera_metric_calibration"
        and operating_025.get("tp") == 1502
        and operating_025.get("fp") == 1102
        and operating_025.get("fn") == 1754
        and operating_025.get("precision") == 0.576804916
        and operating_025.get("recall") == 0.461302211
        and operating_025.get("f1") == 0.512627986
        and tiny.get("metrics", {}).get("recall") == 0.010810811
        and large.get("metrics", {}).get("recall") == 0.91283293
    )
    if not all(integrity.values()):
        return _person_distance_proxy_r1_unavailable(
            "distance_proxy_cross_artifact_contract_invalid",
            integrity=integrity,
        )
    return {
        "evidence_version": "r1",
        "available": True,
        "state": "pixel_scale_proxy_passed_metric_distance_blocked",
        "reason": "missing_per_camera_metric_calibration",
        "metric_distance_established": False,
        "twenty_m_established": False,
        "twenty_five_m_established": False,
        "twenty_to_twenty_five_m_status": (
            "blocked_missing_per_camera_metric_calibration"
        ),
        "dataset": {
            "images": 384,
            "ground_truth_persons": 3256,
            "sequences": 32,
            "capture_groups": 26,
            "official_test_opened": False,
            "test_unseen_opened": False,
        },
        "operating_point_score_0_25": {
            "iou_threshold": 0.5,
            "tp": 1502,
            "fp": 1102,
            "fn": 1754,
            "precision": 0.576804916,
            "recall": 0.461302211,
            "f1": 0.512627986,
        },
        "pixel_height_proxy": {
            "le_16_px_recall": 0.010810811,
            "gt_96_px_recall": 0.91283293,
        },
        "calibration_required": {
            "profiles": [640, 960],
            "distances_m": [20.0, 25.0],
            "conditions": ["day", "night"],
            "per_camera": True,
            "pooled_and_worst_camera": True,
        },
        "integrity": integrity,
        "caveats": [
            "Kutu piksel yüksekliği yalnız görünür ölçek proxy'sidir; metre cinsinden mesafe değildir.",
            "20 m ve 25 m kabulü kamera başına intrinsik/ekstrinsik kalibrasyon ve ölçülmüş mesafe etiketleri gelene kadar kapalıdır.",
        ],
    }


def _person_full_training_r10_unavailable(
    reason: str, *, integrity: dict[str, bool]
) -> dict[str, Any]:
    return {
        "evidence_version": "r10-full-60e-001",
        "available": False,
        "state": "full_training_evidence_not_available",
        "reason": reason,
        "full_training_started": None,
        "completion_evidence_present": False,
        "full_training_complete": False,
        "export_complete": False,
        "tensorrt_complete": False,
        "deepstream9_complete": False,
        "twelve_camera_capacity_complete": False,
        "production_ready": False,
        "integrity": integrity,
        "caveats": [
            "Tam eğitim receipt zinciri doğrulanamadı; tamamlanma iddiası fail-closed kaldı.",
        ],
    }


def _person_full_training_r10(reader: ArtifactReader) -> dict[str, Any]:
    integrity = {
        "host_receipt_exact_pin_verified": False,
        "container_receipt_exact_pin_verified": False,
        "events_exact_pin_verified": False,
        "best_checkpoint_exact_pin_verified": False,
        "host_fingerprint_replayed": False,
        "container_fingerprint_replayed": False,
        "cross_artifact_bindings_verified": False,
        "host_semantics_verified": False,
        "container_semantics_verified": False,
        "epoch_event_replay_verified": False,
        "best_checkpoint_binding_verified": False,
    }
    values: dict[str, dict[str, Any]] = {}
    for key in ("host_receipt", "container_receipt"):
        pin = PERSON_RTDETR_FULL_TRAINING_R10_PINS[key]
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=str(pin["path"]),
            maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        )
        integrity[f"{key}_exact_pin_verified"] = result.available
        if value is None:
            return _person_full_training_r10_unavailable(
                f"full_training_{key}_{result.state}", integrity=integrity
            )
        values[key] = value

    events_pin = PERSON_RTDETR_FULL_TRAINING_R10_PINS["events"]
    events_read = _read_workspace_pin(
        reader,
        events_pin,
        expected_path=str(events_pin["path"]),
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        collect=True,
    )
    integrity["events_exact_pin_verified"] = events_read.available
    if not events_read.available or events_read.content is None:
        return _person_full_training_r10_unavailable(
            f"full_training_events_{events_read.state}", integrity=integrity
        )
    events: list[dict[str, Any]] = []
    try:
        for raw_line in events_read.content.splitlines():
            if not raw_line:
                continue
            row = strict_json_loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError("event row is not an object")
            events.append(row)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ):
        return _person_full_training_r10_unavailable(
            "full_training_events_invalid_jsonl", integrity=integrity
        )

    checkpoint_pin = PERSON_RTDETR_FULL_TRAINING_R10_PINS[
        "best_checkpoint"
    ]
    checkpoint_read = _read_workspace_pin(
        reader,
        checkpoint_pin,
        expected_path=str(checkpoint_pin["path"]),
        maximum_bytes=PERSON_RTDETR_FULL_TRAINING_R10_MAX_CHECKPOINT_BYTES,
        collect=False,
    )
    integrity["best_checkpoint_exact_pin_verified"] = (
        checkpoint_read.available
    )
    if not checkpoint_read.available:
        return _person_full_training_r10_unavailable(
            f"full_training_best_checkpoint_{checkpoint_read.state}",
            integrity=integrity,
        )

    host = values["host_receipt"]
    container = values["container_receipt"]
    integrity["host_fingerprint_replayed"] = _self_fingerprint_matches(host)
    integrity["container_fingerprint_replayed"] = _self_fingerprint_matches(
        container
    )
    expected_plan = {
        "bytes": PERSON_RTDETR_GPU_R10_PINS["plan"]["bytes"],
        "fingerprint_sha256": PERSON_RTDETR_GPU_R10_PLAN_FINGERPRINT,
        "path": (
            "models/person/training-lanes/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/execution-plan.json"
        ),
        "sha256": PERSON_RTDETR_GPU_R10_PINS["plan"]["sha256"],
    }
    expected_container_pin = {
        "bytes": PERSON_RTDETR_FULL_TRAINING_R10_PINS["container_receipt"][
            "bytes"
        ],
        "fingerprint_sha256": container.get("fingerprint_sha256"),
        "path": "container-receipt.json",
        "sha256": PERSON_RTDETR_FULL_TRAINING_R10_PINS[
            "container_receipt"
        ]["sha256"],
    }
    host_artifacts = {
        item.get("path"): item
        for item in host.get("artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    container_artifacts = {
        item.get("path"): item
        for item in container.get("artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    integrity["cross_artifact_bindings_verified"] = bool(
        host.get("plan") == expected_plan
        and host.get("build_receipt")
        == PERSON_RTDETR_GPU_R10_PINS["build_receipt"]
        and host.get("container_receipt") == expected_container_pin
        and container.get("plan", {}).get("bytes")
        == PERSON_RTDETR_GPU_R10_PINS["plan"]["bytes"]
        and container.get("plan", {}).get("sha256")
        == PERSON_RTDETR_GPU_R10_PINS["plan"]["sha256"]
        and container.get("plan", {}).get("fingerprint_sha256")
        == PERSON_RTDETR_GPU_R10_PLAN_FINGERPRINT
        and container.get("resolved_image_id")
        == PERSON_RTDETR_GPU_R10_IMAGE_ID
        and host.get("resolved_image", {}).get("id")
        == PERSON_RTDETR_GPU_R10_IMAGE_ID
        and host_artifacts.get("container-receipt.json")
        == {
            "bytes": PERSON_RTDETR_FULL_TRAINING_R10_PINS[
                "container_receipt"
            ]["bytes"],
            "path": "container-receipt.json",
            "sha256": PERSON_RTDETR_FULL_TRAINING_R10_PINS[
                "container_receipt"
            ]["sha256"],
        }
        and host_artifacts.get("events.jsonl")
        == {
            "bytes": events_pin["bytes"],
            "path": "events.jsonl",
            "sha256": events_pin["sha256"],
        }
        and container_artifacts.get("events.jsonl")
        == {
            "bytes": events_pin["bytes"],
            "path": "events.jsonl",
            "sha256": events_pin["sha256"],
        }
    )

    isolation = host.get("runtime_isolation", {})
    gpu_lease = isolation.get("gpu_lease", {}) if isinstance(isolation, dict) else {}
    integrity["host_semantics_verified"] = bool(
        host.get("schema_version")
        == "deepsafe.person-rtdetrv4-gpu-host-receipt/v1"
        and host.get("receipt_id")
        == "rtdetrv4-s-r-livit-person-r1-gpu-v1-full-60e-001-host"
        and host.get("run_id") == "full-60e-001"
        and host.get("mode") == "full_run"
        and host.get("status") == "passed"
        and host.get("docker_exit_status") == 0
        and host.get("launch_error") is None
        and host.get("validation_error") is None
        and host.get("terminated_by_signal") is None
        and host.get("inputs", {}).get("unchanged") is True
        and isinstance(host.get("duration_seconds"), (int, float))
        and not isinstance(host.get("duration_seconds"), bool)
        and 4800 <= host["duration_seconds"] <= 4900
        and host.get("gpu", {}).get("name")
        == "NVIDIA RTX A5000 Laptop GPU"
        and isolation.get("network") == "none"
        and isolation.get("root_filesystem_read_only") is True
        and isolation.get("dataset_mount_read_only") is True
        and isolation.get("publisher_checkpoint_mount_read_only") is True
        and isolation.get("no_new_privileges") is True
        and isolation.get("capabilities_dropped") == "ALL"
        and gpu_lease.get("required") is True
        and gpu_lease.get("owner_kind") == "person_training"
    )

    execution = container.get("execution", {})
    dataset = container.get("dataset", {})
    final_metrics = execution.get("final_metrics", {})
    final_validation = final_metrics.get("validation", {}).get(
        "coco_eval_bbox", []
    )
    eval_buffers = execution.get("eval_position_buffers", {})
    checkpoints = execution.get("checkpoints", [])
    milestones = execution.get("milestone_checkpoints", [])
    expected_final_metrics = [
        0.43601712785856694,
        0.7766765567535977,
        0.42477936368623376,
        0.3108083595649871,
        0.6237666627187156,
        0.686473344812774,
        0.07678132678132679,
        0.41885749385749377,
        0.5418304668304669,
        0.4504803073967338,
        0.70009765625,
        0.7306666666666667,
    ]
    integrity["container_semantics_verified"] = bool(
        container.get("schema_version")
        == "deepsafe.person-rtdetrv4-gpu-container-receipt/v1"
        and container.get("receipt_id")
        == "rtdetrv4-s-r-livit-person-r1-gpu-v1-full-60e-001"
        and container.get("run_id") == "full-60e-001"
        and container.get("mode") == "full_run"
        and container.get("status") == "passed"
        and container.get("container_exit_status") == 0
        and container.get("official_test_opened") is False
        and container.get("test_unseen_opened") is False
        and dataset.get("official_test_opened") is False
        and dataset.get("test_unseen_opened") is False
        and dataset.get("train_val_capture_group_overlap") == 0
        and dataset.get("train_val_sequence_overlap") == 0
        and dataset.get("train", {}).get("images") == 1524
        and dataset.get("val")
        == {
            "annotations": 3256,
            "capture_groups": 26,
            "images": 384,
            "sequences": 32,
        }
        and execution.get("mode") == "full_run"
        and execution.get("start_epoch") == 0
        and execution.get("final_epoch_completed") == 59
        and execution.get("epochs_executed_this_session") == 60
        and execution.get("total_epochs_contract") == 60
        and execution.get("validation_each_epoch") is True
        and execution.get("fixed_training_resolution") == [640, 640]
        and execution.get("total_batch_size") == 8
        and execution.get("train_batches_per_epoch") == 190
        and execution.get("training_images") == 1524
        and execution.get("validation_images") == 384
        and execution.get("amp") is True
        and execution.get("ema") is True
        and execution.get("teacher_model") is False
        and execution.get("distillation_loss") is False
        and execution.get("resume") is None
        and execution.get("best_coco_ap") == 0.43638015456949064
        and final_metrics.get("epoch") == 59
        and final_metrics.get("coco_ap") == 0.43601712785856694
        and final_metrics.get("best_coco_ap") == 0.43638015456949064
        and final_metrics.get("ema_updates") == 11400
        and final_validation == expected_final_metrics
        and execution.get("max_cuda_memory_allocated_bytes") == 4021828608
        and isinstance(checkpoints, list)
        and len(checkpoints) == 14
        and isinstance(milestones, list)
        and len(milestones) == 12
        and [item.get("path") for item in milestones]
        == [
            f"/output/checkpoints/epoch-{epoch:04d}.pth"
            for epoch in range(5, 61, 5)
        ]
        and all(
            eval_buffers.get(owner, {}).get("policy_id")
            == "rtdetrv4-hybrid-encoder-eval-position-buffer-v1"
            and eval_buffers.get(owner, {}).get("persistent") is False
            and eval_buffers.get(owner, {}).get(
                "state_dict_semantics_changed"
            )
            is False
            and eval_buffers.get(owner, {})
            .get("after_model_to", {})
            .get("expected_device")
            == "cuda:0"
            for owner in ("model", "ema")
        )
    )

    epoch_events = [row for row in events if row.get("event") == "epoch_completed"]
    start_events = [row for row in events if row.get("event") == "run_started"]
    complete_events = [row for row in events if row.get("event") == "run_completed"]
    best_event = (
        max(epoch_events, key=lambda row: float(row.get("coco_ap", -1)))
        if epoch_events
        else {}
    )
    final_event = epoch_events[-1] if epoch_events else {}
    expected_best_metrics = [
        0.43638015456949064,
        0.775500084455937,
        0.4244025805465684,
        0.30936732938144523,
        0.6244955582420204,
        0.6952413929145632,
        0.07702702702702702,
        0.41919533169533174,
        0.5417383292383293,
        0.4504803073967339,
        0.69892578125,
        0.7366666666666666,
    ]
    integrity["epoch_event_replay_verified"] = bool(
        len(events) == 62
        and len(start_events) == 1
        and len(complete_events) == 1
        and len(epoch_events) == 60
        and [row.get("epoch") for row in epoch_events] == list(range(60))
        and all(
            row.get("schema_version")
            == "deepsafe.person-rtdetrv4-training-event/v1"
            and isinstance(row.get("validation", {}).get("coco_eval_bbox"), list)
            and len(row["validation"]["coco_eval_bbox"]) == 12
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in row["validation"]["coco_eval_bbox"]
            )
            for row in epoch_events
        )
        and start_events[0].get("run_id") == "full-60e-001"
        and start_events[0].get("mode") == "full_run"
        and start_events[0].get("plan_fingerprint_sha256")
        == PERSON_RTDETR_GPU_R10_PLAN_FINGERPRINT
        and complete_events[0].get("run_id") == "full-60e-001"
        and complete_events[0].get("mode") == "full_run"
        and complete_events[0].get("status") == "passed"
        and best_event.get("epoch") == 53
        and best_event.get("coco_ap") == 0.43638015456949064
        and best_event.get("best_improved") is True
        and best_event.get("validation", {}).get("coco_eval_bbox")
        == expected_best_metrics
        and final_event.get("epoch") == 59
        and final_event.get("coco_ap") == 0.43601712785856694
        and final_event.get("validation", {}).get("coco_eval_bbox")
        == expected_final_metrics
        and final_event.get("ema_updates") == 11400
    )

    best_core = {
        "bytes": checkpoint_pin["bytes"],
        "sha256": checkpoint_pin["sha256"],
    }
    best_container = execution.get("best_checkpoint", {})
    best_host = host_artifacts.get("checkpoints/best.pth", {})
    integrity["best_checkpoint_binding_verified"] = bool(
        best_container.get("path") == "/output/checkpoints/best.pth"
        and {
            "bytes": best_container.get("bytes"),
            "sha256": best_container.get("sha256"),
        }
        == best_core
        and best_host
        == {
            "bytes": checkpoint_pin["bytes"],
            "path": "checkpoints/best.pth",
            "sha256": checkpoint_pin["sha256"],
        }
        and best_event.get("last_checkpoint", {}).get("sha256")
        == checkpoint_pin["sha256"]
    )
    # The final epoch and last checkpoint are intentionally distinct from the
    # best epoch; only the exact best artifact is promoted by this projection.
    integrity["best_checkpoint_binding_verified"] = bool(
        integrity["best_checkpoint_binding_verified"]
        and execution.get("last_checkpoint", {}).get("path")
        == "/output/checkpoints/last.pth"
        and execution.get("last_checkpoint", {}).get("sha256")
        == "3bdbaa89cbbbe1984e1f5d11412d2c3174bb725c33f8a929c83b66dcbe8adaec"
    )
    if not all(integrity.values()):
        return _person_full_training_r10_unavailable(
            "full_training_cross_artifact_contract_invalid",
            integrity=integrity,
        )

    metric_names = (
        "ap_50_95",
        "ap_50",
        "ap_75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_max_det_1",
        "ar_max_det_10",
        "ar_max_det_100",
        "ar_small",
        "ar_medium",
        "ar_large",
    )
    return {
        "evidence_version": "r10-full-60e-001",
        "available": True,
        "state": "full_training_completed_internal_validation_only",
        "reason": "deployment_and_independent_acceptance_pending",
        "run_id": "full-60e-001",
        "full_training_started": True,
        "completion_evidence_present": True,
        "full_training_complete": True,
        "epochs": {
            "completed": 60,
            "contract": 60,
            "validation_runs": 60,
            "best_zero_based_epoch": 53,
            "final_zero_based_epoch": 59,
        },
        "duration": {
            "host_seconds": host.get("duration_seconds"),
            "container_seconds": execution.get("duration_seconds"),
        },
        "dataset_boundary": {
            "training_images": 1524,
            "validation_images": 384,
            "validation_annotations": 3256,
            "official_test_opened": False,
            "test_unseen_opened": False,
            "independent_final_test": False,
        },
        "coco": {
            "metric_order": list(metric_names),
            "best": dict(zip(metric_names, expected_best_metrics, strict=True)),
            "final": dict(zip(metric_names, expected_final_metrics, strict=True)),
        },
        "checkpoints": {
            "best_exact_pin_verified": True,
            "best_checkpoint_present": True,
            "last_checkpoint_present": True,
            "milestone_checkpoint_count": 12,
            "published_checkpoint_artifact_count": 14,
            "best_is_final": False,
        },
        "gpu": {
            "max_cuda_memory_allocated_bytes": 4021828608,
            "max_cuda_memory_allocated_gib": round(
                4021828608 / (1024**3), 3
            ),
        },
        "export_complete": False,
        "tensorrt_complete": False,
        "deepstream9_complete": False,
        "twelve_camera_capacity_complete": False,
        "independent_ground_truth_quality_passed": False,
        "production_ready": False,
        "integrity": integrity,
        "caveats": [
            "60/60 eğitim ve her epoch iç validation tamamlandı; metrikler grup-safe iç validation ayrımına aittir.",
            "Resmî test ve test-unseen açılmadı; bu receipt bağımsız final test veya ürün kabulü değildir.",
            "Best checkpoint exact-pinlidir; yeni ONNX/TensorRT export, DeepStream 9, 12-kamera kapasite ve production kapıları kapalıdır.",
        ],
    }


def _person_export_plan_r11_unavailable(
    reason: str, *, integrity: dict[str, bool]
) -> dict[str, Any]:
    return {
        "evidence_version": "r11",
        "available": False,
        "state": "export_plan_evidence_not_available",
        "reason": reason,
        "plan_ready": False,
        "execution_authorized": False,
        "export_executed": False,
        "model_loaded": False,
        "onnx_exported": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "twelve_camera_capacity_passed": False,
        "production_ready": False,
        "integrity": integrity,
    }


def _person_export_plan_r11(
    reader: ArtifactReader, *, full_training_verified: bool
) -> dict[str, Any]:
    """Project the exact R11 plan without promoting any execution gate."""

    integrity = {
        "plan_exact_pin_verified": False,
        "contract_exact_pin_verified": False,
        "plan_fingerprint_replayed": False,
        "contract_fingerprint_replayed": False,
        "plan_contract_binding_verified": False,
        "training_lineage_verified": False,
        "full_training_gate_verified": False,
        "plan_semantics_verified": False,
        "contract_semantics_verified": False,
        "profiles_verified": False,
        "execution_boundary_verified": False,
    }
    values: dict[str, dict[str, Any]] = {}
    for key, pin in PERSON_RTDETR_EXPORT_R11_PINS.items():
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=str(pin["path"]),
            maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        )
        integrity[f"{key}_exact_pin_verified"] = result.available
        if value is None:
            return _person_export_plan_r11_unavailable(
                f"export_r11_{key}_{result.state}", integrity=integrity
            )
        values[key] = value

    plan = values["plan"]
    contract = values["contract"]
    integrity["plan_fingerprint_replayed"] = bool(
        _self_fingerprint_matches(plan)
        and plan.get("fingerprint_sha256")
        == PERSON_RTDETR_EXPORT_R11_PLAN_FINGERPRINT
    )
    integrity["contract_fingerprint_replayed"] = bool(
        _self_fingerprint_matches(contract)
        and contract.get("fingerprint_sha256")
        == PERSON_RTDETR_EXPORT_R11_CONTRACT_FINGERPRINT
    )
    integrity["plan_contract_binding_verified"] = bool(
        plan.get("contract")
        == {
            **PERSON_RTDETR_EXPORT_R11_PINS["contract"],
            "fingerprint_sha256": (
                PERSON_RTDETR_EXPORT_R11_CONTRACT_FINGERPRINT
            ),
        }
    )

    lineage = plan.get("training_lineage", {})
    expected_r10_plan = {
        **PERSON_RTDETR_GPU_R10_PINS["plan"],
        "fingerprint_sha256": PERSON_RTDETR_GPU_R10_PLAN_FINGERPRINT,
    }
    expected_build = {
        **PERSON_RTDETR_GPU_R10_PINS["build_receipt"],
        "fingerprint_sha256": (
            "ea144d449951860496a905f8337d0a664b4a64d2733e0b3606c8092548b1a0dd"
        ),
    }
    expected_smoke_host = {
        **PERSON_RTDETR_GPU_R10_PINS["smoke_host_receipt"],
        "fingerprint_sha256": (
            "b0da4e25459b543733e150989a1dfdeb957b6c21d2eed6482fe5c7d789985f8b"
        ),
    }
    expected_baseline_host = {
        **PERSON_RTDETR_GPU_R10_PINS["baseline_host_receipt"],
        "fingerprint_sha256": (
            "6da1daaebe3be5feb8d0875a6b03e8f0e92f3a342bd70b2a2b4108a3d59e275c"
        ),
    }
    expected_full_host = {
        **PERSON_RTDETR_FULL_TRAINING_R10_PINS["host_receipt"],
        "fingerprint_sha256": (
            "5dda919138b25a766dbd263c9a0de68f3a11e937e4c1b637ac9c1a9c0a676679"
        ),
    }
    expected_full_container = {
        **PERSON_RTDETR_FULL_TRAINING_R10_PINS["container_receipt"],
        "fingerprint_sha256": (
            "377e4543f31ab1522366d2fedc1af5ef361be90949c67b0cc77ae3a7f74f84c5"
        ),
    }
    integrity["training_lineage_verified"] = bool(
        isinstance(lineage, dict)
        and lineage.get("r10_execution_plan") == expected_r10_plan
        and lineage.get("r10_build_receipt") == expected_build
        and lineage.get("smoke_host_receipt") == expected_smoke_host
        and lineage.get("baseline_host_receipt") == expected_baseline_host
        and lineage.get("full_host_receipt") == expected_full_host
        and lineage.get("full_container_receipt")
        == expected_full_container
        and lineage.get("best_checkpoint")
        == PERSON_RTDETR_FULL_TRAINING_R10_PINS["best_checkpoint"]
        and lineage.get("checkpoint_selection")
        == {
            "best_coco_ap": 0.43638015456949064,
            "checkpoint_payload": "ema.module",
            "final_epoch_completed": 59,
            "official_test_opened": False,
            "selection_metric": "best_coco_ap",
            "test_unseen_opened": False,
            "total_epochs_contract": 60,
        }
    )
    integrity["full_training_gate_verified"] = full_training_verified is True

    expected_stages = [
        "onnx_export_640",
        "onnx_export_960",
        "threshold_calibration",
        "tensorrt_fp16_640",
        "tensorrt_fp16_960",
        "numerical_parity_640",
        "numerical_parity_960",
        "deepstream_parser_parity_640",
        "deepstream_parser_parity_960",
    ]
    calibration = plan.get("calibration_contract", {})
    integrity["plan_semantics_verified"] = bool(
        plan.get("schema_version")
        == "deepsafe.person-rtdetrv4-trained-export-plan/r11"
        and plan.get("plan_id") == "rtdetrv4-s-r-livit-person-r11"
        and plan.get("status") == "authorized_not_executed"
        and plan.get("fingerprint_sha256")
        == PERSON_RTDETR_EXPORT_R11_PLAN_FINGERPRINT
        and plan.get("stage_order") == expected_stages
        and calibration.get("kind")
        == "person_score_threshold_calibration_not_int8"
        and calibration.get("profiles_calibrated_separately") == [640, 960]
        and calibration.get("official_test_opened") is False
        and calibration.get("test_unseen_opened") is False
        and calibration.get("quality_claim_from_calibration") is False
    )
    gate = contract.get("full_run_gate", {})
    integrity["contract_semantics_verified"] = bool(
        contract.get("schema_version")
        == "deepsafe.person-rtdetrv4-trained-export-contract/r11"
        and contract.get("contract_id") == "rtdetrv4-s-r-livit-person-r11"
        and contract.get("status")
        == "prepared_waiting_for_passed_full_run"
        and contract.get("fingerprint_sha256")
        == PERSON_RTDETR_EXPORT_R11_CONTRACT_FINGERPRINT
        and gate.get("required_run_id") == "full-60e-001"
        and gate.get("required_mode") == "full_run"
        and gate.get("required_status") == "passed"
        and gate.get("required_final_epoch_completed") == 59
        and gate.get("required_total_epochs_contract") == 60
        and gate.get("required_checkpoint_relative_path")
        == "checkpoints/best.pth"
        and gate.get("plan_creation_before_gate") is False
        and gate.get("execution_before_gate") is False
    )

    profiles = plan.get("profiles", {})

    def profile_valid(profile: str, spatial: int) -> bool:
        item = profiles.get(profile, {})
        return bool(
            item.get("spatial") == [spatial, spatial]
            and item.get("precision") == "FP16"
            and item.get("deepstream_batch_size") == 12
            and item.get("images_profile")
            == {
                "min": [1, 3, spatial, spatial],
                "opt": [12, 3, spatial, spatial],
                "max": [12, 3, spatial, spatial],
            }
            and item.get("orig_target_sizes_profile")
            == {
                "min": [1, 2],
                "opt": [12, 2],
                "max": [12, 2],
            }
            and contract.get("profiles", {}).get(profile) == item
        )

    integrity["profiles_verified"] = bool(
        set(profiles) == {"640", "960"}
        and profile_valid("640", 640)
        and profile_valid("960", 960)
    )
    integrity["execution_boundary_verified"] = bool(
        plan.get("execution")
        == {
            "deepstream9": False,
            "docker": False,
            "gpu": False,
            "model_loaded": False,
            "onnx_export": False,
            "performed_during_preparation": False,
            "tensorrt": False,
        }
        and plan.get("acceptance")
        == {
            "deepstream_parser_parity_640_passed": False,
            "deepstream_parser_parity_960_passed": False,
            "exact_25m_passed": False,
            "numerical_parity_640_passed": False,
            "numerical_parity_960_passed": False,
            "onnx_640_passed": False,
            "onnx_960_passed": False,
            "production_ready": False,
            "tensorrt_fp16_640_passed": False,
            "tensorrt_fp16_960_passed": False,
            "threshold_calibration_passed": False,
            "twelve_camera_capacity_passed": False,
        }
        and contract.get("execution_history")
        == {
            "docker": False,
            "gpu": False,
            "model_loaded": False,
            "onnx_export": False,
            "tensorrt": False,
            "deepstream9": False,
        }
        and contract.get("claim_boundary")
        == {
            "quality": False,
            "exact_25m": False,
            "capacity": False,
            "three_module_full_stack": False,
            "production_ready": False,
        }
    )
    if not all(integrity.values()):
        return _person_export_plan_r11_unavailable(
            "export_r11_cross_artifact_contract_invalid",
            integrity=integrity,
        )

    return {
        "evidence_version": "r11",
        "available": True,
        "state": "authorized_plan_ready_not_executed",
        "reason": "export_execution_pending",
        "plan_ready": True,
        "execution_authorized": True,
        "export_executed": False,
        "model_loaded": False,
        "onnx_exported": False,
        "profiles": {
            "640": {"spatial": [640, 640], "batch": 12, "precision": "FP16"},
            "960": {"spatial": [960, 960], "batch": 12, "precision": "FP16"},
        },
        "threshold_calibration_executed": False,
        "tensorrt_executed": False,
        "numerical_parity_passed": False,
        "deepstream9_executed": False,
        "deepstream_parser_parity_passed": False,
        "twelve_camera_capacity_passed": False,
        "exact_25m_passed": False,
        "production_ready": False,
        "integrity": integrity,
        "caveats": [
            "R11 yalnız exact-pinli, yetkilendirilmiş yürütme planıdır; model yükleme ve export yapılmadı.",
            "ONNX, TensorRT FP16, DeepStream 9 parser parity ve 12-kamera kapasite kapıları çalıştırılana kadar kapalıdır.",
        ],
    }


def _person_onnx_r12_unavailable(
    reason: str, *, integrity: dict[str, bool] | None = None
) -> dict[str, Any]:
    return {
        "evidence_version": "r12",
        "available": False,
        "state": "no_published_onnx_receipt_fail_closed",
        "reason": reason,
        "profiles": {
            "640": {"receipt_verified": False, "onnx_exported": False},
            "960": {"receipt_verified": False, "onnx_exported": False},
        },
        "onnx_640_exported": False,
        "onnx_960_exported": False,
        "both_profiles_exported": False,
        "model_loaded": None,
        "gpu_executed": None,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "quality_passed": False,
        "twelve_camera_capacity_passed": False,
        "production_ready": False,
        "integrity": integrity or {},
        "caveats": [
            "Yalnız exact file-pinli, passed R12 stage receipt'i ONNX başarısı sayılır; tek başına ONNX veya recovery intent taranmaz.",
            "Receipt öncesi anonymous/pre-publication çalışma başarı sayılmaz; durable receipt yokken model-load/compute etkinliği bu kartta bilinmiyor kalır.",
        ],
    }


def _person_onnx_r12_expected_bindings(profile: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "images",
            "io": "input",
            "dtype": "FLOAT32",
            "shape": ["batch", 3, profile, profile],
        },
        {
            "name": "orig_target_sizes",
            "io": "input",
            "dtype": "INT64",
            "shape": ["batch", 2],
        },
        {
            "name": "labels",
            "io": "output",
            "dtype": "INT64",
            "shape": ["batch", 300],
        },
        {
            "name": "boxes",
            "io": "output",
            "dtype": "FLOAT32",
            "shape": ["batch", 300, 4],
        },
        {
            "name": "scores",
            "io": "output",
            "dtype": "FLOAT32",
            "shape": ["batch", 300],
        },
    ]


def _person_onnx_r12_receipt_semantics_valid(
    value: Any,
    *,
    profile: int,
    schema: dict[str, Any],
    expected_prior: list[dict[str, Any]],
) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        _validate_schema_node(value, schema, schema)
    except (TypeError, ValueError, RecursionError):
        return False
    payload = value.get("payload", {})
    parity = payload.get("framework_onnx_parity", {})
    plan_pin = {
        **PERSON_RTDETR_EXPORT_R11_PINS["plan"],
        "fingerprint_sha256": PERSON_RTDETR_EXPORT_R11_PLAN_FINGERPRINT,
    }
    checkpoint = PERSON_RTDETR_FULL_TRAINING_R10_PINS["best_checkpoint"]
    expected_onnx_path = (
        "models/person/export-lanes/rtdetrv4-s-r-livit-person-r11/"
        f"onnx/{profile}/rtdetrv4-s-r11-{profile}-bdynamic-opset18.onnx"
    )
    return bool(
        value.get("schema_version")
        == "deepsafe.person-rtdetrv4-trained-export-evidence/r11"
        and value.get("receipt_id")
        == f"rtdetrv4-s-r11-onnx-export-{profile}"
        and value.get("status") == "passed"
        and value.get("stage") == f"onnx_export_{profile}"
        and _self_fingerprint_matches(value)
        and value.get("plan") == plan_pin
        and value.get("prior_receipts") == expected_prior
        and value.get("execution")
        == {
            "docker": False,
            "gpu": False,
            "model_loaded": True,
            "onnx": True,
            "tensorrt": False,
            "deepstream9": False,
            "network_downloads": 0,
        }
        and value.get("claim_boundary")
        == {
            "quality": False,
            "exact_25m": False,
            "twelve_camera_capacity": False,
            "three_module_full_stack": False,
            "production_ready": False,
        }
        and payload.get("profile") == profile
        and payload.get("checkpoint") == checkpoint
        and payload.get("checkpoint_payload") == "ema.module"
        and payload.get("opset") == 18
        and payload.get("checker_passed") is True
        and payload.get("bindings")
        == _person_onnx_r12_expected_bindings(profile)
        and payload.get("batch12_shape_finite") is True
        and payload.get("passed") is True
        and payload.get("onnx", {}).get("path") == expected_onnx_path
        and isinstance(payload.get("onnx", {}).get("bytes"), int)
        and not isinstance(payload.get("onnx", {}).get("bytes"), bool)
        and 1 <= payload["onnx"]["bytes"] <= PERSON_RTDETR_ONNX_R12_MAX_ONNX_BYTES
        and _sha256(payload.get("onnx", {}).get("sha256")) is not None
        and parity.get("batches") == [1, 12]
        and parity.get("labels_class_exact") is True
        and parity.get("passed") is True
        and isinstance(parity.get("boxes_max_abs"), (int, float))
        and not isinstance(parity.get("boxes_max_abs"), bool)
        and math.isfinite(float(parity["boxes_max_abs"]))
        and 0 <= float(parity["boxes_max_abs"]) <= 0.02
        and isinstance(parity.get("scores_max_abs"), (int, float))
        and not isinstance(parity.get("scores_max_abs"), bool)
        and math.isfinite(float(parity["scores_max_abs"]))
        and 0 <= float(parity["scores_max_abs"]) <= 0.0002
    )


def _person_onnx_r12(reader: ArtifactReader) -> dict[str, Any]:
    configured = PERSON_RTDETR_ONNX_R12_RECEIPT_PINS
    if not isinstance(configured, dict) or not set(configured).issubset(
        {640, 960}
    ):
        return _person_onnx_r12_unavailable("r12_receipt_pin_set_invalid")
    if not configured:
        return _person_onnx_r12_unavailable(
            "r12_no_independently_pinned_receipt"
        )
    if 960 in configured and 640 not in configured:
        return _person_onnx_r12_unavailable(
            "r12_960_without_pinned_640_predecessor"
        )

    integrity = {
        "schema_exact_pin_verified": False,
        "schema_identity_verified": False,
        "stage_order_verified": False,
    }
    for profile in configured:
        integrity.update(
            {
                f"receipt_{profile}_exact_pin_verified": False,
                f"receipt_{profile}_fingerprint_verified": False,
                f"receipt_{profile}_schema_semantics_verified": False,
                f"onnx_{profile}_live_pin_verified": False,
            }
        )
    schema_read, schema = _workspace_pin_json(
        reader,
        PERSON_RTDETR_ONNX_R12_SCHEMA_PIN,
        expected_path=PERSON_RTDETR_ONNX_R12_SCHEMA_PIN["path"],
        maximum_bytes=PERSON_RTDETR_ONNX_R12_MAX_RECEIPT_BYTES,
    )
    integrity["schema_exact_pin_verified"] = schema_read.available
    if schema is None:
        return _person_onnx_r12_unavailable(
            f"r12_schema_{schema_read.state}", integrity=integrity
        )
    integrity["schema_identity_verified"] = bool(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "person-rtdetrv4-trained-export-evidence-r11.schema.json"
        )
    )
    if not integrity["schema_identity_verified"]:
        return _person_onnx_r12_unavailable(
            "r12_schema_identity_invalid", integrity=integrity
        )

    receipts: dict[int, dict[str, Any]] = {}
    document_pins: dict[int, dict[str, Any]] = {}
    for profile in (640, 960):
        if profile not in configured:
            continue
        descriptor = configured[profile]
        pin = _person_pin_core(descriptor)
        if pin is None:
            return _person_onnx_r12_unavailable(
                f"r12_receipt_{profile}_pin_invalid", integrity=integrity
            )
        receipt_read, receipt = _workspace_pin_json(
            reader,
            pin,
            expected_path=PERSON_RTDETR_ONNX_R12_RECEIPT_PATHS[profile],
            maximum_bytes=PERSON_RTDETR_ONNX_R12_MAX_RECEIPT_BYTES,
        )
        integrity[f"receipt_{profile}_exact_pin_verified"] = (
            receipt_read.available
        )
        if receipt is None:
            return _person_onnx_r12_unavailable(
                f"r12_receipt_{profile}_{receipt_read.state}",
                integrity=integrity,
            )
        fingerprint = descriptor.get("fingerprint_sha256")
        integrity[f"receipt_{profile}_fingerprint_verified"] = bool(
            _sha256(fingerprint) is not None
            and receipt.get("fingerprint_sha256") == fingerprint
            and _self_fingerprint_matches(receipt)
        )
        expected_prior = [] if profile == 640 else [document_pins[640]]
        integrity[f"receipt_{profile}_schema_semantics_verified"] = (
            _person_onnx_r12_receipt_semantics_valid(
                receipt,
                profile=profile,
                schema=schema,
                expected_prior=expected_prior,
            )
        )
        onnx_pin = receipt.get("payload", {}).get("onnx")
        onnx_read = _read_workspace_pin(
            reader,
            onnx_pin,
            expected_path=str(onnx_pin.get("path", ""))
            if isinstance(onnx_pin, dict)
            else "",
            maximum_bytes=PERSON_RTDETR_ONNX_R12_MAX_ONNX_BYTES,
            collect=False,
        )
        integrity[f"onnx_{profile}_live_pin_verified"] = onnx_read.available
        if not all(
            integrity[key]
            for key in (
                f"receipt_{profile}_exact_pin_verified",
                f"receipt_{profile}_fingerprint_verified",
                f"receipt_{profile}_schema_semantics_verified",
                f"onnx_{profile}_live_pin_verified",
            )
        ):
            return _person_onnx_r12_unavailable(
                f"r12_profile_{profile}_contract_invalid",
                integrity=integrity,
            )
        receipts[profile] = receipt
        document_pins[profile] = {
            **pin,
            "fingerprint_sha256": fingerprint,
        }

    integrity["stage_order_verified"] = bool(
        640 in receipts
        and (
            960 not in receipts
            or receipts[960].get("prior_receipts") == [document_pins[640]]
        )
    )
    if not integrity["stage_order_verified"]:
        return _person_onnx_r12_unavailable(
            "r12_stage_order_invalid", integrity=integrity
        )
    both = set(receipts) == {640, 960}
    profiles: dict[str, dict[str, Any]] = {}
    for profile in (640, 960):
        receipt = receipts.get(profile)
        payload = receipt.get("payload", {}) if receipt else {}
        parity = payload.get("framework_onnx_parity", {})
        profiles[str(profile)] = {
            "receipt_verified": receipt is not None,
            "onnx_exported": receipt is not None,
            "dynamic_batch": [1, 12] if receipt is not None else [],
            "fixed_spatial": profile if receipt is not None else None,
            "opset": 18 if receipt is not None else None,
            "checker_passed": (
                payload.get("checker_passed") is True if receipt else False
            ),
            "batch12_shape_finite": (
                payload.get("batch12_shape_finite") is True
                if receipt
                else False
            ),
            "framework_onnx_parity_passed": (
                parity.get("passed") is True if receipt else False
            ),
        }
    return {
        "evidence_version": "r12",
        "available": True,
        "state": (
            "onnx_640_960_passed_claims_still_closed"
            if both
            else "onnx_640_passed_960_not_independently_pinned"
        ),
        "reason": (
            "later_deployment_stages_pending"
            if both
            else "r12_960_receipt_not_independently_pinned"
        ),
        "profiles": profiles,
        "onnx_640_exported": True,
        "onnx_960_exported": both,
        "both_profiles_exported": both,
        "model_loaded": True,
        "gpu_executed": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "quality_passed": False,
        "twelve_camera_capacity_passed": False,
        "production_ready": False,
        "integrity": integrity,
        "caveats": [
            "R12 yalnız ONNX checker, batch 1/12 şekil-sonluluk ve framework/ORT sayısal parite kanıtıdır.",
            "TensorRT, DeepStream 9, kalite, 20–25 m, 12-kamera kapasite ve production kapıları kapalıdır.",
        ],
    }


def _person_threshold_r13b_unavailable(
    reason: str, *, integrity: dict[str, bool] | None = None
) -> dict[str, Any]:
    return {
        "evidence_version": "r13b",
        "available": False,
        "state": "threshold_calibration_not_exactly_pinned_fail_closed",
        "reason": reason,
        "profiles": {
            "640": {"threshold": None, "full_sweep_exact_pin_verified": False},
            "960": {"threshold": None, "full_sweep_exact_pin_verified": False},
        },
        "threshold_calibration_executed": False,
        "cpu_only": None,
        "gpu_executed": None,
        "quality_passed": False,
        "metric_distance_passed": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "twelve_camera_capacity_passed": False,
        "production_ready": False,
        "integrity": integrity or {},
        "caveats": [
            "R13B planı, iki full-sweep dosyası ve final receipt exact-pin zinciri birlikte doğrulanmadan kalibrasyon tamamlanmış sayılmaz.",
        ],
    }


def _person_threshold_r13b(reader: ArtifactReader) -> dict[str, Any]:
    """Project the accepted R13B CPU threshold lane without loading sweep bodies.

    The two full sweeps total roughly 76 MB.  Their independently compiled
    byte/SHA pins are replayed from open file descriptors, while the small
    final R11 receipt supplies the selected thresholds and binds those exact
    sweep pins.  This keeps the admin endpoint bounded and does not reinterpret
    calibration as independent quality, distance, capacity or production proof.
    """

    integrity = {
        "plan_exact_pin_verified": False,
        "plan_fingerprint_replayed": False,
        "executor_exact_pin_verified": False,
        "sweep_schema_exact_pin_verified": False,
        "sweep_schema_identity_verified": False,
        "r11_evidence_schema_exact_pin_verified": False,
        "final_receipt_exact_pin_verified": False,
        "final_receipt_fingerprint_replayed": False,
        "final_receipt_schema_replayed": False,
        "sweep_640_exact_pin_verified": False,
        "sweep_960_exact_pin_verified": False,
        "cross_artifact_bindings_verified": False,
        "claim_boundary_verified": False,
    }

    plan_descriptor = PERSON_RTDETR_THRESHOLD_R13B_PINS["plan"]
    plan_pin = _person_pin_core(plan_descriptor)
    plan_read, plan = _workspace_pin_json(
        reader,
        plan_pin,
        expected_path=str(plan_descriptor["path"]),
        maximum_bytes=PERSON_RTDETR_THRESHOLD_R13B_MAX_JSON_BYTES,
    )
    integrity["plan_exact_pin_verified"] = plan_read.available
    if plan is None:
        return _person_threshold_r13b_unavailable(
            f"r13b_plan_{plan_read.state}", integrity=integrity
        )
    integrity["plan_fingerprint_replayed"] = bool(
        plan.get("fingerprint_sha256")
        == plan_descriptor["fingerprint_sha256"]
        and _self_fingerprint_matches(plan)
    )

    executor = PERSON_RTDETR_THRESHOLD_R13B_PINS["executor"]
    executor_read = _read_workspace_pin(
        reader,
        _person_pin_core(executor),
        expected_path=str(executor["path"]),
        maximum_bytes=PERSON_RTDETR_THRESHOLD_R13B_MAX_JSON_BYTES,
        collect=False,
    )
    integrity["executor_exact_pin_verified"] = executor_read.available

    sweep_schema_descriptor = PERSON_RTDETR_THRESHOLD_R13B_PINS["sweep_schema"]
    sweep_schema_read, sweep_schema = _workspace_pin_json(
        reader,
        _person_pin_core(sweep_schema_descriptor),
        expected_path=str(sweep_schema_descriptor["path"]),
        maximum_bytes=PERSON_RTDETR_THRESHOLD_R13B_MAX_JSON_BYTES,
    )
    integrity["sweep_schema_exact_pin_verified"] = sweep_schema_read.available
    integrity["sweep_schema_identity_verified"] = bool(
        isinstance(sweep_schema, dict)
        and sweep_schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and sweep_schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "person-rtdetrv4-threshold-sweep-r13b.schema.json"
        )
    )

    r11_schema_read, r11_schema = _workspace_pin_json(
        reader,
        PERSON_RTDETR_ONNX_R12_SCHEMA_PIN,
        expected_path=PERSON_RTDETR_ONNX_R12_SCHEMA_PIN["path"],
        maximum_bytes=PERSON_RTDETR_THRESHOLD_R13B_MAX_JSON_BYTES,
    )
    integrity["r11_evidence_schema_exact_pin_verified"] = r11_schema_read.available

    for profile in (640, 960):
        descriptor = PERSON_RTDETR_THRESHOLD_R13B_PINS[f"sweep_{profile}"]
        sweep_read = _read_workspace_pin(
            reader,
            _person_pin_core(descriptor),
            expected_path=str(descriptor["path"]),
            maximum_bytes=PERSON_RTDETR_THRESHOLD_R13B_MAX_SWEEP_BYTES,
            collect=False,
        )
        integrity[f"sweep_{profile}_exact_pin_verified"] = sweep_read.available

    receipt_descriptor = PERSON_RTDETR_THRESHOLD_R13B_PINS["final_receipt"]
    receipt_read, receipt = _workspace_pin_json(
        reader,
        _person_pin_core(receipt_descriptor),
        expected_path=str(receipt_descriptor["path"]),
        maximum_bytes=PERSON_RTDETR_THRESHOLD_R13B_MAX_JSON_BYTES,
    )
    integrity["final_receipt_exact_pin_verified"] = receipt_read.available
    if receipt is None:
        return _person_threshold_r13b_unavailable(
            f"r13b_final_receipt_{receipt_read.state}", integrity=integrity
        )
    integrity["final_receipt_fingerprint_replayed"] = bool(
        receipt.get("fingerprint_sha256")
        == receipt_descriptor["fingerprint_sha256"]
        and _self_fingerprint_matches(receipt)
    )
    if isinstance(r11_schema, dict):
        try:
            _validate_schema_node(receipt, r11_schema, r11_schema)
        except (TypeError, ValueError, RecursionError):
            pass
        else:
            integrity["final_receipt_schema_replayed"] = True

    expected_r11_plan = {
        **PERSON_RTDETR_EXPORT_R11_PINS["plan"],
        "fingerprint_sha256": PERSON_RTDETR_EXPORT_R11_PLAN_FINGERPRINT,
    }
    expected_onnx_receipts = [
        dict(PERSON_RTDETR_ONNX_R12_RECEIPT_PINS[profile])
        for profile in (640, 960)
    ]
    expected_sweep_pins = {
        str(profile): _person_pin_core(
            PERSON_RTDETR_THRESHOLD_R13B_PINS[f"sweep_{profile}"]
        )
        for profile in (640, 960)
    }
    payload = receipt.get("payload", {})
    profiles = payload.get("profiles", {})
    outputs = plan.get("outputs", {})
    provider = plan.get("provider_contract", {})
    plan_execution = plan.get("execution", {})
    integrity["cross_artifact_bindings_verified"] = bool(
        plan.get("schema_version")
        == "deepsafe.person-rtdetrv4-threshold-calibration-plan/r13b"
        and plan.get("plan_id") == "rtdetrv4-s-r11-threshold-calibration-r13b"
        and plan.get("status")
        == "authorized_not_executed_supersedes_r13_provider_inventory_preflight"
        and plan.get("implementation", {}).get("executor")
        == _person_pin_core(executor)
        and plan.get("inputs", {}).get("sweep_schema")
        == _person_pin_core(sweep_schema_descriptor)
        and plan.get("inputs", {}).get("r11_plan") == expected_r11_plan
        and plan.get("inputs", {}).get("onnx_receipts")
        == {
            str(profile): PERSON_RTDETR_ONNX_R12_RECEIPT_PINS[profile]
            for profile in (640, 960)
        }
        and outputs.get("full_sweep_receipts")
        == {
            str(profile): expected_sweep_pins[str(profile)]["path"]
            for profile in (640, 960)
        }
        and outputs.get("final_r11_receipt") == receipt_descriptor["path"]
        and provider.get("available_provider_inventory")
        == ["AzureExecutionProvider", "CPUExecutionProvider"]
        and provider.get("inference_session_requested_providers")
        == ["CPUExecutionProvider"]
        and provider.get("inference_session_active_providers")
        == ["CPUExecutionProvider"]
        and provider.get("azure_provider_active") is False
        and plan_execution.get("performed_during_preparation") is False
        and plan_execution.get("inference") is False
        and plan_execution.get("gpu") is False
        and receipt.get("schema_version")
        == "deepsafe.person-rtdetrv4-trained-export-evidence/r11"
        and receipt.get("receipt_id")
        == "rtdetrv4-s-r11-threshold-calibration-r13b"
        and receipt.get("status") == "passed"
        and receipt.get("stage") == "threshold_calibration"
        and receipt.get("plan") == expected_r11_plan
        and receipt.get("prior_receipts") == expected_onnx_receipts
        and receipt.get("execution")
        == {
            "deepstream9": False,
            "docker": False,
            "gpu": False,
            "model_loaded": True,
            "network_downloads": 0,
            "onnx": True,
            "tensorrt": False,
        }
        and payload.get("kind") == "person_score_threshold_calibration_not_int8"
        and payload.get("source_role")
        == "development_validation_seen_during_model_selection_not_independent_test"
        and payload.get("images") == 384
        and payload.get("capture_groups") == 26
        and payload.get("int8_calibration") is False
        and payload.get("official_test_opened") is False
        and payload.get("test_unseen_opened") is False
        and payload.get("passed") is True
        and isinstance(profiles, dict)
        and set(profiles) == {"640", "960"}
        and all(
            profiles[str(profile)].get("full_sweep_receipt")
            == expected_sweep_pins[str(profile)]
            and profiles[str(profile)].get("objective")
            == "max_f1_tie_break_higher_recall_then_lower_threshold"
            and profiles[str(profile)].get("threshold_finite") is True
            and isinstance(profiles[str(profile)].get("threshold"), (int, float))
            and not isinstance(profiles[str(profile)].get("threshold"), bool)
            and math.isfinite(float(profiles[str(profile)]["threshold"]))
            and 0 <= float(profiles[str(profile)]["threshold"]) <= 1
            for profile in (640, 960)
        )
    )
    expected_claim_boundary = {
        "exact_25m": False,
        "production_ready": False,
        "quality": False,
        "three_module_full_stack": False,
        "twelve_camera_capacity": False,
    }
    integrity["claim_boundary_verified"] = bool(
        receipt.get("claim_boundary") == expected_claim_boundary
        and plan.get("claim_boundary")
        == {
            "exact_25m": False,
            "metric_distance": False,
            "production_ready": False,
            "quality": False,
            "three_module_full_stack": False,
            "twelve_camera_capacity": False,
        }
    )
    if not all(integrity.values()):
        return _person_threshold_r13b_unavailable(
            "r13b_cross_artifact_contract_invalid", integrity=integrity
        )

    return {
        "evidence_version": "r13b",
        "available": True,
        "state": "threshold_calibration_640_960_passed_internal_validation_only",
        "reason": "tensorrt_deepstream_capacity_and_independent_quality_pending",
        "profiles": {
            str(profile): {
                "threshold": float(profiles[str(profile)]["threshold"]),
                "objective": profiles[str(profile)]["objective"],
                "full_sweep_exact_pin_verified": True,
                "full_sweep_fingerprint_sha256": (
                    PERSON_RTDETR_THRESHOLD_R13B_PINS[f"sweep_{profile}"][
                        "fingerprint_sha256"
                    ]
                ),
            }
            for profile in (640, 960)
        },
        "dataset": {
            "role": payload["source_role"],
            "images": 384,
            "capture_groups": 26,
            "official_test_opened": False,
            "test_unseen_opened": False,
        },
        "providers": {
            "available": ["AzureExecutionProvider", "CPUExecutionProvider"],
            "active_session": ["CPUExecutionProvider"],
            "azure_active": False,
        },
        "threshold_calibration_executed": True,
        "cpu_only": True,
        "gpu_executed": False,
        "quality_passed": False,
        "metric_distance_passed": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "twelve_camera_capacity_passed": False,
        "production_ready": False,
        "final_receipt_fingerprint_sha256": receipt["fingerprint_sha256"],
        "integrity": integrity,
        "caveats": [
            "Eşikler 384 görüntülü, model seçiminde görülmüş geliştirme validation setinden gelir; bağımsız kalite testi değildir.",
            "R13B yalnız score-threshold kalibrasyonudur; INT8, metre/25 m, TensorRT, DeepStream 9, 12 kamera veya production kanıtı değildir.",
        ],
    }


def _person_rtdetr_gpu_r10(reader: ArtifactReader) -> dict[str, Any]:
    integrity = {
        **{
            f"{key}_exact_pin_verified": False
            for key in PERSON_RTDETR_GPU_R10_PINS
        },
        "plan_fingerprint_replayed": False,
        "build_fingerprint_replayed": False,
        "smoke_host_fingerprint_replayed": False,
        "smoke_container_fingerprint_replayed": False,
        "baseline_host_fingerprint_replayed": False,
        "baseline_container_fingerprint_replayed": False,
        "cross_artifact_bindings_verified": False,
        "plan_semantics_verified": False,
        "build_semantics_verified": False,
        "smoke_semantics_verified": False,
        "baseline_semantics_verified": False,
    }
    values: dict[str, dict[str, Any]] = {}
    for key, pin in PERSON_RTDETR_GPU_R10_PINS.items():
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=str(pin["path"]),
            maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        )
        integrity[f"{key}_exact_pin_verified"] = result.available
        if value is None:
            return _person_rtdetr_gpu_r10_unavailable(
                f"r10_{key}_{result.state}", integrity=integrity
            )
        values[key] = value

    plan = values["plan"]
    build = values["build_receipt"]
    smoke_host = values["smoke_host_receipt"]
    smoke_container = values["smoke_container_receipt"]
    baseline_host = values["baseline_host_receipt"]
    baseline_container = values["baseline_container_receipt"]
    integrity["plan_fingerprint_replayed"] = _self_fingerprint_matches(plan)
    integrity["build_fingerprint_replayed"] = _self_fingerprint_matches(build)
    integrity["smoke_host_fingerprint_replayed"] = (
        _self_fingerprint_matches(smoke_host)
    )
    integrity["smoke_container_fingerprint_replayed"] = (
        _self_fingerprint_matches(smoke_container)
    )
    integrity["baseline_host_fingerprint_replayed"] = (
        _self_fingerprint_matches(baseline_host)
    )
    integrity["baseline_container_fingerprint_replayed"] = (
        _self_fingerprint_matches(baseline_container)
    )

    plan_binding = {
        "bytes": PERSON_RTDETR_GPU_R10_PINS["plan"]["bytes"],
        "fingerprint_sha256": PERSON_RTDETR_GPU_R10_PLAN_FINGERPRINT,
        "path": (
            "models/person/training-lanes/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/execution-plan.json"
        ),
        "sha256": PERSON_RTDETR_GPU_R10_PINS["plan"]["sha256"],
    }
    smoke_container_pin = {
        "bytes": PERSON_RTDETR_GPU_R10_PINS["smoke_container_receipt"][
            "bytes"
        ],
        "fingerprint_sha256": smoke_container.get("fingerprint_sha256"),
        "path": "container-receipt.json",
        "sha256": PERSON_RTDETR_GPU_R10_PINS["smoke_container_receipt"][
            "sha256"
        ],
    }
    baseline_container_pin = {
        "bytes": PERSON_RTDETR_GPU_R10_PINS["baseline_container_receipt"][
            "bytes"
        ],
        "fingerprint_sha256": baseline_container.get("fingerprint_sha256"),
        "path": "container-receipt.json",
        "sha256": PERSON_RTDETR_GPU_R10_PINS["baseline_container_receipt"][
            "sha256"
        ],
    }
    integrity["cross_artifact_bindings_verified"] = bool(
        smoke_host.get("plan") == plan_binding
        and baseline_host.get("plan") == plan_binding
        and smoke_host.get("build_receipt")
        == PERSON_RTDETR_GPU_R10_PINS["build_receipt"]
        and baseline_host.get("build_receipt")
        == PERSON_RTDETR_GPU_R10_PINS["build_receipt"]
        and smoke_host.get("container_receipt") == smoke_container_pin
        and baseline_host.get("container_receipt") == baseline_container_pin
        and build.get("plan") == plan_binding
        and smoke_container.get("plan", {}).get("sha256")
        == PERSON_RTDETR_GPU_R10_PINS["plan"]["sha256"]
        and baseline_container.get("plan", {}).get("sha256")
        == PERSON_RTDETR_GPU_R10_PINS["plan"]["sha256"]
    )
    modes = plan.get("modes", {})
    integrity["plan_semantics_verified"] = bool(
        plan.get("schema_version")
        == "deepsafe.person-rtdetrv4-gpu-execution-plan/v1"
        and plan.get("plan_id") == "rtdetrv4-s-r-livit-person-r1-gpu-v1"
        and plan.get("status") == "ready_not_executed"
        and plan.get("fingerprint_sha256")
        == PERSON_RTDETR_GPU_R10_PLAN_FINGERPRINT
        and plan.get("authorization")
        == {
            "user_goal_training_approved": True,
            "explicit_mode_and_fingerprint_required": True,
            "this_plan_executes_work": False,
        }
        and plan.get("execution")
        == {
            "performed_during_preparation": False,
            "docker_build_executed": False,
            "gpu_queried": False,
            "gpu_workload_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        }
        and modes.get("smoke_one_step", {}).get("next_run_id")
        == "smoke-one-step-006"
        and modes.get("baseline_eval", {}).get("next_run_id")
        == "baseline-eval-002"
        and modes.get("full_run", {}).get("epochs") == 60
        and modes.get("full_run", {}).get("validation_each_epoch") is True
        and modes.get("full_run", {}).get("last_checkpoint_each_epoch")
        is True
    )
    child = build.get("child_image", {})
    build_execution = build.get("execution", {})
    integrity["build_semantics_verified"] = bool(
        build.get("schema_version")
        == "deepsafe.person-rtdetrv4-gpu-build-receipt/v1"
        and build.get("receipt_id")
        == (
            "rtdetrv4-s-r-livit-person-r1-gpu-v1-image-build-"
            "eval-device-r10-001"
        )
        and build.get("build_attempt_id") == "eval-device-r10-001"
        and build.get("status") == "passed"
        and build.get("docker_exit_status") == 0
        and build.get("validation_error") is None
        and build.get("image_reference") == "deepsafe-rtdetrv4-person:r10"
        and child.get("id") == PERSON_RTDETR_GPU_R10_IMAGE_ID
        and child.get("repo_tags") == ["deepsafe-rtdetrv4-person:r10"]
        and child.get("labels", {}).get("io.deepsafe.plan.fingerprint")
        == PERSON_RTDETR_GPU_R10_PLAN_FINGERPRINT
        and build_execution.get("docker_build_executed") is True
        and build_execution.get("gpu_exposed") is False
        and build_execution.get("gpu_queried") is False
        and build_execution.get("training_executed") is False
        and build_execution.get("evaluation_executed") is False
        and build_execution.get("runtime_container_started") is False
    )

    def host_valid(
        value: dict[str, Any], *, run_id: str, mode: str
    ) -> bool:
        isolation = value.get("runtime_isolation", {})
        lease = isolation.get("gpu_lease", {}) if isinstance(isolation, dict) else {}
        return bool(
            value.get("schema_version")
            == "deepsafe.person-rtdetrv4-gpu-host-receipt/v1"
            and value.get("run_id") == run_id
            and value.get("mode") == mode
            and value.get("status") == "passed"
            and value.get("docker_exit_status") == 0
            and value.get("launch_error") is None
            and value.get("validation_error") is None
            and value.get("terminated_by_signal") is None
            and value.get("inputs", {}).get("unchanged") is True
            and value.get("resolved_image", {}).get("id")
            == PERSON_RTDETR_GPU_R10_IMAGE_ID
            and value.get("gpu", {}).get("name")
            == "NVIDIA RTX A5000 Laptop GPU"
            and isolation.get("network") == "none"
            and isolation.get("root_filesystem_read_only") is True
            and isolation.get("dataset_mount_read_only") is True
            and isolation.get("publisher_checkpoint_mount_read_only") is True
            and isolation.get("no_new_privileges") is True
            and isolation.get("capabilities_dropped") == "ALL"
            and lease.get("required") is True
            and lease.get("owner_kind") == "person_training"
        )

    smoke_execution = smoke_container.get("execution", {})
    amp = smoke_execution.get("amp", {})
    attempts = amp.get("attempts", []) if isinstance(amp, dict) else []
    final_attempt = attempts[-1] if isinstance(attempts, list) and attempts else {}
    final_gradients = final_attempt.get("gradients", {})
    final_state = final_attempt.get("state_integrity", {})
    integrity["smoke_semantics_verified"] = bool(
        host_valid(smoke_host, run_id="smoke-one-step-006", mode="smoke_one_step")
        and smoke_container.get("schema_version")
        == "deepsafe.person-rtdetrv4-gpu-container-receipt/v1"
        and smoke_container.get("run_id") == "smoke-one-step-006"
        and smoke_container.get("mode") == "smoke_one_step"
        and smoke_container.get("status") == "passed"
        and smoke_container.get("resolved_image_id")
        == PERSON_RTDETR_GPU_R10_IMAGE_ID
        and smoke_container.get("official_test_opened") is False
        and smoke_container.get("test_unseen_opened") is False
        and smoke_execution.get("mode") == "smoke_one_step"
        and smoke_execution.get("quality_metric_computed") is False
        and smoke_execution.get("full_run_resumable_checkpoint") is False
        and smoke_execution.get("optimizer", {}).get("step_count") == 1
        and smoke_execution.get("ema", {}).get("update_count") == 1
        and amp.get("enabled") is True
        and amp.get("policy_id")
        == "rtdetrv4-smoke-bounded-amp-backoff-v1"
        and amp.get("attempt_count") == 9
        and amp.get("overflow_backoff_count") == 8
        and amp.get("persistent_nonfinite_fail_closed") is True
        and len(attempts) == 9
        and final_attempt.get("attempt") == 9
        and final_attempt.get("outcome") == "finite_optimizer_ema_update"
        and final_attempt.get("scaler", {}).get("scale_after") == 256.0
        and final_gradients.get("all_gradients_finite") is True
        and final_gradients.get("gradient_tensor_count") == 488
        and final_gradients.get("nonfinite_gradient_tensor_count") == 0
        and final_state.get("optimizer_step_skipped") is False
        and final_state.get("ema_update_executed") is True
        and smoke_execution.get("checkpoint", {}).get("sha256")
        == "888905b91d83f654f8d01edd9957e36aaf7a5ebeb944c4ddf4f7086c0e741dba"
    )
    baseline_execution = baseline_container.get("execution", {})
    baseline_dataset = baseline_container.get("dataset", {})
    coco = baseline_execution.get("coco", {})
    point = baseline_execution.get("person_operating_point", {})
    position_buffers = baseline_execution.get("eval_position_buffers", {})
    integrity["baseline_semantics_verified"] = bool(
        host_valid(baseline_host, run_id="baseline-eval-002", mode="baseline_eval")
        and baseline_container.get("schema_version")
        == "deepsafe.person-rtdetrv4-gpu-container-receipt/v1"
        and baseline_container.get("run_id") == "baseline-eval-002"
        and baseline_container.get("mode") == "baseline_eval"
        and baseline_container.get("status") == "passed"
        and baseline_container.get("resolved_image_id")
        == PERSON_RTDETR_GPU_R10_IMAGE_ID
        and baseline_container.get("official_test_opened") is False
        and baseline_container.get("test_unseen_opened") is False
        and baseline_dataset.get("official_test_opened") is False
        and baseline_dataset.get("test_unseen_opened") is False
        and baseline_dataset.get("train_val_capture_group_overlap") == 0
        and baseline_dataset.get("train_val_sequence_overlap") == 0
        and baseline_dataset.get("val")
        == {
            "annotations": 3256,
            "capture_groups": 26,
            "images": 384,
            "sequences": 32,
        }
        and baseline_execution.get("scope")
        == "group_safe_validation_calibration_baseline_not_official_test_or_test_unseen"
        and baseline_execution.get("optimizer_constructed") is False
        and baseline_execution.get("optimizer_step_executed") is False
        and baseline_execution.get("backward_executed") is False
        and coco.get("AP_IoU_0.50_0.95") == 0.25337032843844887
        and coco.get("AP_IoU_0.50") == 0.48542442375363937
        and coco.get("AP_IoU_0.75") == 0.22921644467688862
        and point.get("score_threshold") == 0.5
        and point.get("iou_threshold") == 0.5
        and point.get("true_positive") == 850
        and point.get("false_positive") == 80
        and point.get("false_negative") == 2406
        and point.get("precision") == 0.9139784946236559
        and point.get("recall") == 0.26105651105651106
        and point.get("f1") == 0.40611562350692787
        and position_buffers.get("policy_id")
        == "rtdetrv4-hybrid-encoder-eval-position-buffer-v1"
        and position_buffers.get("persistent") is False
        and position_buffers.get("state_dict_semantics_changed") is False
        and position_buffers.get("after_model_to", {}).get("expected_device")
        == "cuda:0"
    )
    if not all(integrity.values()):
        return _person_rtdetr_gpu_r10_unavailable(
            "r10_cross_artifact_contract_invalid", integrity=integrity
        )
    distance_proxy = _person_distance_proxy_r1(reader)
    full_training = _person_full_training_r10(reader)
    full_training_complete = bool(
        full_training.get("available") is True
        and full_training.get("completion_evidence_present") is True
        and full_training.get("full_training_complete") is True
    )
    export_plan_r11 = _person_export_plan_r11(
        reader, full_training_verified=full_training_complete
    )
    export_plan_ready = bool(
        export_plan_r11.get("available") is True
        and export_plan_r11.get("plan_ready") is True
        and export_plan_r11.get("export_executed") is False
    )
    onnx_export_r12 = (
        _person_onnx_r12(reader)
        if full_training_complete and export_plan_ready
        else _person_onnx_r12_unavailable(
            "r12_full_training_or_r11_plan_prerequisite_invalid"
        )
    )
    onnx_pair_complete = bool(
        onnx_export_r12.get("available") is True
        and onnx_export_r12.get("both_profiles_exported") is True
    )
    threshold_calibration_r13b = (
        _person_threshold_r13b(reader)
        if onnx_pair_complete
        else _person_threshold_r13b_unavailable(
            "r13b_onnx_r12_prerequisite_invalid"
        )
    )
    threshold_calibration_complete = bool(
        threshold_calibration_r13b.get("available") is True
        and threshold_calibration_r13b.get("threshold_calibration_executed") is True
    )
    return {
        "evidence_version": "r10",
        "available": True,
        "state": (
            "full_training_completed_internal_validation_only"
            if full_training_complete
            else (
                "smoke_and_internal_baseline_passed_"
                "full_training_result_not_available"
            )
        ),
        "reason": (
            "deployment_and_independent_acceptance_pending"
            if full_training_complete
            else "full_training_receipt_not_projected"
        ),
        "ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "candidate": "RT-DETRv4-S",
        "hardware": {
            "gpu": "NVIDIA RTX A5000 Laptop GPU",
            "cuda_memory_mib": 16384,
        },
        "plan": {
            "training_resolution": 640,
            "deployment_profiles": [640, 960],
            "epochs": 60,
            "validation_each_epoch": True,
            "full_training_started": (
                True if full_training_complete else None
            ),
            "full_training_completion_evidence_present": (
                full_training_complete
            ),
            "full_training_complete": full_training_complete,
        },
        "image_build": {
            "status": "passed",
            "image_version": "r10",
            "exact_immutable_identity_verified": True,
            "gpu_exposed_during_build": False,
        },
        "smoke_one_step": {
            "run_id": "smoke-one-step-006",
            "status": "passed",
            "duration_seconds": smoke_host.get("duration_seconds"),
            "amp_attempts": 9,
            "overflow_backoffs": 8,
            "accepted_scale": 256.0,
            "finite_gradient_tensors": 488,
            "nonfinite_gradient_tensors": 0,
            "optimizer_steps": 1,
            "ema_updates": 1,
            "quality_measured": False,
            "full_run_checkpoint": False,
        },
        "internal_baseline": {
            "run_id": "baseline-eval-002",
            "status": "passed",
            "duration_seconds": baseline_host.get("duration_seconds"),
            "scope": (
                "group_safe_validation_calibration_not_official_test_or_test_unseen"
            ),
            "images": 384,
            "annotations": 3256,
            "official_test_opened": False,
            "test_unseen_opened": False,
            "coco": {
                "ap_50_95": 0.25337032843844887,
                "ap_50": 0.48542442375363937,
                "ap_75": 0.22921644467688862,
                "ap_small": 0.10382480175678808,
                "ap_medium": 0.4806160420907464,
                "ap_large": 0.5542921856182349,
            },
            "operating_point_score_0_5_iou_0_5": {
                "tp": 850,
                "fp": 80,
                "fn": 2406,
                "precision": 0.9139784946236559,
                "recall": 0.26105651105651106,
                "f1": 0.40611562350692787,
            },
        },
        "full_training_r10": full_training,
        "export_plan_r11": export_plan_r11,
        "onnx_export_r12": onnx_export_r12,
        "threshold_calibration_r13b": threshold_calibration_r13b,
        "distance_proxy_r1": distance_proxy,
        "full_training_complete": full_training_complete,
        "export_plan_ready": export_plan_ready,
        "export_complete": False,
        "onnx_export_complete": onnx_pair_complete,
        "threshold_calibration_complete": threshold_calibration_complete,
        "onnx_640_exported": bool(
            onnx_export_r12.get("onnx_640_exported") is True
        ),
        "onnx_960_exported": bool(
            onnx_export_r12.get("onnx_960_exported") is True
        ),
        "tensorrt_complete": False,
        "deepstream9_complete": False,
        "twelve_camera_capacity_complete": False,
        "production_ready": False,
        "integrity": integrity,
        "caveats": [
            "baseline-eval-002 grup-safe iç kalibrasyon ayrımıdır; resmî test veya test-unseen sonucu değildir.",
            "Tek adımlı smoke eğitim kalitesi, 60 epoch fine-tune tamamlanması veya deploy edilebilir checkpoint kanıtı değildir.",
            (
                "R10 full-60e-001 exact receipt zinciri 60/60 eğitimi doğrular; kalite kapsamı iç validation ile sınırlıdır."
                if full_training_complete
                else "R10 zincirinde full-train tamamlanma receipt'i henüz yoktur."
            ),
            (
                "R11 planının 640/960 ONNX aşamaları R12 exact-pin receipt'leriyle tamamlandı; sonraki aşamalar bekliyor."
                if onnx_pair_complete
                else (
                    "R11 640/960, batch-12 FP16 export planı exact-pinli ve yürütmeye hazırdır; iki-profil ONNX tamamlanması kanıtlanmadı."
                    if export_plan_ready
                    else "R11 export planı doğrulanamadı; yürütme kapısı kapalıdır."
                )
            ),
            (
                "R12 exact-pinli 640/960 ONNX receipt çifti doğrulandı; bu yalnız ONNX aşamasıdır."
                if onnx_pair_complete
                else "R12 ONNX aşaması exact-pinli iki profil receipt'i oluşana kadar tamamlanmış sayılmaz."
            ),
            (
                "R13B CPU-only 640/960 score eşiği exact sweep/final receipt zinciriyle tamamlandı; bağımsız kalite değildir."
                if threshold_calibration_complete
                else "R13B threshold kalibrasyonu exact plan/sweep/final receipt zinciriyle tamamlanmadı."
            ),
            "TensorRT export, DeepStream 9 ve 12-kamera kapasite sonucu henüz yoktur.",
        ],
    }


def _person_model_upgrade_readiness(reader: ArtifactReader) -> dict[str, Any]:
    """Project pinned preparation facts without exposing execution material.

    The checked-in plan hash is the trust anchor.  Every dependent artifact is
    opened below the workspace with no-follow directory descriptors and must
    match the plan's size/hash pin.  No result artifact is consulted and no
    model, network, Docker, GPU, training, or export action is available here.
    """

    plan_read, plan = _workspace_pin_json(
        reader,
        PERSON_UPGRADE_PLAN_PIN,
        expected_path=PERSON_UPGRADE_PLAN_PIN["path"],
    )
    integrity = {
        "upgrade_plan_verified": plan_read.available,
        "dataset_manifest_verified": False,
        "training_plan_verified": False,
        "challenger_provenance_verified": False,
        "challenger_checkpoint_verified": False,
        "challenger_structural_receipt_file_verified": False,
        "challenger_structural_schema_verified": False,
        "challenger_structural_validator_verified": False,
        "challenger_framework_receipt_file_verified": False,
        "challenger_framework_schema_verified": False,
        "challenger_framework_validator_verified": False,
        "challenger_onnx_export_plan_file_verified": False,
        "challenger_onnx_exporter_verified": False,
        "challenger_onnx_receipt_schema_verified": False,
        "challenger_onnx_640_receipt_file_verified": False,
        "challenger_onnx_640_artifact_verified": False,
        "challenger_onnx_960_receipt_file_verified": False,
        "challenger_onnx_960_artifact_verified": False,
        "challenger_real_image_parity_plan_file_verified": False,
        "challenger_real_image_parity_receipt_file_verified": False,
        "challenger_real_image_parity_schema_verified": False,
        "challenger_real_image_parity_validator_verified": False,
        "challenger_onnx_batch12_receipt_file_verified": False,
        "challenger_onnx_batch12_schema_verified": False,
        "challenger_onnx_batch12_validator_verified": False,
        "challenger_ds9_parser_receipt_file_verified": False,
        "challenger_ds9_parser_artifact_verified": False,
        **{
            f"challenger_ds9_parser_{key}_verified": False
            for key in PERSON_UPGRADE_DS9_PARSER_SOURCE_PINS
        },
    }
    if plan is None:
        return _person_upgrade_unavailable(
            f"upgrade_plan_{plan_read.state}", integrity=integrity
        )

    try:
        primary = plan["training_data"]["primary"]
        contract = plan["training_and_export_contract"]
        generated = contract["generated_artifacts"]
        challenger = plan["upstream"]["rtdetrv4"]
        readiness = plan["readiness"]
        license_gate = plan["license_gate"]
        manifest_pin = _person_pin_core(generated["dataset_manifest"])
        training_pin = _person_pin_core(generated["training_plan"])
        provenance_pin = _person_pin_core(challenger["provenance"])
        checkpoint_pin = _person_pin_core(challenger["checkpoint"])
        structural_receipt_pin = _person_pin_core(
            challenger["structural_load_receipt"]
        )
        framework_receipt_pin = _person_pin_core(
            challenger["framework_profiles_receipt"]
        )
        export_plan_pin = _person_pin_core(challenger["export_plan"])
        onnx_receipt_pins = {
            profile: _person_pin_core(
                challenger["onnx_profile_receipts"][str(profile)]
            )
            for profile in (640, 960)
        }
        real_image_parity_plan_pin = _person_pin_core(
            challenger["real_image_parity_plan"]
        )
        real_image_parity_receipt_pin = _person_pin_core(
            challenger["real_image_parity_receipt"]
        )
        parser_receipt_pin = _person_pin_core(
            challenger["parser_build_receipt"]
        )
        parser_artifact_pin = _person_pin_core(challenger["parser_artifact"])
        batch12_receipt_pin = _person_pin_core(
            challenger["onnx_batch12_receipt"]
        )
    except (KeyError, TypeError):
        return _person_upgrade_unavailable(
            "upgrade_plan_contract_invalid", integrity=integrity
        )

    plan_semantic_valid = bool(
        plan.get("schema_version") == PERSON_UPGRADE_PLAN_SCHEMA
        and plan.get("status")
        == "training_data_and_frozen_plan_prepared_license_and_training_not_started"
        and plan.get("task") == "person_detection"
        and plan.get("decision", {}).get("control") == "yolo11s"
        and plan.get("decision", {}).get("primary_candidate") == "yolo26s"
        and plan.get("decision", {}).get("permissive_license_challenger")
        == "rtdetrv4-s"
        and license_gate.get("decision") is None
        and license_gate.get("download_and_training_authorized") is False
        and license_gate.get("ultralytics_options")
        == ["AGPL-3.0-compatible-project", "Ultralytics-Enterprise"]
        and readiness
        == {
            "model_selected": False,
            "license_selected": False,
            "training_data_prepared": True,
            "training_complete": False,
            "deepstream_parity_complete": False,
            "quality_gates_passed": False,
            "capacity_gates_passed": False,
            "production_ready": False,
        }
        and contract.get("profiles") == [640, 960]
        and contract.get("deepstream_batch") == 12
        and contract.get("person_deepstream_interval") == 0
        and generated.get("weights") is None
        and generated.get("training_receipt") is None
        and generated.get("onnx") is None
        and generated.get("engine_640") is None
        and generated.get("engine_960") is None
        and generated.get("parity_receipt") is None
        and generated.get("quality_receipt") is None
        and generated.get("capacity_receipt") is None
        and manifest_pin is not None
        and training_pin is not None
        and provenance_pin is not None
        and checkpoint_pin is not None
        and structural_receipt_pin is not None
        and framework_receipt_pin is not None
        and export_plan_pin is not None
        and all(pin is not None for pin in onnx_receipt_pins.values())
        and real_image_parity_plan_pin is not None
        and real_image_parity_receipt_pin is not None
        and parser_receipt_pin is not None
        and parser_artifact_pin is not None
        and batch12_receipt_pin is not None
        and manifest_pin.get("path") == PERSON_UPGRADE_MANIFEST_PATH
        and training_pin.get("path") == PERSON_UPGRADE_TRAINING_PLAN_PATH
        and provenance_pin.get("path") == PERSON_UPGRADE_RTDETR_PROVENANCE_PATH
        and checkpoint_pin.get("path") == PERSON_UPGRADE_RTDETR_CHECKPOINT_PATH
        and structural_receipt_pin
        == _person_pin_core(PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN)
        and challenger.get("structural_load_verified") is True
        and challenger.get("structural_load_receipt", {}).get(
            "receipt_sha256"
        )
        == PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["receipt_sha256"]
        and framework_receipt_pin
        == _person_pin_core(PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN)
        and challenger.get("framework_profiles_verified") is True
        and challenger.get("framework_profiles_receipt", {}).get(
            "receipt_sha256"
        )
        == PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN["receipt_sha256"]
        and export_plan_pin
        == _person_pin_core(PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN)
        and challenger.get("export_plan", {}).get("fingerprint_sha256")
        == PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN["fingerprint_sha256"]
        and challenger.get("onnx_profiles_exported") == [640, 960]
        and challenger.get("synthetic_onnx_parity_passed") is True
        and challenger.get("real_image_framework_parity_passed") is False
        and challenger.get("real_image_parity_evidence_verified") is True
        and challenger.get("real_image_parity_failure_count") == 4
        and real_image_parity_plan_pin
        == _person_pin_core(PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN)
        and challenger.get("real_image_parity_plan", {}).get(
            "fingerprint_sha256"
        )
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN["fingerprint_sha256"]
        and real_image_parity_receipt_pin
        == _person_pin_core(PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN)
        and challenger.get("real_image_parity_receipt", {}).get(
            "receipt_sha256"
        )
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN["receipt_sha256"]
        and all(
            onnx_receipt_pins[profile]
            == _person_pin_core(
                PERSON_UPGRADE_ONNX_PROFILE_PINS[profile]["receipt"]
            )
            and challenger["onnx_profile_receipts"][str(profile)].get(
                "receipt_sha256"
            )
            == PERSON_UPGRADE_ONNX_PROFILE_PINS[profile]["receipt"][
                "receipt_sha256"
            ]
            for profile in (640, 960)
        )
        and parser_receipt_pin
        == _person_pin_core(PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN)
        and challenger.get("parser_build_receipt", {}).get("receipt_sha256")
        == PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN["receipt_sha256"]
        and parser_artifact_pin == PERSON_UPGRADE_DS9_PARSER_ARTIFACT_PIN
        and batch12_receipt_pin
        == _person_pin_core(PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN)
        and challenger.get("onnx_batch12_receipt", {}).get("receipt_sha256")
        == PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN["receipt_sha256"]
        and challenger.get("onnx_batch12_shape_verified") is True
        and challenger.get("parser_cpu_contract_ready") is True
        and challenger.get("deepstream9_real_inference_validated") is False
        and _person_pin_core(primary.get("prepared_dataset_manifest"))
        == manifest_pin
        and challenger.get("declared_code_license") == "Apache-2.0"
        and challenger.get("status")
        == "official_checkpoint_acquired_onnx_profiles_exported_real_image_parity_failed_not_evaluated"
    )
    if not plan_semantic_valid:
        return _person_upgrade_unavailable(
            "upgrade_plan_contract_invalid", integrity=integrity
        )

    manifest_read, manifest = _workspace_pin_json(
        reader,
        manifest_pin,
        expected_path=PERSON_UPGRADE_MANIFEST_PATH,
    )
    training_read, training = _workspace_pin_json(
        reader,
        training_pin,
        expected_path=PERSON_UPGRADE_TRAINING_PLAN_PATH,
    )
    provenance_read, provenance = _workspace_pin_json(
        reader,
        provenance_pin,
        expected_path=PERSON_UPGRADE_RTDETR_PROVENANCE_PATH,
    )
    checkpoint_read = _read_workspace_pin(
        reader,
        checkpoint_pin,
        expected_path=PERSON_UPGRADE_RTDETR_CHECKPOINT_PATH,
        maximum_bytes=MAX_PINNED_FILE_BYTES,
        collect=False,
    )
    structural_receipt_read, structural_receipt = _workspace_pin_json(
        reader,
        structural_receipt_pin,
        expected_path=PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    structural_schema_read, structural_schema = _workspace_pin_json(
        reader,
        PERSON_UPGRADE_STRUCTURAL_SCHEMA_PIN,
        expected_path=PERSON_UPGRADE_STRUCTURAL_SCHEMA_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    structural_validator_read = _read_workspace_pin(
        reader,
        PERSON_UPGRADE_STRUCTURAL_VALIDATOR_PIN,
        expected_path=PERSON_UPGRADE_STRUCTURAL_VALIDATOR_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        collect=False,
    )
    framework_receipt_read, framework_receipt = _workspace_pin_json(
        reader,
        framework_receipt_pin,
        expected_path=PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    framework_schema_read, framework_schema = _workspace_pin_json(
        reader,
        PERSON_UPGRADE_FRAMEWORK_SCHEMA_PIN,
        expected_path=PERSON_UPGRADE_FRAMEWORK_SCHEMA_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    framework_validator_read = _read_workspace_pin(
        reader,
        PERSON_UPGRADE_FRAMEWORK_VALIDATOR_PIN,
        expected_path=PERSON_UPGRADE_FRAMEWORK_VALIDATOR_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        collect=False,
    )
    onnx_export_plan_read, onnx_export_plan = _workspace_pin_json(
        reader,
        export_plan_pin,
        expected_path=PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    onnx_exporter_read = _read_workspace_pin(
        reader,
        PERSON_UPGRADE_ONNX_EXPORTER_PIN,
        expected_path=PERSON_UPGRADE_ONNX_EXPORTER_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        collect=False,
    )
    onnx_receipt_schema_read, onnx_receipt_schema = _workspace_pin_json(
        reader,
        PERSON_UPGRADE_ONNX_RECEIPT_SCHEMA_PIN,
        expected_path=PERSON_UPGRADE_ONNX_RECEIPT_SCHEMA_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    onnx_receipt_reads: dict[int, WorkspacePinRead] = {}
    onnx_receipts: dict[int, dict[str, Any] | None] = {}
    onnx_artifact_reads: dict[int, WorkspacePinRead] = {}
    for profile in (640, 960):
        profile_receipt_pin = onnx_receipt_pins[profile]
        assert profile_receipt_pin is not None
        receipt_read, receipt = _workspace_pin_json(
            reader,
            profile_receipt_pin,
            expected_path=PERSON_UPGRADE_ONNX_PROFILE_PINS[profile][
                "receipt"
            ]["path"],
            maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        )
        onnx_receipt_reads[profile] = receipt_read
        onnx_receipts[profile] = receipt
        onnx_artifact_reads[profile] = _read_workspace_pin(
            reader,
            PERSON_UPGRADE_ONNX_PROFILE_PINS[profile]["onnx"],
            expected_path=PERSON_UPGRADE_ONNX_PROFILE_PINS[profile]["onnx"][
                "path"
            ],
            maximum_bytes=MAX_PINNED_FILE_BYTES,
            collect=False,
        )
    assert real_image_parity_plan_pin is not None
    assert real_image_parity_receipt_pin is not None
    real_image_plan_read, real_image_plan = _workspace_pin_json(
        reader,
        real_image_parity_plan_pin,
        expected_path=PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    real_image_receipt_read, real_image_receipt = _workspace_pin_json(
        reader,
        real_image_parity_receipt_pin,
        expected_path=PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    real_image_schema_read, real_image_schema = _workspace_pin_json(
        reader,
        PERSON_UPGRADE_REAL_IMAGE_PARITY_SCHEMA_PIN,
        expected_path=PERSON_UPGRADE_REAL_IMAGE_PARITY_SCHEMA_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    real_image_validator_read = _read_workspace_pin(
        reader,
        PERSON_UPGRADE_REAL_IMAGE_PARITY_VALIDATOR_PIN,
        expected_path=PERSON_UPGRADE_REAL_IMAGE_PARITY_VALIDATOR_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        collect=False,
    )
    assert parser_receipt_pin is not None
    assert parser_artifact_pin is not None
    assert batch12_receipt_pin is not None
    batch12_receipt_read, batch12_receipt = _workspace_pin_json(
        reader,
        batch12_receipt_pin,
        expected_path=PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    batch12_schema_read, batch12_schema = _workspace_pin_json(
        reader,
        PERSON_UPGRADE_ONNX_BATCH12_SCHEMA_PIN,
        expected_path=PERSON_UPGRADE_ONNX_BATCH12_SCHEMA_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    batch12_validator_read = _read_workspace_pin(
        reader,
        PERSON_UPGRADE_ONNX_BATCH12_VALIDATOR_PIN,
        expected_path=PERSON_UPGRADE_ONNX_BATCH12_VALIDATOR_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
        collect=False,
    )
    parser_receipt_read, parser_receipt = _workspace_pin_json(
        reader,
        parser_receipt_pin,
        expected_path=PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN["path"],
        maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
    )
    parser_artifact_read = _read_workspace_pin(
        reader,
        parser_artifact_pin,
        expected_path=PERSON_UPGRADE_DS9_PARSER_ARTIFACT_PIN["path"],
        maximum_bytes=MAX_PINNED_FILE_BYTES,
        collect=False,
    )
    parser_source_reads = {
        key: _read_workspace_pin(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=PERSON_UPGRADE_MAX_JSON_BYTES,
            collect=False,
        )
        for key, pin in PERSON_UPGRADE_DS9_PARSER_SOURCE_PINS.items()
    }
    integrity.update(
        {
            "dataset_manifest_verified": manifest_read.available,
            "training_plan_verified": training_read.available,
            "challenger_provenance_verified": provenance_read.available,
            "challenger_checkpoint_verified": checkpoint_read.available,
            "challenger_structural_receipt_file_verified": (
                structural_receipt_read.available
            ),
            "challenger_structural_schema_verified": (
                structural_schema_read.available
            ),
            "challenger_structural_validator_verified": (
                structural_validator_read.available
            ),
            "challenger_framework_receipt_file_verified": (
                framework_receipt_read.available
            ),
            "challenger_framework_schema_verified": (
                framework_schema_read.available
            ),
            "challenger_framework_validator_verified": (
                framework_validator_read.available
            ),
            "challenger_onnx_export_plan_file_verified": (
                onnx_export_plan_read.available
            ),
            "challenger_onnx_exporter_verified": onnx_exporter_read.available,
            "challenger_onnx_receipt_schema_verified": (
                onnx_receipt_schema_read.available
            ),
            "challenger_onnx_640_receipt_file_verified": (
                onnx_receipt_reads[640].available
            ),
            "challenger_onnx_640_artifact_verified": (
                onnx_artifact_reads[640].available
            ),
            "challenger_onnx_960_receipt_file_verified": (
                onnx_receipt_reads[960].available
            ),
            "challenger_onnx_960_artifact_verified": (
                onnx_artifact_reads[960].available
            ),
            "challenger_real_image_parity_plan_file_verified": (
                real_image_plan_read.available
            ),
            "challenger_real_image_parity_receipt_file_verified": (
                real_image_receipt_read.available
            ),
            "challenger_real_image_parity_schema_verified": (
                real_image_schema_read.available
            ),
            "challenger_real_image_parity_validator_verified": (
                real_image_validator_read.available
            ),
            "challenger_onnx_batch12_receipt_file_verified": (
                batch12_receipt_read.available
            ),
            "challenger_onnx_batch12_schema_verified": (
                batch12_schema_read.available
            ),
            "challenger_onnx_batch12_validator_verified": (
                batch12_validator_read.available
            ),
            "challenger_ds9_parser_receipt_file_verified": (
                parser_receipt_read.available
            ),
            "challenger_ds9_parser_artifact_verified": (
                parser_artifact_read.available
            ),
            **{
                f"challenger_ds9_parser_{key}_verified": read.available
                for key, read in parser_source_reads.items()
            },
        }
    )
    if not all(integrity.values()):
        failed = next(key for key, valid in integrity.items() if not valid)
        states = {
            "dataset_manifest_verified": manifest_read.state,
            "training_plan_verified": training_read.state,
            "challenger_provenance_verified": provenance_read.state,
            "challenger_checkpoint_verified": checkpoint_read.state,
            "challenger_structural_receipt_file_verified": (
                structural_receipt_read.state
            ),
            "challenger_structural_schema_verified": (
                structural_schema_read.state
            ),
            "challenger_structural_validator_verified": (
                structural_validator_read.state
            ),
            "challenger_framework_receipt_file_verified": (
                framework_receipt_read.state
            ),
            "challenger_framework_schema_verified": framework_schema_read.state,
            "challenger_framework_validator_verified": (
                framework_validator_read.state
            ),
            "challenger_onnx_export_plan_file_verified": (
                onnx_export_plan_read.state
            ),
            "challenger_onnx_exporter_verified": onnx_exporter_read.state,
            "challenger_onnx_receipt_schema_verified": (
                onnx_receipt_schema_read.state
            ),
            "challenger_onnx_640_receipt_file_verified": (
                onnx_receipt_reads[640].state
            ),
            "challenger_onnx_640_artifact_verified": (
                onnx_artifact_reads[640].state
            ),
            "challenger_onnx_960_receipt_file_verified": (
                onnx_receipt_reads[960].state
            ),
            "challenger_onnx_960_artifact_verified": (
                onnx_artifact_reads[960].state
            ),
            "challenger_real_image_parity_plan_file_verified": (
                real_image_plan_read.state
            ),
            "challenger_real_image_parity_receipt_file_verified": (
                real_image_receipt_read.state
            ),
            "challenger_real_image_parity_schema_verified": (
                real_image_schema_read.state
            ),
            "challenger_real_image_parity_validator_verified": (
                real_image_validator_read.state
            ),
            "challenger_onnx_batch12_receipt_file_verified": (
                batch12_receipt_read.state
            ),
            "challenger_onnx_batch12_schema_verified": (
                batch12_schema_read.state
            ),
            "challenger_onnx_batch12_validator_verified": (
                batch12_validator_read.state
            ),
            "challenger_ds9_parser_receipt_file_verified": (
                parser_receipt_read.state
            ),
            "challenger_ds9_parser_artifact_verified": (
                parser_artifact_read.state
            ),
            **{
                f"challenger_ds9_parser_{key}_verified": read.state
                for key, read in parser_source_reads.items()
            },
        }
        return _person_upgrade_unavailable(
            f"{failed.removesuffix('_verified')}_{states.get(failed, 'invalid')}",
            integrity=integrity,
        )
    if (
        manifest is None
        or training is None
        or provenance is None
        or structural_receipt is None
        or structural_schema is None
        or framework_receipt is None
        or framework_schema is None
        or onnx_export_plan is None
        or onnx_receipt_schema is None
        or any(receipt is None for receipt in onnx_receipts.values())
        or real_image_plan is None
        or real_image_receipt is None
        or real_image_schema is None
        or batch12_receipt is None
        or batch12_schema is None
        or parser_receipt is None
    ):
        return _person_upgrade_unavailable(
            "person_upgrade_json_invalid", integrity=integrity
        )

    structural_schema_contract_valid = (
        _person_structural_schema_contract_valid(structural_schema)
    )
    try:
        _validate_schema_node(
            structural_receipt,
            structural_schema,
            structural_schema,
        )
    except (TypeError, ValueError, RecursionError):
        structural_schema_replay_valid = False
    else:
        structural_schema_replay_valid = True
    structural_unsigned = dict(structural_receipt)
    structural_observed_self_hash = structural_unsigned.pop(
        "receipt_sha256", None
    )
    structural_self_hash_valid = bool(
        structural_observed_self_hash
        == PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["receipt_sha256"]
        and _canonical_sha256(structural_unsigned)
        == PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["receipt_sha256"]
    )
    structural_semantics_valid = _person_structural_receipt_semantics_valid(
        structural_receipt,
        checkpoint_pin=checkpoint_pin,
    )
    integrity.update(
        {
            "challenger_structural_schema_contract_verified": (
                structural_schema_contract_valid
            ),
            "challenger_structural_schema_replay_verified": (
                structural_schema_replay_valid
            ),
            "challenger_structural_self_hash_verified": (
                structural_self_hash_valid
            ),
            "challenger_structural_semantic_verified": (
                structural_semantics_valid
            ),
        }
    )
    if not (
        structural_schema_contract_valid
        and structural_schema_replay_valid
        and structural_self_hash_valid
        and structural_semantics_valid
    ):
        return _person_upgrade_unavailable(
            "challenger_structural_receipt_invalid", integrity=integrity
        )

    framework_schema_contract_valid = _person_closed_receipt_schema_valid(
        framework_schema,
        schema_id="deepsafe.person-framework-profiles-receipt/v1",
        schema_version="deepsafe.person-framework-profiles-receipt/v1",
    )
    onnx_schema_contract_valid = _person_closed_receipt_schema_valid(
        onnx_receipt_schema,
        schema_id="deepsafe.rtdetrv4-onnx-export-receipt/v1",
        schema_version="deepsafe.rtdetrv4-onnx-export-receipt/v1",
    )
    try:
        _validate_schema_node(
            framework_receipt,
            framework_schema,
            framework_schema,
        )
    except (TypeError, ValueError, RecursionError):
        framework_schema_replay_valid = False
    else:
        framework_schema_replay_valid = True
    onnx_schema_replay_valid: dict[int, bool] = {}
    for profile in (640, 960):
        profile_receipt = onnx_receipts[profile]
        assert profile_receipt is not None
        try:
            _validate_schema_node(
                profile_receipt,
                onnx_receipt_schema,
                onnx_receipt_schema,
            )
        except (TypeError, ValueError, RecursionError):
            onnx_schema_replay_valid[profile] = False
        else:
            onnx_schema_replay_valid[profile] = True
    framework_self_hash_valid = _external_receipt_self_hash_matches(
        framework_receipt,
        expected=PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN["receipt_sha256"],
    )
    framework_semantics_valid = _person_framework_receipt_semantics_valid(
        framework_receipt,
        checkpoint_pin=checkpoint_pin,
    )
    onnx_plan_semantics_valid = _person_onnx_export_plan_semantics_valid(
        onnx_export_plan,
        checkpoint_pin=checkpoint_pin,
    )
    onnx_self_hash_valid: dict[int, bool] = {}
    onnx_semantics_valid: dict[int, bool] = {}
    for profile in (640, 960):
        profile_receipt = onnx_receipts[profile]
        assert profile_receipt is not None
        onnx_self_hash_valid[profile] = (
            _external_receipt_self_hash_matches(
                profile_receipt,
                expected=PERSON_UPGRADE_ONNX_PROFILE_PINS[profile][
                    "receipt"
                ]["receipt_sha256"],
            )
        )
        onnx_semantics_valid[profile] = _person_onnx_receipt_semantics_valid(
            profile_receipt,
            profile=profile,
            checkpoint_pin=checkpoint_pin,
        )
    evidence_timestamps_valid = bool(
        isinstance(structural_receipt.get("created_at"), str)
        and isinstance(framework_receipt.get("created_at"), str)
        and all(
            isinstance(onnx_receipts[profile], dict)
            and isinstance(onnx_receipts[profile].get("created_at"), str)
            for profile in (640, 960)
        )
        and structural_receipt["created_at"] < framework_receipt["created_at"]
        and framework_receipt["created_at"]
        < onnx_receipts[640]["created_at"]
        <= onnx_receipts[960]["created_at"]
    )
    integrity.update(
        {
            "challenger_framework_schema_contract_verified": (
                framework_schema_contract_valid
            ),
            "challenger_framework_schema_replay_verified": (
                framework_schema_replay_valid
            ),
            "challenger_framework_self_hash_verified": (
                framework_self_hash_valid
            ),
            "challenger_framework_semantic_verified": (
                framework_semantics_valid
            ),
            "challenger_onnx_export_plan_semantic_verified": (
                onnx_plan_semantics_valid
            ),
            "challenger_onnx_receipt_schema_contract_verified": (
                onnx_schema_contract_valid
            ),
            "challenger_onnx_640_schema_replay_verified": (
                onnx_schema_replay_valid[640]
            ),
            "challenger_onnx_640_self_hash_verified": (
                onnx_self_hash_valid[640]
            ),
            "challenger_onnx_640_semantic_verified": (
                onnx_semantics_valid[640]
            ),
            "challenger_onnx_960_schema_replay_verified": (
                onnx_schema_replay_valid[960]
            ),
            "challenger_onnx_960_self_hash_verified": (
                onnx_self_hash_valid[960]
            ),
            "challenger_onnx_960_semantic_verified": (
                onnx_semantics_valid[960]
            ),
            "challenger_onnx_evidence_timestamps_verified": (
                evidence_timestamps_valid
            ),
        }
    )
    if not (
        framework_schema_contract_valid
        and framework_schema_replay_valid
        and framework_self_hash_valid
        and framework_semantics_valid
        and onnx_plan_semantics_valid
        and onnx_schema_contract_valid
        and all(onnx_schema_replay_valid.values())
        and all(onnx_self_hash_valid.values())
        and all(onnx_semantics_valid.values())
        and evidence_timestamps_valid
    ):
        return _person_upgrade_unavailable(
            "challenger_onnx_evidence_invalid", integrity=integrity
        )

    real_image_schema_contract_valid = _person_closed_receipt_schema_valid(
        real_image_schema,
        schema_id=(
            "deepsafe.person-rtdetrv4-real-image-parity-receipt/v1"
        ),
        schema_version=(
            "deepsafe.person-rtdetrv4-real-image-parity-receipt/v1"
        ),
    )
    try:
        _validate_schema_node(
            real_image_receipt,
            real_image_schema,
            real_image_schema,
        )
    except (TypeError, ValueError, RecursionError):
        real_image_schema_replay_valid = False
    else:
        real_image_schema_replay_valid = True
    real_image_plan_semantics_valid = (
        _person_real_image_parity_plan_semantics_valid(
            real_image_plan,
            checkpoint_pin=checkpoint_pin,
        )
    )
    real_image_self_hash_valid = _external_receipt_self_hash_matches(
        real_image_receipt,
        expected=PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN[
            "receipt_sha256"
        ],
    )
    real_image_semantics_valid = (
        _person_real_image_parity_receipt_semantics_valid(
            real_image_receipt,
            plan=real_image_plan,
            checkpoint_pin=checkpoint_pin,
        )
    )
    real_image_timestamp_valid = bool(
        isinstance(real_image_receipt.get("created_at"), str)
        and isinstance(onnx_receipts[960], dict)
        and isinstance(onnx_receipts[960].get("created_at"), str)
        and onnx_receipts[960]["created_at"]
        < real_image_receipt["created_at"]
    )
    integrity.update(
        {
            "challenger_real_image_parity_schema_contract_verified": (
                real_image_schema_contract_valid
            ),
            "challenger_real_image_parity_schema_replay_verified": (
                real_image_schema_replay_valid
            ),
            "challenger_real_image_parity_plan_semantic_verified": (
                real_image_plan_semantics_valid
            ),
            "challenger_real_image_parity_self_hash_verified": (
                real_image_self_hash_valid
            ),
            "challenger_real_image_parity_semantic_verified": (
                real_image_semantics_valid
            ),
            "challenger_real_image_parity_timestamp_verified": (
                real_image_timestamp_valid
            ),
        }
    )
    if not (
        real_image_schema_contract_valid
        and real_image_schema_replay_valid
        and real_image_plan_semantics_valid
        and real_image_self_hash_valid
        and real_image_semantics_valid
        and real_image_timestamp_valid
    ):
        return _person_upgrade_unavailable(
            "challenger_real_image_parity_evidence_invalid",
            integrity=integrity,
        )

    batch12_schema_contract_valid = _person_closed_receipt_schema_valid(
        batch12_schema,
        schema_id="deepsafe.person-onnx-batch12-receipt/v1",
        schema_version="deepsafe.person-onnx-batch12-receipt/v1",
    )
    try:
        _validate_schema_node(
            batch12_receipt,
            batch12_schema,
            batch12_schema,
        )
    except (TypeError, ValueError, RecursionError):
        batch12_schema_replay_valid = False
    else:
        batch12_schema_replay_valid = True
    batch12_self_hash_valid = _external_receipt_self_hash_matches(
        batch12_receipt,
        expected=PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN["receipt_sha256"],
    )
    batch12_semantics_valid = _person_onnx_batch12_receipt_semantics_valid(
        batch12_receipt
    )
    batch12_timestamp_valid = bool(
        isinstance(batch12_receipt.get("created_at"), str)
        and isinstance(onnx_receipts[960], dict)
        and isinstance(onnx_receipts[960].get("created_at"), str)
        and onnx_receipts[960]["created_at"] < batch12_receipt["created_at"]
    )
    integrity.update(
        {
            "challenger_onnx_batch12_schema_contract_verified": (
                batch12_schema_contract_valid
            ),
            "challenger_onnx_batch12_schema_replay_verified": (
                batch12_schema_replay_valid
            ),
            "challenger_onnx_batch12_self_hash_verified": (
                batch12_self_hash_valid
            ),
            "challenger_onnx_batch12_semantic_verified": (
                batch12_semantics_valid
            ),
            "challenger_onnx_batch12_timestamp_verified": (
                batch12_timestamp_valid
            ),
        }
    )
    if not (
        batch12_schema_contract_valid
        and batch12_schema_replay_valid
        and batch12_self_hash_valid
        and batch12_semantics_valid
        and batch12_timestamp_valid
    ):
        return _person_upgrade_unavailable(
            "challenger_onnx_batch12_evidence_invalid", integrity=integrity
        )

    parser_self_hash_valid = _external_receipt_self_hash_matches(
        parser_receipt,
        expected=PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN["receipt_sha256"],
    )
    parser_semantics_valid = _person_ds9_parser_receipt_semantics_valid(
        parser_receipt
    )
    parser_timestamp_valid = bool(
        isinstance(parser_receipt.get("created_at"), str)
        and isinstance(onnx_receipts[960], dict)
        and isinstance(onnx_receipts[960].get("created_at"), str)
        and onnx_receipts[960]["created_at"] < parser_receipt["created_at"]
    )
    integrity.update(
        {
            "challenger_ds9_parser_self_hash_verified": (
                parser_self_hash_valid
            ),
            "challenger_ds9_parser_semantic_verified": (
                parser_semantics_valid
            ),
            "challenger_ds9_parser_timestamp_verified": (
                parser_timestamp_valid
            ),
        }
    )
    if not (
        parser_self_hash_valid
        and parser_semantics_valid
        and parser_timestamp_valid
    ):
        return _person_upgrade_unavailable(
            "challenger_ds9_parser_evidence_invalid", integrity=integrity
        )

    excluded = manifest.get("splits", {}).get("official_test_exclusion", {})
    quarantined = manifest.get("splits", {}).get(
        "quarantined_official_train", {}
    )
    manifest_execution = manifest.get("execution", {})
    manifest_valid = bool(
        manifest.get("schema_version")
        == "deepsafe.rlivit-person-finetune-manifest/v1"
        and manifest.get("status") == "prepared_cpu_only"
        and _self_fingerprint_matches(manifest)
        and manifest.get("fingerprint_sha256")
        == primary["prepared_dataset_manifest"].get("fingerprint_sha256")
        and manifest_execution
        == {
            "docker_executed": False,
            "gpu_executed": False,
            "model_inference_executed": False,
            "model_training_executed": False,
        }
        and manifest.get("qa", {}).get("output_frames") == 1908
        and manifest.get("qa", {}).get("persons") == 16652
        and manifest.get("qa", {}).get("train_calibration_sequence_overlap")
        == 0
        and manifest.get("splits", {}).get("train", {}).get("sequence_count")
        == 127
        and manifest.get("splits", {}).get("train", {}).get("frames") == 1524
        and manifest.get("splits", {}).get("calibration", {}).get(
            "sequence_count"
        )
        == 32
        and manifest.get("splits", {}).get("calibration", {}).get("frames")
        == 384
        and excluded.get("sequence_count") == 40
        and excluded.get("included_output_frames") == 0
        and excluded.get("included_output_labels") == 0
        and excluded.get("rgb_images_read") == 0
        and excluded.get("object_labels_read_for_conversion_or_selection") == 0
        and quarantined.get("sequences") == ["064"]
        and quarantined.get("included_output_frames") == 0
    )

    training_valid = bool(
        training.get("schema_version")
        == "deepsafe.yolo26-person-training-plan/v1"
        and training.get("status") == "planned_license_required_not_executed"
        and _self_fingerprint_matches(training)
        and training.get("fingerprint_sha256")
        == generated["training_plan"].get("fingerprint_sha256")
        and training.get("candidate", {}).get("model_id") == "yolo26s"
        and _person_pin_core(training.get("dataset", {}).get("manifest"))
        == manifest_pin
        and training.get("dataset", {}).get("manifest", {}).get(
            "fingerprint_sha256"
        )
        == manifest.get("fingerprint_sha256")
        and training.get("dataset", {}).get("official_test_output_frames") == 0
        and training.get("dataset", {}).get("train_frames") == 1524
        and training.get("dataset", {}).get("calibration_frames") == 384
        and training.get("dataset", {}).get("person_instances") == 16652
        and training.get("runtime", {}).get("gpu_required_on_execute") is True
        and training.get("training_arguments", {}).get("imgsz") == 960
        and training.get("training_arguments", {}).get("task") == "detect"
        and training.get("license_gate", {}).get("required_for_execute") is True
        and training.get("license_gate", {}).get("decision_recorded_in_plan")
        is False
        and training.get("held_out_guardrails", {}).get(
            "r_livit_official_test_used"
        )
        is False
        and training.get("held_out_guardrails", {}).get("loaf_20_to_25m_used")
        is False
        and training.get("acceptance_effect")
        == "none_until_parity_quality_capacity_and_full_stack_gates_pass"
    )

    deployment = provenance.get("deployment_contract", {})
    challenger_training = provenance.get("training_contract", {})
    acceptance = provenance.get("acceptance", {})
    provenance_structural = provenance.get("structural_load_receipt", {})
    provenance_framework = provenance.get("framework_profiles", {})
    provenance_onnx = provenance.get("onnx_export_evidence", {})
    provenance_real_image = provenance.get("real_image_parity_evidence", {})
    provenance_batch12 = provenance.get("onnx_batch12_evidence", {})
    provenance_parser = provenance.get("deepstream9_parser_evidence", {})
    provenance_valid = bool(
        provenance.get("schema_version")
        == "deepsafe.person-challenger-provenance/v1"
        and provenance.get("status")
        == "official_checkpoint_acquired_onnx_profiles_exported_real_image_parity_failed_not_evaluated"
        and provenance.get("candidate_id") == "rtdetrv4-s"
        and _person_pin_core(provenance.get("checkpoint")) == checkpoint_pin
        and provenance.get("checkpoint", {}).get("download_complete") is True
        and provenance.get("checkpoint", {}).get("structural_load_verified")
        is True
        and provenance.get("checkpoint", {}).get("structural_load_blocker")
        is None
        and isinstance(provenance_structural, dict)
        and _person_pin_core(provenance_structural)
        == structural_receipt_pin
        and provenance_structural.get("receipt_sha256")
        == PERSON_UPGRADE_STRUCTURAL_RECEIPT_PIN["receipt_sha256"]
        and provenance_structural.get("schema")
        == PERSON_UPGRADE_STRUCTURAL_SCHEMA_PIN
        and provenance_structural.get("validator")
        == PERSON_UPGRADE_STRUCTURAL_VALIDATOR_PIN
        and provenance_structural.get("status")
        == "verified_cpu_strict_load_not_exported_not_evaluated"
        and isinstance(provenance_framework, dict)
        and provenance_framework.get("verified") is True
        and provenance_framework.get("profiles") == [640, 960]
        and provenance_framework.get("receipt")
        == PERSON_UPGRADE_FRAMEWORK_RECEIPT_PIN
        and provenance_framework.get("schema")
        == PERSON_UPGRADE_FRAMEWORK_SCHEMA_PIN
        and provenance_framework.get("validator")
        == PERSON_UPGRADE_FRAMEWORK_VALIDATOR_PIN
        and provenance_framework.get("real_image_inference_executed") is True
        and isinstance(provenance_onnx, dict)
        and provenance_onnx.get("export_plan")
        == PERSON_UPGRADE_ONNX_EXPORT_PLAN_PIN
        and provenance_onnx.get("exporter")
        == PERSON_UPGRADE_ONNX_EXPORTER_PIN
        and provenance_onnx.get("receipt_schema")
        == PERSON_UPGRADE_ONNX_RECEIPT_SCHEMA_PIN
        and provenance_onnx.get("fixed_spatial_profiles") is True
        and provenance_onnx.get("dynamic_batch_axis") is True
        and provenance_onnx.get("synthetic_seeded_prng_parity_passed") is True
        and provenance_onnx.get("real_image_framework_parity_passed") is False
        and all(
            provenance_onnx.get("profiles", {}).get(str(profile))
            == PERSON_UPGRADE_ONNX_PROFILE_PINS[profile]
            for profile in (640, 960)
        )
        and isinstance(provenance_real_image, dict)
        and provenance_real_image.get("executed") is True
        and provenance_real_image.get("status")
        == "real_image_framework_onnx_parity_failed_not_quality_not_performance"
        and provenance_real_image.get("profiles")
        == {
            "640": {
                "batch1_passed": True,
                "batch2_passed": True,
                "passed": True,
            },
            "960": {
                "batch1_passed": False,
                "batch2_passed": True,
                "passed": False,
            },
        }
        and provenance_real_image.get("batches") == [1, 2]
        and provenance_real_image.get("selected_frame_count") == 11
        and provenance_real_image.get("unique_scene_count") == 11
        and provenance_real_image.get("unique_primary_video_type_count")
        == 11
        and provenance_real_image.get("medium_close_present") is True
        and provenance_real_image.get("high_oblique_present") is True
        and provenance_real_image.get("top_view_present") is True
        and provenance_real_image.get("plan")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_PLAN_PIN
        and provenance_real_image.get("receipt")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_RECEIPT_PIN
        and provenance_real_image.get("schema")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_SCHEMA_PIN
        and provenance_real_image.get("validator")
        == PERSON_UPGRADE_REAL_IMAGE_PARITY_VALIDATOR_PIN
        and provenance_real_image.get(
            "source_manifest_asset_sha_license_bindings_passed"
        )
        is True
        and provenance_real_image.get("exact_real_pixel_inputs_pinned")
        is True
        and provenance_real_image.get("failure_count") == 4
        and provenance_real_image.get("tolerances_relaxed") is False
        and provenance_real_image.get(
            "topk_tie_diagnostics_override_acceptance"
        )
        is False
        and provenance_real_image.get(
            "real_image_framework_onnx_parity_passed"
        )
        is False
        and provenance_real_image.get("quality_passed") is False
        and provenance_real_image.get("latency_or_fps_passed") is False
        and provenance_real_image.get("production_ready") is False
        and isinstance(provenance_batch12, dict)
        and provenance_batch12.get("shape_and_finite_verified") is True
        and provenance_batch12.get("profiles") == [640, 960]
        and provenance_batch12.get("receipt")
        == PERSON_UPGRADE_ONNX_BATCH12_RECEIPT_PIN
        and provenance_batch12.get("schema")
        == PERSON_UPGRADE_ONNX_BATCH12_SCHEMA_PIN
        and provenance_batch12.get("validator")
        == PERSON_UPGRADE_ONNX_BATCH12_VALIDATOR_PIN
        and provenance_batch12.get("latency_or_fps_claimed") is False
        and provenance_batch12.get("real_image_inference_executed") is False
        and provenance_batch12.get("tensorrt_batch12_verified") is False
        and provenance_batch12.get("deepstream9_batch12_verified") is False
        and provenance_batch12.get("capacity_passed") is False
        and isinstance(provenance_parser, dict)
        and provenance_parser.get("cpu_contract_ready") is True
        and provenance_parser.get("build_receipt")
        == PERSON_UPGRADE_DS9_PARSER_RECEIPT_PIN
        and provenance_parser.get("artifact")
        == PERSON_UPGRADE_DS9_PARSER_ARTIFACT_PIN
        and provenance_parser.get("abi_version")
        == "DEEPSAFE_RTDETRV4_PARSER_1.0"
        and provenance_parser.get("contract_test_passed") is True
        and provenance_parser.get("tensorrt_engines_built") is False
        and provenance_parser.get("gpu_integration_validated") is False
        and provenance_parser.get("deepstream9_real_inference_validated")
        is False
        and provenance_parser.get("real_image_parity_passed") is False
        and provenance.get("upstream", {}).get("license", {}).get("spdx")
        == "Apache-2.0"
        and deployment.get("profiles_to_export_and_measure") == [640, 960]
        and deployment.get("deepstream_batch") == 12
        and deployment.get("onnx_status")
        == "profiles_640_960_exported_synthetic_parity_passed_real_image_parity_failed"
        and deployment.get("custom_deepstream_tensor_adapter_status")
        == "cpu_parser_contract_built_not_gpu_integrated"
        and deployment.get("tensorrt_status") == "not_built"
        and deployment.get("deepstream9_status") == "not_run"
        and challenger_training.get("training_status") == "not_started"
        and challenger_training.get("official_r_livit_test_excluded") is True
        and challenger_training.get("loaf_20_to_25m_excluded") is True
        and isinstance(acceptance, dict)
        and len(acceptance) == 10
        and all(value is False for value in acceptance.values())
    )
    if not (manifest_valid and training_valid and provenance_valid):
        return _person_upgrade_unavailable(
            "person_upgrade_contract_invalid", integrity=integrity
        )

    return {
        "label": "Kişi modeli yükseltme hazırlığı",
        "available": True,
        "state": "prepared_not_evaluated",
        "reason": "license_basis_not_selected",
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "license": {
            "decision": None,
            "selected": False,
            "download_and_training_authorized": False,
            "allowed_bases": ["AGPL-3.0 compatible", "Ultralytics Enterprise"],
        },
        "selection": {
            "control": "YOLO11s",
            "primary_candidate": "YOLO26s",
            "permissive_challenger": "RT-DETRv4-S",
            "production_model_selected": False,
        },
        "preparation": {
            "training_data_prepared": True,
            "frozen_training_plan_verified": True,
            "permissive_checkpoint_acquired": True,
        },
        "dataset": {
            "dataset_id": manifest.get("dataset_id"),
            "license": primary.get("license"),
            "train_sequences": 127,
            "train_frames": 1524,
            "calibration_sequences": 32,
            "calibration_frames": 384,
            "person_instances": 16652,
            "official_test_output_frames": 0,
            "quarantined_neighbor_sequences": 1,
            "gpu_or_model_execution_occurred": False,
        },
        "training_plan": {
            "candidate": "YOLO26s",
            "status": "license_required_not_executed",
            "training_input_size": 960,
            "deployment_profiles": [640, 960],
            "deepstream_batch": 12,
            "training_executed": False,
            "export_executed": False,
        },
        "permissive_challenger": {
            "candidate": "RT-DETRv4-S",
            "declared_code_license": "Apache-2.0",
            "checkpoint_integrity_verified": True,
            "structural_load_verified": True,
            "structural_receipt_verified": True,
            "forward_pass_executed": False,
            "framework_profiles_verified": True,
            "onnx_profiles_exported": [640, 960],
            "synthetic_onnx_parity_passed": True,
            "onnx_batch12_shape_verified": True,
            "onnx_batch12_profiles": [640, 960],
            "onnx_batch12_performance_claimed": False,
            "real_image_framework_onnx_evidence_verified": True,
            "real_image_inference_executed": True,
            "real_image_selected_frame_count": 11,
            "real_image_unique_video_type_count": 11,
            "real_image_profiles": {
                "640": {
                    "batch1_passed": True,
                    "batch2_passed": True,
                    "passed": True,
                },
                "960": {
                    "batch1_passed": False,
                    "batch2_passed": True,
                    "passed": False,
                },
            },
            "real_image_failure_count": 4,
            "real_image_tolerances_relaxed": False,
            "real_image_framework_onnx_parity_passed": False,
            "parser_cpu_contract_ready": True,
            "parser_contract_test_passed": True,
            "parser_max_batch_contract": 12,
            "training_executed": False,
            "tensorrt_built": False,
            "deepstream9_executed": False,
            "production_selected": False,
        },
        "gpu_execution_r10": _person_rtdetr_gpu_r10(reader),
        "gates": {
            "model_selected": False,
            "license_selected": False,
            "training_complete": False,
            "framework_parity_passed": False,
            "onnx_export_complete": False,
            "onnx_parity_passed": False,
            "tensorrt_engines_complete": False,
            "tensorrt_parity_passed": False,
            "deepstream9_parity_passed": False,
            "independent_ground_truth_quality_passed": False,
            "exact_25m_passed": False,
            "twelve_camera_640_passed": False,
            "twelve_camera_960_passed": False,
            "three_module_full_stack_passed": False,
            "acceptance_passed": False,
            "production_ready": False,
        },
        "integrity": integrity,
        "caveats": [
            "Hazır veri ve dondurulmuş plan, eğitim veya model kabulü değildir.",
            "Lisans temeli seçilmeden YOLO26 indirme ve eğitim yetkisi kapalıdır.",
            "RT-DETRv4-S checkpoint'i CPU-only strict model/EMA yükleme receipt'iyle doğrulandı; TensorRT ve DeepStream 9 kanıtı yoktur.",
            "11 açık-lisanslı gerçek karede framework/ONNX kanıtı doğrulandı: 640 geçti, 960 batch-1 dört strict fark nedeniyle kaldı; tolerans gevşetilmedi ve parity gate'i kapalıdır.",
            "Gerçek-kare receipt'i kalite, mesafe, latency, FPS veya kapasite kanıtı değildir; TopK tie tanısı kabul sonucunu değiştirmez.",
            "640/960 ONNX profilleri CPU ORT'de sentetik batch-12 şekil/finite kontrolünü geçti; FPS, latency, TensorRT, DS9 veya kapasite başarısı değildir.",
            "DS9 parser paylaşımlı kütüphanesi CPU-only ABI/contract testini geçti; TensorRT engine, GPU entegrasyonu veya gerçek DeepStream 9 inference kanıtı değildir.",
            "Sürümlü R10 GPU kartı, tarihsel YOLO26 hazırlık alanlarını değiştirmeden RT-DETRv4-S smoke ve iç baseline kanıtını ayrı yayınlar.",
        ],
        "evidence": [],
    }


def _ppe_seed_unavailable(
    reason: str,
    *,
    integrity: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "label": "PPE veri tohumu hazırlığı",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "preparation": {
            "source_manifest_verified": False,
            "receipt_contracts_verified": False,
            "data_acquired": False,
            "quarantine_complete": False,
        },
        "source_contract": {},
        "receipts": {
            "acquisition": {
                "pin_declared": True,
                "verified": False,
                "accepted": False,
            },
            "quarantine": {
                "pin_declared": True,
                "verified": False,
                "accepted": False,
            },
        },
        "provenance_review": {
            "evidence_verified": False,
            "mechanical_audit_complete": False,
            "training_eligible": False,
        },
        "normalization": {
            "evidence_verified": False,
            "provenance_review_evidence_present": False,
            "provenance_mechanical_audit_replayed": False,
            "provenance_review_approved": False,
            "embedded_rights_review_approved": False,
            "camera_group_split_approved": False,
            "normalization_ready": False,
            "source_training_eligible": False,
            "normalized_training_eligible": False,
            "independent_bbox_out_of_range_count": None,
            "bbox_overflow_severity": None,
        },
        "gates": {
            "source_contract_verified": False,
            "acquired": False,
            "quarantined": False,
            "rights_audit_complete": False,
            "camera_group_split_audit_complete": False,
            "training_eligible": False,
            "training_complete": False,
            "export_complete": False,
            "deepstream9_evaluated": False,
            "ground_truth_quality_passed": False,
            "twelve_camera_640_passed": False,
            "twelve_camera_960_passed": False,
            "acceptance_passed": False,
            "production_ready": False,
        },
        "integrity": integrity or {},
        "caveats": [
            "PPE seed kanıt zinciri doğrulanamadı; edinimden ürün kabulüne kadar bütün kapılar kapalıdır.",
        ],
        "evidence": [],
    }


def _ppe_seed_schema_contract_valid(
    schema: Any,
    *,
    schema_id: str,
    schema_version: str,
    operation: str | None = None,
) -> bool:
    if not isinstance(schema, dict):
        return False
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return False
    valid = bool(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id") == schema_id
        and schema.get("type") == "object"
        and schema.get("additionalProperties") is False
        and properties.get("schema_version", {}).get("const")
        == schema_version
    )
    if operation is None:
        return valid
    required = schema.get("required")
    receipt_fingerprint = properties.get("receipt_sha256", {})
    return bool(
        valid
        and properties.get("operation", {}).get("const") == operation
        and properties.get("training_eligible", {}).get("const") is False
        and isinstance(required, list)
        and "receipt_sha256" in required
        and receipt_fingerprint.get("type") == "string"
        and receipt_fingerprint.get("pattern") == "^[0-9a-f]{64}$"
    )


def _ppe_seed_pinned_asset(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    byte_count = value.get("bytes")
    digest = value.get("sha256")
    url = value.get("url")
    return bool(
        isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count > 0
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and isinstance(url, str)
        and url.startswith("https://")
    )


def _ppe_receipt_self_hash_matches(
    value: Any,
    *,
    expected: str,
) -> bool:
    if not isinstance(value, dict):
        return False
    observed = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    return bool(
        observed == expected
        and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
        and _canonical_sha256(unsigned) == expected
    )


def _ppe_provenance_schema_contract_valid(
    schema: Any,
    *,
    receipt: bool,
) -> bool:
    if not isinstance(schema, dict):
        return False
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return False
    suffix = "receipt" if receipt else "plan"
    schema_version = f"deepsafe.ppe-seed-provenance-review-{suffix}/v1"
    valid = bool(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            f"ppe-seed-provenance-review-{suffix}-v1.schema.json"
        )
        and schema.get("type") == "object"
        and schema.get("additionalProperties") is False
        and properties.get("schema_version", {}).get("const")
        == schema_version
        and "schema_version" in required
    )
    if not receipt:
        return bool(
            valid
            and {
                "inputs",
                "component_sources",
                "rights_policy",
                "filename_family_policy",
                "near_duplicate_policy",
                "expected_observations",
                "required_blockers",
            }.issubset(required)
        )
    return bool(
        valid
        and properties.get("operation", {}).get("const")
        == "audit_ppe_seed_provenance_rights_and_split_without_extraction"
        and properties.get("training_eligible", {}).get("const") is False
        and properties.get("receipt_sha256", {}).get("$ref")
        == "#/$defs/sha256"
        and {
            "rights_review",
            "split_review",
            "duplicate_review",
            "observations",
            "gates",
            "training_eligible",
            "receipt_sha256",
        }.issubset(required)
    )


def _ppe_provenance_r2_semantics_valid(
    plan: Any,
    receipt: Any,
    *,
    acquisition: dict[str, Any],
    quarantine: dict[str, Any],
) -> bool:
    if not isinstance(plan, dict) or not isinstance(receipt, dict):
        return False
    expected_archive = {
        "path": "data/raw/ppe/mendeley-ppe-v6/20250731-PPE2286y.zip",
        "bytes": 236065015,
        "sha256": (
            "7a22e5cbc0327971f56358e14f1fce88fec1d6ab30cfac642de0746202ae1f0d"
        ),
    }
    expected_observations = {
        "images": 2286,
        "train_images": 1829,
        "validation_images": 457,
        "filename_families": 6,
        "cross_split_filename_families": 6,
        "exact_content_duplicate_groups": 0,
        "exact_original_key_duplicate_groups": 0,
        "strict_cross_split_candidate_pairs": 7,
        "strict_cross_split_validation_members": 4,
        "high_confidence_automated_candidate_pairs": 2,
    }
    expected_blockers = [
        "authoritative_quarantine_structural_failure",
        "item_level_source_mapping_incomplete",
        "embedded_third_party_rights_review_incomplete",
        "depicted_person_consent_or_lawful_basis_not_item_verified",
        "location_owner_capture_rights_not_item_verified",
        "article_and_dataset_license_scopes_require_separate_review",
        "camera_site_session_metadata_absent",
        "published_random_file_level_split_not_group_safe",
        "cross_split_filename_family_overlap",
        "cross_split_near_duplicate_candidates_detected",
        "near_duplicate_human_review_incomplete",
    ]
    expected_quarantine_plan_pin = {
        "path": PPE_SEED_RECEIPT_PINS["quarantine"]["path"],
        "file_sha256": PPE_SEED_RECEIPT_PINS["quarantine"]["sha256"],
        "receipt_sha256": PPE_SEED_RECEIPT_PINS["quarantine"][
            "receipt_sha256"
        ],
        "structural_pass": False,
    }
    expected_component_ids = {
        "gdut-hwd-selected",
        "helmetvest-roboflow-selected",
        "kaggle-safety-helmet-reflective-jacket-selected",
        "authors-real-world-photographs",
    }
    components = plan.get("component_sources")
    families = plan.get("filename_family_policy", {}).get(
        "expected_families"
    )
    research = plan.get("research_evidence")
    plan_valid = bool(
        plan.get("schema_version")
        == "deepsafe.ppe-seed-provenance-review-plan/v1"
        and plan.get("plan_id")
        == "mendeley-ppe-v6-provenance-rights-split-review-r2"
        and plan.get("status") == "planned_fail_closed_training_blocked"
        and plan.get("source_id") == "mendeley-ppe-v6-20250731"
        and plan.get("research_cutoff") == "2026-07-17"
        and plan.get("inputs", {}).get("archive") == expected_archive
        and plan.get("inputs", {}).get("quarantine_receipt")
        == expected_quarantine_plan_pin
        and isinstance(plan.get("inputs", {}).get("embedded_members"), list)
        and len(plan["inputs"]["embedded_members"]) == 4
        and isinstance(research, list)
        and len(research) == 11
        and isinstance(components, list)
        and len(components) == 4
        and {item.get("id") for item in components if isinstance(item, dict)}
        == expected_component_ids
        and sum(
            item.get("declared_images", 0)
            for item in components
            if isinstance(item, dict)
        )
        == 2286
        and all(
            isinstance(item, dict)
            and item.get("item_level_member_mapping_complete") is False
            and item.get("embedded_media_chain_of_title_verified") is False
            and item.get("depicted_person_rights_verified") is False
            and item.get("location_capture_rights_verified") is False
            for item in components
        )
        and plan.get("rights_policy", {}).get(
            "aggregate_license_is_not_item_level_clearance"
        )
        is True
        and plan.get("rights_policy", {}).get(
            "publication_license_is_not_dataset_license"
        )
        is True
        and plan.get("rights_policy", {}).get(
            "commercial_training_requires_item_level_chain_of_title"
        )
        is True
        and plan.get("rights_policy", {}).get(
            "person_and_location_rights_require_separate_review"
        )
        is True
        and plan.get("rights_policy", {}).get(
            "embedded_third_party_rights_review_complete"
        )
        is False
        and isinstance(families, list)
        and len(families) == 6
        and sum(
            item.get("images", 0)
            for item in families
            if isinstance(item, dict)
        )
        == 2286
        and all(
            isinstance(item, dict)
            and item.get("train", 0) + item.get("validation", 0)
            == item.get("images")
            for item in families
        )
        and plan.get("near_duplicate_policy")
        == {
            "method": (
                "opencv_grayscale_dct_phash63_plus_horizontal_dhash64_"
                "cross_split_v1"
            ),
            "strict_thresholds": {
                "phash_hamming_max": 4,
                "dhash_hamming_max": 4,
            },
            "high_confidence_thresholds": {
                "phash_hamming_max": 1,
                "dhash_hamming_max": 1,
                "resized_rgb_mae_max": 2.0,
                "grayscale_correlation_min": 0.999,
            },
            "max_candidate_pairs": 10000,
            "human_confirmation_required": True,
        }
        and plan.get("expected_observations") == expected_observations
        and plan.get("required_blockers") == expected_blockers
    )
    if not plan_valid:
        return False

    gate_rows = receipt.get("gates")
    expected_gate_states = {
        "plan_schema_and_pin_verified": True,
        "archive_identity_verified": True,
        "authoritative_quarantine_receipt_verified": True,
        "authoritative_quarantine_structural_pass": False,
        "embedded_evidence_member_pins_verified": True,
        "component_source_counts_reconcile": True,
        "filename_family_ledger_replayed": True,
        "exact_content_duplicate_free": True,
        "exact_original_key_duplicate_free": True,
        "cross_split_perceptual_candidate_free": False,
        "item_level_source_mapping_complete": False,
        "embedded_third_party_rights_review_complete": False,
        "camera_group_split_review_complete": False,
    }
    gate_by_id = {
        item.get("id"): item.get("passed")
        for item in gate_rows or []
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    split = receipt.get("split_review")
    duplicate = receipt.get("duplicate_review")
    rights = receipt.get("rights_review")
    if not all(isinstance(item, dict) for item in (split, duplicate, rights)):
        return False
    assert isinstance(split, dict)
    assert isinstance(duplicate, dict)
    assert isinstance(rights, dict)
    candidates = duplicate.get("cross_split_candidates")
    candidate_validation_paths = {
        item.get("validation", {}).get("path")
        for item in candidates or []
        if isinstance(item, dict)
    }
    receipt_quarantine = receipt.get("inputs", {}).get(
        "quarantine_receipt"
    )
    receipt_plan = receipt.get("plan")
    receipt_valid = bool(
        receipt.get("schema_version")
        == "deepsafe.ppe-seed-provenance-review-receipt/v1"
        and receipt.get("operation")
        == "audit_ppe_seed_provenance_rights_and_split_without_extraction"
        and receipt.get("source_id") == "mendeley-ppe-v6-20250731"
        and isinstance(receipt.get("created_at"), str)
        and isinstance(receipt_plan, dict)
        and receipt_plan.get("path") == PPE_PROVENANCE_PLAN_PIN["path"]
        and receipt_plan.get("sha256") == PPE_PROVENANCE_PLAN_PIN["sha256"]
        and receipt_plan.get("schema_sha256")
        == PPE_PROVENANCE_SCHEMA_PINS["plan"]["sha256"]
        and str(receipt_plan.get("schema_path", "")).endswith(
            PPE_PROVENANCE_SCHEMA_PINS["plan"]["path"]
        )
        and receipt.get("inputs", {}).get("archive") == expected_archive
        and receipt.get("inputs", {}).get("embedded_members")
        == plan["inputs"]["embedded_members"]
        and receipt_quarantine
        == {
            **expected_quarantine_plan_pin,
            "accepted_to_quarantine": False,
            "training_eligible": False,
        }
        and receipt.get("research_evidence") == plan["research_evidence"]
        and receipt.get("component_sources") == components
        and receipt.get("observations") == expected_observations
        and len(gate_by_id) == len(expected_gate_states)
        and gate_by_id == expected_gate_states
        and rights
        == {
            "aggregate_dataset_license_metadata_recorded": True,
            "article_license_scope_separated_from_dataset_metadata": True,
            "item_level_source_mapping_complete": False,
            "embedded_third_party_rights_review_complete": False,
            "depicted_person_consent_or_lawful_basis_verified": False,
            "location_owner_capture_rights_verified": False,
            "commercial_training_clearance": False,
            "review_note": plan["rights_policy"]["review_note"],
        }
        and split.get("published_split_method")
        == "unseeded_random_file_level_sample_and_move"
        and split.get("site_camera_session_metadata_present") is False
        and split.get("camera_group_split_review_complete") is False
        and split.get("cross_split_family_count") == 6
        and split.get("all_images_in_cross_split_families") == 2286
        and split.get("family_mapping_is_item_level_proof") is False
        and isinstance(split.get("families"), list)
        and len(split["families"]) == 6
        and all(item.get("cross_split") is True for item in split["families"])
        and duplicate.get("image_ledger_members") == 2286
        and duplicate.get("exact_content_duplicate_groups") == []
        and duplicate.get("exact_original_key_duplicate_groups") == []
        and duplicate.get("strict_candidate_pairs") == 7
        and duplicate.get("high_confidence_automated_candidate_pairs") == 2
        and duplicate.get("human_review_complete") is False
        and isinstance(candidates, list)
        and len(candidates) == 7
        and len(candidate_validation_paths) == 4
        and sum(
            item.get("high_confidence_automated_candidate") is True
            for item in candidates
            if isinstance(item, dict)
        )
        == 2
        and all(
            item.get("human_confirmed") is False
            for item in candidates
            if isinstance(item, dict)
        )
        and receipt.get("mechanical_audit_complete") is True
        and receipt.get("embedded_third_party_rights_review_complete")
        is False
        and receipt.get("camera_group_split_review_complete") is False
        and receipt.get("training_eligible") is False
        and receipt.get("status") == "blocked"
        and receipt.get("blockers") == expected_blockers
    )
    acquisition_observed = acquisition.get("observed", {})
    quarantine_archive = quarantine.get("archive", {})
    return bool(
        receipt_valid
        and acquisition_observed.get("bytes") == expected_archive["bytes"]
        and acquisition_observed.get("sha256") == expected_archive["sha256"]
        and quarantine_archive.get("bytes") == expected_archive["bytes"]
        and quarantine_archive.get("sha256") == expected_archive["sha256"]
        and quarantine.get("structural_pass") is False
        and quarantine.get("accepted_to_quarantine") is False
        and quarantine.get("training_eligible") is False
        and isinstance(quarantine.get("created_at"), str)
        and quarantine["created_at"] < receipt["created_at"]
    )


def _ppe_normalization_schema_contract_valid(
    schema: Any,
    *,
    schema_id: str,
    schema_version: str,
    operation: str | None = None,
) -> bool:
    if not isinstance(schema, dict):
        return False
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return False
    valid = bool(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id") == schema_id
        and schema.get("type") == "object"
        and schema.get("additionalProperties") is False
        and properties.get("schema_version", {}).get("const")
        == schema_version
        and "schema_version" in required
    )
    if operation is None:
        return valid
    return bool(
        valid
        and properties.get("operation", {}).get("const") == operation
        and properties.get("source_archive_training_eligible", {}).get(
            "const"
        )
        is False
        and properties.get("normalized_dataset_training_eligible", {}).get(
            "const"
        )
        is False
        and properties.get("normalization_ready", {}).get("const") is False
        and {
            "operation",
            "source_archive_training_eligible",
            "normalized_dataset_training_eligible",
            "normalization_ready",
            "receipt_sha256",
        }.issubset(required)
    )


def _ppe_normalization_legacy_r1_semantics_valid(
    plan: Any,
    assessment: Any,
    *,
    acquisition: dict[str, Any],
    quarantine: dict[str, Any],
) -> bool:
    if not isinstance(plan, dict) or not isinstance(assessment, dict):
        return False

    expected_archive = {
        "path": "data/raw/ppe/mendeley-ppe-v6/20250731-PPE2286y.zip",
        "bytes": 236065015,
        "sha256": (
            "7a22e5cbc0327971f56358e14f1fce88fec1d6ab30cfac642de0746202ae1f0d"
        ),
    }
    expected_plan_id = (
        "mendeley-ppe-v6-person-equipment-decisions-v2-overlay-r1"
    )
    expected_histogram = {"0": 2421, "1": 1174, "2": 1058, "3": 1385}
    expected_observations = {
        "image_count": 2286,
        "label_file_count": 2286,
        "label_row_count": 6038,
        "split_image_counts": {"train": 1829, "validation": 457},
        "class_id_histogram": expected_histogram,
        "bbox_out_of_range_count": 52,
        "declared_dimension_mismatch_count": 1814,
    }
    expected_blockers = [
        "authoritative_quarantine_rejected",
        "declared_dimensions_mismatch_manual_resolution_required",
        "bbox_out_of_range_manual_review_required",
        "provenance_review_required",
        "embedded_third_party_rights_review_required",
        "camera_group_metadata_review_required",
        "reviewed_person_roi_association_required",
        "unknown_not_visible_ambiguous_preservation_review_required",
        "near_duplicate_and_upstream_provenance_review_required",
        "canonical_v2_dataset_not_generated",
        "deterministic_group_split_and_leakage_audit_required",
    ]
    expected_overflow = {
        "count": 52,
        "maximum": 0.00003597122302156919,
        "median": 0.000007812500000037303,
        "minimum": 0.0000031249999998816946,
        "p95": 0.000020833333333358794,
        "p95_method": "sorted_values[round(0.95*(n-1))]",
    }
    expected_inputs = {
        "seed_manifest": {
            "path": PPE_SEED_MANIFEST_PIN["path"],
            "sha256": PPE_SEED_MANIFEST_PIN["sha256"],
        },
        "source_archive": expected_archive,
        "acquisition_receipt": {
            "path": PPE_SEED_RECEIPT_PINS["acquisition"]["path"],
            "receipt_sha256": PPE_SEED_RECEIPT_PINS["acquisition"][
                "receipt_sha256"
            ],
        },
        "authoritative_quarantine_receipt": {
            "path": PPE_SEED_RECEIPT_PINS["quarantine"]["path"],
            "receipt_sha256": PPE_SEED_RECEIPT_PINS["quarantine"][
                "receipt_sha256"
            ],
        },
        "superseded_quarantine_receipt": {
            "path": PPE_SUPERSEDED_QUARANTINE_LINEAGE["path"],
            "receipt_sha256": PPE_SUPERSEDED_QUARANTINE_LINEAGE[
                "receipt_sha256"
            ],
        },
        "canonical_schema": {
            "path": PPE_NORMALIZATION_SCHEMA_PINS["canonical_dataset"][
                "path"
            ],
            "sha256": PPE_NORMALIZATION_SCHEMA_PINS["canonical_dataset"][
                "sha256"
            ],
        },
    }
    overlay = plan.get("overlay")
    semantic_policy = plan.get("semantic_policy")
    split_policy = plan.get("split_policy")
    review_evidence = plan.get("review_evidence")
    plan_valid = bool(
        plan.get("schema_version") == "deepsafe.ppe-normalization-plan/v1"
        and plan.get("plan_id") == expected_plan_id
        and plan.get("status")
        == "blocked_pending_manual_review_and_canonical_dataset"
        and plan.get("source_id") == "mendeley-ppe-v6-20250731"
        and plan.get("inputs") == expected_inputs
        and isinstance(overlay, dict)
        and overlay.get("mode")
        == "non_mutating_archive_relative_metadata_overlay"
        and overlay.get("upstream_yaml_path_disposition")
        == "ignore_unsafe_absolute_path_preserve_original_member"
        and overlay.get("classes")
        == ["Helmet", "NoHelmet", "NoVest", "Vest"]
        and overlay.get("max_label_bytes") == 5242880
        and overlay.get("bbox_repair_policy")
        == {
            "silent_clipping": False,
            "automatic_drop": False,
            "required_before_repair": (
                "immutable_original_and_repaired_coordinate_ledger_plus_"
                "per_row_human_or_frozen_policy_approval_plus_derived_"
                "artifact_hash"
            ),
        }
        and overlay.get("dimension_policy")
        == "preserve_decoded_per_image_dimensions_declared_640x640_is_not_ground_truth"
        and semantic_policy
        == {
            "target_schema_version": "ppe-person-equipment-decisions-v2.0",
            "source_archive_training_eligible": False,
            "raw_object_box_conversion": (
                "forbidden_without_reviewed_person_roi_association"
            ),
            "person_roi_requirement": (
                "every_known_helmet_or_hi_vis_decision_bbox_must_be_inside_"
                "one_reviewed_person_bbox"
            ),
            "unknown_policy": (
                "preserve_unknown_not_visible_too_small_and_ambiguous_"
                "without_emitting_training_label"
            ),
            "canonical_label_emission": (
                "disabled_until_all_normalization_gates_pass"
            ),
        }
        and split_policy
        == {
            "method": (
                "connected-site-camera-session-and-provenance-groups-v1"
            ),
            "seed": "deepsafe-ppe-mendeley-v6-v1",
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "exact_duplicate_policy": (
                "remove_all_but_one_before_group_split_and_reject_cross_"
                "split_duplicates"
            ),
            "near_duplicate_policy": "review_and_group_before_split",
        }
        and isinstance(review_evidence, dict)
        and set(review_evidence)
        == {
            "provenance",
            "embedded_rights",
            "camera_group",
            "person_roi_association",
            "unknown_state_preservation",
            "near_duplicate",
        }
        and all(value is None for value in review_evidence.values())
        and plan.get("expected_observations") == expected_observations
    )
    if not plan_valid:
        return False

    gate_rows = assessment.get("gates")
    if not isinstance(gate_rows, list):
        return False
    gate_by_id = {
        row.get("id"): row
        for row in gate_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    expected_gate_states = {
        "plan_schema_and_external_pin_verified": True,
        "seed_manifest_exact_pin_verified": True,
        "source_archive_exact_pin_verified": True,
        "canonical_v2_schema_exact_pin_verified": True,
        "acquisition_receipt_exact_pin_verified": True,
        "authoritative_quarantine_receipt_exact_pin_verified": True,
        "superseded_quarantine_lineage_exact_pin_verified": True,
        "source_identity_chain_consistent": True,
        "authoritative_quarantine_accepted": False,
        "safe_archive_prerequisites_passed": True,
        "declared_image_dimensions_match_payload": False,
        "declared_dimension_audit_reproduces_expected_count": True,
        "overlay_archive_roots_and_pairing_verified": True,
        "overlay_class_contract_verified": True,
        "offline_label_audit_reproduces_expected_counts": True,
        "all_raw_bboxes_strictly_inside_normalized_frame": False,
        "provenance_review_approved": False,
        "embedded_rights_review_approved": False,
        "camera_group_review_approved": False,
        "person_roi_association_review_approved": False,
        "unknown_state_preservation_review_approved": False,
        "near_duplicate_review_approved": False,
        "canonical_person_equipment_decisions_v2_dataset_present": False,
        "deterministic_group_split_and_leakage_audit_passed": False,
    }
    gates_valid = bool(
        len(gate_rows) == len(gate_by_id) == len(expected_gate_states)
        and all(
            isinstance(gate_by_id.get(gate_id), dict)
            and gate_by_id[gate_id].get("passed") is expected
            and isinstance(gate_by_id[gate_id].get("details"), dict)
            for gate_id, expected in expected_gate_states.items()
        )
    )
    if not gates_valid:
        return False

    def details(gate_id: str) -> dict[str, Any]:
        value = gate_by_id[gate_id].get("details")
        return value if isinstance(value, dict) else {}

    acquisition_gate = details("acquisition_receipt_exact_pin_verified")
    quarantine_gate = details(
        "authoritative_quarantine_receipt_exact_pin_verified"
    )
    superseded_gate = details(
        "superseded_quarantine_lineage_exact_pin_verified"
    )
    lineage_gates_valid = bool(
        details("seed_manifest_exact_pin_verified")
        == {
            "bytes": PPE_SEED_MANIFEST_PIN["bytes"],
            "expected_sha256": PPE_SEED_MANIFEST_PIN["sha256"],
            "observed_sha256": PPE_SEED_MANIFEST_PIN["sha256"],
            "path": PPE_SEED_MANIFEST_PIN["path"],
        }
        and details("source_archive_exact_pin_verified")
        == {
            "bytes": expected_archive["bytes"],
            "expected_bytes": expected_archive["bytes"],
            "expected_sha256": expected_archive["sha256"],
            "observed_sha256": expected_archive["sha256"],
            "path": expected_archive["path"],
        }
        and details("canonical_v2_schema_exact_pin_verified")
        == {
            "bytes": PPE_NORMALIZATION_SCHEMA_PINS["canonical_dataset"][
                "bytes"
            ],
            "expected_sha256": PPE_NORMALIZATION_SCHEMA_PINS[
                "canonical_dataset"
            ]["sha256"],
            "observed_sha256": PPE_NORMALIZATION_SCHEMA_PINS[
                "canonical_dataset"
            ]["sha256"],
            "path": PPE_NORMALIZATION_SCHEMA_PINS["canonical_dataset"][
                "path"
            ],
        }
        and acquisition_gate.get("path")
        == PPE_SEED_RECEIPT_PINS["acquisition"]["path"]
        and acquisition_gate.get("file_sha256")
        == PPE_SEED_RECEIPT_PINS["acquisition"]["sha256"]
        and acquisition_gate.get("expected_receipt_sha256")
        == PPE_SEED_RECEIPT_PINS["acquisition"]["receipt_sha256"]
        and acquisition_gate.get("observed_receipt_sha256")
        == PPE_SEED_RECEIPT_PINS["acquisition"]["receipt_sha256"]
        and acquisition_gate.get("external_pin_verified") is True
        and quarantine_gate.get("path")
        == PPE_SEED_RECEIPT_PINS["quarantine"]["path"]
        and quarantine_gate.get("file_sha256")
        == PPE_SEED_RECEIPT_PINS["quarantine"]["sha256"]
        and quarantine_gate.get("expected_receipt_sha256")
        == PPE_SEED_RECEIPT_PINS["quarantine"]["receipt_sha256"]
        and quarantine_gate.get("observed_receipt_sha256")
        == PPE_SEED_RECEIPT_PINS["quarantine"]["receipt_sha256"]
        and quarantine_gate.get("external_pin_verified") is True
        and superseded_gate.get("path")
        == PPE_SUPERSEDED_QUARANTINE_LINEAGE["path"]
        and superseded_gate.get("file_sha256")
        == PPE_SUPERSEDED_QUARANTINE_LINEAGE["sha256"]
        and superseded_gate.get("expected_receipt_sha256")
        == PPE_SUPERSEDED_QUARANTINE_LINEAGE["receipt_sha256"]
        and superseded_gate.get("observed_receipt_sha256")
        == PPE_SUPERSEDED_QUARANTINE_LINEAGE["receipt_sha256"]
        and superseded_gate.get("external_pin_verified") is True
        and details("source_identity_chain_consistent")
        == {
            "acquisition_archive_sha256": expected_archive["sha256"],
            "archive_sha256": expected_archive["sha256"],
            "quarantine_archive_sha256": expected_archive["sha256"],
        }
    )
    if not lineage_gates_valid:
        return False

    audit = assessment.get("observations", {}).get(
        "offline_overlay_label_audit"
    )
    if not isinstance(audit, dict):
        return False
    issues = audit.get("issues")
    issue_rows_valid = bool(
        isinstance(issues, list)
        and len(issues) == 52
        and all(
            isinstance(item, dict)
            and item.get("code") == "bbox_out_of_range"
            and item.get("class_id") in {0, 1, 2, 3}
            and isinstance(item.get("normalized_overflow"), (int, float))
            and not isinstance(item.get("normalized_overflow"), bool)
            and 0 < item["normalized_overflow"] <= expected_overflow["maximum"]
            for item in issues
        )
    )
    audit_valid = bool(
        audit.get("image_count") == 2286
        and audit.get("label_file_count") == 2286
        and audit.get("label_row_count") == 6038
        and audit.get("split_image_counts")
        == {"train": 1829, "validation": 457}
        and audit.get("class_id_histogram") == expected_histogram
        and sum(expected_histogram.values()) == 6038
        and audit.get("issue_counts") == {"bbox_out_of_range": 52}
        and audit.get("bbox_overflow") == expected_overflow
        and issue_rows_valid
        and audit.get("missing_labels") == []
        and audit.get("orphan_labels") == []
        and audit.get("canonical_labels_emitted") is False
        and audit.get("automatic_repairs_applied") == []
        and audit.get("overlay_is_non_mutating") is True
        and audit.get("silent_bbox_clipping_applied") is False
        and audit.get("source_archive_sha256") == expected_archive["sha256"]
    )
    review_gate_ids = (
        "provenance_review_approved",
        "embedded_rights_review_approved",
        "camera_group_review_approved",
        "person_roi_association_review_approved",
        "unknown_state_preservation_review_approved",
        "near_duplicate_review_approved",
    )
    assessment_semantics_valid = bool(
        assessment.get("schema_version")
        == "deepsafe.ppe-normalization-assessment-receipt/v1"
        and assessment.get("operation")
        == "assess_non_mutating_ppe_normalization_overlay"
        and assessment.get("source_id") == "mendeley-ppe-v6-20250731"
        and assessment.get("plan")
        == {
            "path": PPE_NORMALIZATION_PLAN_PIN["path"],
            "file_sha256": PPE_NORMALIZATION_PLAN_PIN["sha256"],
            "expected_file_sha256": PPE_NORMALIZATION_PLAN_PIN["sha256"],
            "external_pin_verified": True,
            "plan_id": expected_plan_id,
        }
        and assessment.get("source_archive_training_eligible") is False
        and assessment.get("normalized_dataset_training_eligible") is False
        and assessment.get("normalization_ready") is False
        and assessment.get("eligibility_blockers") == expected_blockers
        and assessment.get("failure", {}).get("code")
        == "normalization_blocked"
        and assessment.get("failure", {}).get("details")
        == {"blockers": expected_blockers}
        and lineage_gates_valid
        and details("authoritative_quarantine_accepted")
        == {
            "accepted_to_quarantine": False,
            "failed_gate_ids": [
                "declared_image_dimensions_match",
                "valid_yolo_yaml",
                "yolo_split_paths_resolve_in_archive",
                "declared_classes_match_yaml",
                "valid_yolo_detection_labels",
            ],
            "structural_pass": False,
        }
        and details("declared_image_dimensions_match_payload")
        == {
            "declared": {"height": 640, "width": 640},
            "decoded_count": 2286,
            "mismatch_count": 1814,
            "policy": "preserve_decoded_per_image_dimensions_no_resize_claim",
        }
        and details("declared_dimension_audit_reproduces_expected_count")
        == {
            "decoded_count": 2286,
            "expected_mismatch_count": 1814,
            "observed_mismatch_count": 1814,
        }
        and details("offline_label_audit_reproduces_expected_counts")
        == {
            "expected_bbox_out_of_range_count": 52,
            "expected_label_row_count": 6038,
            "issue_counts": {"bbox_out_of_range": 52},
            "label_row_count": 6038,
        }
        and details("all_raw_bboxes_strictly_inside_normalized_frame")
        == {
            "automatic_clipping_permitted": False,
            "issue_count": 52,
            "overflow": expected_overflow,
            "repair_requirement": (
                "immutable_coordinate_ledger_plus_per_row_approval"
            ),
        }
        and all(
            details(gate_id)
            == {"exact_pin_verified": False, "state": "missing"}
            for gate_id in review_gate_ids
        )
        and details(
            "canonical_person_equipment_decisions_v2_dataset_present"
        ).get("state")
        == "not_generated"
        and details(
            "deterministic_group_split_and_leakage_audit_passed"
        ).get("state")
        == (
            "blocked_until_camera_session_provenance_metadata_and_"
            "canonical_dataset_exist"
        )
        and audit_valid
        and isinstance(assessment.get("created_at"), str)
        and isinstance(quarantine.get("created_at"), str)
        and quarantine["created_at"] < assessment["created_at"]
        and acquisition.get("receipt_sha256")
        == PPE_SEED_RECEIPT_PINS["acquisition"]["receipt_sha256"]
        and quarantine.get("receipt_sha256")
        == PPE_SEED_RECEIPT_PINS["quarantine"]["receipt_sha256"]
    )
    return assessment_semantics_valid


def _ppe_normalization_r2_semantics_valid(
    plan: Any,
    assessment: Any,
    *,
    provenance_plan: dict[str, Any],
    provenance_receipt: dict[str, Any],
    quarantine: dict[str, Any],
) -> bool:
    if not isinstance(plan, dict) or not isinstance(assessment, dict):
        return False
    expected_archive = {
        "path": "data/raw/ppe/mendeley-ppe-v6/20250731-PPE2286y.zip",
        "bytes": 236065015,
        "sha256": (
            "7a22e5cbc0327971f56358e14f1fce88fec1d6ab30cfac642de0746202ae1f0d"
        ),
    }
    expected_blockers = [
        "authoritative_quarantine_structural_failure",
        "provenance_mechanical_evidence_is_not_human_approval",
        "item_level_source_mapping_incomplete",
        "embedded_third_party_rights_review_incomplete",
        "depicted_person_and_location_rights_not_item_verified",
        "camera_site_session_metadata_absent",
        "published_random_file_split_not_group_safe",
        "all_filename_families_cross_split",
        "cross_split_near_duplicate_candidates_require_human_review",
        "bbox_out_of_range_manual_review_required",
        "declared_image_dimensions_mismatch_payload",
        "reviewed_person_roi_association_required",
        "unknown_not_visible_ambiguous_preservation_review_required",
        "canonical_person_equipment_decisions_v2_dataset_not_generated",
        "deterministic_group_split_and_leakage_audit_not_approved",
    ]
    expected_provenance = {
        "images": 2286,
        "train_images": 1829,
        "validation_images": 457,
        "filename_families": 6,
        "cross_split_filename_families": 6,
        "images_in_cross_split_filename_families": 2286,
        "exact_content_duplicate_groups": 0,
        "exact_original_key_duplicate_groups": 0,
        "strict_cross_split_candidate_pairs": 7,
        "strict_cross_split_validation_members": 4,
        "high_confidence_automated_candidate_pairs": 2,
    }
    expected_plan_observations = {
        "provenance": expected_provenance,
        "label_audit": {
            "images": 2286,
            "label_files": 2286,
            "label_rows": 6038,
            "class_id_histogram": {
                "0": 2421,
                "1": 1174,
                "2": 1058,
                "3": 1385,
            },
            "bbox_out_of_range": 52,
        },
        "dimension_audit": {
            "decoded_images": 2286,
            "declared_dimension_mismatches": 1814,
            "declared": {"width": 640, "height": 640},
        },
    }
    expected_history_plan = {
        "path": PPE_NORMALIZATION_SUPERSEDED_R1_PINS["plan"]["path"],
        "sha256": PPE_NORMALIZATION_SUPERSEDED_R1_PINS["plan"]["sha256"],
    }
    expected_history_receipt = {
        "path": PPE_NORMALIZATION_SUPERSEDED_R1_PINS["assessment"]["path"],
        "sha256": PPE_NORMALIZATION_SUPERSEDED_R1_PINS["assessment"][
            "sha256"
        ],
        "receipt_sha256": PPE_NORMALIZATION_SUPERSEDED_R1_PINS[
            "assessment"
        ]["receipt_sha256"],
    }
    expected_inputs = {
        "source_archive": expected_archive,
        "authoritative_quarantine_receipt": {
            "path": PPE_SEED_RECEIPT_PINS["quarantine"]["path"],
            "sha256": PPE_SEED_RECEIPT_PINS["quarantine"]["sha256"],
            "receipt_sha256": PPE_SEED_RECEIPT_PINS["quarantine"][
                "receipt_sha256"
            ],
        },
        "provenance_plan": {
            "path": PPE_PROVENANCE_PLAN_PIN["path"],
            "sha256": PPE_PROVENANCE_PLAN_PIN["sha256"],
        },
        "provenance_plan_schema": {
            "path": PPE_PROVENANCE_SCHEMA_PINS["plan"]["path"],
            "sha256": PPE_PROVENANCE_SCHEMA_PINS["plan"]["sha256"],
        },
        "provenance_receipt": {
            "path": PPE_PROVENANCE_RECEIPT_PIN["path"],
            "sha256": PPE_PROVENANCE_RECEIPT_PIN["sha256"],
            "receipt_sha256": PPE_PROVENANCE_RECEIPT_PIN[
                "receipt_sha256"
            ],
        },
        "provenance_receipt_schema": {
            "path": PPE_PROVENANCE_SCHEMA_PINS["receipt"]["path"],
            "sha256": PPE_PROVENANCE_SCHEMA_PINS["receipt"]["sha256"],
        },
        "canonical_schema": {
            "path": PPE_NORMALIZATION_SCHEMA_PINS["canonical_dataset"][
                "path"
            ],
            "sha256": PPE_NORMALIZATION_SCHEMA_PINS["canonical_dataset"][
                "sha256"
            ],
        },
    }
    plan_valid = bool(
        plan.get("schema_version") == "deepsafe.ppe-normalization-plan/v2"
        and plan.get("plan_id")
        == "mendeley-ppe-v6-person-equipment-decisions-v2-normalization-r2"
        and plan.get("status")
        == "provenance_r2_evidence_present_normalization_blocked"
        and plan.get("source_id") == "mendeley-ppe-v6-20250731"
        and plan.get("history")
        == {
            "predecessor_plan": expected_history_plan,
            "predecessor_receipt": expected_history_receipt,
            "preservation_policy": (
                "r1_plan_and_0440_receipt_are_immutable_predecessors_"
                "never_rewritten"
            ),
        }
        and plan.get("inputs") == expected_inputs
        and plan.get("policy")
        == {
            "provenance_review_evidence_present": True,
            "provenance_review_approved": False,
            "embedded_third_party_rights_review_approved": False,
            "camera_site_session_group_split_approved": False,
            "source_archive_training_eligible": False,
            "normalized_dataset_training_eligible": False,
            "normalization_ready": False,
            "silent_bbox_clamp": False,
            "automatic_annotation_drop": False,
            "raw_object_box_conversion": (
                "forbidden_without_reviewed_person_roi_association"
            ),
            "unknown_policy": (
                "preserve_unknown_not_visible_too_small_and_ambiguous_"
                "without_emitting_training_label"
            ),
        }
        and plan.get("expected_observations") == expected_plan_observations
        and plan.get("required_blockers") == expected_blockers
    )
    if not plan_valid:
        return False

    expected_gate_states = {
        "r2_plan_schema_and_external_pin_verified": True,
        "r1_history_exact_pins_verified_without_mutation": True,
        "source_archive_exact_pin_verified": True,
        "canonical_v2_schema_exact_pin_verified": True,
        "authoritative_quarantine_r2_exact_pin_verified": True,
        "authoritative_quarantine_structural_pass": False,
        "provenance_r2_plan_and_schema_exact_pins_verified": True,
        "provenance_r2_receipt_self_external_file_and_schema_pins_verified": True,
        "provenance_review_evidence_present": True,
        "provenance_mechanical_audit_replayed_and_exactly_matched": True,
        "published_filename_families_group_split_safe": False,
        "exact_duplicate_groups_absent": True,
        "cross_split_near_duplicate_candidates_absent": False,
        "r1_label_audit_exactly_replayed_without_repair": True,
        "all_raw_bboxes_strictly_inside_normalized_frame": False,
        "declared_image_dimensions_match_payload": False,
        "silent_repair_clamp_or_drop_absent": True,
        "provenance_review_approved": False,
        "embedded_third_party_rights_review_approved": False,
        "camera_site_session_group_split_approved": False,
        "canonical_person_equipment_decisions_v2_dataset_present": False,
    }
    gate_rows = assessment.get("gates")
    gate_by_id = {
        item.get("id"): item.get("passed")
        for item in gate_rows or []
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    observations = assessment.get("observations")
    if not isinstance(observations, dict):
        return False
    authoritative = observations.get("provenance_authoritative")
    replay = observations.get("provenance_replay")
    label = observations.get("label_audit_replay")
    dimension = observations.get("dimension_audit")
    if not all(
        isinstance(item, dict)
        for item in (authoritative, replay, label, dimension)
    ):
        return False
    assert isinstance(authoritative, dict)
    assert isinstance(replay, dict)
    assert isinstance(label, dict)
    assert isinstance(dimension, dict)
    duplicate = provenance_receipt.get("duplicate_review", {})
    expected_provenance_with_digests = {
        **expected_provenance,
        "mechanical_audit_complete": True,
        "human_review_complete": False,
        "image_ledger_sha256": duplicate.get("image_ledger_sha256"),
        "cross_split_candidates_sha256": duplicate.get(
            "cross_split_candidates_sha256"
        ),
    }
    overflow = label.get("bbox_overflow")
    expected_overflow = {
        "count": 52,
        "maximum": 0.00003597122302156919,
        "median": 0.000007812500000037303,
        "minimum": 0.0000031249999998816946,
        "p95": 0.000020833333333358794,
        "p95_method": "sorted_values[round(0.95*(n-1))]",
    }
    history = assessment.get("history")
    assessment_valid = bool(
        assessment.get("schema_version")
        == "deepsafe.ppe-normalization-assessment-receipt/v2"
        and assessment.get("operation")
        == "assess_history_preserving_ppe_normalization_with_provenance_r2_replay"
        and assessment.get("source_id") == "mendeley-ppe-v6-20250731"
        and isinstance(assessment.get("created_at"), str)
        and assessment.get("plan")
        == {
            "path": PPE_NORMALIZATION_PLAN_PIN["path"],
            "file_sha256": PPE_NORMALIZATION_PLAN_PIN["sha256"],
            "expected_file_sha256": PPE_NORMALIZATION_PLAN_PIN["sha256"],
            "external_pin_verified": True,
            "plan_id": plan["plan_id"],
        }
        and history
        == {
            "predecessor_plan": {
                "path": PPE_NORMALIZATION_SUPERSEDED_R1_PINS["plan"][
                    "path"
                ],
                "bytes": PPE_NORMALIZATION_SUPERSEDED_R1_PINS["plan"][
                    "bytes"
                ],
                "expected_sha256": PPE_NORMALIZATION_SUPERSEDED_R1_PINS[
                    "plan"
                ]["sha256"],
                "observed_sha256": PPE_NORMALIZATION_SUPERSEDED_R1_PINS[
                    "plan"
                ]["sha256"],
            },
            "predecessor_receipt": {
                "path": PPE_NORMALIZATION_SUPERSEDED_R1_PINS[
                    "assessment"
                ]["path"],
                "bytes": PPE_NORMALIZATION_SUPERSEDED_R1_PINS[
                    "assessment"
                ]["bytes"],
                "expected_sha256": PPE_NORMALIZATION_SUPERSEDED_R1_PINS[
                    "assessment"
                ]["sha256"],
                "observed_sha256": PPE_NORMALIZATION_SUPERSEDED_R1_PINS[
                    "assessment"
                ]["sha256"],
                "external_receipt_pin_verified": True,
                "receipt_sha256": PPE_NORMALIZATION_SUPERSEDED_R1_PINS[
                    "assessment"
                ]["receipt_sha256"],
                "schema_version": (
                    "deepsafe.ppe-normalization-assessment-receipt/v1"
                ),
            },
            "predecessor_semantics_preserved": True,
            "preservation_policy": (
                "r1_plan_and_0440_receipt_are_immutable_predecessors_"
                "never_rewritten"
            ),
        }
        and len(gate_by_id) == len(expected_gate_states)
        and gate_by_id == expected_gate_states
        and authoritative == expected_provenance_with_digests
        and replay == expected_provenance_with_digests
        and label.get("image_count") == 2286
        and label.get("label_file_count") == 2286
        and label.get("label_row_count") == 6038
        and label.get("split_image_counts")
        == {"train": 1829, "validation": 457}
        and label.get("class_id_histogram")
        == {"0": 2421, "1": 1174, "2": 1058, "3": 1385}
        and label.get("issue_counts") == {"bbox_out_of_range": 52}
        and overflow == expected_overflow
        and label.get("overlay_is_non_mutating") is True
        and label.get("silent_bbox_clipping_applied") is False
        and label.get("automatic_repairs_applied") == []
        and label.get("canonical_labels_emitted") is False
        and dimension
        == {
            "declared": {"height": 640, "width": 640},
            "decoded_count": 2286,
            "gate_passed": False,
            "mismatch_count": 1814,
            "policy": (
                "preserve_decoded_per_image_dimensions_no_resize_claim"
            ),
        }
        and assessment.get("provenance_review_evidence_present") is True
        and assessment.get("provenance_mechanical_audit_replayed") is True
        and assessment.get("provenance_review_approved") is False
        and assessment.get("embedded_third_party_rights_review_approved")
        is False
        and assessment.get("camera_site_session_group_split_approved")
        is False
        and assessment.get("source_archive_training_eligible") is False
        and assessment.get("normalized_dataset_training_eligible") is False
        and assessment.get("normalization_ready") is False
        and assessment.get("eligibility_blockers") == expected_blockers
        and assessment.get("failure", {}).get("code")
        == "normalization_blocked"
        and assessment.get("failure", {}).get("details", {}).get("blockers")
        == expected_blockers
        and quarantine.get("receipt_sha256")
        == PPE_SEED_RECEIPT_PINS["quarantine"]["receipt_sha256"]
        and provenance_plan.get("required_blockers")
        == provenance_receipt.get("blockers")
        and isinstance(provenance_receipt.get("created_at"), str)
        and provenance_receipt["created_at"] < assessment["created_at"]
    )
    return assessment_valid


def _ppe_seed_readiness(reader: ArtifactReader) -> dict[str, Any]:
    """Project the exact checked-in PPE seed and receipt evidence chain.

    The manifest and all three schemas are exact workspace pins and are read
    through component-by-component no-follow descriptors.  The acquisition and
    authoritative R2 quarantine receipts are independently exact-pinned,
    schema-replayed, self-hash-recomputed and lineage-bound.  The reader never
    scans mutable result or raw-data directories, and receipt-embedded hashes
    are not trust anchors on their own.
    """

    manifest_read, manifest = _workspace_pin_json(
        reader,
        PPE_SEED_MANIFEST_PIN,
        expected_path=PPE_SEED_MANIFEST_PIN["path"],
        maximum_bytes=PPE_SEED_MAX_JSON_BYTES,
    )
    schema_reads: dict[str, WorkspacePinRead] = {}
    schemas: dict[str, dict[str, Any] | None] = {}
    for key, pin in PPE_SEED_SCHEMA_PINS.items():
        schema_read, schema = _workspace_pin_json(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=PPE_SEED_MAX_JSON_BYTES,
        )
        schema_reads[key] = schema_read
        schemas[key] = schema

    integrity = {
        "source_manifest_fingerprint_verified": manifest_read.available,
        "source_manifest_schema_pin_verified": schema_reads["manifest"].available,
        "acquisition_receipt_schema_pin_verified": schema_reads[
            "acquisition_receipt"
        ].available,
        "quarantine_receipt_schema_pin_verified": schema_reads[
            "quarantine_receipt"
        ].available,
        "source_manifest_schema_replay_verified": False,
        "source_manifest_semantic_contract_verified": False,
        "acquisition_receipt_file_pin_verified": False,
        "acquisition_receipt_schema_replay_verified": False,
        "acquisition_receipt_self_hash_verified": False,
        "quarantine_receipt_file_pin_verified": False,
        "quarantine_receipt_schema_replay_verified": False,
        "quarantine_receipt_self_hash_verified": False,
        "receipt_lineage_verified": False,
        "provenance_plan_file_pin_verified": False,
        "provenance_receipt_file_pin_verified": False,
        "provenance_code_pin_verified": False,
        "provenance_plan_schema_pin_verified": False,
        "provenance_receipt_schema_pin_verified": False,
        "provenance_schema_contract_verified": False,
        "provenance_plan_schema_replay_verified": False,
        "provenance_receipt_schema_replay_verified": False,
        "provenance_receipt_self_hash_verified": False,
        "provenance_semantic_lineage_verified": False,
        "normalization_r1_plan_history_pin_verified": False,
        "normalization_r1_assessment_history_pin_verified": False,
        "normalization_r1_assessment_self_hash_verified": False,
        "normalization_plan_file_pin_verified": False,
        "normalization_assessment_file_pin_verified": False,
        "normalization_plan_schema_pin_verified": False,
        "normalization_assessment_schema_pin_verified": False,
        "normalization_canonical_schema_pin_verified": False,
        "normalization_schema_contract_verified": False,
        "normalization_plan_schema_replay_verified": False,
        "normalization_assessment_schema_replay_verified": False,
        "normalization_assessment_self_hash_verified": False,
        "normalization_semantic_lineage_verified": False,
    }
    if manifest is None:
        return _ppe_seed_unavailable(
            f"source_manifest_{manifest_read.state}", integrity=integrity
        )
    for key, schema_read in schema_reads.items():
        if not schema_read.available or schemas[key] is None:
            return _ppe_seed_unavailable(
                f"{key}_schema_{schema_read.state}", integrity=integrity
            )

    manifest_schema = schemas["manifest"]
    acquisition_schema = schemas["acquisition_receipt"]
    quarantine_schema = schemas["quarantine_receipt"]
    assert manifest_schema is not None
    assert acquisition_schema is not None
    assert quarantine_schema is not None
    schema_contracts_valid = bool(
        _ppe_seed_schema_contract_valid(
            manifest_schema,
            schema_id=(
                "https://deepsafe.local/schemas/"
                "ppe-training-seed-sources-v1.schema.json"
            ),
            schema_version=PPE_SEED_MANIFEST_SCHEMA,
        )
        and _ppe_seed_schema_contract_valid(
            acquisition_schema,
            schema_id=(
                "https://deepsafe.local/schemas/"
                "ppe-seed-acquisition-receipt-v1.schema.json"
            ),
            schema_version="deepsafe.ppe-seed-acquisition-receipt/v1",
            operation="acquire_pinned_seed_asset",
        )
        and _ppe_seed_schema_contract_valid(
            quarantine_schema,
            schema_id=(
                "https://deepsafe.local/schemas/"
                "ppe-seed-quarantine-receipt-v1.schema.json"
            ),
            schema_version="deepsafe.ppe-seed-quarantine-receipt/v1",
            operation="inspect_ppe_seed_zip_without_extraction",
        )
    )
    try:
        _validate_schema_node(manifest, manifest_schema, manifest_schema)
    except (TypeError, ValueError, RecursionError):
        schema_contracts_valid = False
    else:
        integrity["source_manifest_schema_replay_verified"] = bool(
            schema_contracts_valid
        )
    if not schema_contracts_valid:
        return _ppe_seed_unavailable(
            "ppe_seed_schema_contract_invalid", integrity=integrity
        )

    policy = manifest.get("policy")
    sources = manifest.get("sources")
    if not isinstance(policy, dict) or not isinstance(sources, list):
        return _ppe_seed_unavailable(
            "ppe_seed_manifest_contract_invalid", integrity=integrity
        )
    source_by_id = {
        source.get("id"): source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    }
    if len(source_by_id) != len(sources):
        return _ppe_seed_unavailable(
            "ppe_seed_manifest_contract_invalid", integrity=integrity
        )

    pinned_asset_count = 0
    pinned_archive_count = 0
    blockers: set[str] = set()
    source_contracts_valid = True
    for source in sources:
        license_contract = source.get("license", {})
        eligibility = source.get("eligibility", {})
        declared_content = source.get("declared_content", {})
        source_contracts_valid = bool(
            source_contracts_valid
            and license_contract.get("spdx") == "CC-BY-4.0"
            and license_contract.get("repository_metadata_verified") is True
            and license_contract.get("embedded_third_party_audit_complete")
            is False
            and eligibility.get("download") is True
            and eligibility.get("quarantine_inspection") is True
            and eligibility.get("training") is False
            and eligibility.get("final_validation_or_test") is False
            and isinstance(eligibility.get("blockers"), list)
            and bool(eligibility.get("blockers"))
            and declared_content.get("video_ground_truth") is False
            and declared_content.get("track_ground_truth") is False
            and declared_content.get("distance_ground_truth") is False
        )
        blockers.update(
            item
            for item in eligibility.get("blockers", [])
            if isinstance(item, str)
        )
        for asset_key in ("artifact", "data_yaml", "archive"):
            asset = source.get(asset_key)
            if asset is None:
                continue
            if not _ppe_seed_pinned_asset(asset):
                source_contracts_valid = False
                continue
            pinned_asset_count += 1
            if asset_key in {"artifact", "archive"}:
                pinned_archive_count += 1

    primary = source_by_id.get("mendeley-ppe-v6-20250731", {})
    secondary = source_by_id.get("mendeley-ppe-five-class-v1", {})
    semantic_valid = bool(
        manifest.get("schema_version") == PPE_SEED_MANIFEST_SCHEMA
        and manifest.get("status")
        == "acquisition_planned_not_training_approved"
        and manifest.get("research_cutoff") == "2026-07-17"
        and policy.get("required_classes") == list(PPE_SEED_REQUIRED_CLASSES)
        and policy.get("split_rule")
        == "site_camera_session_and_provenance_group_disjoint"
        and policy.get("public_seed_in_final_test") is False
        and tuple(source_by_id) == PPE_SEED_SOURCE_IDS
        and source_contracts_valid
        and pinned_asset_count == 2
        and pinned_archive_count == 1
        and _ppe_seed_pinned_asset(primary.get("artifact"))
        and primary.get("artifact", {}).get("bytes") == 236065015
        and primary.get("artifact", {}).get("sha256")
        == "7a22e5cbc0327971f56358e14f1fce88fec1d6ab30cfac642de0746202ae1f0d"
        and primary.get("declared_content", {}).get("images") == 2286
        and primary.get("declared_content", {}).get("classes")
        == ["Helmet", "NoHelmet", "NoVest", "Vest"]
        and secondary.get("archive") is None
        and _ppe_seed_pinned_asset(secondary.get("data_yaml"))
        and secondary.get("data_yaml", {}).get("bytes") == 336
        and secondary.get("data_yaml", {}).get("sha256")
        == "321f49d674e08a84283da1a52a9f288c31b7c40773955c4f75efe8cd71258cf1"
        and secondary.get("declared_content", {}).get("classes")
        == ["helmet", "no_helmet", "no_vest", "person", "vest"]
        and "download_all_archive_size_and_sha256_not_yet_pinned" in blockers
        and "embedded_third_party_rights_audit_incomplete" in blockers
    )
    if not semantic_valid:
        return _ppe_seed_unavailable(
            "ppe_seed_manifest_contract_invalid", integrity=integrity
        )
    integrity["source_manifest_semantic_contract_verified"] = True

    receipt_values: dict[str, dict[str, Any] | None] = {}
    receipt_reads: dict[str, WorkspacePinRead] = {}
    for key, trusted_pin in PPE_SEED_RECEIPT_PINS.items():
        pin = _person_pin_core(trusted_pin)
        assert pin is not None
        receipt_read, receipt = _workspace_pin_json(
            reader,
            pin,
            expected_path=trusted_pin["path"],
            maximum_bytes=PPE_SEED_MAX_RECEIPT_BYTES,
        )
        receipt_reads[key] = receipt_read
        receipt_values[key] = receipt
        integrity[f"{key}_receipt_file_pin_verified"] = (
            receipt_read.available
        )
    if any(not item.available for item in receipt_reads.values()):
        failed_key, failed_read = next(
            (key, value)
            for key, value in receipt_reads.items()
            if not value.available
        )
        return _ppe_seed_unavailable(
            f"{failed_key}_receipt_{failed_read.state}",
            integrity=integrity,
        )

    acquisition = receipt_values["acquisition"]
    quarantine = receipt_values["quarantine"]
    if acquisition is None or quarantine is None:
        return _ppe_seed_unavailable(
            "ppe_seed_receipt_json_invalid", integrity=integrity
        )
    try:
        _validate_schema_node(acquisition, acquisition_schema, acquisition_schema)
    except (TypeError, ValueError, RecursionError):
        acquisition_schema_valid = False
    else:
        acquisition_schema_valid = True
    try:
        _validate_schema_node(quarantine, quarantine_schema, quarantine_schema)
    except (TypeError, ValueError, RecursionError):
        quarantine_schema_valid = False
    else:
        quarantine_schema_valid = True
    integrity["acquisition_receipt_schema_replay_verified"] = (
        acquisition_schema_valid
    )
    integrity["quarantine_receipt_schema_replay_verified"] = (
        quarantine_schema_valid
    )
    acquisition_self_hash_valid = _ppe_receipt_self_hash_matches(
        acquisition,
        expected=PPE_SEED_RECEIPT_PINS["acquisition"]["receipt_sha256"],
    )
    quarantine_self_hash_valid = _ppe_receipt_self_hash_matches(
        quarantine,
        expected=PPE_SEED_RECEIPT_PINS["quarantine"]["receipt_sha256"],
    )
    integrity["acquisition_receipt_self_hash_verified"] = (
        acquisition_self_hash_valid
    )
    integrity["quarantine_receipt_self_hash_verified"] = (
        quarantine_self_hash_valid
    )
    if not (
        acquisition_schema_valid
        and quarantine_schema_valid
        and acquisition_self_hash_valid
        and quarantine_self_hash_valid
    ):
        return _ppe_seed_unavailable(
            "ppe_seed_receipt_contract_invalid", integrity=integrity
        )

    primary_artifact = primary.get("artifact", {})
    expected_asset = {
        "bytes": primary_artifact.get("bytes"),
        "sha256": primary_artifact.get("sha256"),
    }
    acquisition_manifest = acquisition.get("manifest")
    acquisition_request = acquisition.get("request")
    acquisition_observed = acquisition.get("observed")
    quarantine_manifest = quarantine.get("manifest")
    quarantine_archive = quarantine.get("archive")
    if not all(
        isinstance(item, dict)
        for item in (
            acquisition_manifest,
            acquisition_request,
            acquisition_observed,
            quarantine_manifest,
            quarantine_archive,
        )
    ):
        return _ppe_seed_unavailable(
            "ppe_seed_receipt_shape_invalid", integrity=integrity
        )
    assert isinstance(acquisition_manifest, dict)
    assert isinstance(acquisition_request, dict)
    assert isinstance(acquisition_observed, dict)
    assert isinstance(quarantine_manifest, dict)
    assert isinstance(quarantine_archive, dict)

    gate_rows = quarantine.get("gates")
    gate_by_id = {
        row.get("id"): row
        for row in gate_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    } if isinstance(gate_rows, list) else {}
    required_gate_states = {
        "pinned_archive_identity": True,
        "zip_readable": True,
        "safe_member_paths": True,
        "no_links_or_special_files": True,
        "no_encrypted_members": True,
        "image_payloads_decodable": True,
        "all_member_payloads_readable": True,
        "declared_image_dimensions_match": False,
        "valid_yolo_yaml": False,
        "yolo_split_paths_resolve_in_archive": False,
        "declared_classes_match_yaml": False,
        "valid_yolo_detection_labels": False,
        "image_label_pairing": True,
        "declared_image_counts_match": True,
    }
    gate_semantics_valid = bool(
        len(gate_by_id) == len(gate_rows or [])
        and all(
            isinstance(gate_by_id.get(gate_id), dict)
            and gate_by_id[gate_id].get("passed") is expected
            for gate_id, expected in required_gate_states.items()
        )
    )
    yaml_gate = gate_by_id.get("valid_yolo_yaml", {})
    yaml_details = (
        yaml_gate.get("details")
        if isinstance(yaml_gate.get("details"), dict)
        else {}
    )
    yaml_errors = (
        yaml_details.get("errors")
        if isinstance(yaml_details.get("errors"), dict)
        else {}
    )
    yaml_error_items = yaml_errors.get("items")
    yaml_path_failure_bound = bool(
        isinstance(yaml_error_items, list)
        and len(yaml_error_items) == 1
        and isinstance(yaml_error_items[0], dict)
        and "guvensiz archive-ici yol" in str(yaml_error_items[0].get("error"))
        and "C:/Users/" in str(yaml_error_items[0].get("error"))
    )
    image_gate_details = gate_by_id.get(
        "image_payloads_decodable", {}
    ).get("details", {})
    pairing_gate_details = gate_by_id.get("image_label_pairing", {}).get(
        "details", {}
    )
    label_gate_details = gate_by_id.get(
        "valid_yolo_detection_labels", {}
    ).get("details", {})
    dimension_gate_details = gate_by_id.get(
        "declared_image_dimensions_match", {}
    ).get("details", {})
    observed_dimensions = (
        dimension_gate_details.get("observed_dimensions")
        if isinstance(dimension_gate_details, dict)
        else None
    )
    dimension_mismatches = (
        dimension_gate_details.get("mismatches")
        if isinstance(dimension_gate_details, dict)
        else None
    )
    exact_declared_dimension_count = (
        sum(
            row.get("count", 0)
            for row in observed_dimensions
            if isinstance(row, dict)
            and row.get("width") == 640
            and row.get("height") == 640
            and isinstance(row.get("count"), int)
            and not isinstance(row.get("count"), bool)
        )
        if isinstance(observed_dimensions, list)
        else None
    )
    yolo_summary = quarantine.get("yolo")
    inventory = quarantine.get("inventory")
    failure = quarantine.get("failure")
    if not isinstance(yolo_summary, dict):
        yolo_summary = {}
    if not isinstance(inventory, dict):
        inventory = {}
    if not isinstance(failure, dict):
        failure = {}

    acquisition_semantic_valid = bool(
        acquisition.get("schema_version")
        == "deepsafe.ppe-seed-acquisition-receipt/v1"
        and acquisition.get("operation") == "acquire_pinned_seed_asset"
        and acquisition.get("source_id") == "mendeley-ppe-v6-20250731"
        and acquisition.get("asset_key") == "artifact"
        and acquisition_manifest
        == {
            "path": PPE_SEED_MANIFEST_PIN["path"],
            "sha256": PPE_SEED_MANIFEST_PIN["sha256"],
        }
        and acquisition_request.get("expected_bytes")
        == expected_asset["bytes"]
        and acquisition_request.get("expected_sha256")
        == expected_asset["sha256"]
        and acquisition_observed
        == {
            "bytes": expected_asset["bytes"],
            "sha256": expected_asset["sha256"],
            "payload_kind": "zip",
        }
        and acquisition.get("accepted") is True
        and isinstance(acquisition.get("published_path"), str)
        and acquisition.get("published_path", "").endswith(
            "/data/raw/ppe/mendeley-ppe-v6/20250731-PPE2286y.zip"
        )
        and acquisition.get("training_eligible") is False
        and acquisition.get("failure") is None
    )
    quarantine_semantic_valid = bool(
        quarantine.get("schema_version")
        == "deepsafe.ppe-seed-quarantine-receipt/v1"
        and quarantine.get("operation")
        == "inspect_ppe_seed_zip_without_extraction"
        and quarantine.get("source_id") == "mendeley-ppe-v6-20250731"
        and quarantine.get("asset_key") == "artifact"
        and quarantine_manifest == acquisition_manifest
        and quarantine_archive.get("bytes") == expected_asset["bytes"]
        and quarantine_archive.get("sha256") == expected_asset["sha256"]
        and quarantine.get("structural_pass") is False
        and quarantine.get("accepted_to_quarantine") is False
        and quarantine.get("training_eligible") is False
        and failure.get("code") == "structural_gates_failed"
        and failure.get("details", {}).get("failed_gates")
        == [
            "declared_image_dimensions_match",
            "valid_yolo_yaml",
            "yolo_split_paths_resolve_in_archive",
            "declared_classes_match_yaml",
            "valid_yolo_detection_labels",
        ]
        and gate_semantics_valid
        and yaml_path_failure_bound
        and inventory.get("entry_count") == 4583
        and yolo_summary.get("decoded_image_count") == 2286
        and yolo_summary.get("image_count") == 2286
        and yolo_summary.get("label_file_count") == 2286
        and yolo_summary.get("label_row_count") == 6038
        and isinstance(image_gate_details, dict)
        and image_gate_details.get("decoded_count") == 2286
        and image_gate_details.get("image_count") == 2286
        and isinstance(pairing_gate_details, dict)
        and pairing_gate_details.get("image_count") == 2286
        and pairing_gate_details.get("label_count") == 2286
        and isinstance(label_gate_details, dict)
        and label_gate_details.get("label_files") == 2286
        and label_gate_details.get("rows") == 6038
        and isinstance(dimension_gate_details, dict)
        and dimension_gate_details.get("declared")
        == {"height": 640, "width": 640}
        and dimension_gate_details.get("decoded_count") == 2286
        and isinstance(dimension_mismatches, dict)
        and dimension_mismatches.get("count") == 1814
        and isinstance(observed_dimensions, list)
        and len(observed_dimensions) == 905
        and sum(
            row.get("count", 0)
            for row in observed_dimensions
            if isinstance(row, dict)
            and isinstance(row.get("count"), int)
            and not isinstance(row.get("count"), bool)
        )
        == 2286
        and exact_declared_dimension_count == 472
        and "structural_gates_failed"
        in quarantine.get("eligibility_blockers", [])
        and "embedded_third_party_rights_review_required"
        in quarantine.get("eligibility_blockers", [])
        and "camera_group_split_review_required"
        in quarantine.get("eligibility_blockers", [])
    )
    receipt_lineage_valid = bool(
        acquisition_semantic_valid
        and quarantine_semantic_valid
        and acquisition_observed.get("bytes")
        == quarantine_archive.get("bytes")
        and acquisition_observed.get("sha256")
        == quarantine_archive.get("sha256")
        and acquisition.get("published_path") == quarantine_archive.get("path")
        and isinstance(acquisition.get("created_at"), str)
        and isinstance(quarantine.get("created_at"), str)
        and acquisition["created_at"] < quarantine["created_at"]
    )
    integrity["receipt_lineage_verified"] = receipt_lineage_valid
    if not receipt_lineage_valid:
        return _ppe_seed_unavailable(
            "ppe_seed_receipt_lineage_invalid", integrity=integrity
        )

    provenance_plan_read, provenance_plan = _workspace_pin_json(
        reader,
        PPE_PROVENANCE_PLAN_PIN,
        expected_path=PPE_PROVENANCE_PLAN_PIN["path"],
        maximum_bytes=PPE_SEED_MAX_JSON_BYTES,
    )
    provenance_receipt_core_pin = _person_pin_core(
        PPE_PROVENANCE_RECEIPT_PIN
    )
    assert provenance_receipt_core_pin is not None
    provenance_receipt_read, provenance_receipt = _workspace_pin_json(
        reader,
        provenance_receipt_core_pin,
        expected_path=PPE_PROVENANCE_RECEIPT_PIN["path"],
        maximum_bytes=PPE_SEED_MAX_JSON_BYTES,
    )
    provenance_code_read = _read_workspace_pin(
        reader,
        PPE_PROVENANCE_CODE_PIN,
        expected_path=PPE_PROVENANCE_CODE_PIN["path"],
        maximum_bytes=PPE_SEED_MAX_JSON_BYTES,
        collect=False,
    )
    provenance_schema_reads: dict[str, WorkspacePinRead] = {}
    provenance_schemas: dict[str, dict[str, Any] | None] = {}
    for key, pin in PPE_PROVENANCE_SCHEMA_PINS.items():
        schema_read, schema = _workspace_pin_json(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=PPE_SEED_MAX_JSON_BYTES,
        )
        provenance_schema_reads[key] = schema_read
        provenance_schemas[key] = schema
    integrity.update(
        {
            "provenance_plan_file_pin_verified": provenance_plan_read.available,
            "provenance_receipt_file_pin_verified": (
                provenance_receipt_read.available
            ),
            "provenance_code_pin_verified": provenance_code_read.available,
            "provenance_plan_schema_pin_verified": (
                provenance_schema_reads["plan"].available
            ),
            "provenance_receipt_schema_pin_verified": (
                provenance_schema_reads["receipt"].available
            ),
        }
    )
    provenance_reads = {
        "plan": provenance_plan_read,
        "receipt": provenance_receipt_read,
        "code": provenance_code_read,
        "plan_schema": provenance_schema_reads["plan"],
        "receipt_schema": provenance_schema_reads["receipt"],
    }
    for key, read in provenance_reads.items():
        if not read.available:
            return _ppe_seed_unavailable(
                f"provenance_{key}_{read.state}", integrity=integrity
            )
    if (
        provenance_plan is None
        or provenance_receipt is None
        or provenance_schemas["plan"] is None
        or provenance_schemas["receipt"] is None
    ):
        return _ppe_seed_unavailable(
            "provenance_json_invalid", integrity=integrity
        )
    provenance_plan_schema = provenance_schemas["plan"]
    provenance_receipt_schema = provenance_schemas["receipt"]
    assert provenance_plan_schema is not None
    assert provenance_receipt_schema is not None
    provenance_schema_contract_valid = bool(
        _ppe_provenance_schema_contract_valid(
            provenance_plan_schema,
            receipt=False,
        )
        and _ppe_provenance_schema_contract_valid(
            provenance_receipt_schema,
            receipt=True,
        )
    )
    integrity["provenance_schema_contract_verified"] = (
        provenance_schema_contract_valid
    )
    try:
        _validate_schema_node(
            provenance_plan,
            provenance_plan_schema,
            provenance_plan_schema,
        )
    except (TypeError, ValueError, RecursionError):
        provenance_plan_schema_valid = False
    else:
        provenance_plan_schema_valid = True
    try:
        _validate_schema_node(
            provenance_receipt,
            provenance_receipt_schema,
            provenance_receipt_schema,
        )
    except (TypeError, ValueError, RecursionError):
        provenance_receipt_schema_valid = False
    else:
        provenance_receipt_schema_valid = True
    integrity["provenance_plan_schema_replay_verified"] = (
        provenance_plan_schema_valid
    )
    integrity["provenance_receipt_schema_replay_verified"] = (
        provenance_receipt_schema_valid
    )
    if not (
        provenance_schema_contract_valid
        and provenance_plan_schema_valid
        and provenance_receipt_schema_valid
    ):
        return _ppe_seed_unavailable(
            "ppe_provenance_schema_contract_invalid", integrity=integrity
        )
    provenance_receipt_self_hash_valid = _ppe_receipt_self_hash_matches(
        provenance_receipt,
        expected=PPE_PROVENANCE_RECEIPT_PIN["receipt_sha256"],
    )
    integrity["provenance_receipt_self_hash_verified"] = (
        provenance_receipt_self_hash_valid
    )
    if not provenance_receipt_self_hash_valid:
        return _ppe_seed_unavailable(
            "ppe_provenance_receipt_invalid", integrity=integrity
        )
    provenance_semantic_lineage_valid = _ppe_provenance_r2_semantics_valid(
        provenance_plan,
        provenance_receipt,
        acquisition=acquisition,
        quarantine=quarantine,
    )
    integrity["provenance_semantic_lineage_verified"] = (
        provenance_semantic_lineage_valid
    )
    if not provenance_semantic_lineage_valid:
        return _ppe_seed_unavailable(
            "ppe_provenance_semantic_lineage_invalid", integrity=integrity
        )

    normalization_r1_plan_read = _read_workspace_pin(
        reader,
        PPE_NORMALIZATION_SUPERSEDED_R1_PINS["plan"],
        expected_path=PPE_NORMALIZATION_SUPERSEDED_R1_PINS["plan"]["path"],
        maximum_bytes=PPE_NORMALIZATION_MAX_JSON_BYTES,
        collect=False,
    )
    normalization_r1_assessment_pin = _person_pin_core(
        PPE_NORMALIZATION_SUPERSEDED_R1_PINS["assessment"]
    )
    assert normalization_r1_assessment_pin is not None
    normalization_r1_assessment_read, normalization_r1_assessment = (
        _workspace_pin_json(
            reader,
            normalization_r1_assessment_pin,
            expected_path=PPE_NORMALIZATION_SUPERSEDED_R1_PINS[
                "assessment"
            ]["path"],
            maximum_bytes=PPE_NORMALIZATION_MAX_JSON_BYTES,
        )
    )
    integrity["normalization_r1_plan_history_pin_verified"] = (
        normalization_r1_plan_read.available
    )
    integrity["normalization_r1_assessment_history_pin_verified"] = (
        normalization_r1_assessment_read.available
    )
    if not normalization_r1_plan_read.available:
        return _ppe_seed_unavailable(
            f"normalization_r1_plan_history_{normalization_r1_plan_read.state}",
            integrity=integrity,
        )
    if (
        not normalization_r1_assessment_read.available
        or normalization_r1_assessment is None
    ):
        return _ppe_seed_unavailable(
            "normalization_r1_assessment_history_"
            f"{normalization_r1_assessment_read.state}",
            integrity=integrity,
        )
    normalization_r1_self_hash_valid = _ppe_receipt_self_hash_matches(
        normalization_r1_assessment,
        expected=PPE_NORMALIZATION_SUPERSEDED_R1_PINS["assessment"][
            "receipt_sha256"
        ],
    )
    integrity["normalization_r1_assessment_self_hash_verified"] = (
        normalization_r1_self_hash_valid
    )
    if not normalization_r1_self_hash_valid:
        return _ppe_seed_unavailable(
            "normalization_r1_assessment_history_invalid",
            integrity=integrity,
        )

    normalization_plan_read, normalization_plan = _workspace_pin_json(
        reader,
        PPE_NORMALIZATION_PLAN_PIN,
        expected_path=PPE_NORMALIZATION_PLAN_PIN["path"],
        maximum_bytes=PPE_NORMALIZATION_MAX_JSON_BYTES,
    )
    normalization_assessment_pin = _person_pin_core(
        PPE_NORMALIZATION_ASSESSMENT_PIN
    )
    assert normalization_assessment_pin is not None
    normalization_assessment_read, normalization_assessment = (
        _workspace_pin_json(
            reader,
            normalization_assessment_pin,
            expected_path=PPE_NORMALIZATION_ASSESSMENT_PIN["path"],
            maximum_bytes=PPE_NORMALIZATION_MAX_JSON_BYTES,
        )
    )
    integrity["normalization_plan_file_pin_verified"] = (
        normalization_plan_read.available
    )
    integrity["normalization_assessment_file_pin_verified"] = (
        normalization_assessment_read.available
    )
    normalization_schema_reads: dict[str, WorkspacePinRead] = {}
    normalization_schemas: dict[str, dict[str, Any] | None] = {}
    for key, pin in PPE_NORMALIZATION_SCHEMA_PINS.items():
        schema_read, schema = _workspace_pin_json(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=PPE_NORMALIZATION_MAX_JSON_BYTES,
        )
        normalization_schema_reads[key] = schema_read
        normalization_schemas[key] = schema
    integrity["normalization_plan_schema_pin_verified"] = (
        normalization_schema_reads["plan"].available
    )
    integrity["normalization_assessment_schema_pin_verified"] = (
        normalization_schema_reads["assessment"].available
    )
    integrity["normalization_canonical_schema_pin_verified"] = (
        normalization_schema_reads["canonical_dataset"].available
    )
    if not normalization_plan_read.available or normalization_plan is None:
        return _ppe_seed_unavailable(
            f"normalization_plan_{normalization_plan_read.state}",
            integrity=integrity,
        )
    if (
        not normalization_assessment_read.available
        or normalization_assessment is None
    ):
        return _ppe_seed_unavailable(
            f"normalization_assessment_{normalization_assessment_read.state}",
            integrity=integrity,
        )
    for key, schema_read in normalization_schema_reads.items():
        if not schema_read.available or normalization_schemas[key] is None:
            return _ppe_seed_unavailable(
                f"normalization_{key}_schema_{schema_read.state}",
                integrity=integrity,
            )

    normalization_plan_schema = normalization_schemas["plan"]
    normalization_assessment_schema = normalization_schemas["assessment"]
    normalization_canonical_schema = normalization_schemas[
        "canonical_dataset"
    ]
    assert normalization_plan_schema is not None
    assert normalization_assessment_schema is not None
    assert normalization_canonical_schema is not None
    normalization_schema_contract_valid = bool(
        _ppe_normalization_schema_contract_valid(
            normalization_plan_schema,
            schema_id=(
                "https://deepsafe.local/schemas/"
                "ppe-normalization-plan-v2.schema.json"
            ),
            schema_version="deepsafe.ppe-normalization-plan/v2",
        )
        and _ppe_normalization_schema_contract_valid(
            normalization_assessment_schema,
            schema_id=(
                "https://deepsafe.local/schemas/"
                "ppe-normalization-assessment-receipt-v2.schema.json"
            ),
            schema_version=(
                "deepsafe.ppe-normalization-assessment-receipt/v2"
            ),
            operation=(
                "assess_history_preserving_ppe_normalization_with_"
                "provenance_r2_replay"
            ),
        )
        and _ppe_normalization_schema_contract_valid(
            normalization_canonical_schema,
            schema_id=(
                "https://deepsafe.local/schemas/"
                "person-equipment-decisions-v2.schema.json"
            ),
            schema_version="ppe-person-equipment-decisions-v2.0",
        )
    )
    integrity["normalization_schema_contract_verified"] = (
        normalization_schema_contract_valid
    )
    try:
        _validate_schema_node(
            normalization_plan,
            normalization_plan_schema,
            normalization_plan_schema,
        )
    except (TypeError, ValueError, RecursionError):
        normalization_plan_schema_valid = False
    else:
        normalization_plan_schema_valid = True
    try:
        _validate_schema_node(
            normalization_assessment,
            normalization_assessment_schema,
            normalization_assessment_schema,
        )
    except (TypeError, ValueError, RecursionError):
        normalization_assessment_schema_valid = False
    else:
        normalization_assessment_schema_valid = True
    integrity["normalization_plan_schema_replay_verified"] = (
        normalization_plan_schema_valid
    )
    integrity["normalization_assessment_schema_replay_verified"] = (
        normalization_assessment_schema_valid
    )
    if not (
        normalization_schema_contract_valid
        and normalization_plan_schema_valid
        and normalization_assessment_schema_valid
    ):
        return _ppe_seed_unavailable(
            "ppe_normalization_schema_contract_invalid",
            integrity=integrity,
        )
    normalization_assessment_self_hash_valid = (
        _ppe_receipt_self_hash_matches(
            normalization_assessment,
            expected=PPE_NORMALIZATION_ASSESSMENT_PIN["receipt_sha256"],
        )
    )
    integrity["normalization_assessment_self_hash_verified"] = (
        normalization_assessment_self_hash_valid
    )
    if not normalization_assessment_self_hash_valid:
        return _ppe_seed_unavailable(
            "ppe_normalization_assessment_receipt_invalid",
            integrity=integrity,
        )
    normalization_semantic_lineage_valid = (
        _ppe_normalization_r2_semantics_valid(
            normalization_plan,
            normalization_assessment,
            provenance_plan=provenance_plan,
            provenance_receipt=provenance_receipt,
            quarantine=quarantine,
        )
    )
    integrity["normalization_semantic_lineage_verified"] = (
        normalization_semantic_lineage_valid
    )
    if not normalization_semantic_lineage_valid:
        return _ppe_seed_unavailable(
            "ppe_normalization_semantic_lineage_invalid",
            integrity=integrity,
        )

    return {
        "label": "PPE veri tohumu hazırlığı",
        "available": True,
        "state": "acquired_quarantine_failed",
        "reason": "quarantine_structural_gates_failed",
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "preparation": {
            "source_manifest_verified": True,
            "receipt_contracts_verified": True,
            "data_acquired": True,
            "quarantine_complete": False,
        },
        "source_contract": {
            "source_count": 2,
            "required_classes": list(PPE_SEED_REQUIRED_CLASSES),
            "fully_pinned_asset_count": 2,
            "fully_pinned_archive_count": 1,
            "declared_license": "CC-BY-4.0",
            "repository_license_metadata_verified_sources": 2,
            "embedded_third_party_rights_audited_sources": 0,
            "training_eligible_sources": 0,
            "final_validation_eligible_sources": 0,
            "video_ground_truth_sources": 0,
            "track_ground_truth_sources": 0,
            "distance_ground_truth_sources": 0,
        },
        "receipts": {
            "acquisition": {
                "pin_declared": True,
                "verified": True,
                "accepted": True,
            },
            "quarantine": {
                "pin_declared": True,
                "verified": True,
                "accepted": False,
            },
        },
        "provenance_review": {
            "evidence_verified": True,
            "mechanical_audit_complete": True,
            "images": 2286,
            "train_images": 1829,
            "validation_images": 457,
            "exact_content_duplicate_groups": 0,
            "exact_original_key_duplicate_groups": 0,
            "filename_families": 6,
            "cross_split_filename_families": 6,
            "images_in_cross_split_filename_families": 2286,
            "strict_near_duplicate_candidate_pairs": 7,
            "strict_near_duplicate_validation_members": 4,
            "high_confidence_near_duplicate_candidate_pairs": 2,
            "item_level_source_mapping_complete": False,
            "embedded_rights_review_complete": False,
            "camera_site_session_metadata_present": False,
            "human_near_duplicate_review_complete": False,
            "training_eligible": False,
        },
        "normalization": {
            "evidence_verified": True,
            "provenance_review_evidence_present": True,
            "provenance_mechanical_audit_replayed": True,
            "provenance_review_approved": False,
            "embedded_rights_review_approved": False,
            "camera_group_split_approved": False,
            "normalization_ready": False,
            "source_training_eligible": False,
            "normalized_training_eligible": False,
            "independent_bbox_out_of_range_count": 52,
            "bbox_overflow_severity": {
                "classification": "small_but_blocking",
                "minimum": 0.0000031249999998816946,
                "median": 0.000007812500000037303,
                "p95": 0.000020833333333358794,
                "maximum": 0.00003597122302156919,
            },
        },
        "quarantine_review": {
            "archive_entries": 4583,
            "decoded_images": 2286,
            "paired_images": 2286,
            "label_rows": 6038,
            "declared_image_width": 640,
            "declared_image_height": 640,
            "exact_declared_dimension_images": 472,
            "dimension_mismatch_images": 1814,
            "distinct_observed_dimensions": 905,
            "failed_structural_gate_count": 5,
            "failed_gate_ids": [
                "declared_image_dimensions_match",
                "valid_yolo_yaml",
                "yolo_split_paths_resolve_in_archive",
                "declared_classes_match_yaml",
                "valid_yolo_detection_labels",
            ],
            "absolute_windows_yaml_path_rejected": True,
            "independent_bbox_out_of_range_count": 52,
        },
        "gates": {
            "source_contract_verified": True,
            "acquired": True,
            "quarantined": False,
            "rights_audit_complete": False,
            "camera_group_split_audit_complete": False,
            "training_eligible": False,
            "training_complete": False,
            "export_complete": False,
            "deepstream9_evaluated": False,
            "ground_truth_quality_passed": False,
            "twelve_camera_640_passed": False,
            "twelve_camera_960_passed": False,
            "acceptance_passed": False,
            "production_ready": False,
        },
        "integrity": integrity,
        "caveats": [
            "Exact-pinli acquisition receipt edinimi doğrular; training_eligible yine false'dur.",
            "R2 karantina receipt'i; 1814 boyut uyuşmazlığı, archive-içi mutlak Windows data.yaml yolu ve YOLO label kapıları nedeniyle reddedildi.",
            "Yetkili provenance R2 mekanik denetimi 2286 görüntünün tamamını çapraz-split filename ailelerinde buldu; 7 yakın-kopya adayı ve 2 yüksek güvenli aday insan incelemesi bekliyor.",
            "Yetkili normalization R2, provenance R2'yi birebir replay eder ve 52 küçük fakat bloke edici bbox taşmasını yeniden üretir; R1 yalnız değişmez tarihsel öncüldür.",
            "Öğe-seviyesi kaynak/hak ve kamera-site-session grup onayları tamamlanmadı; eğitim, export ve DeepStream 9 kanıtı yoktur.",
            "Statik seed kaynakları final video, track, zamanlama, mesafe veya ürün kabul ground truth'u değildir.",
        ],
        "evidence": [],
    }


def _ppe_five_class_gates() -> dict[str, bool]:
    return {key: False for key in PPE_FIVE_CLASS_GATE_KEYS}


def _ppe_five_class_unavailable(
    reason: str,
    *,
    integrity: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "label": "PPE 5-Class R2 karantina",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "dataset": {},
        "source_receipt": {
            "stream_pin_verified": False,
            "parsed_by_admin": False,
        },
        "quarantine": {
            "structural_pass": False,
            "accepted": False,
            "training_eligible": False,
        },
        "normalization_group_split": {
            "current": False,
            "mechanical_group_split_complete": False,
            "training_eligible": False,
            "final_validation_or_test_eligible": False,
        },
        "quarantine_history": {
            "r1_stream_pin_verified": False,
            "r1_superseded": True,
            "r2_stream_pin_verified": False,
            "r2_authoritative": True,
        },
        "gates": _ppe_five_class_gates(),
        "integrity": integrity or {},
        "caveats": [
            "PPE 5-Class R2 exact-pin veya compact receipt zinciri doğrulanamadı; karantina ve bütün ürün kapıları kapalıdır.",
        ],
        "evidence": [],
    }


def _ppe_five_class_manifest_semantics_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        return False
    source = sources[0]
    if not isinstance(source, dict):
        return False
    archive = source.get("archive", {})
    declared = source.get("declared_content", {})
    license_value = source.get("license", {})
    eligibility = source.get("eligibility", {})
    history = value.get("policy", {}).get("history", {})
    return bool(
        value.get("schema_version")
        == "deepsafe.ppe-training-seed-sources/v1"
        and value.get("status")
        == "acquisition_planned_not_training_approved"
        and source.get("id") == "mendeley-ppe-five-class-v1"
        and source.get("doi") == "10.17632/8vf7z6v5sb.1"
        and license_value.get("spdx") == "CC-BY-4.0"
        and license_value.get("repository_metadata_verified") is True
        and license_value.get("embedded_third_party_audit_complete") is False
        and archive.get("bytes") == 208799718
        and archive.get("sha256")
        == "bf9af5cefc9a35e5fa6158b0d72789c13c1b4fcb564e223d3ced02f8f41f6e26"
        and declared.get("train_images") == 2069
        and declared.get("validation_images") == 517
        and declared.get("published_test_directory_present") is False
        and declared.get("classes")
        == ["helmet", "no_helmet", "no_vest", "person", "vest"]
        and eligibility.get("download") is True
        and eligibility.get("quarantine_inspection") is True
        and eligibility.get("training") is False
        and eligibility.get("final_validation_or_test") is False
        and history.get("authoritative_archive_sha256")
        == "bf9af5cefc9a35e5fa6158b0d72789c13c1b4fcb564e223d3ced02f8f41f6e26"
        and history.get("supersedes_manifest_sha256")
        == "e65eeb283bc765525b270a41f25dcf3667750833e6eabfc51ec014098054a5cc"
        and history.get("r1_receipt_sha256")
        == PPE_FIVE_CLASS_ADMIN_PINS["superseded_r1_receipt"]["sha256"]
    )


def _ppe_five_class_compact_semantics_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    inputs = value.get("inputs", {})
    verification = value.get("verification", {})
    dataset = value.get("dataset", {})
    quarantine = value.get("quarantine", {})
    expected_blockers = [
        "provenance_review_required",
        "embedded_third_party_rights_review_required",
        "camera_group_split_review_required",
        "upstream_roboflow_provenance_and_duplicates_not_audited",
        "embedded_third_party_rights_audit_incomplete",
        "camera_site_session_grouping_not_verified",
        "person_equipment_semantics_not_normalized",
        "published_test_split_missing",
        "exact_duplicate_image_review_required",
    ]
    expected_source_pin = PPE_FIVE_CLASS_ADMIN_PINS["authoritative_receipt"]
    return bool(
        value.get("schema_version")
        == "deepsafe.ppe-five-class-admin-projection-receipt/v1"
        and value.get("status")
        == "quarantine_structural_pass_training_blocked"
        and value.get("source_id") == "mendeley-ppe-five-class-v1"
        and isinstance(value.get("created_at"), str)
        and _external_receipt_self_hash_matches(
            value,
            expected=PPE_FIVE_CLASS_ADMIN_PINS["projection_receipt"][
                "receipt_sha256"
            ],
        )
        and _person_pin_core(inputs.get("manifest"))
        == PPE_FIVE_CLASS_ADMIN_PINS["manifest"]
        and inputs.get("authoritative_quarantine_receipt")
        == expected_source_pin
        and inputs.get("archive")
        == {
            "path": (
                "data/raw/ppe/mendeley-ppe-five-class-v1/"
                "8vf7z6v5sb-1.zip"
            ),
            "bytes": 208799718,
            "sha256": (
                "bf9af5cefc9a35e5fa6158b0d72789c13c1b4fcb564e223d3ced02f8f41f6e26"
            ),
        }
        and _person_pin_core(inputs.get("validator"))
        == PPE_FIVE_CLASS_ADMIN_PINS["validator"]
        and _person_pin_core(inputs.get("schema"))
        == PPE_FIVE_CLASS_ADMIN_PINS["schema"]
        and verification
        == {
            "manifest_exact_pin_verified": True,
            "manifest_semantics_verified": True,
            "authoritative_receipt_exact_pin_verified": True,
            "authoritative_receipt_bounded_offline_parse_verified": True,
            "authoritative_receipt_canonical_self_hash_verified": True,
            "archive_exact_pin_verified": True,
            "admin_authoritative_receipt_policy": (
                "stream_hash_stat_only_no_json_parse"
            ),
            "network_executed": False,
            "gpu_executed": False,
            "docker_executed": False,
            "training_executed": False,
        }
        and dataset
        == {
            "title": "Mendeley PPE Detection Dataset (5-Class)",
            "doi": "10.17632/8vf7z6v5sb.1",
            "repository_license": "CC-BY-4.0",
            "repository_metadata_verified": True,
            "embedded_third_party_rights_audit_complete": False,
            "published_independent_test_split_present": False,
        }
        and quarantine
        == {
            "structural_pass": True,
            "accepted_to_quarantine": True,
            "training_eligible": False,
            "images": 2586,
            "decoded_images": 2586,
            "label_files": 2586,
            "bounding_boxes": 17827,
            "splits": {"train": 2069, "validation": 517},
            "class_names": [
                "helmet",
                "no_helmet",
                "no_vest",
                "person",
                "vest",
            ],
            "class_bbox_counts": {
                "helmet": 5036,
                "no_helmet": 1026,
                "no_vest": 3116,
                "person": 5955,
                "vest": 2694,
            },
            "exact_duplicate_groups": 31,
            "cross_split_exact_duplicate_groups": 10,
            "structural_gate_count": 24,
            "structural_gate_pass_count": 24,
        }
        and value.get("eligibility_blockers") == expected_blockers
        and value.get("gates") == _ppe_five_class_gates()
    )


def _ppe_five_class_normalization_blockers() -> list[str]:
    return [
        "provenance_review_required",
        "embedded_third_party_rights_review_required",
        "camera_group_split_review_required",
        "upstream_roboflow_provenance_and_duplicates_not_audited",
        "embedded_third_party_rights_audit_incomplete",
        "camera_site_session_grouping_not_verified",
        "person_equipment_semantics_not_normalized",
        "published_test_split_missing",
        "exact_duplicate_image_review_required",
        "heuristic_capture_groups_not_camera_verified",
        "vest_to_hi_vis_semantic_review_required",
        "no_vest_to_no_hi_vis_visibility_review_required",
        "person_equipment_association_review_required",
        "independent_final_test_missing",
        "exact_duplicate_annotation_adjudication_required",
    ]


def _ppe_five_class_normalization_plan_semantics_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    expected_archive = {
        "path": (
            "data/raw/ppe/mendeley-ppe-five-class-v1/"
            "8vf7z6v5sb-1.zip"
        ),
        "bytes": 208799718,
        "sha256": (
            "bf9af5cefc9a35e5fa6158b0d72789c13c1b4fcb564e223d3ced02f8f41f6e26"
        ),
    }
    inputs = value.get("inputs", {})
    expected_observations = {
        "images": 2586,
        "label_rows": 17827,
        "source_class_bbox_counts": {
            "helmet": 5036,
            "no_helmet": 1026,
            "no_vest": 3116,
            "person": 5955,
            "vest": 2694,
        },
        "exact_duplicate_groups": 31,
        "cross_upstream_split_exact_duplicate_groups": 10,
    }
    return bool(
        value.get("schema_version")
        == "deepsafe.ppe-five-class-normalization-plan/v1"
        and value.get("plan_id")
        == "mendeley-ppe-five-class-v1-normalization-group-split-r2"
        and value.get("source_id") == "mendeley-ppe-five-class-v1"
        and value.get("operation")
        == "plan_dry_run_group_safe_normalization_without_extraction"
        and inputs.get("archive") == expected_archive
        and _person_pin_core(inputs.get("acquisition_manifest"))
        == PPE_FIVE_CLASS_ADMIN_PINS["manifest"]
        and inputs.get("quarantine_receipt")
        == PPE_FIVE_CLASS_ADMIN_PINS["authoritative_receipt"]
        and value.get("expected_source_observations")
        == expected_observations
        and value.get("grouping")
        == {
            "algorithm": (
                "roboflow_original_stem_numeric_window_plus_exact_sha_union_v1"
            ),
            "numeric_window_size": 32,
            "roboflow_suffix_policy": "strip_optional_jpg_dot_rf_dot_32hex",
            "pure_numeric_width_partition": True,
            "exact_duplicate_policy": "union_all_members_before_assignment",
            "duplicate_annotation_policy": (
                "retain_all_grouped_pending_adjudication"
            ),
        }
        and value.get("assignment", {}).get("roles")
        == [
            {
                "id": "train",
                "target_basis_points": 8000,
                "claim": "model_fitting_candidate_training_blocked",
            },
            {
                "id": "calibration",
                "target_basis_points": 1000,
                "claim": (
                    "internal_tensorrt_calibration_candidate_training_blocked"
                ),
            },
            {
                "id": "test",
                "target_basis_points": 1000,
                "claim": (
                    "internal_heldout_audit_only_not_independent_final_test"
                ),
            },
        ]
        and value.get("materialization")
        == {
            "mode": "dry_run_before_materialization",
            "archive_extraction": False,
            "dataset_write": False,
            "overwrite": False,
        }
        and value.get("eligibility_policy", {}).get("required_blockers")
        == _ppe_five_class_normalization_blockers()
        and value.get("eligibility_policy", {}).get("training_eligible")
        is False
        and value.get("eligibility_policy", {}).get(
            "final_validation_or_test_eligible"
        )
        is False
    )


def _ppe_five_class_normalization_group_replay_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        observations = value["source_observations"]
        grouping = value["grouping"]
        assignment = value["assignment"]
        groups = assignment["groups"]
        if groups != sorted(groups, key=lambda item: item["group_id"]):
            return False
        if _canonical_sha256({"groups": groups}) != grouping[
            "assignment_ledger_sha256"
        ]:
            return False

        group_ids: set[str] = set()
        all_paths: list[str] = []
        role_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        roles_by_path: defaultdict[str, set[str]] = defaultdict(set)
        roles_by_capture: defaultdict[str, set[str]] = defaultdict(set)
        roles_by_duplicate: defaultdict[str, set[str]] = defaultdict(set)
        source_totals: Counter[str] = Counter()
        for group in groups:
            group_id = group["group_id"]
            role = group["role"]
            paths = group["member_paths"]
            if (
                group_id in group_ids
                or role not in {"train", "calibration", "test"}
                or paths != sorted(paths)
                or len(paths) != len(set(paths))
                or group["image_count"] != len(paths)
            ):
                return False
            group_ids.add(group_id)
            role_groups[role].append(group)
            source_counts = group["source_class_bbox_counts"]
            expected_projection = {
                "helmet": source_counts["helmet"],
                "no_helmet": source_counts["no_helmet"],
                "hi_vis": source_counts["vest"],
                "no_hi_vis": source_counts["no_vest"],
            }
            if (
                group["canonical_projection_bbox_counts"]
                != expected_projection
                or group["person_association_anchor_bbox_count"]
                != source_counts["person"]
            ):
                return False
            source_totals.update(source_counts)
            for path in paths:
                all_paths.append(path)
                roles_by_path[path].add(role)
            for capture in group["capture_keys"]:
                roles_by_capture[capture].add(role)
            for digest in group["exact_duplicate_group_sha256s"]:
                roles_by_duplicate[digest].add(role)

        expected_role_counts = {
            "train": {
                "group_count": 145,
                "image_count": 2068,
                "source_class_bbox_counts": {
                    "helmet": 4032,
                    "no_helmet": 820,
                    "no_vest": 2496,
                    "person": 4758,
                    "vest": 2156,
                },
            },
            "calibration": {
                "group_count": 75,
                "image_count": 259,
                "source_class_bbox_counts": {
                    "helmet": 502,
                    "no_helmet": 105,
                    "no_vest": 310,
                    "person": 608,
                    "vest": 271,
                },
            },
            "test": {
                "group_count": 72,
                "image_count": 259,
                "source_class_bbox_counts": {
                    "helmet": 502,
                    "no_helmet": 101,
                    "no_vest": 310,
                    "person": 589,
                    "vest": 267,
                },
            },
        }
        for role, expected in expected_role_counts.items():
            selected = role_groups[role]
            replayed_counts: Counter[str] = Counter()
            for group in selected:
                replayed_counts.update(group["source_class_bbox_counts"])
            summary = assignment["roles"][role]
            if (
                len(selected) != expected["group_count"]
                or sum(group["image_count"] for group in selected)
                != expected["image_count"]
                or dict(replayed_counts)
                != expected["source_class_bbox_counts"]
                or summary["group_count"] != expected["group_count"]
                or summary["image_count"] != expected["image_count"]
                or summary["source_class_bbox_counts"]
                != expected["source_class_bbox_counts"]
            ):
                return False

        histogram = {
            str(size): count
            for size, count in sorted(
                Counter(group["image_count"] for group in groups).items()
            )
        }
        leakage = {
            "unique_image_path_coverage": (
                len(all_paths) == len(set(all_paths)) == observations["images"]
            ),
            "image_path_role_leakage_count": sum(
                len(roles) > 1 for roles in roles_by_path.values()
            ),
            "capture_key_role_leakage_count": sum(
                len(roles) > 1 for roles in roles_by_capture.values()
            ),
            "exact_duplicate_role_leakage_count": sum(
                len(roles) > 1 for roles in roles_by_duplicate.values()
            ),
            "group_role_leakage_count": 0,
            "leakage_zero": True,
        }
        if any(
            leakage[key] != 0
            for key in (
                "image_path_role_leakage_count",
                "capture_key_role_leakage_count",
                "exact_duplicate_role_leakage_count",
                "group_role_leakage_count",
            )
        ):
            leakage["leakage_zero"] = False
        return bool(
            len(group_ids) == grouping["final_group_count"] == 292
            and len(all_paths) == observations["images"] == 2586
            and dict(source_totals)
            == observations["source_class_bbox_counts"]
            and len(roles_by_capture) == grouping["base_capture_key_count"]
            == 321
            and len(roles_by_duplicate) == grouping["exact_duplicate_groups"]
            == 31
            and histogram == grouping["group_size_histogram"]
            and max(group["image_count"] for group in groups)
            == grouping["maximum_group_image_count"]
            == 216
            and len(grouping["annotation_row_variant_group_sha256s"])
            == grouping["duplicate_groups_with_annotation_row_variants"]
            == 31
            and len(grouping["class_histogram_conflict_group_sha256s"])
            == grouping["duplicate_groups_with_class_histogram_conflicts"]
            == 2
            and leakage == value["leakage_audit"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ppe_five_class_normalization_receipt_semantics_valid(
    value: Any,
    plan: dict[str, Any],
) -> bool:
    if not isinstance(value, dict):
        return False
    expected_receipt = PPE_FIVE_CLASS_NORMALIZATION_R2_PINS["receipt"]
    expected_observations = {
        "images": 2586,
        "label_rows": 17827,
        "source_class_bbox_counts": {
            "helmet": 5036,
            "no_helmet": 1026,
            "no_vest": 3116,
            "person": 5955,
            "vest": 2694,
        },
        "exact_duplicate_groups": 31,
        "cross_upstream_split_exact_duplicate_groups": 10,
        "upstream_source_split_image_counts": {"train": 2069, "validation": 517},
        "upstream_validation_policy": (
            "legacy_source_membership_only_not_independent_test"
        ),
    }
    expected_execution = {
        "mode": "dry_run_only",
        "archive_extracted": False,
        "dataset_materialized": False,
        "materialized_dataset_files": 0,
        "archive_redownloaded": False,
        "network_executed": False,
        "gpu_executed": False,
        "docker_executed": False,
        "training_executed": False,
        "admin_service_restarted": False,
    }
    expected_readiness = {
        "mechanical_group_split_complete": True,
        "rights_approved": False,
        "embedded_third_party_rights_audit_complete": False,
        "camera_site_session_grouping_verified": False,
        "canonical_person_equipment_semantics_approved": False,
        "published_independent_test_available": False,
        "normalized_dataset_training_eligible": False,
        "final_validation_or_test_eligible": False,
        "production_ready": False,
    }
    try:
        canonical = value["canonical_contract"]
        mapping = canonical["mapping"]
        mechanical = value["mechanical_gates"]
        receipt_plan = value["plan"]
        inputs = value["inputs"]
        return bool(
            value.get("schema_version")
            == "deepsafe.ppe-five-class-normalization-dry-run-receipt/v1"
            and value.get("operation")
            == (
                "dry_run_group_safe_ppe_five_class_normalization_without_extraction"
            )
            and value.get("status")
            == "dry_run_group_split_complete_training_blocked"
            and value.get("source_id") == "mendeley-ppe-five-class-v1"
            and _external_receipt_self_hash_matches(
                value, expected=expected_receipt["receipt_sha256"]
            )
            and receipt_plan
            == {
                "path": PPE_FIVE_CLASS_NORMALIZATION_R2_PINS["plan"]["path"],
                "bytes": PPE_FIVE_CLASS_NORMALIZATION_R2_PINS["plan"]["bytes"],
                "sha256": PPE_FIVE_CLASS_NORMALIZATION_R2_PINS["plan"]["sha256"],
                "expected_sha256": PPE_FIVE_CLASS_NORMALIZATION_R2_PINS[
                    "plan"
                ]["sha256"],
                "external_pin_verified": True,
                "plan_id": plan["plan_id"],
            }
            and _person_pin_core(inputs.get("acquisition_manifest"))
            == PPE_FIVE_CLASS_ADMIN_PINS["manifest"]
            and inputs.get("quarantine_receipt")
            == PPE_FIVE_CLASS_ADMIN_PINS["authoritative_receipt"]
            and _person_pin_core(inputs.get("implementation"))
            == PPE_FIVE_CLASS_NORMALIZATION_R2_PINS["implementation"]
            and _person_pin_core(inputs.get("plan_schema"))
            == PPE_FIVE_CLASS_NORMALIZATION_R2_PINS["plan_schema"]
            and _person_pin_core(inputs.get("receipt_schema"))
            == PPE_FIVE_CLASS_NORMALIZATION_R2_PINS["receipt_schema"]
            and value.get("execution") == expected_execution
            and value.get("source_observations") == expected_observations
            and canonical.get("decision_class_order")
            == ["helmet", "no_helmet", "hi_vis", "no_hi_vis"]
            and [item.get("source_name") for item in mapping]
            == ["helmet", "no_helmet", "no_vest", "person", "vest"]
            and [item.get("canonical_id") for item in mapping]
            == [0, 1, 3, None, 2]
            and canonical.get("mapping_is_training_ready") is False
            and canonical.get(
                "person_is_association_anchor_not_decision_class"
            )
            is True
            and canonical.get("absence_is_not_inferred_from_missing_detection")
            is True
            and set(mechanical)
            == {
                "input_exact_pins_verified",
                "quarantine_self_hash_verified",
                "zip_inventory_replayed",
                "label_hash_ledger_replayed",
                "source_observations_match_plan",
                "all_exact_duplicates_single_group",
                "all_capture_keys_single_group",
                "all_groups_single_role",
                "assignment_balance_tolerance_passed",
                "upstream_validation_not_promoted_to_independent_test",
                "dry_run_no_dataset_materialization",
            }
            and all(item is True for item in mechanical.values())
            and value.get("readiness") == expected_readiness
            and value.get("eligibility_blockers")
            == _ppe_five_class_normalization_blockers()
            and value.get("assignment", {}).get("max_abs_feature_share_error_ppm")
            == 2339
            and value.get("assignment", {}).get("within_balance_tolerance")
            is True
            and _ppe_five_class_normalization_group_replay_valid(value)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ppe_five_class_semantic_r4_semantics_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        summary = value["selection"]["summary"]
        review = value["review"]
        mapping = value["source_to_canonical_mapping"]
        access = value["payload_access"]
        readiness = value["readiness"]
        execution = value["execution"]
        expected = PPE_FIVE_CLASS_SEMANTIC_R4_PINS["receipt"]
        return bool(
            value.get("schema_version")
            == "deepsafe.ppe-five-class-semantic-audit-receipt/v1"
            and value.get("receipt_id")
            == "mendeley-ppe-five-class-v1-semantic-audit-r4"
            and value.get("status")
            == "ai_semantic_audit_complete_human_adjudication_required"
            and _external_receipt_self_hash_matches(
                value, expected=expected["receipt_sha256"]
            )
            and value.get("plan")
            == {
                **PPE_FIVE_CLASS_SEMANTIC_R4_PINS["plan"],
                "external_sha256_verified": (
                    PPE_FIVE_CLASS_SEMANTIC_R4_PINS["plan"]["sha256"]
                ),
            }
            and summary.get("images") == 20
            and summary.get("groups") == 18
            and summary.get("roles") == {"calibration": 6, "train": 14}
            and summary.get("canonical_class_bbox_counts")
            == {
                "helmet": 109,
                "hi_vis": 37,
                "no_helmet": 58,
                "no_hi_vis": 121,
                "person": 163,
            }
            and value["selection"].get("minimum_groups_satisfied") is True
            and value["selection"].get(
                "all_canonical_class_and_size_targets_satisfied"
            )
            is True
            and mapping
            == {
                "mapping": {
                    "helmet": "helmet",
                    "no_helmet": "no_helmet",
                    "no_vest": "no_hi_vis",
                    "person": "person",
                    "vest": "hi_vis",
                },
                "rows_checked": 488,
                "selected_source_labels_opened": 20,
                "source_to_canonical_geometry_exact": True,
            }
            and review.get("records") == 20
            and review.get("review_status") == "ai_reviewed_needs_human_qa"
            and review.get("overall_decisions")
            == {
                "accept_for_development_with_guardrails": 2,
                "questionable_needs_adjudication": 15,
                "reject_from_development_candidate": 3,
            }
            and review.get("issue_code_counts")
            == {
                "association_ambiguity": 5,
                "helmet_semantic_ambiguity": 2,
                "no_vest_no_hi_vis_semantic_risk": 17,
                "occlusion_limits_review": 4,
                "person_box_issue": 1,
                "vest_hi_vis_semantic_risk": 3,
            }
            and review.get("mapping_decisions", {}).get("hi_vis")
            == {"acceptable": 4, "incorrect": 1, "questionable": 2}
            and review.get("mapping_decisions", {}).get("no_hi_vis")
            == {"questionable": 17}
            and access.get("authorized_materialized_labels_opened") == 2327
            and access.get("selected_materialized_images_opened") == 20
            and access.get("selected_source_label_members_opened") == 20
            and access.get("development_holdout_payload_files_opened") == 0
            and execution
            == {
                "admin_rebuilt_or_restarted": False,
                "docker_executed": False,
                "export_executed": False,
                "gpu_executed": False,
                "gpu_queried": False,
                "mode": "cpu_only_semantic_audit_finalize",
                "network_executed": False,
                "overwrite": False,
                "training_executed": False,
            }
            and readiness
            == {
                "ai_review_complete": True,
                "dataset_rights_cleared": False,
                "development_holdout_opened": False,
                "development_semantic_audit_complete": True,
                "human_adjudication_required": True,
                "independent_final_test_available": False,
                "production_ready": False,
                "semantic_mapping_approved": False,
                "training_authorized_by_this_audit": False,
            }
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ppe_yolo11s_semantic_launch_gate_r3_semantics_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        expected = PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS["gate"]
        unsigned = dict(value)
        observed_fingerprint = unsigned.pop("fingerprint_sha256", None)
        blocked = {
            "allowed": False,
            "reason": (
                "semantic_r4_human_adjudication_and_new_authorization_required"
            ),
        }
        return bool(
            value.get("schema_version")
            == "deepsafe.ppe-yolo11s-semantic-launch-gate/v1"
            and value.get("gate_id")
            == "ppe-yolo11s-r2-semantic-launch-gate-r3"
            and value.get("status")
            == (
                "blocked_pending_human_semantic_adjudication_"
                "and_new_authorization"
            )
            and observed_fingerprint == expected["fingerprint_sha256"]
            and _canonical_sha256(unsigned) == observed_fingerprint
            and value.get("scope")
            == {
                "historical_plan_immutable": True,
                "historical_plan_authorization_not_rewritten": True,
                "current_repository_launch_policy": True,
                "image_build_preparation_may_continue": True,
                "data_or_model_execution_requires_this_gate": True,
            }
            and value.get("inputs", {}).get("historical_r2_execution_plan")
            == {
                "path": (
                    "models/ppe/training-lanes/"
                    "yolo11s-mendeley-five-class-internal-eval-r2/"
                    "execution-plan-r2.json"
                ),
                "bytes": 12135,
                "sha256": (
                    "d0d2a0b239c0575e8b7ff46b470b18c9fba5e568a85863a862ae399d45db7a27"
                ),
                "fingerprint_sha256": (
                    "4d2c089624eaf53f8a8b33ef326b20e3a16dc8ee4af56d84fb12577c98a11118"
                ),
            }
            and value.get("inputs", {}).get("semantic_audit_r4_receipt")
            == PPE_FIVE_CLASS_SEMANTIC_R4_PINS["receipt"]
            and value.get("inputs", {}).get("semantic_audit_r4_plan")
            == PPE_FIVE_CLASS_SEMANTIC_R4_PINS["plan"]
            and value.get("inputs", {}).get("gate_schema")
            == PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS["schema"]
            and value.get("inputs", {}).get("gate_verifier")
            == PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS["verifier"]
            and value.get("semantic_evidence")
            == {
                "sample_images": 20,
                "source_groups": 18,
                "bbox_rows_checked": 488,
                "questionable_needs_adjudication": 15,
                "rejected_development_candidates": 3,
                "accepted_with_guardrails": 2,
                "development_holdout_payload_files_opened": 0,
                "critical_findings": [
                    "vest_to_hi_vis_harness_misclassification",
                    "helmet_worn_vs_carried_ambiguous",
                    "no_vest_to_no_hi_vis_unproven",
                ],
            }
            and value.get("launch_policy")
            == {
                "image_build": {
                    "allowed": True,
                    "scope": (
                        "container_preparation_only_no_dataset_or_model_execution"
                    ),
                },
                "smoke_train": blocked,
                "baseline_calibration": blocked,
                "full_train_150e": blocked,
                "resume": blocked,
                "evaluation": blocked,
                "export": blocked,
            }
            and value.get("release_requirements")
            == {
                "human_adjudication_complete": False,
                "semantic_mapping_approved": False,
                "held_vs_worn_helmet_policy_approved": False,
                "training_subset_remediated_or_exclusions_pinned": False,
                "new_authorization_receipt": None,
                "all_satisfied": False,
            }
            and value.get("production_effect")
            == {
                "training_ready": False,
                "production_ready": False,
                "commercially_cleared": False,
                "acceptance_effect": "none_blocking_evidence_only",
            }
            and value.get("execution_history")
            == {
                "docker": False,
                "gpu": False,
                "training": False,
                "evaluation": False,
                "export": False,
            }
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ppe_human_qa_r6_unavailable(
    reason: str, *, integrity: dict[str, bool] | None = None
) -> dict[str, Any]:
    return {
        "evidence_version": "r6",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "packet_prepared": False,
        "samples": None,
        "contact_sheets": None,
        "groups": None,
        "role_samples": {"train": None, "calibration": None},
        "holdout_payload_access": {
            "image_files_opened": None,
            "label_files_opened": None,
        },
        "human_qa_complete": False,
        "permanent_review_record_present": False,
        "permanent_approval_present": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "export_authorized": False,
        "production_ready": False,
        "integrity": integrity or {},
        "caveats": [
            "R6 insan-QA paketi exact-pin zinciri doğrulanamadı; insan onayı, eğitim ve üretim kapıları kapalıdır.",
        ],
    }


def _ppe_human_qa_r6_relative_pin(
    key: str, relative_name: str
) -> dict[str, Any]:
    descriptor = PPE_HUMAN_QA_R6_PINS[key]
    return {
        "path": relative_name,
        "bytes": descriptor["bytes"],
        "sha256": descriptor["sha256"],
    }


def _ppe_human_qa_r6_schema_contract_valid(
    sample_schema: Any, adjudication_schema: Any
) -> bool:
    if not isinstance(sample_schema, dict) or not isinstance(
        adjudication_schema, dict
    ):
        return False
    sample_properties = sample_schema.get("properties")
    adjudication_properties = adjudication_schema.get("properties")
    if not isinstance(sample_properties, dict) or not isinstance(
        adjudication_properties, dict
    ):
        return False
    return bool(
        sample_schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and sample_schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "ppe-human-qa-sample-r6-v1.schema.json"
        )
        and sample_schema.get("type") == "object"
        and sample_schema.get("additionalProperties") is False
        and sample_properties.get("schema_version")
        == {"const": "deepsafe.ppe-human-qa-sample-r6/v1"}
        and sample_properties.get("training_authorization_effect")
        == {"const": "none_human_qa_sample_only"}
        and adjudication_schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and adjudication_schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "ppe-human-qa-adjudication-r6-v1.schema.json"
        )
        and adjudication_schema.get("type") == "object"
        and adjudication_schema.get("additionalProperties") is False
        and adjudication_properties.get("schema_version")
        == {"const": "deepsafe.ppe-human-qa-adjudication-r6/v1"}
        and adjudication_properties.get("training_authorized")
        == {"const": False}
        and adjudication_properties.get("production_ready")
        == {"const": False}
        and adjudication_properties.get("authorization_effect")
        == {
            "const": (
                "none_human_qa_only_new_exact_training_"
                "authorization_required"
            )
        }
    )


def _ppe_human_qa_r6_semantics_valid(plan: Any, receipt: Any) -> bool:
    if not isinstance(plan, dict) or not isinstance(receipt, dict):
        return False
    expected_categories = {
        "candidate_zero_label_context_all": 18,
        "helmet_head_zone_boundary_quarantined_below": 50,
        "helmet_head_zone_boundary_retained": 50,
        "helmet_worn_candidate_random": 200,
        "hi_vis_worn_candidate_random": 200,
        "no_helmet_explicit_random": 100,
        "quarantine_reason_stratified": 100,
    }
    expected_execution = {
        "admin_rebuilt_or_restarted": False,
        "docker_executed": False,
        "evaluation_executed": False,
        "export_executed": False,
        "gpu_executed": False,
        "gpu_queried": False,
        "mode": "cpu_only_visual_packet_materialization",
        "network_executed": False,
        "training_executed": False,
    }
    expected_readiness = {
        "authorization_effect": "none_packet_only",
        "evaluation_authorized": False,
        "export_authorized": False,
        "human_qa_complete": False,
        "new_exact_training_authorization_required_after_human_qa": True,
        "packet_prepared": True,
        "production_ready": False,
        "training_authorized": False,
    }
    plan_pin = PPE_HUMAN_QA_R6_PINS["plan"]
    try:
        plan_inputs = plan["inputs"]
        selection = receipt["selection"]
        payload_access = receipt["payload_access"]
        artifacts = receipt["artifacts"]
        receipt_inputs = receipt["inputs"]
        receipt_input_keys = (
            "candidate_file_ledger",
            "candidate_quarantine",
            "r4_receipt",
            "r5_human_qa_request",
            "r5_receipt",
            "r5_transform_contract",
        )
        return bool(
            plan.get("schema_version")
            == "deepsafe.ppe-human-qa-packet-plan-r6/v1"
            and plan.get("plan_id")
            == "mendeley-ppe-four-class-human-qa-r6"
            and plan.get("status")
            == "human_qa_packet_planned_not_adjudicated"
            and plan.get("sampling", {}).get("expected_total_samples") == 718
            and plan.get("sampling", {}).get("authorized_roles")
            == ["train", "calibration"]
            and plan.get("sampling", {}).get("excluded_role")
            == "development_holdout"
            and plan.get("sampling", {}).get(
                "camera_group_stratification_required"
            )
            is True
            and plan.get("execution_constraints")
            == {
                "admin_rebuild_allowed": False,
                "cpu_only": True,
                "development_holdout_payload_allowed": False,
                "docker_allowed": False,
                "evaluation_allowed": False,
                "export_allowed": False,
                "gpu_allowed": False,
                "gpu_query_allowed": False,
                "network_allowed": False,
                "training_allowed": False,
            }
            and plan.get("readiness")
            == {
                "authorization_effect": "none",
                "human_qa_complete": False,
                "human_qa_packet_only": True,
                "production_ready": False,
                "training_authorized": False,
            }
            and plan_inputs.get("r5_receipt")
            == PPE_FOUR_CLASS_SEMANTIC_R5_PINS["receipt"]
            and plan_inputs.get("r5_transform_contract")
            == PPE_FOUR_CLASS_SEMANTIC_R5_PINS["contract"]
            and plan_inputs.get("r4_receipt")
            == PPE_FIVE_CLASS_SEMANTIC_R4_PINS["receipt"]
            and receipt.get("schema_version")
            == "deepsafe.ppe-human-qa-packet-receipt-r6/v1"
            and receipt.get("receipt_id")
            == "mendeley-ppe-four-class-human-qa-r6"
            and receipt.get("status")
            == "human_qa_packet_prepared_not_adjudicated"
            and receipt.get("plan")
            == {
                **plan_pin,
                "external_sha256_verified": plan_pin["sha256"],
            }
            and set(receipt_inputs) == set(receipt_input_keys)
            and all(
                receipt_inputs.get(key) == plan_inputs.get(key)
                for key in receipt_input_keys
            )
            and selection
            == {
                "all_record_ids_unique": True,
                "all_sample_ids_unique": True,
                "categories": expected_categories,
                "groups": 213,
                "roles": {"calibration": 332, "train": 386},
                "samples": _ppe_human_qa_r6_relative_pin(
                    "samples", "samples.jsonl"
                ),
                "total_samples": 718,
                "unique_record_ids": 718,
            }
            and payload_access
            == {
                "candidate_image_payload_files_opened": 497,
                "candidate_label_payload_files_opened": 2327,
                "development_holdout_image_payload_files_opened": 0,
                "development_holdout_label_payload_files_opened": 0,
                "ledger": _ppe_human_qa_r6_relative_pin(
                    "payload_access", "payload-access.jsonl"
                ),
                "source_label_payload_files_opened": 2327,
            }
            and artifacts
            == {
                "contact_sheets": 45,
                "manifest": _ppe_human_qa_r6_relative_pin(
                    "artifact_manifest", "artifact-manifest.jsonl"
                ),
                "tiles": 718,
            }
            and receipt.get("execution") == expected_execution
            and receipt.get("readiness") == expected_readiness
            and receipt.get("r4_visual_context", {}).get(
                "human_must_reinspect"
            )
            is True
            and receipt.get("r4_visual_context", {}).get("known_risks")
            == [
                "carried_or_held_helmet_may_be_mislabeled_as_worn",
                (
                    "fall_arrest_harness_or_ordinary_garment_may_be_"
                    "mislabeled_as_hi_vis"
                ),
                "no_vest_proxy_is_not_no_hi_vis_ground_truth",
            ]
            and not any(
                key in receipt
                for key in (
                    "adjudication",
                    "approval",
                    "authorization_receipt",
                    "human_review",
                )
            )
            and _external_receipt_self_hash_matches(
                receipt,
                expected=PPE_HUMAN_QA_R6_PINS["receipt"][
                    "receipt_sha256"
                ],
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ppe_human_qa_r6(reader: ArtifactReader) -> dict[str, Any]:
    integrity = {
        f"{key}_exact_pin_verified": False for key in PPE_HUMAN_QA_R6_PINS
    }
    integrity.update(
        {
            "samples_not_collected": False,
            "artifact_manifest_not_collected": False,
            "payload_access_not_collected": False,
            "implementation_not_collected": False,
            "plan_schema_replayed": False,
            "receipt_schema_replayed": False,
            "sample_and_adjudication_schema_contracts_verified": False,
            "receipt_self_hash_verified": False,
            "cross_artifact_semantics_verified": False,
        }
    )
    values: dict[str, dict[str, Any]] = {}
    json_keys = (
        "plan",
        "receipt",
        "plan_schema",
        "receipt_schema",
        "sample_schema",
        "adjudication_schema",
    )
    for key in json_keys:
        descriptor = PPE_HUMAN_QA_R6_PINS[key]
        pin = _person_pin_core(descriptor)
        assert pin is not None
        result = _read_workspace_pin(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=PPE_HUMAN_QA_R6_JSON_MAX_BYTES,
            collect=True,
        )
        integrity[f"{key}_exact_pin_verified"] = result.available
        if not result.available or result.content is None:
            return _ppe_human_qa_r6_unavailable(
                f"r6_{key}_{result.state}", integrity=integrity
            )
        try:
            value = strict_json_loads(result.content)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            return _ppe_human_qa_r6_unavailable(
                f"r6_{key}_invalid_json", integrity=integrity
            )
        if not isinstance(value, dict):
            return _ppe_human_qa_r6_unavailable(
                f"r6_{key}_invalid_shape", integrity=integrity
            )
        values[key] = value

    for key in (
        "samples",
        "artifact_manifest",
        "payload_access",
        "implementation",
    ):
        descriptor = PPE_HUMAN_QA_R6_PINS[key]
        pin = _person_pin_core(descriptor)
        assert pin is not None
        maximum_bytes = (
            PPE_HUMAN_QA_R6_JSON_MAX_BYTES
            if key == "implementation"
            else PPE_HUMAN_QA_R6_STREAM_MAX_BYTES
        )
        result = _read_workspace_pin(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=maximum_bytes,
            collect=False,
        )
        integrity[f"{key}_exact_pin_verified"] = result.available
        integrity[f"{key}_not_collected"] = bool(
            result.available and result.content is None
        )
        if not result.available:
            return _ppe_human_qa_r6_unavailable(
                f"r6_{key}_{result.state}", integrity=integrity
            )

    plan = values["plan"]
    receipt = values["receipt"]
    plan_schema = values["plan_schema"]
    receipt_schema = values["receipt_schema"]
    try:
        _validate_schema_node(plan, plan_schema, plan_schema)
    except (TypeError, ValueError, RecursionError):
        pass
    else:
        integrity["plan_schema_replayed"] = bool(
            plan_schema.get("$schema")
            == "https://json-schema.org/draft/2020-12/schema"
            and plan_schema.get("$id")
            == (
                "https://deepsafe.local/schemas/"
                "ppe-human-qa-packet-plan-r6-v1.schema.json"
            )
        )
    try:
        _validate_schema_node(receipt, receipt_schema, receipt_schema)
    except (TypeError, ValueError, RecursionError):
        pass
    else:
        integrity["receipt_schema_replayed"] = bool(
            receipt_schema.get("$schema")
            == "https://json-schema.org/draft/2020-12/schema"
            and receipt_schema.get("$id")
            == (
                "https://deepsafe.local/schemas/"
                "ppe-human-qa-packet-receipt-r6-v1.schema.json"
            )
        )
    integrity["sample_and_adjudication_schema_contracts_verified"] = (
        _ppe_human_qa_r6_schema_contract_valid(
            values["sample_schema"], values["adjudication_schema"]
        )
    )
    integrity["receipt_self_hash_verified"] = (
        _external_receipt_self_hash_matches(
            receipt,
            expected=PPE_HUMAN_QA_R6_PINS["receipt"]["receipt_sha256"],
        )
    )
    integrity["cross_artifact_semantics_verified"] = (
        _ppe_human_qa_r6_semantics_valid(plan, receipt)
    )
    if not all(integrity.values()):
        return _ppe_human_qa_r6_unavailable(
            "r6_cross_artifact_contract_invalid", integrity=integrity
        )

    return {
        "evidence_version": "r6",
        "available": True,
        "state": "human_qa_packet_prepared_not_adjudicated",
        "reason": "permanent_human_review_and_new_authorization_required",
        "packet_prepared": True,
        "samples": 718,
        "contact_sheets": 45,
        "groups": 213,
        "role_samples": {"train": 386, "calibration": 332},
        "holdout_payload_access": {
            "image_files_opened": 0,
            "label_files_opened": 0,
        },
        "human_qa_complete": False,
        "permanent_review_record_present": False,
        "permanent_approval_present": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "export_authorized": False,
        "production_ready": False,
        "execution": {
            "cpu_only_packet_materialization": True,
            "gpu_executed": False,
            "docker_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
            "export_executed": False,
            "admin_rebuilt_or_restarted": False,
        },
        "integrity": integrity,
        "caveats": [
            "718 örnek ve 45 contact sheet yalnız inceleme paketidir; kalıcı insan kararı veya onay receipt'i değildir.",
            "Development holdout görüntü/etiket payload erişimi 0/0 kaldı.",
            "İnsan QA tamamlansa bile eğitim için ayrı ve yeni exact-pin yetki gerekir.",
        ],
    }


def _ppe_four_class_r5_unavailable(
    reason: str, *, integrity: dict[str, bool] | None = None
) -> dict[str, Any]:
    return {
        "evidence_version": "r5",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "semantic_remediation_prepared": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "export_authorized": False,
        "production_ready": False,
        "holdout_payload_access": {
            "image_files_opened": None,
            "label_files_opened": None,
        },
        "integrity": integrity or {},
        "caveats": [
            "R5 exact-pin semantik remediation zinciri doğrulanamadı; eğitim ve sonraki tüm kapılar kapalıdır.",
        ],
    }


def _ppe_four_class_r5_semantics_valid(
    contract: Any, receipt: Any
) -> bool:
    if not isinstance(contract, dict) or not isinstance(receipt, dict):
        return False
    expected_counts = {
        "helmet_worn_candidate": 4365,
        "hi_vis_worn_candidate": 2403,
        "no_helmet_explicit": 925,
        "person": 5366,
    }
    expected_quarantine = {
        "helmet_center_below_associated_person_top_35_percent": 90,
        "helmet_no_person_center_association": 79,
        "hi_vis_group_semantic_quarantine_r4_harness_or_uncertain": 24,
        "no_vest_proxy_removed_no_runtime_no_hi_vis_class": 2806,
    }
    try:
        transform = contract["semantic_transform"]
        candidate = receipt["candidate"]
        payload_access = receipt["payload_access"]
        readiness = receipt["readiness"]
        human_qa = receipt["human_qa"]
        execution = receipt["execution"]
        return bool(
            contract.get("schema_version")
            == "deepsafe.ppe-four-class-remediation-contract/v1"
            and contract.get("contract_id")
            == "mendeley-ppe-five-class-to-four-class-remediation-r5"
            and contract.get("status")
            == "semantic_remediation_prepared_not_training_authorized"
            and contract.get("execution_constraints")
            == {
                "admin_rebuild_allowed": False,
                "docker_allowed": False,
                "evaluation_allowed": False,
                "export_allowed": False,
                "gpu_allowed": False,
                "gpu_query_allowed": False,
                "network_allowed": False,
                "training_allowed": False,
            }
            and transform.get("authorized_roles") == ["train", "calibration"]
            and transform.get("excluded_role") == "development_holdout"
            and transform.get("candidate_classes")
            == [
                "person",
                "helmet_worn_candidate",
                "no_helmet_explicit",
                "hi_vis_worn_candidate",
            ]
            and transform.get("helmet", {}).get("head_zone_max_vertical_fraction")
            == 0.35
            and transform.get("helmet", {}).get("semantic_label")
            == "helmet_worn_candidate_not_ground_truth"
            and transform.get("hi_vis", {}).get("semantic_label")
            == "hi_vis_worn_candidate_not_ground_truth"
            and transform.get("no_hi_vis")
            == {
                "absence_policy": (
                    "calibrated_person_association_policy_or_unknown_only"
                ),
                "action": "quarantine_all_no_vest_proxy_boxes",
                "runtime_detector_class_created": False,
                "source_class_id": 4,
            }
            and contract.get("expected", {}).get("candidate")
            == {
                key: candidate.get(key)
                for key in contract.get("expected", {}).get("candidate", {})
            }
            and receipt.get("schema_version")
            == "deepsafe.ppe-four-class-remediation-receipt/v1"
            and receipt.get("receipt_id")
            == "mendeley-ppe-four-class-remediated-r5"
            and receipt.get("status")
            == "semantic_remediation_prepared_not_training_authorized"
            and _external_receipt_self_hash_matches(
                receipt,
                expected=PPE_FOUR_CLASS_SEMANTIC_R5_PINS["receipt"][
                    "receipt_sha256"
                ],
            )
            and receipt.get("contract")
            == {
                **PPE_FOUR_CLASS_SEMANTIC_R5_PINS["contract"],
                "external_pin_verified": True,
            }
            and receipt.get("source")
            == {
                "authorized_rows_materialized": 2327,
                "dataset_id": "mendeley-ppe-five-class-v1-development-r3",
                "development_holdout_metadata_rows_seen": 259,
                "metadata_rows_read": 2586,
                "source_file_ledger": {
                    "bytes": 3674002,
                    "path": (
                        "data/derived/ppe/"
                        "mendeley-ppe-five-class-v1-development-r3/"
                        "metadata/file-ledger.jsonl"
                    ),
                    "sha256": (
                        "bb0aa478596a6c787bcc1289329f953e1786c926ccbbc4a5488ac7364feb1959"
                    ),
                },
            }
            and candidate.get("images") == 2327
            and candidate.get("groups") == 220
            and candidate.get("retained_bbox_rows") == 13059
            and candidate.get("quarantined_bbox_rows") == 2999
            and candidate.get("retained_class_counts") == expected_counts
            and candidate.get("quarantine_reason_counts")
            == expected_quarantine
            and candidate.get("roles", {}).get("train", {}).get("images")
            == 2068
            and candidate.get("roles", {}).get("calibration", {}).get(
                "images"
            )
            == 259
            and payload_access
            == {
                "calibration_image_payload_files_opened": 259,
                "calibration_label_payload_files_opened": 259,
                "development_holdout_image_payload_files_opened": 0,
                "development_holdout_label_payload_files_opened": 0,
                "train_image_payload_files_opened": 2068,
                "train_label_payload_files_opened": 2068,
            }
            and readiness
            == {
                "evaluation_authorized": False,
                "export_authorized": False,
                "human_qa_complete": False,
                "production_ready": False,
                "semantic_remediation_prepared": True,
                "training_authorized": False,
            }
            and human_qa.get("required") is True
            and human_qa.get("complete") is False
            and human_qa.get("authorization_receipt") is None
            and human_qa.get("new_training_authorization_required") is True
            and execution
            == {
                "admin_rebuilt_or_restarted": False,
                "docker_executed": False,
                "evaluation_executed": False,
                "export_executed": False,
                "gpu_executed": False,
                "gpu_queried": False,
                "mode": "cpu_only_no_overwrite_candidate_materialization",
                "network_executed": False,
                "training_executed": False,
            }
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ppe_four_class_r5(reader: ArtifactReader) -> dict[str, Any]:
    integrity = {
        f"{key}_exact_pin_verified": False
        for key in PPE_FOUR_CLASS_SEMANTIC_R5_PINS
    }
    integrity.update(
        {
            "contract_schema_replayed": False,
            "receipt_schema_replayed": False,
            "receipt_self_hash_verified": False,
            "cross_artifact_semantics_verified": False,
        }
    )
    values: dict[str, dict[str, Any]] = {}

    def read_optional_json(
        key: str, pin: dict[str, Any]
    ) -> tuple[WorkspacePinRead, dict[str, Any] | None]:
        result = _read_workspace_pin(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=PPE_FOUR_CLASS_SEMANTIC_R5_MAX_BYTES,
            collect=True,
        )
        if not result.available or result.content is None:
            return result, None
        try:
            value = strict_json_loads(result.content)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            return WorkspacePinRead("invalid_json"), None
        if not isinstance(value, dict):
            return WorkspacePinRead("invalid_shape"), None
        return result, value

    for key in ("contract", "receipt", "contract_schema", "receipt_schema"):
        descriptor = PPE_FOUR_CLASS_SEMANTIC_R5_PINS[key]
        pin = _person_pin_core(descriptor)
        assert pin is not None
        result, value = read_optional_json(key, pin)
        integrity[f"{key}_exact_pin_verified"] = result.available
        if value is None:
            return _ppe_four_class_r5_unavailable(
                f"r5_{key}_{result.state}", integrity=integrity
            )
        values[key] = value
    implementation = PPE_FOUR_CLASS_SEMANTIC_R5_PINS["implementation"]
    implementation_read = _read_workspace_pin(
        reader,
        implementation,
        expected_path=implementation["path"],
        maximum_bytes=PPE_FOUR_CLASS_SEMANTIC_R5_MAX_BYTES,
        collect=False,
    )
    integrity["implementation_exact_pin_verified"] = (
        implementation_read.available
    )
    if not implementation_read.available:
        return _ppe_four_class_r5_unavailable(
            f"r5_implementation_{implementation_read.state}",
            integrity=integrity,
        )

    contract = values["contract"]
    receipt = values["receipt"]
    contract_schema = values["contract_schema"]
    receipt_schema = values["receipt_schema"]
    try:
        _validate_schema_node(contract, contract_schema, contract_schema)
    except (TypeError, ValueError, RecursionError):
        pass
    else:
        integrity["contract_schema_replayed"] = bool(
            contract_schema.get("$schema")
            == "https://json-schema.org/draft/2020-12/schema"
            and contract_schema.get("$id")
            == (
                "https://deepsafe.local/schemas/"
                "ppe-four-class-remediation-contract-v1.schema.json"
            )
        )
    try:
        _validate_schema_node(receipt, receipt_schema, receipt_schema)
    except (TypeError, ValueError, RecursionError):
        pass
    else:
        integrity["receipt_schema_replayed"] = bool(
            receipt_schema.get("$schema")
            == "https://json-schema.org/draft/2020-12/schema"
            and receipt_schema.get("$id")
            == (
                "https://deepsafe.local/schemas/"
                "ppe-four-class-remediation-receipt-v1.schema.json"
            )
        )
    integrity["receipt_self_hash_verified"] = (
        _external_receipt_self_hash_matches(
            receipt,
            expected=PPE_FOUR_CLASS_SEMANTIC_R5_PINS["receipt"][
                "receipt_sha256"
            ],
        )
    )
    integrity["cross_artifact_semantics_verified"] = (
        _ppe_four_class_r5_semantics_valid(contract, receipt)
    )
    if not all(integrity.values()):
        return _ppe_four_class_r5_unavailable(
            "r5_cross_artifact_contract_invalid", integrity=integrity
        )
    candidate = receipt["candidate"]
    access = receipt["payload_access"]
    return {
        "evidence_version": "r5",
        "available": True,
        "state": "semantic_remediation_prepared_not_training_authorized",
        "reason": "human_qa_and_new_training_authorization_required",
        "semantic_remediation_prepared": True,
        "candidate": {
            "images": candidate["images"],
            "groups": candidate["groups"],
            "retained_bbox_rows": candidate["retained_bbox_rows"],
            "quarantined_bbox_rows": candidate["quarantined_bbox_rows"],
            "retained_class_counts": candidate["retained_class_counts"],
            "quarantine_reason_counts": candidate[
                "quarantine_reason_counts"
            ],
            "roles": {
                role: {
                    "groups": row["groups"],
                    "images": row["images"],
                    "retained_bbox_rows": row["retained_bbox_rows"],
                }
                for role, row in candidate["roles"].items()
            },
        },
        "holdout_payload_access": {
            "image_files_opened": access[
                "development_holdout_image_payload_files_opened"
            ],
            "label_files_opened": access[
                "development_holdout_label_payload_files_opened"
            ],
            "metadata_rows_seen": receipt["source"][
                "development_holdout_metadata_rows_seen"
            ],
        },
        "runtime_no_hi_vis_detector_class_created": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "export_authorized": False,
        "human_qa_complete": False,
        "new_training_authorization_required": True,
        "production_ready": False,
        "execution": {
            "cpu_only_materialization": True,
            "gpu_executed": False,
            "docker_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
            "export_executed": False,
        },
        "human_qa_packet_r6": _ppe_human_qa_r6(reader),
        "integrity": integrity,
        "caveats": [
            "R5 dört sınıflı development adayıdır; worn/explicit sınıflar ground truth veya production etiketi değildir.",
            "no_vest proxy kutuları karantinaya alındı; runtime no_hi_vis detector sınıfı oluşturulmadı.",
            "Development holdout için 259 metadata satırı görüldü, fakat görüntü/etiket payload erişimi 0/0 kaldı.",
        ],
    }


def _ppe_five_class_readiness(reader: ArtifactReader) -> dict[str, Any]:
    """Project normalization R2 while retaining immutable R1/R2 history."""

    integrity = {
        "manifest_exact_pin_verified": False,
        "manifest_semantics_verified": False,
        "authoritative_receipt_stream_pin_verified": False,
        "authoritative_receipt_not_collected": False,
        "superseded_r1_receipt_stream_pin_verified": False,
        "superseded_r1_receipt_not_collected": False,
        "compact_receipt_exact_pin_verified": False,
        "compact_receipt_self_hash_verified": False,
        "compact_schema_exact_pin_verified": False,
        "compact_schema_replay_verified": False,
        "compact_validator_exact_pin_verified": False,
        "compact_semantics_verified": False,
        "normalization_plan_exact_pin_verified": False,
        "normalization_plan_schema_exact_pin_verified": False,
        "normalization_plan_schema_replay_verified": False,
        "normalization_plan_semantics_verified": False,
        "normalization_receipt_exact_pin_verified": False,
        "normalization_receipt_self_hash_verified": False,
        "normalization_receipt_schema_exact_pin_verified": False,
        "normalization_receipt_schema_replay_verified": False,
        "normalization_implementation_exact_pin_verified": False,
        "normalization_group_ledger_replay_verified": False,
        "normalization_semantics_verified": False,
        "semantic_r4_plan_exact_pin_verified": False,
        "semantic_r4_receipt_exact_pin_verified": False,
        "semantic_r4_receipt_self_hash_verified": False,
        "semantic_r4_receipt_schema_exact_pin_verified": False,
        "semantic_r4_receipt_schema_replay_verified": False,
        "semantic_r4_implementation_exact_pin_verified": False,
        "semantic_r4_semantics_verified": False,
        "semantic_launch_gate_r3_exact_pin_verified": False,
        "semantic_launch_gate_r3_fingerprint_verified": False,
        "semantic_launch_gate_r3_schema_exact_pin_verified": False,
        "semantic_launch_gate_r3_schema_replay_verified": False,
        "semantic_launch_gate_r3_verifier_exact_pin_verified": False,
        "semantic_launch_gate_r3_semantics_verified": False,
    }
    manifest_pin = PPE_FIVE_CLASS_ADMIN_PINS["manifest"]
    manifest_read, manifest = _workspace_pin_json(
        reader,
        manifest_pin,
        expected_path=manifest_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_COMPACT_MAX_BYTES,
    )
    integrity["manifest_exact_pin_verified"] = manifest_read.available

    source_descriptor = PPE_FIVE_CLASS_ADMIN_PINS["authoritative_receipt"]
    source_pin = _person_pin_core(source_descriptor)
    assert source_pin is not None
    source_read = _read_workspace_pin(
        reader,
        source_pin,
        expected_path=source_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_SOURCE_RECEIPT_MAX_BYTES,
        collect=False,
    )
    integrity["authoritative_receipt_stream_pin_verified"] = (
        source_read.available
    )
    integrity["authoritative_receipt_not_collected"] = bool(
        source_read.available and source_read.content is None
    )

    r1_descriptor = PPE_FIVE_CLASS_ADMIN_PINS["superseded_r1_receipt"]
    r1_pin = _person_pin_core(r1_descriptor)
    assert r1_pin is not None
    r1_read = _read_workspace_pin(
        reader,
        r1_pin,
        expected_path=r1_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_SOURCE_RECEIPT_MAX_BYTES,
        collect=False,
    )
    integrity["superseded_r1_receipt_stream_pin_verified"] = (
        r1_read.available
    )
    integrity["superseded_r1_receipt_not_collected"] = bool(
        r1_read.available and r1_read.content is None
    )

    compact_descriptor = PPE_FIVE_CLASS_ADMIN_PINS["projection_receipt"]
    compact_pin = _person_pin_core(compact_descriptor)
    assert compact_pin is not None
    compact_read, compact = _workspace_pin_json(
        reader,
        compact_pin,
        expected_path=compact_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_COMPACT_MAX_BYTES,
    )
    integrity["compact_receipt_exact_pin_verified"] = compact_read.available

    schema_pin = PPE_FIVE_CLASS_ADMIN_PINS["schema"]
    schema_read, schema = _workspace_pin_json(
        reader,
        schema_pin,
        expected_path=schema_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_COMPACT_MAX_BYTES,
    )
    integrity["compact_schema_exact_pin_verified"] = schema_read.available
    validator_pin = PPE_FIVE_CLASS_ADMIN_PINS["validator"]
    validator_read = _read_workspace_pin(
        reader,
        validator_pin,
        expected_path=validator_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_COMPACT_MAX_BYTES,
        collect=False,
    )
    integrity["compact_validator_exact_pin_verified"] = (
        validator_read.available
    )

    normalization_plan_pin = PPE_FIVE_CLASS_NORMALIZATION_R2_PINS["plan"]
    normalization_plan_read, normalization_plan = _workspace_pin_json(
        reader,
        normalization_plan_pin,
        expected_path=normalization_plan_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_NORMALIZATION_MAX_BYTES,
    )
    integrity["normalization_plan_exact_pin_verified"] = (
        normalization_plan_read.available
    )
    normalization_plan_schema_pin = PPE_FIVE_CLASS_NORMALIZATION_R2_PINS[
        "plan_schema"
    ]
    normalization_plan_schema_read, normalization_plan_schema = (
        _workspace_pin_json(
            reader,
            normalization_plan_schema_pin,
            expected_path=normalization_plan_schema_pin["path"],
            maximum_bytes=PPE_FIVE_CLASS_NORMALIZATION_MAX_BYTES,
        )
    )
    integrity["normalization_plan_schema_exact_pin_verified"] = (
        normalization_plan_schema_read.available
    )
    normalization_receipt_descriptor = PPE_FIVE_CLASS_NORMALIZATION_R2_PINS[
        "receipt"
    ]
    normalization_receipt_pin = _person_pin_core(
        normalization_receipt_descriptor
    )
    assert normalization_receipt_pin is not None
    normalization_receipt_read, normalization_receipt = _workspace_pin_json(
        reader,
        normalization_receipt_pin,
        expected_path=normalization_receipt_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_NORMALIZATION_MAX_BYTES,
    )
    integrity["normalization_receipt_exact_pin_verified"] = (
        normalization_receipt_read.available
    )
    normalization_receipt_schema_pin = PPE_FIVE_CLASS_NORMALIZATION_R2_PINS[
        "receipt_schema"
    ]
    normalization_receipt_schema_read, normalization_receipt_schema = (
        _workspace_pin_json(
            reader,
            normalization_receipt_schema_pin,
            expected_path=normalization_receipt_schema_pin["path"],
            maximum_bytes=PPE_FIVE_CLASS_NORMALIZATION_MAX_BYTES,
        )
    )
    integrity["normalization_receipt_schema_exact_pin_verified"] = (
        normalization_receipt_schema_read.available
    )
    normalization_implementation_pin = PPE_FIVE_CLASS_NORMALIZATION_R2_PINS[
        "implementation"
    ]
    normalization_implementation_read = _read_workspace_pin(
        reader,
        normalization_implementation_pin,
        expected_path=normalization_implementation_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_NORMALIZATION_MAX_BYTES,
        collect=False,
    )
    integrity["normalization_implementation_exact_pin_verified"] = (
        normalization_implementation_read.available
    )

    semantic_plan_pin = PPE_FIVE_CLASS_SEMANTIC_R4_PINS["plan"]
    semantic_plan_read = _read_workspace_pin(
        reader,
        semantic_plan_pin,
        expected_path=semantic_plan_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_SEMANTIC_MAX_BYTES,
        collect=False,
    )
    integrity["semantic_r4_plan_exact_pin_verified"] = semantic_plan_read.available
    semantic_receipt_descriptor = PPE_FIVE_CLASS_SEMANTIC_R4_PINS["receipt"]
    semantic_receipt_pin = _person_pin_core(semantic_receipt_descriptor)
    assert semantic_receipt_pin is not None
    semantic_receipt_read, semantic_receipt = _workspace_pin_json(
        reader,
        semantic_receipt_pin,
        expected_path=semantic_receipt_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_SEMANTIC_MAX_BYTES,
    )
    integrity["semantic_r4_receipt_exact_pin_verified"] = (
        semantic_receipt_read.available
    )
    semantic_schema_pin = PPE_FIVE_CLASS_SEMANTIC_R4_PINS["receipt_schema"]
    semantic_schema_read, semantic_schema = _workspace_pin_json(
        reader,
        semantic_schema_pin,
        expected_path=semantic_schema_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_SEMANTIC_MAX_BYTES,
    )
    integrity["semantic_r4_receipt_schema_exact_pin_verified"] = (
        semantic_schema_read.available
    )
    semantic_implementation_pin = PPE_FIVE_CLASS_SEMANTIC_R4_PINS[
        "implementation"
    ]
    semantic_implementation_read = _read_workspace_pin(
        reader,
        semantic_implementation_pin,
        expected_path=semantic_implementation_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_SEMANTIC_MAX_BYTES,
        collect=False,
    )
    integrity["semantic_r4_implementation_exact_pin_verified"] = (
        semantic_implementation_read.available
    )

    launch_gate_descriptor = PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS["gate"]
    launch_gate_pin = _person_pin_core(launch_gate_descriptor)
    assert launch_gate_pin is not None
    launch_gate_read, launch_gate = _workspace_pin_json(
        reader,
        launch_gate_pin,
        expected_path=launch_gate_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_SEMANTIC_MAX_BYTES,
    )
    integrity["semantic_launch_gate_r3_exact_pin_verified"] = (
        launch_gate_read.available
    )
    launch_gate_schema_pin = PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS["schema"]
    launch_gate_schema_read, launch_gate_schema = _workspace_pin_json(
        reader,
        launch_gate_schema_pin,
        expected_path=launch_gate_schema_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_SEMANTIC_MAX_BYTES,
    )
    integrity["semantic_launch_gate_r3_schema_exact_pin_verified"] = (
        launch_gate_schema_read.available
    )
    launch_gate_verifier_pin = PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS[
        "verifier"
    ]
    launch_gate_verifier_read = _read_workspace_pin(
        reader,
        launch_gate_verifier_pin,
        expected_path=launch_gate_verifier_pin["path"],
        maximum_bytes=PPE_FIVE_CLASS_SEMANTIC_MAX_BYTES,
        collect=False,
    )
    integrity["semantic_launch_gate_r3_verifier_exact_pin_verified"] = (
        launch_gate_verifier_read.available
    )

    reads = {
        "manifest": manifest_read,
        "authoritative_receipt": source_read,
        "superseded_r1_receipt": r1_read,
        "compact_receipt": compact_read,
        "compact_schema": schema_read,
        "compact_validator": validator_read,
        "normalization_plan": normalization_plan_read,
        "normalization_plan_schema": normalization_plan_schema_read,
        "normalization_receipt": normalization_receipt_read,
        "normalization_receipt_schema": normalization_receipt_schema_read,
        "normalization_implementation": normalization_implementation_read,
        "semantic_r4_plan": semantic_plan_read,
        "semantic_r4_receipt": semantic_receipt_read,
        "semantic_r4_receipt_schema": semantic_schema_read,
        "semantic_r4_implementation": semantic_implementation_read,
        "semantic_launch_gate_r3": launch_gate_read,
        "semantic_launch_gate_r3_schema": launch_gate_schema_read,
        "semantic_launch_gate_r3_verifier": launch_gate_verifier_read,
    }
    if any(not result.available for result in reads.values()):
        key, result = next(
            (key, result)
            for key, result in reads.items()
            if not result.available
        )
        return _ppe_five_class_unavailable(
            f"{key}_{result.state}", integrity=integrity
        )
    if (
        manifest is None
        or compact is None
        or schema is None
        or normalization_plan is None
        or normalization_plan_schema is None
        or normalization_receipt is None
        or normalization_receipt_schema is None
        or semantic_receipt is None
        or semantic_schema is None
        or launch_gate is None
        or launch_gate_schema is None
    ):
        return _ppe_five_class_unavailable(
            "five_class_json_contract_invalid", integrity=integrity
        )

    manifest_valid = _ppe_five_class_manifest_semantics_valid(manifest)
    integrity["manifest_semantics_verified"] = manifest_valid
    schema_identity_valid = bool(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "ppe-five-class-admin-projection-receipt-v1.schema.json"
        )
    )
    try:
        _validate_schema_node(compact, schema, schema)
    except (TypeError, ValueError, RecursionError):
        schema_replay_valid = False
    else:
        schema_replay_valid = schema_identity_valid
    integrity["compact_schema_replay_verified"] = schema_replay_valid
    self_hash_valid = _external_receipt_self_hash_matches(
        compact,
        expected=compact_descriptor["receipt_sha256"],
    )
    integrity["compact_receipt_self_hash_verified"] = self_hash_valid
    compact_valid = _ppe_five_class_compact_semantics_valid(compact)
    integrity["compact_semantics_verified"] = compact_valid

    normalization_plan_schema_identity = bool(
        normalization_plan_schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and normalization_plan_schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "ppe-five-class-normalization-plan-v1.schema.json"
        )
    )
    try:
        _validate_schema_node(
            normalization_plan,
            normalization_plan_schema,
            normalization_plan_schema,
        )
    except (TypeError, ValueError, RecursionError):
        normalization_plan_schema_valid = False
    else:
        normalization_plan_schema_valid = normalization_plan_schema_identity
    integrity["normalization_plan_schema_replay_verified"] = (
        normalization_plan_schema_valid
    )
    normalization_plan_valid = (
        _ppe_five_class_normalization_plan_semantics_valid(
            normalization_plan
        )
    )
    integrity["normalization_plan_semantics_verified"] = (
        normalization_plan_valid
    )

    normalization_receipt_schema_identity = bool(
        normalization_receipt_schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and normalization_receipt_schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "ppe-five-class-normalization-dry-run-receipt-v1.schema.json"
        )
    )
    try:
        _validate_schema_node(
            normalization_receipt,
            normalization_receipt_schema,
            normalization_receipt_schema,
        )
    except (TypeError, ValueError, RecursionError):
        normalization_receipt_schema_valid = False
    else:
        normalization_receipt_schema_valid = (
            normalization_receipt_schema_identity
        )
    integrity["normalization_receipt_schema_replay_verified"] = (
        normalization_receipt_schema_valid
    )
    normalization_self_hash_valid = _external_receipt_self_hash_matches(
        normalization_receipt,
        expected=normalization_receipt_descriptor["receipt_sha256"],
    )
    integrity["normalization_receipt_self_hash_verified"] = (
        normalization_self_hash_valid
    )
    normalization_group_replay_valid = (
        _ppe_five_class_normalization_group_replay_valid(
            normalization_receipt
        )
    )
    integrity["normalization_group_ledger_replay_verified"] = (
        normalization_group_replay_valid
    )
    normalization_semantics_valid = (
        _ppe_five_class_normalization_receipt_semantics_valid(
            normalization_receipt,
            normalization_plan,
        )
    )
    integrity["normalization_semantics_verified"] = (
        normalization_semantics_valid
    )

    semantic_schema_identity = bool(
        semantic_schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and semantic_schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "ppe-five-class-semantic-audit-receipt-v1.schema.json"
        )
    )
    try:
        _validate_schema_node(
            semantic_receipt,
            semantic_schema,
            semantic_schema,
        )
    except (TypeError, ValueError, RecursionError):
        semantic_schema_valid = False
    else:
        semantic_schema_valid = semantic_schema_identity
    integrity["semantic_r4_receipt_schema_replay_verified"] = (
        semantic_schema_valid
    )
    semantic_self_hash_valid = _external_receipt_self_hash_matches(
        semantic_receipt,
        expected=semantic_receipt_descriptor["receipt_sha256"],
    )
    integrity["semantic_r4_receipt_self_hash_verified"] = (
        semantic_self_hash_valid
    )
    semantic_r4_valid = _ppe_five_class_semantic_r4_semantics_valid(
        semantic_receipt
    )
    integrity["semantic_r4_semantics_verified"] = semantic_r4_valid

    launch_gate_schema_identity = bool(
        launch_gate_schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and launch_gate_schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "ppe-yolo11s-semantic-launch-gate-v1.schema.json"
        )
    )
    try:
        _validate_schema_node(
            launch_gate,
            launch_gate_schema,
            launch_gate_schema,
        )
    except (TypeError, ValueError, RecursionError):
        launch_gate_schema_valid = False
    else:
        launch_gate_schema_valid = launch_gate_schema_identity
    integrity["semantic_launch_gate_r3_schema_replay_verified"] = (
        launch_gate_schema_valid
    )
    launch_gate_valid = _ppe_yolo11s_semantic_launch_gate_r3_semantics_valid(
        launch_gate
    )
    integrity["semantic_launch_gate_r3_semantics_verified"] = launch_gate_valid
    integrity["semantic_launch_gate_r3_fingerprint_verified"] = bool(
        launch_gate_valid
        and launch_gate.get("fingerprint_sha256")
        == launch_gate_descriptor["fingerprint_sha256"]
    )
    if not (
        manifest_valid
        and schema_replay_valid
        and self_hash_valid
        and compact_valid
        and normalization_plan_schema_valid
        and normalization_plan_valid
        and normalization_receipt_schema_valid
        and normalization_self_hash_valid
        and normalization_group_replay_valid
        and normalization_semantics_valid
        and semantic_schema_valid
        and semantic_self_hash_valid
        and semantic_r4_valid
        and launch_gate_schema_valid
        and launch_gate_valid
    ):
        return _ppe_five_class_unavailable(
            "five_class_projection_normalization_or_semantic_gate_invalid",
            integrity=integrity,
        )

    return {
        "label": "PPE 5-Class R2 normalizasyon + karantina",
        "available": True,
        "state": "dry_run_group_split_complete_training_blocked",
        "reason": "rights_camera_semantics_and_independent_test_gates_incomplete",
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "dataset": {
            "source": "Mendeley PPE Detection Dataset (5-Class)",
            "repository_license": "CC-BY-4.0",
            "repository_metadata_verified": True,
            "embedded_third_party_rights_audit_complete": False,
            "published_independent_test_split_present": False,
            "archive_bytes": 208799718,
            "archive_exact_pin_verified_at_projection_generation": True,
            "archive_read_by_admin": False,
        },
        "source_receipt": {
            "bytes": 5456369,
            "stream_pin_verified": True,
            "canonical_self_hash_verified_at_projection_generation": True,
            "parsed_by_admin": False,
            "compact_projection_verified": True,
        },
        "quarantine": {
            "structural_pass": True,
            "accepted": True,
            "training_eligible": False,
            "structural_gates_passed": 24,
            "structural_gates_total": 24,
            "images": 2586,
            "decoded_images": 2586,
            "label_files": 2586,
            "bounding_boxes": 17827,
            "train_images": 2069,
            "validation_images": 517,
            "class_bbox_counts": {
                "helmet": 5036,
                "no_helmet": 1026,
                "no_vest": 3116,
                "person": 5955,
                "vest": 2694,
            },
            "exact_duplicate_groups": 31,
            "cross_split_exact_duplicate_groups": 10,
        },
        "normalization_group_split": {
            "current": True,
            "status": "dry_run_group_split_complete_training_blocked",
            "created_at_utc": _timestamp(
                normalization_receipt.get("created_at")
            ),
            "dry_run_only": True,
            "dataset_materialized": False,
            "gpu_executed": False,
            "mechanical_group_split_complete": True,
            "heuristic_camera_grouping_verified": False,
            "final_group_count": 292,
            "base_capture_key_count": 321,
            "maximum_group_image_count": 216,
            "roles": {
                "train": {"groups": 145, "images": 2068},
                "calibration": {"groups": 75, "images": 259},
                "test": {
                    "groups": 72,
                    "images": 259,
                    "claim": "internal_heldout_audit_only",
                },
            },
            "leakage": {
                "image_path_role": 0,
                "capture_key_role": 0,
                "exact_duplicate_role": 0,
                "group_role": 0,
                "zero": True,
            },
            "exact_duplicate_groups": 31,
            "cross_upstream_split_exact_duplicate_groups": 10,
            "duplicate_annotation_adjudication_pending": True,
            "canonical_decision_classes": [
                "helmet",
                "no_helmet",
                "hi_vis",
                "no_hi_vis",
            ],
            "mapping_training_ready": False,
            "training_eligible": False,
            "final_validation_or_test_eligible": False,
            "blocker_codes": _ppe_five_class_normalization_blockers(),
        },
        "semantic_audit_r4": {
            "status": (
                "ai_semantic_audit_complete_human_adjudication_required"
            ),
            "exact_evidence_verified": True,
            "sample_images": 20,
            "source_groups": 18,
            "bbox_rows_checked": 488,
            "roles": {"train": 14, "calibration": 6},
            "decisions": {
                "accepted_with_guardrails": 2,
                "questionable_needs_adjudication": 15,
                "rejected_development_candidates": 3,
            },
            "issue_counts": {
                "vest_hi_vis_semantic_risk": 3,
                "helmet_semantic_ambiguity": 2,
                "no_vest_no_hi_vis_semantic_risk": 17,
            },
            "development_holdout_payload_files_opened": 0,
            "human_adjudication_required": True,
            "semantic_mapping_approved": False,
            "training_authorized_by_this_audit": False,
            "production_ready": False,
            "critical_findings": [
                "vest_to_hi_vis_harness_misclassification",
                "helmet_worn_vs_carried_ambiguous",
                "no_vest_to_no_hi_vis_unproven",
            ],
        },
        "semantic_remediation_r5": _ppe_four_class_r5(reader),
        "semantic_launch_gate_r3": {
            "status": (
                "blocked_pending_human_semantic_adjudication_"
                "and_new_authorization"
            ),
            "historical_r2_plan_immutable": True,
            "current_repository_launch_policy": True,
            "image_build_preparation_allowed": True,
            "image_build_scope": (
                "container_preparation_only_no_dataset_or_model_execution"
            ),
            "blocked_modes": [
                "smoke_train",
                "baseline_calibration",
                "full_train_150e",
                "resume",
                "evaluation",
                "export",
            ],
            "new_authorization_receipt_present": False,
            "release_requirements_satisfied": False,
            "training_ready": False,
            "production_ready": False,
            "blocker_codes": [
                "semantic_r4_human_adjudication_required",
                "vest_to_hi_vis_harness_misclassification",
                "helmet_worn_vs_carried_ambiguous",
                "no_vest_to_no_hi_vis_unproven",
                "remediated_subset_or_exact_exclusions_required",
                "new_training_authorization_receipt_required",
            ],
        },
        "quarantine_history": {
            "r1_stream_pin_verified": True,
            "r1_superseded": True,
            "r1_authoritative": False,
            "r1_training_eligible": False,
            "r2_stream_pin_verified": True,
            "r2_authoritative": True,
            "r2_structural_pass": True,
            "r2_accepted": True,
            "r2_training_eligible": False,
            "normalization_r2_current": True,
        },
        "eligibility": {
            "embedded_rights_audit_complete": False,
            "camera_site_session_group_safe": False,
            "person_equipment_semantics_normalized": False,
            "published_independent_test_split_ready": False,
            "training_eligible": False,
            "final_validation_or_test_eligible": False,
        },
        "gates": _ppe_five_class_gates(),
        "integrity": integrity,
        "caveats": [
            "24/24 arşiv ve YOLO yapısal kapısı geçti; bu yalnız karantinaya kabul kanıtıdır, eğitim veya ürün kabulü değildir.",
            "R2 dry-run 292 grubu 2068 train, 259 calibration ve 259 internal-audit test görüntüsüne sıfır rol sızıntısıyla ayırdı; veri materialize edilmedi.",
            "R4 exact-pin semantik audit 20 görüntü, 18 grup ve 488 bbox üzerinde 15 insan adjudication adayı ile 3 development reddi buldu; development_holdout payload erişimi sıfır kaldı.",
            "R4; harness kutularının hi-vis sayılması, elde taşınan baretin takılı baret sayılması ve no_vest etiketinden no_hi_vis çıkarılması risklerini doğruladı.",
            "Sürümlü R3 launch gate yalnız image-build hazırlığını açık bırakır; smoke, calibration/evaluation, full train, resume ve export insan adjudication ile yeni exact-pin yetki receipt'i oluşana kadar kapalıdır.",
            "R5 semantik remediation no_vest proxy'sini detector sınıfına çevirmeden karantinaya alır; insan QA ve yeni eğitim yetkisi olmadan train/export başlatmaz.",
            "Sıfır mekanik sızıntı kamera/site/session bağımsızlığını kanıtlamaz; kullanılan gruplama heuristiktir ve kamera doğrulaması bekler.",
            "31 exact-duplicate grubun 10'u eski upstream train/validation sınırını geçiyor; üyeler aynı yeni grupta tutuldu, anotasyon adjudication tamamlanmadı.",
            "Resmî kayıt CC-BY-4.0 bildirir; gömülü üçüncü taraf medya hak denetimi tamamlanmadı.",
            "Yayınlanmış bağımsız test split'i yoktur; 259 görüntülük test rolü yalnız internal audit'tir, final test veya saha ground truth'u değildir.",
            "R1 superseded ve R2 authoritative karantina receipt'leri büyük JSON olarak parse edilmeden exact stream pinleriyle korunur; R2 normalizasyon receipt'i kapalı şema ve grup-ledger replay ile doğrulanır.",
        ],
        "evidence": [],
    }


def _ppe_lo_cpped_gate_states() -> dict[str, bool]:
    return {
        "dataset_artifact_identity_pinned": False,
        "dataset_package_license_compatible": False,
        "embedded_media_provenance_cleared": False,
        "person_and_location_rights_cleared": False,
        "camera_site_session_or_source_asset_grouping_sufficient": False,
        "person_equipment_semantics_verified": False,
        "independent_unopened_test_identity_verified": False,
        "dataset_acquisition_authorized": False,
        "training_eligible": False,
        "commercial_product_training_eligible": False,
        "independent_validation_eligible": False,
        "final_test_eligible": False,
        "production_ready": False,
    }


def _ppe_lo_cpped_blockers() -> dict[str, list[str]]:
    return {
        "lo_lin_hung_ppe_compliance_11k": [
            "lo_drive_asset_authentication_required",
            "lo_dataset_artifact_name_size_and_sha256_unavailable",
            "lo_dataset_specific_license_missing",
            "lo_embedded_web_and_camera_media_rights_unresolved",
            "lo_person_and_location_rights_unresolved",
            "lo_capture_sequence_and_grouping_metadata_missing",
            "lo_raw_vs_augmented_membership_unknown",
            "lo_person_equipment_association_unverified",
            "lo_independent_test_identity_unverified",
        ],
        "cpped_v1": [
            "cpped_justified_access_request_required",
            "cpped_dataset_file_listing_size_and_sha256_unavailable",
            "cpped_cc_by_nc_incompatible_with_commercial_product_training",
            "cpped_ferdous_and_internet_media_provenance_unresolved",
            "cpped_person_and_location_rights_unresolved",
            "cpped_capture_sequence_and_grouping_metadata_missing",
            "cpped_helmet_and_hi_vis_target_mapping_incomplete",
            "cpped_person_equipment_association_unverified",
            "cpped_independent_test_identity_unverified",
        ],
    }


def _ppe_lo_cpped_manifest_semantics_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    expected_scope = {
        "metadata_only": True,
        "network_metadata_observed": True,
        "remote_metadata_bytes_streamed_for_hash_and_discarded": True,
        "remote_metadata_artifacts_persisted": False,
        "dataset_bytes_streamed_or_persisted": False,
        "dataset_archive_members_opened": False,
        "annotations_opened": False,
        "declared_final_test_labels_opened": False,
        "training_or_inference_executed": False,
        "gpu_used": False,
        "admin_modified": False,
    }
    expected_policy = {
        "public_link_is_training_permission": False,
        "article_license_is_dataset_license": False,
        "repository_readme_license_clears_embedded_media_rights": False,
        "paper_title_or_method_proves_frame_sequence_grouping": False,
        "unknown_class_mapping_is_accepted": False,
        "unknown_capture_or_split_grouping_is_accepted": False,
        "commercial_product_training_requires_explicit_compatible_rights": True,
        "independent_validation_requires_unopened_group_safe_test_identity": True,
    }
    expected_eligibility = {
        "dataset_acquisition_authorized": False,
        "training_eligible": False,
        "commercial_product_training_eligible": False,
        "independent_validation_eligible": False,
        "final_test_eligible": False,
    }
    blockers = _ppe_lo_cpped_blockers()
    try:
        sources = value["sources"]
        if (
            not isinstance(sources, list)
            or len(sources) != 2
            or [source.get("id") for source in sources]
            != ["lo_lin_hung_ppe_compliance_11k", "cpped_v1"]
        ):
            return False
        lo, cpped = sources
        lo_declared = lo["declared_dataset"]
        lo_asset = lo["official_dataset_asset"]
        lo_rights = lo["rights"]
        lo_grouping = lo["grouping_and_semantics"]
        cpped_declared = cpped["declared_dataset"]
        cpped_asset = cpped["official_dataset_asset"]
        cpped_repo = cpped["official_repository_snapshot"]
        cpped_rights = cpped["rights"]
        cpped_grouping = cpped["grouping_and_semantics"]
        decision = value["decision"]
        return bool(
            value.get("schema_version")
            == "deepsafe.ppe-lo-cpped-source-quarantine/v1"
            and value.get("observed_at") == "2026-07-18"
            and value.get("status")
            == (
                "metadata_only_all_dataset_acquisition_training_and_"
                "validation_gates_blocked"
            )
            and value.get("scope") == expected_scope
            and value.get("policy") == expected_policy
            and lo.get("doi") == "10.3390/su15010391"
            and lo_declared.get("media_type") == "static_images"
            and lo_declared.get("image_count") == 11000
            and lo_declared.get("annotation_count") == 88725
            and lo_declared.get("annotation_type") == "bounding_boxes"
            and lo_declared.get("annotation_format")
            == "xml_generated_by_labelimg"
            and lo_declared.get("class_counts")
            == {
                "hard_hat": 27905,
                "no_hard_hat": 26163,
                "high_visibility_vest": 12197,
                "no_high_visibility_vest": 22460,
            }
            and sum(lo_declared["class_counts"].values()) == 88725
            and lo_declared.get("archive_raw_vs_augmented_membership_observed")
            is False
            and lo_declared.get(
                "published_train_validation_test_member_listing_observed"
            )
            is False
            and lo_asset.get("provider") == "google_drive"
            and lo_asset.get("resource_kind") == "file"
            and lo_asset.get("anonymous_access_result")
            == "authentication_required"
            and lo_asset.get("file_listing_observed") is False
            and lo_asset.get("file_name") is None
            and lo_asset.get("bytes") is None
            and lo_asset.get("sha256") is None
            and lo_asset.get("dataset_archive_downloaded") is False
            and lo_asset.get("artifact_identity_pinned") is False
            and lo_rights
            == {
                "article_license": "CC-BY-4.0",
                "article_license_scope_proves_dataset_package_rights": False,
                "dataset_specific_license_status": "missing_or_unknown",
                "commercial_derivative_training_allowed": False,
                "embedded_web_media_provenance_cleared": False,
                "camera_media_provenance_cleared": False,
                "person_release_confirmed": False,
                "location_release_confirmed": False,
                "institutional_review_board_statement": (
                    "not_applicable_in_article"
                ),
                "informed_consent_statement": "not_applicable_in_article",
            }
            and lo_grouping
            == {
                "camera_site_session_identifiers_observed": False,
                "source_video_or_frame_sequence_identifiers_observed": False,
                "exact_or_near_duplicate_groups_observed": False,
                "source_asset_group_safe_split_possible_from_observed_metadata": False,
                "independent_test_identity_observed": False,
                "person_class_present": False,
                "person_to_equipment_association_verified": False,
                "target_label_mapping": {
                    "helmet": "hard_hat",
                    "no_helmet": "no_hard_hat",
                    "hi_vis": "high_visibility_vest",
                    "no_hi_vis": "no_high_visibility_vest",
                },
                "target_label_names_match": True,
                "wearing_semantics_and_person_association_verified": False,
            }
            and lo.get("eligibility") == expected_eligibility
            and lo.get("blockers") == blockers[lo["id"]]
            and cpped.get("doi") == "10.1061/JCEMD4.COENG-15310"
            and cpped_repo.get("repository") == "QHCV/CPPED"
            and cpped_repo.get("commit")
            == "22de323a1ad4e27ed95cf586495399265a821ce7"
            and cpped_repo.get("tree")
            == "f653779bdf1740c5494040d29ea19f7ca6b5fe58"
            and cpped_repo.get("commit_signature_verified") is False
            and cpped_repo.get("file_count") == 17
            and cpped_repo.get("file_bytes") == 754101
            and cpped_repo.get("opaque_spreadsheet_contents_interpreted")
            is False
            and cpped_repo.get(
                "dataset_images_or_labels_present_in_repository_tree"
            )
            is False
            and cpped_declared.get("image_count") == 2612
            and cpped_declared.get("category_count") == 13
            and cpped_declared.get("annotation_count") == 20172
            and cpped_declared.get("annotation_type") == "bounding_boxes"
            and cpped_declared.get("class_counts")
            == {
                "head": 957,
                "person": 5936,
                "no_gloves": 2017,
                "shoes": 1484,
                "glass": 701,
                "yellow": 1600,
                "white": 1563,
                "no_shoes": 347,
                "blue": 604,
                "vest": 2587,
                "gloves": 985,
                "red": 862,
                "mask": 529,
            }
            and sum(cpped_declared["class_counts"].values()) == 20172
            and cpped_declared.get(
                "published_train_validation_test_member_listing_observed"
            )
            is False
            and cpped_asset.get("provider") == "google_drive"
            and cpped_asset.get("resource_kind") == "folder"
            and cpped_asset.get("anonymous_access_result")
            == "authentication_and_justified_access_request_required"
            and cpped_asset.get("file_listing_observed") is False
            and cpped_asset.get("file_names") == []
            and cpped_asset.get("total_bytes") is None
            and cpped_asset.get("archive_sha256") is None
            and cpped_asset.get("dataset_archive_downloaded") is False
            and cpped_asset.get("artifact_identity_pinned") is False
            and cpped_rights
            == {
                "dataset_license_statement": "CC-BY-NC-4.0",
                "dataset_access_request_required": True,
                "access_request_submitted_or_granted": False,
                "commercial_derivative_training_allowed": False,
                "noncommercial_research_use_assumed_authorized_without_access_grant": False,
                "ferdous_source_media_rights_cleared": False,
                "internet_source_media_rights_cleared": False,
                "person_release_confirmed": False,
                "location_release_confirmed": False,
            }
            and cpped_grouping
            == {
                "camera_site_session_identifiers_observed": False,
                "source_video_or_frame_sequence_identifiers_observed": False,
                "exact_or_near_duplicate_groups_observed": False,
                "source_asset_group_safe_split_possible_from_observed_metadata": False,
                "independent_test_identity_observed": False,
                "person_class_present": True,
                "target_label_mapping": {
                    "helmet_candidates": ["yellow", "white", "blue", "red"],
                    "no_helmet": None,
                    "hi_vis_candidate": "vest",
                    "no_hi_vis": None,
                },
                "helmet_color_classes_verified_as_worn_helmet_semantics": False,
                "head_class_assumed_to_mean_no_helmet": False,
                "vest_class_verified_as_high_visibility_and_person_associated": False,
                "person_to_equipment_association_verified": False,
                "complete_project_target_mapping_verified": False,
            }
            and cpped.get("eligibility") == expected_eligibility
            and cpped.get("blockers") == blockers[cpped["id"]]
            and decision.get("source_count") == 2
            and decision.get("dataset_artifact_count") == 0
            and decision.get("download_authorized_source_count") == 0
            and decision.get("training_eligible_source_count") == 0
            and decision.get(
                "commercial_product_training_eligible_source_count"
            )
            == 0
            and decision.get("independent_validation_eligible_source_count")
            == 0
            and decision.get("final_test_eligible_source_count") == 0
            and isinstance(decision.get("next_actions"), list)
            and len(decision["next_actions"]) == 6
        )
    except (KeyError, TypeError, ValueError):
        return False


def _ppe_lo_cpped_receipt_semantics_valid(
    receipt: Any,
    manifest: dict[str, Any],
    *,
    current: bool,
) -> bool:
    if not isinstance(receipt, dict):
        return False
    input_validator = (
        PPE_LO_CPPED_SOURCE_ADMIN_PINS["validator"]
        if current
        else PPE_LO_CPPED_HISTORICAL_VALIDATOR_PIN
    )
    expected_created = (
        "2026-07-18T01:19:52Z" if current else "2026-07-18T01:19:10Z"
    )
    blockers = _ppe_lo_cpped_blockers()
    expected_source_blockers = [
        {"source_id": source_id, "codes": codes}
        for source_id, codes in blockers.items()
    ]
    expected_boundary = {
        "admin_modified": False,
        "annotations_opened": False,
        "dataset_archive_members_opened": False,
        "dataset_bytes_downloaded_or_persisted": False,
        "declared_final_test_labels_opened": False,
        "gpu_used": False,
        "metadata_only": True,
        "network_replayed_during_receipt_build_or_verify": False,
        "opaque_spreadsheet_contents_interpreted": False,
        "training_or_inference_executed": False,
    }
    expected_summary = {
        "all_eligibility_gates_blocked": True,
        "annotations_or_final_test_labels_opened": False,
        "blocker_count": 18,
        "cpped_repository_file_bytes": 754101,
        "cpped_repository_file_count": 17,
        "dataset_artifact_count": 0,
        "dataset_bytes_streamed_or_persisted": False,
        "manifest_bytes": PPE_LO_CPPED_SOURCE_ADMIN_PINS["manifest"]["bytes"],
        "manifest_sha256": PPE_LO_CPPED_SOURCE_ADMIN_PINS["manifest"]["sha256"],
        "metadata_artifact_count": 2,
        "observed_at": "2026-07-18",
        "source_count": 2,
        "source_ids": list(blockers),
        "status": manifest["status"],
        "training_inference_or_gpu_used": False,
        "valid": True,
    }
    expected_inputs = {
        "manifest": PPE_LO_CPPED_SOURCE_ADMIN_PINS["manifest"],
        "schema": PPE_LO_CPPED_SOURCE_ADMIN_PINS["schema"],
        "validator": input_validator,
    }
    return bool(
        receipt.get("schema_version")
        == "deepsafe.ppe-lo-cpped-source-quarantine-receipt/v1"
        and receipt.get("created_at") == expected_created
        and receipt.get("status")
        == (
            "valid_metadata_snapshot_all_dataset_acquisition_training_and_"
            "validation_gates_blocked"
        )
        and receipt.get("inputs") == expected_inputs
        and receipt.get("summary") == expected_summary
        and receipt.get("source_blockers") == expected_source_blockers
        and receipt.get("gates") == _ppe_lo_cpped_gate_states()
        and receipt.get("execution_boundary") == expected_boundary
    )


def _ppe_lo_cpped_unavailable(
    reason: str, *, integrity: dict[str, bool] | None = None
) -> dict[str, Any]:
    return {
        "label": "PPE Lo/CPPED açık kaynak karantinası",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "sources": {},
        "history": {},
        "eligibility": {
            "dataset_acquisition_authorized": False,
            "training_eligible": False,
            "commercial_product_training_eligible": False,
            "independent_validation_eligible": False,
            "final_test_eligible": False,
            "production_ready": False,
        },
        "gates": _ppe_lo_cpped_gate_states(),
        "integrity": integrity or {},
        "caveats": [
            "Lo/CPPED metadata karantina zinciri doğrulanamadı; acquisition, eğitim, validation ve ürün kapıları kapalıdır."
        ],
        "evidence": [],
    }


def _ppe_lo_cpped_source_quarantine(reader: ArtifactReader) -> dict[str, Any]:
    """Project only the small, exact-pinned Lo/CPPED metadata receipts."""

    integrity = {
        "manifest_exact_pin_verified": False,
        "schema_exact_pin_verified": False,
        "validator_exact_pin_verified": False,
        "current_receipt_exact_pin_verified": False,
        "historical_receipt_exact_pin_verified": False,
        "manifest_semantics_replayed": False,
        "current_receipt_schema_replayed": False,
        "historical_receipt_schema_replayed": False,
        "current_receipt_self_hash_replayed": False,
        "historical_receipt_self_hash_replayed": False,
        "current_receipt_semantics_replayed": False,
        "historical_receipt_semantics_replayed": False,
        "r1_superseded_r2_authoritative_verified": False,
    }
    parsed: dict[str, dict[str, Any] | None] = {}
    reads: dict[str, WorkspacePinRead] = {}
    for key in ("manifest", "schema", "current_receipt", "historical_receipt"):
        pin = PPE_LO_CPPED_SOURCE_ADMIN_PINS[key]
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=PPE_LO_CPPED_MAX_JSON_BYTES,
        )
        reads[key] = result
        parsed[key] = value
        integrity[f"{key}_exact_pin_verified"] = result.available

    validator_pin = PPE_LO_CPPED_SOURCE_ADMIN_PINS["validator"]
    validator_read = _read_workspace_pin(
        reader,
        validator_pin,
        expected_path=validator_pin["path"],
        maximum_bytes=PPE_LO_CPPED_MAX_JSON_BYTES,
        collect=False,
    )
    reads["validator"] = validator_read
    integrity["validator_exact_pin_verified"] = validator_read.available
    if any(not result.available for result in reads.values()):
        key, result = next(
            (key, result)
            for key, result in reads.items()
            if not result.available
        )
        return _ppe_lo_cpped_unavailable(
            f"{key}_{result.state}", integrity=integrity
        )

    manifest = parsed["manifest"]
    schema = parsed["schema"]
    current_receipt = parsed["current_receipt"]
    historical_receipt = parsed["historical_receipt"]
    if any(
        item is None
        for item in (manifest, schema, current_receipt, historical_receipt)
    ):
        return _ppe_lo_cpped_unavailable(
            "json_contract_invalid", integrity=integrity
        )
    assert manifest is not None
    assert schema is not None
    assert current_receipt is not None
    assert historical_receipt is not None

    integrity["manifest_semantics_replayed"] = (
        _ppe_lo_cpped_manifest_semantics_valid(manifest)
    )
    schema_identity_valid = bool(
        schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "ppe-lo-cpped-source-quarantine-receipt-v1.schema.json"
        )
    )
    for key, receipt in (
        ("current", current_receipt),
        ("historical", historical_receipt),
    ):
        try:
            _validate_schema_node(receipt, schema, schema)
        except (TypeError, ValueError, RecursionError):
            valid = False
        else:
            valid = schema_identity_valid
        integrity[f"{key}_receipt_schema_replayed"] = valid
        integrity[f"{key}_receipt_self_hash_replayed"] = (
            _external_receipt_self_hash_matches(
                receipt,
                expected=PPE_LO_CPPED_RECEIPT_SELF_SHA256[key],
            )
        )
        integrity[f"{key}_receipt_semantics_replayed"] = (
            _ppe_lo_cpped_receipt_semantics_valid(
                receipt,
                manifest,
                current=key == "current",
            )
        )
    integrity["r1_superseded_r2_authoritative_verified"] = bool(
        current_receipt["created_at"] > historical_receipt["created_at"]
        and current_receipt["inputs"]["validator"]
        == PPE_LO_CPPED_SOURCE_ADMIN_PINS["validator"]
        and historical_receipt["inputs"]["validator"]
        == PPE_LO_CPPED_HISTORICAL_VALIDATOR_PIN
    )
    if not all(integrity.values()):
        return _ppe_lo_cpped_unavailable(
            "exact_pin_schema_or_semantic_replay_invalid",
            integrity=integrity,
        )

    return {
        "label": "PPE Lo/CPPED açık kaynak karantinası",
        "available": True,
        "state": "metadata_only_training_blocked",
        "reason": "rights_access_grouping_semantics_and_test_unresolved",
        "ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "observed_at": "2026-07-18",
        "sources": {
            "lo": {
                "name": "Lo, Lin ve Hung PPE Compliance Detection",
                "media_type": "static_images",
                "images": 11000,
                "bounding_boxes": 88725,
                "target_class_counts": {
                    "helmet": 27905,
                    "no_helmet": 26163,
                    "hi_vis": 12197,
                    "no_hi_vis": 22460,
                },
                "dataset_package_license": "unknown",
                "anonymous_access_available": False,
                "artifact_identity_pinned": False,
                "camera_site_session_grouping_verified": False,
                "person_equipment_association_verified": False,
                "independent_test_identity_verified": False,
                "blocker_count": 9,
            },
            "cpped": {
                "name": "CPPED",
                "media_type": "mixed_source_static_images",
                "images": 2612,
                "categories": 13,
                "bounding_boxes": 20172,
                "repository_metadata_snapshot_verified": True,
                "dataset_package_license": "CC-BY-NC-4.0",
                "access_grant_obtained": False,
                "artifact_identity_pinned": False,
                "commercial_product_training_compatible": False,
                "complete_target_mapping_verified": False,
                "camera_site_session_grouping_verified": False,
                "person_equipment_association_verified": False,
                "independent_test_identity_verified": False,
                "blocker_count": 9,
            },
        },
        "history": {
            "r1_exact_pin_verified": True,
            "r1_schema_and_self_hash_replayed": True,
            "r1_verifier_order_defect_preserved": True,
            "r1_superseded": True,
            "r1_authoritative": False,
            "r2_exact_pin_verified": True,
            "r2_schema_self_hash_and_semantics_replayed": True,
            "r2_authoritative": True,
        },
        "execution_boundary": {
            "metadata_only": True,
            "dataset_bytes_downloaded_or_persisted": False,
            "annotations_or_final_test_labels_opened": False,
            "training_or_inference_executed": False,
            "gpu_used": False,
        },
        "eligibility": {
            "dataset_acquisition_authorized": False,
            "training_eligible": False,
            "commercial_product_training_eligible": False,
            "independent_validation_eligible": False,
            "final_test_eligible": False,
            "production_ready": False,
        },
        "gates": _ppe_lo_cpped_gate_states(),
        "integrity": integrity,
        "caveats": [
            "Lo için 11.000 görüntü ve dört hedef PPE sınıfı yalnız yayıncı metadata'sıdır; veri paketi kimliği, lisansı, kişi-ekipman ilişkisi ve group-safe split kanıtlanmadı.",
            "CPPED için 2.612 görüntü/13 sınıf metadata'sı ve yazar repo snapshot'ı doğrulandı; erişim izni yoktur ve CC-BY-NC-4.0 ticari ürün eğitimine uygun değildir.",
            "Hiçbir dataset byte'ı, annotation veya ilan edilmiş final-test etiketi açılmadı; bu kart training, validation veya ürün kabulü değildir.",
            "Tarihsel R1 anahtar-sırası verifier kusuruyla immutable tutulur; yalnız düzeltilmiş R2 güncel ve authoritative metadata receipt'idir.",
        ],
        "evidence": [],
    }


def _pose_readiness_gates() -> dict[str, bool]:
    return {
        "license_selected": False,
        "model_weights_acquired": False,
        "export_640_complete": False,
        "export_960_complete": False,
        "onnx_640_parity_passed": False,
        "onnx_960_parity_passed": False,
        "tensorrt_engine_640_complete": False,
        "tensorrt_engine_960_complete": False,
        "tensorrt_640_parity_passed": False,
        "tensorrt_960_parity_passed": False,
        "deepstream9_parser_parity_passed": False,
        "independent_ground_truth_ready": False,
        "pck_evaluation_plan_approved": False,
        "pck_640_passed": False,
        "pck_960_passed": False,
        "twelve_camera_640_passed": False,
        "twelve_camera_960_passed": False,
        "three_module_full_stack_passed": False,
        "acceptance_passed": False,
        "production_ready": False,
    }


def _pose_readiness_unavailable(
    reason: str,
    *,
    integrity: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "label": "Pose modeli hazırlığı",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "license": {
            "decision": None,
            "selected": False,
            "download_authorized": False,
            "export_authorized": False,
        },
        "selection": {},
        "permissive_challenger": {},
        "preparation": {
            "provenance_plan_verified": False,
            "frozen_export_plans_verified": False,
            "shared_semantic_contract_verified": False,
            "export_implementation_verified": False,
            "pck_evaluator_contract_verified": False,
            "diagnostic_source_plan_verified": False,
        },
        "model_contract": {},
        "artifacts": {
            "weights_acquired": False,
            "onnx_640_exported": False,
            "onnx_960_exported": False,
            "engine_640_built": False,
            "engine_960_built": False,
        },
        "pck": {
            "evaluator_contract_verified": False,
            "evaluation_plan_pin_declared": False,
            "ground_truth_pin_declared": False,
            "predictions_pin_declared": False,
            "receipt_pin_declared": False,
            "result_available": False,
        },
        "source_readiness": {},
        "gates": _pose_readiness_gates(),
        "integrity": integrity or {},
        "caveats": [
            "Pose hazırlık zinciri doğrulanamadı; export, PCK, kapasite ve ürün kabul kapıları kapalıdır.",
        ],
        "evidence": [],
    }


def _pose_schema_identity(
    schema: Any,
    *,
    schema_id: str,
) -> bool:
    return bool(
        isinstance(schema, dict)
        and schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and schema.get("$id") == schema_id
    )


def _pose_all_false(value: Any, expected_keys: set[str]) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == expected_keys
        and all(item is False for item in value.values())
    )


def _pose_mmpose_raw_profile_valid(value: Any, *, size: int) -> bool:
    if not isinstance(value, dict):
        return False
    levels = (size // 8, size // 16, size // 32)
    expected_groups = (
        ("class_scores", 1),
        ("objectness", 1),
        ("bounding_boxes", 4),
        ("keypoint_offsets", 34),
        ("keypoint_visibility", 17),
    )
    groups = value.get("groups")
    return bool(
        value.get("input_shape") == [1, 3, size, size]
        and value.get("batch") == 1
        and value.get("spatial_size") == size
        and isinstance(groups, list)
        and len(groups) == len(expected_groups)
        and all(
            group
            == {
                "name": name,
                "channels": channels,
                "shapes": [
                    [1, channels, level, level] for level in levels
                ],
                "dtypes": ["float32", "float32", "float32"],
                "all_finite": True,
            }
            for group, (name, channels) in zip(groups, expected_groups)
        )
        and value.get("shape_contract_verified") is True
        and value.get("all_outputs_finite") is True
        and value.get("timing_collected") is False
        and value.get("fps_measured") is False
        and value.get("decoded_predictions_produced") is False
        and value.get("real_images_used") is False
        and value.get("quality_measured") is False
    )


def _pose_mmpose_onnx_profile_valid(value: Any, *, size: int) -> bool:
    if not isinstance(value, dict):
        return False
    return value == {
        "spatial_size": size,
        "dynamic_batch_axis": True,
        "required_batches": [1, 12],
        "expected_input_shapes": [
            [1, 3, size, size],
            [12, 3, size, size],
        ],
        "expected_outputs": [
            {
                "name": "dets",
                "shapes": [[1, 100, 5], [12, 100, 5]],
            },
            {
                "name": "keypoints",
                "shapes": [[1, 100, 17, 3], [12, 100, 17, 3]],
            },
        ],
        "export_attempted": False,
        "onnx_file_published": False,
        "onnx_checker_passed": False,
        "onnxruntime_batch1_passed": False,
        "onnxruntime_batch12_passed": False,
        "pytorch_parity_passed": False,
        "deepstream9_passed": False,
    }


def _pose_mmpose_onnx_preflight_semantics_valid(
    receipt: Any,
    *,
    expected_self_sha256: str,
    expected_blockers: tuple[str, ...],
    expected_checkout_verified: bool,
    schema_valid: bool,
) -> bool:
    """Validate one immutable preflight snapshot without replaying execution."""

    if not isinstance(receipt, dict):
        return False
    execution = receipt.get("execution_boundary", {})
    license_boundary = receipt.get("license_boundary", {})
    official = receipt.get("official_mmdeploy", {})
    tooling_checkout = receipt.get("tooling", {}).get(
        "mmdeploy_checkout", {}
    )
    blockers = receipt.get("blockers")
    profiles = receipt.get("profiles")
    conflict = receipt.get("postprocess_conflict_audit", {})
    conclusions = receipt.get("conclusions", {})
    blocker_codes = (
        tuple(item.get("code") for item in blockers)
        if isinstance(blockers, list)
        and all(isinstance(item, dict) for item in blockers)
        else ()
    )
    expected_execution = {
        "runtime": "cpu_only_preflight",
        "cuda_visible_devices": "",
        "gpu_touched": False,
        "network_downloads_performed": False,
        "docker_touched": False,
        "admin_restarted": False,
        "export_attempted": False,
        "onnx_checker_executed": False,
        "onnxruntime_executed": False,
        "batch12_executed": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
    }
    expected_conclusions = {
        "structural_checkpoint_proof_verified": True,
        "exact_expected_onnx_contract_specified": True,
        "dynamic_batch1_12_acceptance_specified": True,
        "deepstream9_handoff_specified": True,
        "preflight_blocked": True,
        "export_executed": False,
        "onnx_640_verified": False,
        "onnx_960_verified": False,
        "onnx_batch12_verified": False,
        "numeric_parity_passed": False,
        "deepstream9_parser_implemented": False,
        "deepstream9_parity_passed": False,
        "profile_960_quality_claimed": False,
        "production_model_selected": False,
        "production_ready": False,
    }
    return bool(
        schema_valid
        and receipt.get("schema_version")
        == "deepsafe.pose-mmpose-yoloxpose-onnx-lane-receipt/v1"
        and receipt.get("status")
        == "blocked_preflight_no_export_attempted"
        and receipt.get("candidate_id") == "mmpose-yoloxpose-s"
        and _timestamp(receipt.get("created_at")) is not None
        and _external_receipt_self_hash_matches(
            receipt, expected=expected_self_sha256
        )
        and blocker_codes == expected_blockers
        and len(blockers) == len(expected_blockers)
        and all(
            isinstance(item.get("detail"), str)
            and bool(item["detail"])
            and item.get("retryable") is True
            for item in blockers
        )
        and execution == expected_execution
        and license_boundary
        == {
            "challenger_license_spdx": "Apache-2.0",
            "yolo26_license_decision_changed": False,
            "production_model_selected": False,
            "production_ready": False,
        }
        and official.get("repository")
        == "https://github.com/open-mmlab/mmdeploy"
        and official.get("tag") == "v1.3.1"
        and official.get("commit")
        == "bc75c9d6c8940aa03d0e1e5b5962bd930478ba77"
        and official.get("local_source_bytes_verified")
        is expected_checkout_verified
        and tooling_checkout.get("exists") is expected_checkout_verified
        and tooling_checkout.get("expected_commit")
        == "bc75c9d6c8940aa03d0e1e5b5962bd930478ba77"
        and (
            tooling_checkout.get("observed_commit")
            == "bc75c9d6c8940aa03d0e1e5b5962bd930478ba77"
            if expected_checkout_verified
            else tooling_checkout.get("observed_commit") is None
        )
        and isinstance(profiles, dict)
        and set(profiles) == {"640", "960"}
        and _pose_mmpose_onnx_profile_valid(profiles["640"], size=640)
        and _pose_mmpose_onnx_profile_valid(profiles["960"], size=960)
        and conflict
        == {
            "directly_compatible": False,
            "official_outputs": [
                {"name": "dets", "shape": ["B", 100, 5]},
                {
                    "name": "keypoints",
                    "shape": ["B", 100, 17, 3],
                },
            ],
            "existing_input": {
                "name": "single_decoded_tensor",
                "shape": ["B", 300, 57],
            },
            "adapter_required": True,
            "adapter_mapping_specified": True,
            "adapter_implemented": False,
            "adapter_tests_passed": False,
        }
        and conclusions == expected_conclusions
    )


def _pose_mmpose_challenger_semantics_valid(
    plan: Any,
    receipt: Any,
    *,
    plan_pin: dict[str, Any],
    receipt_schema_valid: bool,
) -> bool:
    if not isinstance(plan, dict) or not isinstance(receipt, dict):
        return False
    candidate = plan.get("candidate", {})
    license_contract = plan.get("license", {})
    acquisition = plan.get("acquisition", {})
    structural = plan.get("structural_evidence", {})
    profiles = plan.get("profiles", {})
    deployment = plan.get("deployment_feasibility", {})
    onnx_plan = deployment.get("onnx_plan", {})
    tensorrt_plan = deployment.get("tensorrt_plan", {})
    parser = deployment.get("deepstream9_parser", {})
    quality = plan.get("quality_evaluation_plan", {})
    control = plan.get("control_candidate", {})
    upstream_metrics = plan.get("official_upstream_metrics", {})
    expected_gate_keys = {
        "production_selection_passed",
        "decoded_framework_parity_passed",
        "onnx_640_export_passed",
        "onnx_960_export_passed",
        "onnx_batch12_passed",
        "tensorrt_640_passed",
        "tensorrt_960_passed",
        "deepstream9_parser_parity_passed",
        "pck_640_passed",
        "pck_960_passed",
        "twelve_camera_640_passed",
        "twelve_camera_960_passed",
        "three_module_full_stack_passed",
        "endurance_passed",
        "acceptance_passed",
        "production_ready",
    }

    receipt_pin = structural.get("receipt")
    validator_pin = _person_pin_core(structural.get("validator"))
    schema_pin = _person_pin_core(structural.get("schema"))
    checkpoint_pin = _person_pin_core(acquisition.get("checkpoint"))
    if (
        not isinstance(receipt_pin, dict)
        or _person_pin_core(receipt_pin) is None
        or validator_pin is None
        or schema_pin is None
        or checkpoint_pin is None
    ):
        return False

    inputs = receipt.get("inputs", {})
    execution = receipt.get("execution", {})
    architecture = receipt.get("architecture", {})
    state = receipt.get("checkpoint_structure", {}).get("state_dict", {})
    raw_profiles = receipt.get("raw_forward_profiles", {})
    conclusions = receipt.get("conclusions", {})
    expected_true_conclusions = {
        "checkpoint_integrity_verified",
        "strict_architecture_load_verified",
        "cpu_raw_forward_640_verified",
        "cpu_raw_forward_960_shape_feasibility_verified",
    }
    expected_false_conclusions = {
        "profile_960_upstream_quality_claimed",
        "batch12_feasibility_verified",
        "decoded_framework_parity_passed",
        "onnx_exported",
        "onnx_parity_passed",
        "tensorrt_built",
        "tensorrt_parity_passed",
        "deepstream9_parser_implemented",
        "deepstream9_parity_passed",
        "local_coco_ap_measured",
        "independent_pck_passed",
        "twelve_camera_capacity_passed",
        "production_model_selected",
        "production_ready",
    }
    receipt_valid = bool(
        receipt_schema_valid
        and receipt.get("schema_version")
        == "deepsafe.pose-mmpose-yoloxpose-structural-receipt/v1"
        and receipt.get("status")
        == (
            "verified_cpu_strict_load_and_raw_forward_not_exported_"
            "not_evaluated"
        )
        and receipt.get("candidate_id") == "mmpose-yoloxpose-s"
        and _external_receipt_self_hash_matches(
            receipt,
            expected=POSE_PERMISSIVE_CHALLENGER_RECEIPT_SHA256,
        )
        and receipt_pin.get("receipt_sha256")
        == POSE_PERMISSIVE_CHALLENGER_RECEIPT_SHA256
        and _person_pin_core(inputs.get("checkpoint")) == checkpoint_pin
        and _person_pin_core(inputs.get("validator")) == validator_pin
        and _person_pin_core(inputs.get("schema")) == schema_pin
        and inputs.get("upstream", {}).get("commit")
        == "759b39c13fea6ba094afc1fa932f51dc1b11cbf9"
        and inputs.get("upstream", {}).get("git_tree")
        == "3d214966f1cbaf63682c92514712b7f8ac6d9518"
        and inputs.get("upstream", {}).get("license_spdx") == "Apache-2.0"
        and execution
        == {
            "runtime": "cpu_only",
            "cuda_visible_devices": "",
            "torch_build_cuda": None,
            "torch_cuda_available": False,
            "gpu_touched": False,
            "network_download_calls": 0,
            "weights_only": True,
            "map_location": "cpu",
            "raw_forward_executed": True,
            "raw_forward_batches": [1],
            "decoded_prediction_executed": False,
            "real_image_inference_executed": False,
            "batch12_forward_executed": False,
            "onnx_export_executed": False,
            "tensorrt_executed": False,
            "deepstream9_executed": False,
            "training_executed": False,
            "benchmark_executed": False,
        }
        and architecture.get("class") == "BottomupPoseEstimator"
        and architecture.get("parameter_count") == 10729963
        and architecture.get("buffer_count") == 26285
        and architecture.get("state_tensor_count") == 547
        and architecture.get("parameter_device_types") == ["cpu"]
        and architecture.get("strict_load")
        == {
            "strict": True,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
        }
        and architecture.get("mmcv_ops_import_workaround")
        == {
            "scope": "unrelated_eager_registry_import_only",
            "selected_model_stub_instance_count": 0,
            "deployment_compatibility_claimed": False,
        }
        and state.get("tensor_count") == 547
        and state.get("tensor_value_count") == 10756242
        and isinstance(raw_profiles, dict)
        and set(raw_profiles) == {"640", "960"}
        and _pose_mmpose_raw_profile_valid(raw_profiles["640"], size=640)
        and _pose_mmpose_raw_profile_valid(raw_profiles["960"], size=960)
        and set(conclusions)
        == expected_true_conclusions | expected_false_conclusions
        and all(
            conclusions.get(key) is True
            for key in expected_true_conclusions
        )
        and all(
            conclusions.get(key) is False
            for key in expected_false_conclusions
        )
    )

    return bool(
        plan.get("schema_version")
        == "deepsafe.pose-permissive-challenger-plan/v1"
        and plan.get("status")
        == (
            "checkpoint_acquired_cpu_structural_verified_deployment_not_"
            "started_not_evaluated"
        )
        and _self_fingerprint_matches(plan)
        and plan.get("fingerprint_sha256")
        == plan_pin.get("fingerprint_sha256")
        and candidate
        == {
            "candidate_id": "mmpose-yoloxpose-s",
            "display_name": "MMPose YOLOX-Pose-S",
            "role": "permissive_open_license_challenger",
            "implementation": "official_mmpose_core",
            "task": "bottom_up_multi_person_pose",
            "keypoint_layout": "COCO17",
            "production_model_selected": False,
            "replaces_yolo26_selection": False,
        }
        and license_contract.get("spdx") == "Apache-2.0"
        and license_contract.get("repository_license_verified") is True
        and license_contract.get("does_not_authorize_yolo26") is True
        and license_contract.get("legal_product_approval_claimed") is False
        and acquisition.get("checkpoint_mode") == "0440"
        and acquisition.get("exact_checkpoint_acquired") is True
        and acquisition.get("checkpoint_integrity_verified") is True
        and structural.get("cpu_only") is True
        and structural.get("strict_state_load") is True
        and structural.get("state_tensor_count") == 547
        and structural.get("parameter_count") == 10729963
        and structural.get("raw_forward_batch") == 1
        and structural.get("raw_forward_profiles_verified") == [640, 960]
        and structural.get("real_image_inference_executed") is False
        and structural.get("quality_measured") is False
        and structural.get("performance_measured") is False
        and structural.get("gpu_executed") is False
        and profiles.get("640", {}).get("upstream_trained_resolution") is True
        and profiles.get("640", {}).get(
            "local_cpu_raw_shape_and_finite_verified"
        )
        is True
        and profiles.get("640", {}).get("local_cpu_raw_batch") == 1
        and profiles.get("640", {}).get("batch12_executed") is False
        and profiles.get("640", {}).get("quality_verified") is False
        and profiles.get("640", {}).get("onnx_exported") is False
        and profiles.get("640", {}).get("tensorrt_built") is False
        and profiles.get("640", {}).get("deepstream9_verified") is False
        and profiles.get("960", {}).get("upstream_trained_resolution") is False
        and profiles.get("960", {}).get("upstream_quality_claimed") is False
        and profiles.get("960", {}).get(
            "local_cpu_raw_shape_and_finite_verified"
        )
        is True
        and profiles.get("960", {}).get("feasibility_profile_only") is True
        and profiles.get("960", {}).get("batch12_executed") is False
        and profiles.get("960", {}).get("quality_verified") is False
        and profiles.get("960", {}).get("onnx_exported") is False
        and profiles.get("960", {}).get("tensorrt_built") is False
        and profiles.get("960", {}).get("deepstream9_verified") is False
        and upstream_metrics
        == {
            "source": "official_model_index",
            "dataset": "COCO val2017",
            "input_size": [640, 640],
            "ap": 0.641,
            "ap50": 0.872,
            "ap75": 0.702,
            "ar": 0.682,
            "ar50": 0.902,
            "locally_reproduced": False,
            "local_pck_result": False,
            "product_acceptance_evidence": False,
        }
        and onnx_plan.get("profile_specific_models") == [640, 960]
        and onnx_plan.get("spatial_dimensions_static_per_profile") is True
        and onnx_plan.get("dynamic_batch_target") is True
        and (
            onnx_plan.get("batch_min_target"),
            onnx_plan.get("batch_opt_target"),
            onnx_plan.get("batch_max_target"),
        )
        == (1, 12, 12)
        and onnx_plan.get("export_640_executed") is False
        and onnx_plan.get("export_960_executed") is False
        and onnx_plan.get("onnx_runtime_parity_executed") is False
        and onnx_plan.get("dynamic_batch12_verified") is False
        and tensorrt_plan.get("deepstream_version_target") == "9.0"
        and tensorrt_plan.get("precision_target") == "FP16"
        and tensorrt_plan.get("engine_640_built") is False
        and tensorrt_plan.get("engine_960_built") is False
        and tensorrt_plan.get("batch12_verified") is False
        and tensorrt_plan.get("parity_verified") is False
        and parser.get("existing_yolo26_end2end_adapter_compatible") is False
        and parser.get("custom_tensor_adapter_required") is True
        and parser.get("exact_onnx_output_contract_known") is False
        and parser.get("parser_implemented") is False
        and parser.get("parser_unit_tests_passed") is False
        and parser.get("deepstream9_parity_passed") is False
        and quality.get("keypoint_layout") == "COCO17"
        and quality.get("primary_local_metric") == "PCK@0.2"
        and quality.get("threshold") == 0.8
        and quality.get("profiles") == [640, 960]
        and quality.get("minimum_distinct_video_types") == 10
        and quality.get("required_view_types")
        == ["medium_close", "overhead_security_camera"]
        and quality.get("coco_val2017_can_replace_owner_site_gt") is False
        and quality.get("local_coco_gt_evaluation_executed") is False
        and quality.get("owner_site_gt_ready") is False
        and quality.get("pck_640_passed") is False
        and quality.get("pck_960_passed") is False
        and control.get("candidate_id") == "mmpose-rtmpose-control"
        and control.get("status")
        == "research_only_not_acquired_not_executed"
        and control.get("top_down_requires_person_detector") is True
        and control.get("checkpoint_acquired") is False
        and control.get("structural_load_executed") is False
        and control.get("export_executed") is False
        and control.get("quality_evaluated") is False
        and control.get("selected") is False
        and _pose_all_false(plan.get("gates"), expected_gate_keys)
        and receipt_valid
    )


def _payload_fingerprint_matches(
    value: Any, *, field: str, expected: str
) -> bool:
    if not isinstance(value, dict) or value.get(field) != expected:
        return False
    unsigned = dict(value)
    unsigned.pop(field, None)
    return _canonical_sha256(unsigned) == expected


def _pose_export_r10_failure_unavailable(
    reason: str, *, integrity: dict[str, bool]
) -> dict[str, Any]:
    return {
        "evidence_version": "r10",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "run_status": "unknown",
        "failure_cause_code": None,
        "onnx_640_published": False,
        "onnx_960_published": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "production_ready": False,
        "integrity": integrity,
    }


def _pose_export_r10_failure(reader: ArtifactReader) -> dict[str, Any]:
    integrity = {
        "plan_exact_pin_verified": False,
        "receipt_exact_pin_verified": False,
        "failure_log_exact_pin_verified": False,
        "plan_fingerprint_replayed": False,
        "receipt_fingerprint_replayed": False,
        "failure_log_shape_marker_verified": False,
        "cross_artifact_semantics_verified": False,
    }
    plan_pin = POSE_MMPOSE_EXPORT_R10_FAILURE_PINS["plan"]
    receipt_pin = POSE_MMPOSE_EXPORT_R10_FAILURE_PINS["receipt"]
    log_pin = POSE_MMPOSE_EXPORT_R10_FAILURE_PINS["failure_log"]
    plan_read, plan = _workspace_pin_json(
        reader,
        plan_pin,
        expected_path=plan_pin["path"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
    )
    receipt_read, receipt = _workspace_pin_json(
        reader,
        receipt_pin,
        expected_path=receipt_pin["path"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
    )
    log_read = _read_workspace_pin(
        reader,
        log_pin,
        expected_path=log_pin["path"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
        collect=True,
    )
    integrity["plan_exact_pin_verified"] = plan_read.available
    integrity["receipt_exact_pin_verified"] = receipt_read.available
    integrity["failure_log_exact_pin_verified"] = log_read.available
    for key, result, value in (
        ("plan", plan_read, plan),
        ("receipt", receipt_read, receipt),
    ):
        if value is None:
            return _pose_export_r10_failure_unavailable(
                f"pose_r10_{key}_{result.state}", integrity=integrity
            )
    if not log_read.available or log_read.content is None:
        return _pose_export_r10_failure_unavailable(
            f"pose_r10_failure_log_{log_read.state}", integrity=integrity
        )
    assert plan is not None and receipt is not None
    integrity["plan_fingerprint_replayed"] = _payload_fingerprint_matches(
        plan,
        field="plan_fingerprint_sha256",
        expected=POSE_MMPOSE_EXPORT_R10_PLAN_FINGERPRINT,
    )
    integrity["receipt_fingerprint_replayed"] = (
        _payload_fingerprint_matches(
            receipt,
            field="receipt_fingerprint_sha256",
            expected=POSE_MMPOSE_EXPORT_R10_RECEIPT_FINGERPRINT,
        )
    )
    try:
        log_text = log_read.content.decode("utf-8")
    except UnicodeDecodeError:
        log_text = ""
    integrity["failure_log_shape_marker_verified"] = bool(
        "ProfileExportError: dets shape differs" in log_text
    )
    profiles = receipt.get("profiles", {})
    execution = receipt.get("execution_boundary", {})
    conclusions = receipt.get("conclusions", {})
    integrity["cross_artifact_semantics_verified"] = bool(
        plan.get("schema_version")
        == "deepsafe.pose-mmpose-yoloxpose-onnx-export-plan/r10"
        and plan.get("candidate_id") == "mmpose-yoloxpose-s"
        and plan.get("status") == "planned_cpu_export_not_executed"
        and plan.get("execution_boundary", {}).get("planner_only") is True
        and plan.get("execution_boundary", {}).get("export_executed")
        is False
        and receipt.get("schema_version")
        == "deepsafe.pose-mmpose-yoloxpose-onnx-export-receipt/r10"
        and receipt.get("run_id") == "cpu-export-001"
        and receipt.get("status") == "failed"
        and receipt.get("plan_fingerprint_sha256")
        == POSE_MMPOSE_EXPORT_R10_PLAN_FINGERPRINT
        and receipt.get("error")
        == "ExportR10Error: profile 640 container exited 2"
        and profiles.get("640")
        == {
            "attempted": True,
            "docker_exit_code": 2,
            "dynamic_batch": [1, 12],
            "error": "profile container exited 2",
            "host_onnx_recheck": None,
            "receipt": None,
            "spatial_size": 640,
            "status": "failed",
        }
        and profiles.get("960")
        == {
            "attempted": False,
            "docker_exit_code": None,
            "dynamic_batch": [1, 12],
            "error": None,
            "host_onnx_recheck": None,
            "receipt": None,
            "spatial_size": 960,
            "status": "not_attempted",
        }
        and execution.get("container_runs_attempted") == 1
        and execution.get("gpu_exposed") is False
        and execution.get("gpu_api_queried") is False
        and execution.get("tensorrt_executed") is False
        and execution.get("deepstream_executed") is False
        and conclusions.get("both_profiles_passed") is False
        and conclusions.get("publishable_onnx_pair") is False
        and conclusions.get("tensorrt_verified") is False
        and conclusions.get("deepstream9_verified") is False
        and conclusions.get("production_ready") is False
    )
    if not all(integrity.values()):
        return _pose_export_r10_failure_unavailable(
            "pose_r10_failure_contract_invalid", integrity=integrity
        )
    return {
        "evidence_version": "r10",
        "available": True,
        "state": "failed_640_shape_contract_960_not_attempted",
        "reason": "dets_shape_differs",
        "run_id": "cpu-export-001",
        "run_status": "failed",
        "failure_cause_code": "dets_shape_differs",
        "failure_cause": "ProfileExportError: dets shape differs",
        "profiles": {
            "640": {"attempted": True, "status": "failed"},
            "960": {"attempted": False, "status": "not_attempted"},
        },
        "published_onnx_pair": False,
        "onnx_640_published": False,
        "onnx_960_published": False,
        "gpu_exposed": False,
        "gpu_queried": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "quality_measured": False,
        "production_ready": False,
        "integrity": integrity,
        "caveats": [
            "640 graph oluşturma koşusu sabit K=100 çıktı şekli beklentisinde fail-closed oldu; hiçbir production ONNX yayınlanmadı.",
            "960 profili denenmedi; TensorRT, DeepStream 9, PCK ve kapasite sonucu yoktur.",
        ],
    }


def _pose_shape_diagnostic_r11_unavailable(
    reason: str, *, integrity: dict[str, bool]
) -> dict[str, Any]:
    return {
        "evidence_version": "r11",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "shape_observed": False,
        "production_onnx_publishable": False,
        "existing_fixed_k_packer_compatible": False,
        "contract_change_authorized": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "production_ready": False,
        "integrity": integrity,
    }


def _pose_shape_diagnostic_r11(reader: ArtifactReader) -> dict[str, Any]:
    integrity = {
        **{
            f"{key}_exact_pin_verified": False
            for key in POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PINS
        },
        "plan_fingerprint_replayed": False,
        "receipt_fingerprint_replayed": False,
        "profile_fingerprint_replayed": False,
        "r10_failure_binding_verified": False,
        "receipt_profile_binding_verified": False,
        "dynamic_k_semantics_verified": False,
        "runtime_observation_verified": False,
        "claim_boundary_verified": False,
    }
    values: dict[str, dict[str, Any]] = {}
    for key, pin in POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PINS.items():
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=POSE_MAX_JSON_BYTES,
        )
        integrity[f"{key}_exact_pin_verified"] = result.available
        if value is None:
            return _pose_shape_diagnostic_r11_unavailable(
                f"pose_r11_{key}_{result.state}", integrity=integrity
            )
        values[key] = value
    plan = values["plan"]
    receipt = values["receipt"]
    profile = values["profile_receipt"]
    integrity["plan_fingerprint_replayed"] = _payload_fingerprint_matches(
        plan,
        field="plan_fingerprint_sha256",
        expected=POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PLAN_FINGERPRINT,
    )
    integrity["receipt_fingerprint_replayed"] = (
        _payload_fingerprint_matches(
            receipt,
            field="receipt_fingerprint_sha256",
            expected=POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_RECEIPT_FINGERPRINT,
        )
    )
    integrity["profile_fingerprint_replayed"] = (
        _payload_fingerprint_matches(
            profile,
            field="receipt_fingerprint_sha256",
            expected=POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PROFILE_FINGERPRINT,
        )
    )
    prerequisite = receipt.get("r10_failure_prerequisite", {})
    integrity["r10_failure_binding_verified"] = bool(
        prerequisite.get("valid") is True
        and prerequisite.get("failed_run_id") == "cpu-export-001"
        and prerequisite.get("failed_receipt_status") == "failed"
        and prerequisite.get("failed_receipt_file_sha256")
        == POSE_MMPOSE_EXPORT_R10_FAILURE_PINS["receipt"]["sha256"]
        and prerequisite.get("failed_receipt_fingerprint_sha256")
        == POSE_MMPOSE_EXPORT_R10_RECEIPT_FINGERPRINT
        and prerequisite.get("failed_log_sha256")
        == POSE_MMPOSE_EXPORT_R10_FAILURE_PINS["failure_log"]["sha256"]
        and prerequisite.get("shape_error_marker_observed") is True
        and prerequisite.get("topk_rewrite_marker_observed") is True
        and prerequisite.get("profile_640_attempted") is True
        and prerequisite.get("profile_960_attempted") is False
    )
    diagnostic = receipt.get("diagnostic", {})
    integrity["receipt_profile_binding_verified"] = bool(
        receipt.get("schema_version")
        == "deepsafe.pose-mmpose-yoloxpose-shape-diagnostic-receipt/r11"
        and receipt.get("run_id") == "cpu-shape-diag-001"
        and receipt.get("status") == "passed"
        and receipt.get("error") is None
        and receipt.get("plan_fingerprint_sha256")
        == POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PLAN_FINGERPRINT
        and diagnostic.get("attempted") is True
        and diagnostic.get("status") == "observed"
        and diagnostic.get("contract_change_authorized") is False
        and diagnostic.get("receipt")
        == {
            "bytes": POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PINS[
                "profile_receipt"
            ]["bytes"],
            "path": "diagnostic/shape-diagnostic-receipt.json",
            "sha256": POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PINS[
                "profile_receipt"
            ]["sha256"],
        }
        and profile.get("schema_version")
        == (
            "deepsafe.pose-mmpose-yoloxpose-"
            "shape-diagnostic-profile-receipt/r11"
        )
        and profile.get("run_id") == "cpu-shape-diag-001"
        and profile.get("profile") == 640
        and profile.get("status") == "observed"
        and profile.get("plan_fingerprint_sha256")
        == POSE_MMPOSE_SHAPE_DIAGNOSTIC_R11_PLAN_FINGERPRINT
    )
    classification = profile.get("classification", {})
    interface = classification.get("derived_shape_interface", {})
    evidence = classification.get("evidence", {})
    integrity["dynamic_k_semantics_verified"] = bool(
        classification.get("classification")
        == "other_shape_observation_fail_closed"
        and classification.get("contract_change_authorized") is False
        and classification.get("correction_plan_created") is False
        and classification.get("unexpected_shapes_fail_closed") is True
        and interface
        == {
            "dets": ["B", "K", 5],
            "input": ["B", 3, 640, 640],
            "k_common_to_outputs": True,
            "k_data_and_batch_dependent": True,
            "k_formula_from_pinned_source": "K=min(100,M+1)",
            "k_lower_bound": 1,
            "k_upper_bound": 100,
            "keypoints": ["B", "K", 17, 3],
            "raw_dim_param_name_assumed": False,
        }
        and evidence.get("dynamic_topk_100_semantics_observed") is True
        and evidence.get("shared_nms_topk_output_lineage_observed") is True
        and evidence.get("raw_instance_axis_symbolic_or_unknown") is True
        and evidence.get("runtime_batch1_and_batch12_static_k100") is False
        and evidence.get("runtime_batch1_and_batch12_bounded_dynamic_shapes_valid")
        is True
    )
    runtime = evidence.get("runtime_per_batch", {})
    integrity["runtime_observation_verified"] = bool(
        runtime.get("1", {}).get("dets_shape") == [1, 1, 5]
        and runtime.get("1", {}).get("keypoints_shape")
        == [1, 1, 17, 3]
        and runtime.get("1", {}).get("shared_k") is True
        and runtime.get("1", {}).get("finite") is True
        and runtime.get("12", {}).get("dets_shape") == [12, 1, 5]
        and runtime.get("12", {}).get("keypoints_shape")
        == [12, 1, 17, 3]
        and runtime.get("12", {}).get("shared_k") is True
        and runtime.get("12", {}).get("finite") is True
    )
    conclusions = receipt.get("conclusions", {})
    claims = profile.get("claims", {})
    publication = profile.get("publication", {})
    integrity["claim_boundary_verified"] = bool(
        plan.get("schema_version")
        == "deepsafe.pose-mmpose-yoloxpose-shape-diagnostic-plan/r11"
        and plan.get("candidate_id") == "mmpose-yoloxpose-s"
        and plan.get("status") == "planned_cpu_shape_diagnostic_not_executed"
        and conclusions
        == {
            "contract_change_authorized": False,
            "deepstream9_verified": False,
            "diagnostic_onnx_quarantined": True,
            "existing_fixed_k_packer_compatible": False,
            "production_model_selected": False,
            "production_onnx_publishable": False,
            "production_ready": False,
            "shape_observed": True,
            "tensorrt_verified": False,
        }
        and claims
        == {
            "contract_changed": False,
            "deepstream9_verified": False,
            "production_model_selected": False,
            "production_ready": False,
            "shape_observed": True,
            "tensorrt_verified": False,
        }
        and publication.get("quarantined_diagnostic_only") is True
        and publication.get("no_overwrite") is True
    )
    if not all(integrity.values()):
        return _pose_shape_diagnostic_r11_unavailable(
            "pose_r11_shape_diagnostic_contract_invalid",
            integrity=integrity,
        )
    effective = profile.get("post_processing_configuration", {}).get(
        "effective", {}
    )
    return {
        "evidence_version": "r11",
        "available": True,
        "state": "dynamic_k_shape_observed_diagnostic_quarantined",
        "reason": "production_contract_change_not_authorized",
        "run_id": "cpu-shape-diag-001",
        "shape_observed": True,
        "profile": 640,
        "derived_interface": {
            "input": ["B", 3, 640, 640],
            "dets": ["B", "K", 5],
            "keypoints": ["B", "K", 17, 3],
            "shared_k": True,
            "k_formula_from_pinned_source": "K=min(100,M+1)",
            "k_min": 1,
            "k_max": 100,
            "k_data_and_batch_dependent": True,
        },
        "runtime_blank_probe": {
            "batch_1": {
                "k": 1,
                "dets_shape": [1, 1, 5],
                "keypoints_shape": [1, 1, 17, 3],
            },
            "batch_12": {
                "k": 1,
                "dets_shape": [12, 1, 5],
                "keypoints_shape": [12, 1, 17, 3],
            },
            "all_outputs_finite": True,
            "runtime_k_variation_proven": False,
        },
        "postprocess": {
            "effective_score_threshold": effective.get("score_threshold"),
            "effective_iou_threshold": effective.get("iou_threshold"),
            "pre_top_k": effective.get("pre_top_k"),
            "keep_top_k": effective.get("keep_top_k"),
            "max_output_boxes_per_class": effective.get(
                "max_output_boxes_per_class"
            ),
        },
        "diagnostic_onnx_quarantined": True,
        "production_onnx_publishable": False,
        "existing_fixed_k_packer_compatible": False,
        "contract_change_authorized": False,
        "gpu_exposed": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "quality_measured": False,
        "capacity_measured": False,
        "production_ready": False,
        "integrity": integrity,
        "caveats": [
            "K=1 yalnız deterministik boş tensor probe'unda batch 1 ve 12 için ölçüldü; farklı girdilerde runtime K değişimi henüz gösterilmedi.",
            "K=1..100 ve K=min(100,M+1) pinned graph/source lineage türetimidir; mevcut sabit-K packer ile uyumlu kabul edilmedi.",
            "Diagnostic ONNX karantinadadır ve production export, TensorRT, DeepStream 9, PCK veya FPS kanıtı değildir.",
        ],
    }


def _pose_permission_probe_r9_unavailable(
    reason: str, *, integrity: dict[str, bool]
) -> dict[str, Any]:
    return {
        "evidence_version": "r9",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "export_environment_runtime_ready": False,
        "model_exported": False,
        "onnx_exported": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "production_ready": False,
        "integrity": integrity,
        "caveats": [
            "R9 exact-pin probe zinciri doğrulanamadı; export ortamı hazır ilan edilmedi.",
        ],
    }


def _pose_permission_probe_r9(reader: ArtifactReader) -> dict[str, Any]:
    integrity = {
        **{
            f"{key}_exact_pin_verified": False
            for key in POSE_MMPOSE_PERMISSION_PROBE_R9_PINS
        },
        "plan_payload_fingerprint_replayed": False,
        "attempt_payload_fingerprint_replayed": False,
        "probe_payload_fingerprint_replayed": False,
        "cross_artifact_bindings_verified": False,
        "plan_semantics_verified": False,
        "attempt_semantics_verified": False,
        "runtime_probe_semantics_verified": False,
    }
    values: dict[str, dict[str, Any]] = {}
    for key, pin in POSE_MMPOSE_PERMISSION_PROBE_R9_PINS.items():
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=str(pin["path"]),
            maximum_bytes=POSE_MAX_JSON_BYTES,
        )
        integrity[f"{key}_exact_pin_verified"] = result.available
        if value is None:
            return _pose_permission_probe_r9_unavailable(
                f"pose_r9_{key}_{result.state}", integrity=integrity
            )
        values[key] = value

    plan = values["plan"]
    attempt = values["attempt_receipt"]
    probe = values["probe_receipt"]
    integrity["plan_payload_fingerprint_replayed"] = (
        _payload_fingerprint_matches(
            plan,
            field="plan_sha256",
            expected=POSE_MMPOSE_PERMISSION_PROBE_R9_PLAN_PAYLOAD_SHA256,
        )
    )
    integrity["attempt_payload_fingerprint_replayed"] = (
        _payload_fingerprint_matches(
            attempt,
            field="receipt_payload_sha256",
            expected=POSE_MMPOSE_PERMISSION_PROBE_R9_ATTEMPT_PAYLOAD_SHA256,
        )
    )
    integrity["probe_payload_fingerprint_replayed"] = (
        _payload_fingerprint_matches(
            probe,
            field="receipt_payload_sha256",
            expected=POSE_MMPOSE_PERMISSION_PROBE_R9_PROBE_PAYLOAD_SHA256,
        )
    )
    probe_pin = attempt.get("controls", {}).get("probe_receipt", {})
    integrity["cross_artifact_bindings_verified"] = bool(
        attempt.get("plan_sha256")
        == POSE_MMPOSE_PERMISSION_PROBE_R9_PLAN_PAYLOAD_SHA256
        and probe.get("plan_sha256")
        == POSE_MMPOSE_PERMISSION_PROBE_R9_PLAN_PAYLOAD_SHA256
        and attempt.get("attempt_id") == "child-v8-probe-r9-001"
        and probe.get("attempt_id") == "child-v8-probe-r9-001"
        and probe_pin.get("bytes")
        == POSE_MMPOSE_PERMISSION_PROBE_R9_PINS["probe_receipt"]["bytes"]
        and probe_pin.get("sha256")
        == POSE_MMPOSE_PERMISSION_PROBE_R9_PINS["probe_receipt"]["sha256"]
        and probe_pin.get("path") == "probe/probe-receipt.json"
        and attempt.get("image_id") == POSE_MMPOSE_PERMISSION_PROBE_R9_IMAGE_ID
        and attempt.get("image", {}).get("image_id")
        == POSE_MMPOSE_PERMISSION_PROBE_R9_IMAGE_ID
        and plan.get("image", {}).get("image_id")
        == POSE_MMPOSE_PERMISSION_PROBE_R9_IMAGE_ID
    )
    plan_boundary = plan.get("execution_boundary", {})
    integrity["plan_semantics_verified"] = bool(
        plan.get("schema_version")
        == "deepsafe.pose-mmpose-permission-probe-plan/v1"
        and plan.get("candidate_id") == "mmpose-yoloxpose-s"
        and plan.get("status") == "planned_exact_image_probe_only"
        and plan.get("commands", {}).get("docker_pull") is None
        and plan.get("commands", {}).get("docker_build") is None
        and plan.get("image", {}).get("size") == 14917754987
        and plan.get("image", {}).get("layer_count") == 28
        and plan.get("image", {}).get("local_reference")
        == "deepsafe-mmpose-yoloxpose-export:child-v8-symlink-aware"
        and plan.get("image", {}).get("immutable_reference")
        == (
            "deepsafe-mmpose-yoloxpose-export@"
            "sha256:8ba836b80502277ce999ffb8b0c6a2c29368f09cb12cbe07abc10028e821915f"
        )
        and plan_boundary
        == {
            "planner_only": True,
            "image_pulled": False,
            "image_built": False,
            "container_run": False,
            "gpu_exposed": False,
            "gpu_api_queried": False,
            "gpu_compute_executed": False,
            "model_loaded": False,
            "model_exported": False,
            "tensorrt_executed": False,
            "deepstream_executed": False,
        }
    )
    conclusions = attempt.get("conclusions", {})
    attempt_boundary = attempt.get("execution_boundary", {})
    image_binding = attempt.get("image_binding", {})
    integrity["attempt_semantics_verified"] = bool(
        attempt.get("schema_version")
        == "deepsafe.pose-mmpose-permission-probe-attempt/v1"
        and attempt.get("status") == "passed"
        and attempt.get("stage_reached") == "complete"
        and attempt.get("error") is None
        and attempt.get("pull") == {"performed": False}
        and attempt.get("build") == {"performed": False}
        and conclusions.get("exact_image_binding_stable") is True
        and conclusions.get("export_environment_runtime_ready") is True
        and conclusions.get("image_rebuilt") is False
        and conclusions.get("model_exported") is False
        and conclusions.get("production_ready") is False
        and conclusions.get("rootless_mode_inventory_passed") is True
        and conclusions.get("runtime_probe_passed") is True
        and image_binding.get("stable") is True
        and image_binding.get("before") == image_binding.get("after")
        and attempt.get("post_fix_inventory", {}).get("exit_code") == 0
        and attempt.get("probe", {}).get("exit_code") == 0
        and attempt.get("probe", {}).get("network") == "none"
        and attempt.get("probe", {}).get("root_filesystem") == "read_only"
        and attempt.get("probe", {}).get("user") == "1000:1000"
        and attempt.get("probe", {}).get("gpu_exposed") is False
        and attempt.get("probe", {}).get("gpu_api_queried") is False
        and attempt_boundary
        == {
            "deepstream_executed": False,
            "gpu_api_queried": False,
            "gpu_compute_executed": False,
            "gpu_exposed": False,
            "model_exported": False,
            "model_loaded": False,
            "tensorrt_executed": False,
        }
    )
    environment = probe.get("environment", {})
    distributions = environment.get("distributions", {})
    probe_conclusions = probe.get("conclusions", {})
    probe_boundary = probe.get("execution_boundary", {})
    isolation = probe.get("isolation", {})
    mmcv = probe.get("mmcv", {})
    ort = probe.get("onnxruntime", {})
    integrity["runtime_probe_semantics_verified"] = bool(
        probe.get("schema_version")
        == "deepsafe.pose-mmpose-export-child-probe/v1"
        and probe.get("status") == "passed"
        and probe.get("phase") == "runtime"
        and distributions.get("mmcv") == "2.0.1"
        and distributions.get("mmdeploy") == "1.3.1"
        and distributions.get("mmpose") == "1.3.2"
        and distributions.get("torch") == "2.0.0+cu118"
        and distributions.get("onnx") == "1.14.1"
        and distributions.get("onnxruntime") == "1.15.1"
        and probe.get("pip_check", {}).get("passed") is True
        and mmcv.get("missing_ops") == []
        and mmcv.get("required_ops")
        == ["nms", "batched_nms", "roi_align", "deform_conv2d"]
        and mmcv.get("mmdeploy_yoloxpose_rewrite_imported") is True
        and mmcv.get("yoloxpose_head_imported") is True
        and mmcv.get("cpu_execution") is True
        and ort.get("selected_providers") == ["CPUExecutionProvider"]
        and ort.get("onnx_checker_passed") is True
        and ort.get("cpu_execution") is True
        and isolation.get("effective_uid") == 1000
        and isolation.get("non_root") is True
        and isolation.get("network_none_observed") is True
        and isolation.get("root_read_only") is True
        and isolation.get("gpu_device_nodes") == []
        and isolation.get("gpu_api_query_executed") is False
        and isolation.get("gpu_compute_executed") is False
        and probe_conclusions.get("compiled_mmcv_ext_imported") is True
        and probe_conclusions.get("exact_source_wheels_installed") is True
        and probe_conclusions.get("model_exported") is False
        and probe_conclusions.get("production_ready") is False
        and probe_boundary
        == {
            "deepstream_executed": False,
            "gpu_api_query_executed": False,
            "gpu_compute_executed": False,
            "model_exported": False,
            "model_loaded": False,
            "tensorrt_executed": False,
        }
    )
    if not all(integrity.values()):
        return _pose_permission_probe_r9_unavailable(
            "pose_r9_cross_artifact_contract_invalid", integrity=integrity
        )
    return {
        "evidence_version": "r9",
        "available": True,
        "state": "export_environment_runtime_ready_model_not_exported",
        "reason": "model_export_not_started",
        "candidate": "MMPose YOLOX-Pose-S",
        "export_environment_runtime_ready": True,
        "exact_image": {
            "image_version": "child-v8-symlink-aware",
            "exact_immutable_identity_verified": True,
            "size_bytes": 14917754987,
            "layer_count": 28,
            "binding_stable_before_after_probe": True,
            "rebuilt_during_r9": False,
        },
        "runtime": {
            "python": "3.8.10",
            "mmdeploy": "1.3.1",
            "mmpose": "1.3.2",
            "mmcv": "2.0.1",
            "torch": "2.0.0+cu118",
            "onnx": "1.14.1",
            "onnxruntime": "1.15.1",
            "compiled_mmcv_ops_ready": True,
            "mmdeploy_yoloxpose_rewrite_ready": True,
            "onnxruntime_cpu_probe_passed": True,
        },
        "isolation": {
            "network": "none",
            "root_filesystem": "read_only",
            "non_root_uid": 1000,
            "gpu_exposed": False,
            "gpu_api_queried": False,
        },
        "model_loaded": False,
        "model_exported": False,
        "onnx_640_exported": False,
        "onnx_960_exported": False,
        "dynamic_batch12_verified": False,
        "tensorrt_executed": False,
        "deepstream9_executed": False,
        "quality_measured": False,
        "capacity_measured": False,
        "production_ready": False,
        "integrity": integrity,
        "caveats": [
            "R9 yalnız exact image ve CPU export-runtime probe kanıtıdır; model checkpoint'i bu koşuda yüklenmedi.",
            "Runtime hazır olması ONNX export, TensorRT engine, DeepStream 9 paritesi, PCK kalitesi veya FPS kanıtı değildir.",
        ],
    }


def _pose_readiness(reader: ArtifactReader) -> dict[str, Any]:
    """Project exact pose preparation while refusing result discovery.

    The provenance plan is the export trust anchor.  Its two profile plans,
    shared semantic contract and implementation pins are descriptor-bound.
    Separate exact pins bind the diagnostic GT-source review and CPU-only PCK
    evaluator contracts.  No model/artifact/result directory is scanned, so
    an unpinned weight, ONNX, engine, prediction or receipt cannot promote a
    readiness gate.
    """

    provenance_read, provenance = _workspace_pin_json(
        reader,
        POSE_EXPORT_PROVENANCE_PIN,
        expected_path=POSE_EXPORT_PROVENANCE_PIN["path"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
    )
    integrity: dict[str, bool] = {
        "provenance_plan_verified": provenance_read.available,
        "semantic_contract_verified": False,
        "export_plan_640_verified": False,
        "export_plan_960_verified": False,
        "export_harness_verified": False,
        "export_wrapper_verified": False,
        "onnx_validator_verified": False,
        "gt_source_manifest_verified": False,
        "gt_source_schema_verified": False,
        "gt_source_validator_verified": False,
        "gt_source_schema_replay_verified": False,
        "pck_evaluator_verified": False,
        "pck_ground_truth_schema_verified": False,
        "pck_predictions_schema_verified": False,
        "pck_receipt_schema_verified": False,
        "owner_acceptance_policy_verified": False,
        "permissive_challenger_plan_verified": False,
        "permissive_challenger_checkpoint_verified": False,
        "permissive_challenger_receipt_verified": False,
        "permissive_challenger_validator_verified": False,
        "permissive_challenger_schema_verified": False,
        "permissive_challenger_schema_replay_verified": False,
        "permissive_challenger_semantics_verified": False,
        "mmpose_onnx_preflight_schema_verified": False,
        "mmpose_onnx_preflight_historical_r1_verified": False,
        "mmpose_onnx_preflight_current_r2_verified": False,
        "mmpose_onnx_preflight_historical_r1_schema_replay_verified": False,
        "mmpose_onnx_preflight_current_r2_schema_replay_verified": False,
        "mmpose_onnx_preflight_historical_r1_semantics_verified": False,
        "mmpose_onnx_preflight_current_r2_semantics_verified": False,
        "cross_artifact_semantics_verified": False,
    }
    if provenance is None:
        return _pose_readiness_unavailable(
            f"provenance_plan_{provenance_read.state}", integrity=integrity
        )

    try:
        semantic_pin = _person_pin_core(
            provenance["shared_semantic_contract"]
        )
        frozen_plans = provenance["frozen_export_plans"]
        plan_pins = {
            "640": _person_pin_core(frozen_plans["640"]),
            "960": _person_pin_core(frozen_plans["960"]),
        }
        implementation = provenance["runtime"]["implementation"]
        implementation_pins = {
            "harness": _person_pin_core(implementation["harness"]),
            "wrapper": _person_pin_core(implementation["wrapper"]),
            "onnx_validator": _person_pin_core(
                implementation["onnx_validator"]
            ),
        }
        challenger_descriptor = provenance["permissive_challenger"]
        challenger_plan_descriptor = challenger_descriptor["plan"]
        challenger_plan_pin = _person_pin_core(
            challenger_plan_descriptor
        )
    except (KeyError, TypeError):
        return _pose_readiness_unavailable(
            "provenance_plan_contract_invalid", integrity=integrity
        )

    expected_paths = {
        "semantic": POSE_EXPORT_PATHS["semantic_contract"],
        "640": POSE_EXPORT_PATHS["plan_640"],
        "960": POSE_EXPORT_PATHS["plan_960"],
        **{
            key: POSE_EXPORT_PATHS[key]
            for key in ("harness", "wrapper", "onnx_validator")
        },
    }
    pin_paths_valid = bool(
        semantic_pin is not None
        and semantic_pin.get("path") == expected_paths["semantic"]
        and all(
            plan_pins[profile] is not None
            and plan_pins[profile].get("path") == expected_paths[profile]
            for profile in ("640", "960")
        )
        and all(
            implementation_pins[key] is not None
            and implementation_pins[key].get("path") == expected_paths[key]
            for key in ("harness", "wrapper", "onnx_validator")
        )
        and challenger_plan_pin is not None
        and challenger_plan_pin.get("path")
        == POSE_PERMISSIVE_CHALLENGER_PATHS["plan"]
    )
    if not pin_paths_valid:
        return _pose_readiness_unavailable(
            "provenance_dependent_pin_invalid", integrity=integrity
        )

    assert semantic_pin is not None
    semantic_read, semantic_contract = _workspace_pin_json(
        reader,
        semantic_pin,
        expected_path=expected_paths["semantic"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
    )
    integrity["semantic_contract_verified"] = semantic_read.available
    plans: dict[str, dict[str, Any] | None] = {}
    plan_reads: dict[str, WorkspacePinRead] = {}
    for profile in ("640", "960"):
        pin = plan_pins[profile]
        assert pin is not None
        plan_read, plan = _workspace_pin_json(
            reader,
            pin,
            expected_path=expected_paths[profile],
            maximum_bytes=POSE_MAX_JSON_BYTES,
        )
        plan_reads[profile] = plan_read
        plans[profile] = plan
        integrity[f"export_plan_{profile}_verified"] = plan_read.available

    implementation_reads: dict[str, WorkspacePinRead] = {}
    for key in ("harness", "wrapper", "onnx_validator"):
        pin = implementation_pins[key]
        assert pin is not None
        result = _read_workspace_pin(
            reader,
            pin,
            expected_path=expected_paths[key],
            maximum_bytes=POSE_MAX_JSON_BYTES,
            collect=False,
        )
        implementation_reads[key] = result
        integrity[f"export_{key}_verified"] = result.available
    # Preserve the public integrity key used by the UI for the ONNX validator.
    integrity["onnx_validator_verified"] = integrity.pop(
        "export_onnx_validator_verified"
    )

    assert challenger_plan_pin is not None
    challenger_plan_read, challenger_plan = _workspace_pin_json(
        reader,
        challenger_plan_pin,
        expected_path=POSE_PERMISSIVE_CHALLENGER_PATHS["plan"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
    )
    integrity["permissive_challenger_plan_verified"] = (
        challenger_plan_read.available
    )
    if challenger_plan is None:
        return _pose_readiness_unavailable(
            f"permissive_challenger_plan_{challenger_plan_read.state}",
            integrity=integrity,
        )
    try:
        challenger_acquisition = challenger_plan["acquisition"]
        challenger_structural = challenger_plan["structural_evidence"]
        challenger_checkpoint_pin = _person_pin_core(
            challenger_acquisition["checkpoint"]
        )
        challenger_receipt_descriptor = challenger_structural["receipt"]
        challenger_receipt_pin = _person_pin_core(
            challenger_receipt_descriptor
        )
        challenger_validator_pin = _person_pin_core(
            challenger_structural["validator"]
        )
        challenger_schema_pin = _person_pin_core(
            challenger_structural["schema"]
        )
    except (KeyError, TypeError):
        return _pose_readiness_unavailable(
            "permissive_challenger_contract_invalid", integrity=integrity
        )
    challenger_dependent_pins = {
        "checkpoint": challenger_checkpoint_pin,
        "receipt": challenger_receipt_pin,
        "validator": challenger_validator_pin,
        "schema": challenger_schema_pin,
    }
    if not all(
        pin is not None
        and pin.get("path") == POSE_PERMISSIVE_CHALLENGER_PATHS[key]
        for key, pin in challenger_dependent_pins.items()
    ):
        return _pose_readiness_unavailable(
            "permissive_challenger_dependent_pin_invalid",
            integrity=integrity,
        )
    assert challenger_checkpoint_pin is not None
    assert challenger_receipt_pin is not None
    assert challenger_validator_pin is not None
    assert challenger_schema_pin is not None
    challenger_checkpoint_read = _read_workspace_pin(
        reader,
        challenger_checkpoint_pin,
        expected_path=POSE_PERMISSIVE_CHALLENGER_PATHS["checkpoint"],
        maximum_bytes=POSE_MODEL_MAX_BYTES,
        collect=False,
    )
    integrity["permissive_challenger_checkpoint_verified"] = (
        challenger_checkpoint_read.available
    )
    challenger_receipt_read, challenger_receipt = _workspace_pin_json(
        reader,
        challenger_receipt_pin,
        expected_path=POSE_PERMISSIVE_CHALLENGER_PATHS["receipt"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
    )
    integrity["permissive_challenger_receipt_verified"] = (
        challenger_receipt_read.available
    )
    challenger_validator_read = _read_workspace_pin(
        reader,
        challenger_validator_pin,
        expected_path=POSE_PERMISSIVE_CHALLENGER_PATHS["validator"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
        collect=False,
    )
    integrity["permissive_challenger_validator_verified"] = (
        challenger_validator_read.available
    )
    challenger_schema_read, challenger_schema = _workspace_pin_json(
        reader,
        challenger_schema_pin,
        expected_path=POSE_PERMISSIVE_CHALLENGER_PATHS["schema"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
    )
    integrity["permissive_challenger_schema_verified"] = (
        challenger_schema_read.available
    )

    onnx_preflight_values: dict[str, dict[str, Any] | None] = {}
    onnx_preflight_reads: dict[str, WorkspacePinRead] = {}
    for key in ("historical_r1", "current_r2", "schema"):
        pin = POSE_MMPOSE_ONNX_PREFLIGHT_PINS[key]
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=POSE_MAX_JSON_BYTES,
        )
        onnx_preflight_reads[key] = result
        onnx_preflight_values[key] = value
        integrity_key = (
            "mmpose_onnx_preflight_schema_verified"
            if key == "schema"
            else f"mmpose_onnx_preflight_{key}_verified"
        )
        integrity[integrity_key] = result.available

    gt_values: dict[str, dict[str, Any] | None] = {}
    gt_reads: dict[str, WorkspacePinRead] = {}
    for key in ("source_manifest", "source_schema"):
        pin = POSE_GT_EVIDENCE_PINS[key]
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=POSE_MAX_JSON_BYTES,
        )
        gt_reads[key] = result
        gt_values[key] = value
        integrity[f"gt_{key}_verified"] = result.available
    source_validator_pin = POSE_GT_EVIDENCE_PINS["source_validator"]
    source_validator_read = _read_workspace_pin(
        reader,
        source_validator_pin,
        expected_path=source_validator_pin["path"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
        collect=False,
    )
    integrity["gt_source_validator_verified"] = source_validator_read.available

    pck_values: dict[str, dict[str, Any] | None] = {}
    pck_reads: dict[str, WorkspacePinRead] = {}
    for key in (
        "ground_truth_schema",
        "predictions_schema",
        "receipt_schema",
    ):
        pin = POSE_PCK_EVIDENCE_PINS[key]
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=POSE_MAX_JSON_BYTES,
        )
        pck_reads[key] = result
        pck_values[key] = value
        integrity[f"pck_{key}_verified"] = result.available
    evaluator_pin = POSE_PCK_EVIDENCE_PINS["evaluator"]
    evaluator_read = _read_workspace_pin(
        reader,
        evaluator_pin,
        expected_path=evaluator_pin["path"],
        maximum_bytes=POSE_MAX_JSON_BYTES,
        collect=False,
    )
    integrity["pck_evaluator_verified"] = evaluator_read.available

    try:
        owner_policy = load_approved_policy(reader.workspace_root)
    except (AcceptancePolicyError, OSError, RuntimeError, ValueError):
        owner_policy = None
    integrity["owner_acceptance_policy_verified"] = owner_policy is not None

    read_states = {
        "semantic_contract": semantic_read,
        "export_plan_640": plan_reads["640"],
        "export_plan_960": plan_reads["960"],
        "export_harness": implementation_reads["harness"],
        "export_wrapper": implementation_reads["wrapper"],
        "onnx_validator": implementation_reads["onnx_validator"],
        "permissive_challenger_plan": challenger_plan_read,
        "permissive_challenger_checkpoint": challenger_checkpoint_read,
        "permissive_challenger_receipt": challenger_receipt_read,
        "permissive_challenger_validator": challenger_validator_read,
        "permissive_challenger_schema": challenger_schema_read,
        "mmpose_onnx_preflight_schema": onnx_preflight_reads["schema"],
        "mmpose_onnx_preflight_historical_r1": onnx_preflight_reads[
            "historical_r1"
        ],
        "mmpose_onnx_preflight_current_r2": onnx_preflight_reads[
            "current_r2"
        ],
        "gt_source_manifest": gt_reads["source_manifest"],
        "gt_source_schema": gt_reads["source_schema"],
        "gt_source_validator": source_validator_read,
        "pck_evaluator": evaluator_read,
        **{
            f"pck_{key}": value for key, value in pck_reads.items()
        },
    }
    if any(not result.available for result in read_states.values()):
        failed_key, failed_read = next(
            (key, value)
            for key, value in read_states.items()
            if not value.available
        )
        return _pose_readiness_unavailable(
            f"{failed_key}_{failed_read.state}", integrity=integrity
        )
    if owner_policy is None:
        return _pose_readiness_unavailable(
            "owner_acceptance_policy_invalid", integrity=integrity
        )
    if (
        semantic_contract is None
        or challenger_receipt is None
        or challenger_schema is None
        or any(
            value is None for value in onnx_preflight_values.values()
        )
        or any(plan is None for plan in plans.values())
        or any(value is None for value in gt_values.values())
        or any(value is None for value in pck_values.values())
    ):
        return _pose_readiness_unavailable(
            "pose_json_contract_invalid", integrity=integrity
        )

    source_manifest = gt_values["source_manifest"]
    source_schema = gt_values["source_schema"]
    assert source_manifest is not None
    assert source_schema is not None
    source_schema_valid = _pose_schema_identity(
        source_schema,
        schema_id=(
            "https://deepsafe.local/schemas/"
            "pose-gt-evaluation-sources-v1.schema.json"
        ),
    )
    try:
        _validate_schema_node(source_manifest, source_schema, source_schema)
    except (TypeError, ValueError, RecursionError):
        source_schema_valid = False
    integrity["gt_source_schema_replay_verified"] = source_schema_valid

    pck_schema_valid = bool(
        _pose_schema_identity(
            pck_values["ground_truth_schema"],
            schema_id=(
                "https://deepsafe.local/schemas/"
                "pose-pck-ground-truth-v1.schema.json"
            ),
        )
        and _pose_schema_identity(
            pck_values["predictions_schema"],
            schema_id=(
                "https://deepsafe.local/schemas/"
                "pose-pck-predictions-v1.schema.json"
            ),
        )
        and _pose_schema_identity(
            pck_values["receipt_schema"],
            schema_id=(
                "https://deepsafe.local/schemas/"
                "pose-pck-evaluation-receipt-v1.schema.json"
            ),
        )
    )
    challenger_schema_valid = _pose_schema_identity(
        challenger_schema,
        schema_id=(
            "https://deepsafe.local/schemas/"
            "pose-mmpose-yoloxpose-structural-receipt-v1.schema.json"
        ),
    )
    try:
        _validate_schema_node(
            challenger_receipt,
            challenger_schema,
            challenger_schema,
        )
    except (TypeError, ValueError, RecursionError):
        challenger_schema_valid = False
    integrity["permissive_challenger_schema_replay_verified"] = (
        challenger_schema_valid
    )
    onnx_preflight_schema = onnx_preflight_values["schema"]
    assert onnx_preflight_schema is not None
    onnx_preflight_schema_valid = _pose_schema_identity(
        onnx_preflight_schema,
        schema_id=(
            "https://deepsafe.local/schemas/"
            "pose-mmpose-yoloxpose-onnx-lane-receipt-v1.schema.json"
        ),
    )
    for key in ("historical_r1", "current_r2"):
        receipt = onnx_preflight_values[key]
        assert receipt is not None
        receipt_schema_valid = onnx_preflight_schema_valid
        try:
            _validate_schema_node(
                receipt,
                onnx_preflight_schema,
                onnx_preflight_schema,
            )
        except (TypeError, ValueError, RecursionError):
            receipt_schema_valid = False
        integrity[
            f"mmpose_onnx_preflight_{key}_schema_replay_verified"
        ] = receipt_schema_valid
        onnx_preflight_schema_valid = bool(
            onnx_preflight_schema_valid and receipt_schema_valid
        )
    ground_truth_defs = pck_values["ground_truth_schema"].get("$defs", {})
    prediction_defs = pck_values["predictions_schema"].get("$defs", {})
    receipt_properties = pck_values["receipt_schema"].get("properties", {})
    pck_schema_valid = bool(
        pck_schema_valid
        and ground_truth_defs.get("manifestBase", {})
        .get("properties", {})
        .get("schema_version", {})
        .get("const")
        == "deepsafe.pose-pck-ground-truth/v1"
        and prediction_defs.get("manifestBase", {})
        .get("properties", {})
        .get("schema_version", {})
        .get("const")
        == "deepsafe.pose-pck-predictions/v1"
        and receipt_properties.get("schema_version", {}).get("const")
        == "deepsafe.pose-pck-evaluation-receipt/v1"
        and receipt_properties.get("evaluator_contract", {}).get("const")
        == "deepsafe.pose-pck-evaluator/coco17-pck0.2-v1"
        and receipt_properties.get("scope", {}).get("const")
        == "pose_pck_quality_only"
        and receipt_properties.get("product_acceptance_claimed", {}).get(
            "const"
        )
        is False
    )
    if not (
        source_schema_valid
        and pck_schema_valid
        and challenger_schema_valid
        and onnx_preflight_schema_valid
    ):
        return _pose_readiness_unavailable(
            "pose_schema_contract_invalid", integrity=integrity
        )

    expected_readiness_keys = {
        "model_ready",
        "onnx_parity_passed",
        "tensorrt_engine_ready",
        "tensorrt_parity_passed",
        "deepstream9_parser_parity_passed",
        "independent_pck_passed",
        "twelve_camera_capacity_passed",
        "full_stack_passed",
        "production_ready",
    }
    expected_acceptance_keys = {
        "quality_passed",
        "profile_640_passed",
        "profile_960_passed",
        "capacity_passed",
        "endurance_passed",
        "product_acceptance_passed",
    }
    artifact_state = provenance.get("artifact_state")
    license_contract = (
        provenance.get("license")
        if isinstance(provenance.get("license"), dict)
        else {}
    )
    runtime = (
        provenance.get("runtime")
        if isinstance(provenance.get("runtime"), dict)
        else {}
    )
    selected_model = provenance.get("selected_model")
    provenance_valid = bool(
        provenance.get("schema_version")
        == "deepsafe.pose-export-provenance-plan/v2"
        and provenance.get("status")
        == "frozen_640_960_plans_prepared_license_and_export_not_started"
        and isinstance(selected_model, dict)
        and selected_model.get("model_id") == "yolo26s-pose"
        and selected_model.get("bytes") == 24151790
        and re.fullmatch(
            r"[0-9a-f]{64}", str(selected_model.get("sha256", ""))
        )
        is not None
        and challenger_descriptor
        == {
            "candidate_id": "mmpose-yoloxpose-s",
            "role": (
                "separate_permissive_challenger_does_not_change_"
                "selected_model"
            ),
            "production_model_selected": False,
            "plan": {
                "path": POSE_PERMISSIVE_CHALLENGER_PATHS["plan"],
                "bytes": 9010,
                "sha256": (
                    "69ccbe115f250c5e176f68b7d95dc650b3f58a39fbfa93a06e2343d7fb4a5e4e"
                ),
                "fingerprint_sha256": (
                    "13ba0164871a1e8b20c8c5156eb940aac29c2e7829454ab73caf44f486c84993"
                ),
            },
        }
        and artifact_state
        == {
            "weights_acquired": False,
            "profile_640_onnx_exported": False,
            "profile_960_onnx_exported": False,
            "profile_640_engine_built": False,
            "profile_960_engine_built": False,
            "export_executed": False,
            "docker_executed": False,
            "network_executed": False,
            "gpu_executed": False,
        }
        and license_contract.get("options")
        == ["AGPL-3.0-compatible-project", "Ultralytics-Enterprise"]
        and license_contract.get("decision") is None
        and license_contract.get("download_authorized") is False
        and license_contract.get("export_authorized") is False
        and _pose_all_false(
            provenance.get("readiness"), expected_readiness_keys
        )
        and _pose_all_false(
            provenance.get("acceptance"), expected_acceptance_keys
        )
        and runtime.get("ultralytics_version") == "8.4.99"
        and runtime.get("device") == "cpu"
        and runtime.get("gpu_exposed_to_export_container") is False
        and runtime.get("onnx_opset") == 18
    )

    semantic_keypoints = (
        semantic_contract.get("keypoints")
        if isinstance(semantic_contract.get("keypoints"), dict)
        else {}
    )
    tensor_contract = (
        semantic_contract.get("tensor_contract")
        if isinstance(semantic_contract.get("tensor_contract"), dict)
        else {}
    )
    spatial_profiles = (
        semantic_contract.get("spatial_profiles")
        if isinstance(semantic_contract.get("spatial_profiles"), dict)
        else {}
    )
    tensorrt_profile = (
        semantic_contract.get("tensorrt_profile")
        if isinstance(semantic_contract.get("tensorrt_profile"), dict)
        else {}
    )
    semantic_valid = bool(
        semantic_contract.get("schema_version")
        == "deepsafe.pose-semantic-model-contract/v1"
        and semantic_contract.get("contract_id")
        == "yolo26-coco17-end2end-v1"
        and semantic_contract.get("task") == "pose"
        and semantic_contract.get("model_id") == "yolo26s-pose"
        and semantic_contract.get("classes")
        == [{"id": 0, "name": "person"}]
        and semantic_keypoints.get("layout") == "COCO17"
        and semantic_keypoints.get("shape") == [17, 3]
        and semantic_keypoints.get("names")
        == list(POSE_COCO17_KEYPOINTS)
        and tensor_contract.get("dynamic_batch") is True
        and tensor_contract.get("output_shape") == ["B", 300, 57]
        and tensor_contract.get("end_to_end") is True
        and tensor_contract.get("nms_required") is False
        and set(spatial_profiles) == {"640", "960"}
        and tensorrt_profile.get("batch_min") == 1
        and tensorrt_profile.get("batch_opt") == 12
        and tensorrt_profile.get("batch_max") == 12
        and tensorrt_profile.get("precision") == "FP16"
    )

    plan_semantics_valid = True
    for profile, expected_size in (("640", 640), ("960", 960)):
        plan = plans[profile]
        assert plan is not None
        frozen_pin = frozen_plans[profile]
        export_contract = (
            plan.get("export")
            if isinstance(plan.get("export"), dict)
            else {}
        )
        plan_semantics_valid = bool(
            plan_semantics_valid
            and plan.get("schema_version")
            == "deepsafe.yolo26-pose-export-plan/v2"
            and plan.get("status") == "planned_license_required_not_executed"
            and _self_fingerprint_matches(plan)
            and plan.get("fingerprint_sha256")
            == frozen_pin.get("fingerprint_sha256")
            and _person_pin_core(plan.get("shared_semantic_contract"))
            == semantic_pin
            and plan.get("model") == selected_model
            and plan.get("profile")
            == {
                "name": profile,
                "height": expected_size,
                "width": expected_size,
            }
            and export_contract.get("imgsz") == expected_size
            and export_contract.get("batch") == 12
            and export_contract.get("dynamic_batch") is True
            and export_contract.get("end2end") is True
            and export_contract.get("device") == "cpu"
            and export_contract.get("gpu_exposed_to_container") is False
            and plan.get("license_gate")
            == {
                "required_for_execute": True,
                "allowed_bases": ["agpl-3.0", "enterprise"],
                "decision": None,
                "download_authorized": False,
                "export_authorized": False,
            }
            and _pose_all_false(
                plan.get("readiness"), expected_readiness_keys
            )
            and _pose_all_false(
                plan.get("acceptance"), expected_acceptance_keys
            )
            and plan.get("acceptance_effect")
            == "none_export_is_not_evaluation"
        )

    source_by_id = {
        source.get("id"): source
        for source in source_manifest.get("sources", [])
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    }
    subset = (
        source_manifest.get("selected_subset_plan")
        if isinstance(source_manifest.get("selected_subset_plan"), dict)
        else {}
    )
    subset_readiness = (
        subset.get("readiness")
        if isinstance(subset.get("readiness"), dict)
        else {}
    )
    summary = (
        subset.get("summary")
        if isinstance(subset.get("summary"), dict)
        else {}
    )
    guardrails = (
        source_manifest.get("global_guardrails")
        if isinstance(source_manifest.get("global_guardrails"), dict)
        else {}
    )
    source_valid = bool(
        source_manifest.get("schema_version")
        == "deepsafe.pose-gt-evaluation-sources/v1"
        and source_manifest.get("status")
        == "source_review_complete_small_coco_plan_diagnostic_only_no_production_acceptance"
        and _self_fingerprint_matches(source_manifest)
        and source_manifest.get("fingerprint_sha256")
        == POSE_GT_SOURCE_FINGERPRINT
        and tuple(source_by_id)
        == (
            "coco_2017_keypoints_val",
            "jrdb_pose_2022",
            "jta",
            "mpii_human_pose_v1",
        )
        and all(
            source.get("eligibility", {}).get(
                "pose_pck_product_acceptance"
            )
            is False
            and source.get("eligibility", {}).get("commercial_model_training")
            is False
            and source.get("eligibility", {}).get("exact_25m") is False
            for source in source_by_id.values()
        )
        and subset.get("id")
        == "coco2017-val-license7-visible-pose-3-v1"
        and subset.get("status")
        == "remote_hash_pinned_plan_not_materialized_not_acceptance_evidence"
        and subset.get("purpose")
        == "CPU evaluator/schema/coordinate-contract diagnostic only"
        and summary.get("selected_images") == 3
        and summary.get("selected_person_annotations") == 3
        and summary.get("visible_keypoints") == 38
        and summary.get("video_sequences") == 0
        and summary.get("track_ids") == 0
        and summary.get("overhead_or_security_views") == 0
        and subset_readiness.get("exact_coco17_verified") is True
        and subset_readiness.get("independent_project_human_review_complete")
        is False
        and subset_readiness.get("security_or_overhead_coverage_complete")
        is False
        and subset_readiness.get("minimum_gt_coverage_plan_complete") is False
        and subset_readiness.get("commercial_or_closed_product_rights_cleared")
        is False
        and subset_readiness.get("eligible_for_pose_pck_product_acceptance")
        is False
        and guardrails.get(
            "pose_pck_pass_claim_forbidden_without_new_predictions_and_independent_review"
        )
        is True
        and guardrails.get(
            "final_acceptance_requires_authorized_site_video_with_independent_human_review"
        )
        is True
    )

    pose_policy = owner_policy.get("quality_thresholds", {}).get("pose")
    policy_scope = owner_policy.get("scope", {})
    policy_valid = bool(
        owner_policy.get("pre_run_fingerprint_sha256")
        == APPROVED_POLICY_FINGERPRINT_SHA256
        and pose_policy
        == {
            "scope": "each_profile_visible_ground_truth_keypoints_only",
            "metric": "PCK",
            "threshold_radius": 0.20,
            "normalization": "ground_truth_person_bbox_max_dimension",
            "operator": "gte",
            "threshold": 0.80,
        }
        and policy_scope.get("profiles") == [640, 960]
        and policy_scope.get("camera_count") == 12
        and policy_scope.get("minimum_distinct_video_types") == 10
        and policy_scope.get("required_view_types")
        == ["medium_close", "overhead_security_camera"]
    )
    challenger_valid = _pose_mmpose_challenger_semantics_valid(
        challenger_plan,
        challenger_receipt,
        plan_pin=challenger_plan_descriptor,
        receipt_schema_valid=challenger_schema_valid,
    )
    integrity["permissive_challenger_semantics_verified"] = (
        challenger_valid
    )
    onnx_preflight_semantics: dict[str, bool] = {}
    for key, checkout_verified in (
        ("historical_r1", False),
        ("current_r2", True),
    ):
        receipt = onnx_preflight_values[key]
        assert receipt is not None
        semantic_valid = _pose_mmpose_onnx_preflight_semantics_valid(
            receipt,
            expected_self_sha256=(
                POSE_MMPOSE_ONNX_PREFLIGHT_SELF_SHA256[key]
            ),
            expected_blockers=POSE_MMPOSE_ONNX_PREFLIGHT_BLOCKERS[key],
            expected_checkout_verified=checkout_verified,
            schema_valid=integrity[
                f"mmpose_onnx_preflight_{key}_schema_replay_verified"
            ],
        )
        onnx_preflight_semantics[key] = semantic_valid
        integrity[
            f"mmpose_onnx_preflight_{key}_semantics_verified"
        ] = semantic_valid

    if not (
        provenance_valid
        and semantic_valid
        and plan_semantics_valid
        and source_valid
        and policy_valid
        and challenger_valid
        and all(onnx_preflight_semantics.values())
    ):
        return _pose_readiness_unavailable(
            "pose_cross_artifact_contract_invalid", integrity=integrity
        )
    integrity["cross_artifact_semantics_verified"] = True
    export_attempt_r10 = _pose_export_r10_failure(reader)
    shape_diagnostic_r11 = _pose_shape_diagnostic_r11(reader)

    return {
        "label": "Pose modeli hazırlığı",
        "available": True,
        "state": "planned_license_required_not_exported",
        "reason": "license_basis_not_selected",
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "license": {
            "decision": None,
            "selected": False,
            "download_authorized": False,
            "export_authorized": False,
            "allowed_bases": ["AGPL-3.0 compatible", "Ultralytics Enterprise"],
        },
        "selection": {
            "candidate": "YOLO26s-pose",
            "production_model_selected": False,
        },
        "permissive_challenger": {
            "candidate": "MMPose YOLOX-Pose-S",
            "role": "separate_permissive_challenger",
            "license": "Apache-2.0",
            "production_model_selected": False,
            "replaces_yolo26_selection": False,
            "checkpoint": {
                "acquired": True,
                "integrity_verified": True,
                "immutable_read_only": True,
            },
            "cpu_structural_evidence": {
                "strict_state_load_verified": True,
                "state_tensor_count": 547,
                "parameter_count": 10729963,
                "raw_forward_batch": 1,
                "raw_profiles_verified": [640, 960],
                "real_image_inference_executed": False,
                "quality_measured": False,
                "performance_measured": False,
                "gpu_executed": False,
            },
            "profiles": {
                "640": {
                    "upstream_trained_resolution": True,
                    "cpu_raw_shape_verified": True,
                    "quality_verified": False,
                },
                "960": {
                    "upstream_trained_resolution": False,
                    "cpu_raw_shape_verified": True,
                    "feasibility_only": True,
                    "quality_verified": False,
                },
            },
            "official_metrics": {
                "dataset": "COCO val2017",
                "input_size": 640,
                "ap": 0.641,
                "ap50": 0.872,
                "ap75": 0.702,
                "locally_reproduced": False,
                "product_acceptance_evidence": False,
            },
            "onnx_preflight": {
                "current": {
                    "run_id": "r2",
                    "snapshot_kind": "current_immutable_preflight",
                    "state": "blocked",
                    "status": "blocked_preflight_no_export_attempted",
                    "blocker_count": 2,
                    "blocker_codes": list(
                        POSE_MMPOSE_ONNX_PREFLIGHT_BLOCKERS["current_r2"]
                    ),
                    "mmdeploy_checkout_verified": True,
                    "export_attempted": False,
                    "onnxruntime_executed": False,
                    "batch12_executed": False,
                    "deepstream9_executed": False,
                    "production_ready": False,
                },
                "historical": [
                    {
                        "run_id": "r1",
                        "snapshot_kind": "historical_immutable_preflight",
                        "state": "blocked",
                        "status": (
                            "blocked_preflight_no_export_attempted"
                        ),
                        "blocker_count": 3,
                        "blocker_codes": list(
                            POSE_MMPOSE_ONNX_PREFLIGHT_BLOCKERS[
                                "historical_r1"
                            ]
                        ),
                        "mmdeploy_checkout_verified": False,
                        "export_attempted": False,
                        "onnxruntime_executed": False,
                        "batch12_executed": False,
                        "deepstream9_executed": False,
                        "production_ready": False,
                    }
                ],
                "progress": {
                    "resolved_blocker_codes": [
                        "mmdeploy_checkout_missing"
                    ],
                    "remaining_blocker_codes": list(
                        POSE_MMPOSE_ONNX_PREFLIGHT_BLOCKERS["current_r2"]
                    ),
                },
                "historical_snapshot_is_not_live_environment_state": True,
                "production_ready": False,
            },
            "export_environment_r9": _pose_permission_probe_r9(reader),
            "export_attempt_r10": export_attempt_r10,
            "shape_diagnostic_r11": shape_diagnostic_r11,
            "deployment": {
                "official_mmdeploy_onnxruntime_listed": True,
                "official_mmdeploy_tensorrt_listed": True,
                "exact_onnx_output_contract_known": True,
                "onnx_preflight_blocked": True,
                "onnx_640_exported": False,
                "onnx_960_exported": False,
                "dynamic_batch12_verified": False,
                "engine_640_built": False,
                "engine_960_built": False,
                "existing_yolo26_parser_compatible": False,
                "custom_parser_required": True,
                "custom_parser_implemented": False,
                "deepstream9_parity_passed": False,
            },
            "quality": {
                "metric": "PCK@0.2",
                "profiles": [640, 960],
                "owner_site_ground_truth_ready": False,
                "pck_640_passed": False,
                "pck_960_passed": False,
                "twelve_camera_capacity_passed": False,
            },
            "control": {
                "candidate": "MMPose RTMPose",
                "status": "research_only_not_acquired",
                "requires_person_detector": True,
                "selected": False,
            },
            "production_ready": False,
        },
        "preparation": {
            "provenance_plan_verified": True,
            "frozen_export_plans_verified": True,
            "shared_semantic_contract_verified": True,
            "export_implementation_verified": True,
            "pck_evaluator_contract_verified": True,
            "diagnostic_source_plan_verified": True,
        },
        "model_contract": {
            "task": "pose",
            "layout": "COCO17",
            "keypoint_count": 17,
            "keypoint_fields": ["x", "y", "confidence"],
            "profiles": [640, 960],
            "batch_min": 1,
            "batch_opt": 12,
            "batch_max": 12,
            "planned_precision": "FP16",
            "profile_specific_onnx": True,
            "profile_specific_engines": True,
        },
        "artifacts": {
            "weights_acquired": False,
            "onnx_640_exported": False,
            "onnx_960_exported": False,
            "engine_640_built": False,
            "engine_960_built": False,
        },
        "pck": {
            "metric": "PCK",
            "threshold_radius": 0.20,
            "threshold": 0.80,
            "profiles": [640, 960],
            "minimum_distinct_video_types": 10,
            "required_view_types": [
                "medium_close",
                "overhead_security_camera",
            ],
            "evaluator_contract_verified": True,
            "evaluation_plan_pin_declared": False,
            "ground_truth_pin_declared": False,
            "predictions_pin_declared": False,
            "receipt_pin_declared": False,
            "result_available": False,
        },
        "source_readiness": {
            "reviewed_official_sources": 4,
            "diagnostic_images_planned": 3,
            "diagnostic_person_annotations": 3,
            "diagnostic_visible_keypoints": 38,
            "exact_coco17_verified": True,
            "video_sequences": 0,
            "track_ids": 0,
            "overhead_or_security_views": 0,
            "independent_human_review_complete": False,
            "rights_cleared_for_closed_product": False,
            "eligible_for_product_pck": False,
        },
        "gates": _pose_readiness_gates(),
        "integrity": integrity,
        "caveats": [
            "Dondurulmuş export planları ve semantic contract, model weight veya export sonucu değildir.",
            "Ultralytics lisans temeli seçilmeden model indirme ve export yetkisi kapalıdır.",
            "Apache-2.0 MMPose YOLOX-Pose-S ayrı challenger olarak exact-pinli checkpoint ve CPU strict-load/ham-shape kanıtına sahiptir; YOLO26 lisans kararını veya üretim seçimini değiştirmez.",
            "Mühürlü ONNX preflight R2 güncel snapshot'ında MMDeploy checkout pini doğrulanmıştır; mmdeploy dağıtımı ve derlenmiş mmcv ops eksikliği nedeniyle export denenmeden kapalıdır. R1'in üç blocker'ı yalnız tarihsel snapshot olarak gösterilir.",
            "Challenger 960 doğrulaması yalnız batch-1 ham tensor şekli/sonluluk fizibilitesidir; upstream 960 kalite iddiası, batch-12, ONNX, TensorRT, DeepStream 9 parser/parite veya FPS kanıtı değildir.",
            "PCK evaluator hazırdır; ancak sahibi onaylı saha GT evaluation planı, tahminler ve replay edilebilir PCK receipt'i yoktur.",
            "Üç COCO görüntülük plan yalnız tanısaldır; video/track, üst-güvenlik açısı, hak temizliği veya ürün kabulü sağlamaz.",
            "Sürümlü R9 kartı yalnız export ortamı runtime probe sonucunu ayrı yayınlar; eski R1/R2 preflight alanları tarihsel kanıt olarak korunur.",
            "R10 640 export denemesi exact-pinli failed receipt ile dets şekil uyuşmazlığında kapandı; 960 denenmedi ve production ONNX yayınlanmadı.",
            "R11 640 shape diagnostic dinamik K arayüzünü gözledi; çıktı karantinadadır, sabit-K packer uyumlu değildir ve contract değişikliği yetkilendirilmedi.",
        ],
        "evidence": [],
    }


def _full_stack_benchmark_integrity() -> dict[str, bool]:
    return {
        "plan_exact_pin_verified": False,
        "plan_schema_exact_pin_verified": False,
        "receipt_schema_exact_pin_verified": False,
        "validator_exact_pin_verified": False,
        "plan_schema_replayed": False,
        "plan_fingerprint_replayed": False,
        "plan_semantics_replayed": False,
        "receipt_contract_replayed": False,
    }


def _full_stack_benchmark_unavailable(
    reason: str, *, integrity: dict[str, bool] | None = None
) -> dict[str, Any]:
    return {
        "label": "Üç modül 12-kamera / 5-dakika benchmark planı",
        "available": False,
        "state": "artifact_error",
        "reason": reason,
        "ready": False,
        "execution_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "scope": {},
        "runtime": {},
        "profiles": {},
        "blocker_count": None,
        "blockers": [],
        "measurement": {
            "executed": False,
            "full_stack_result_available": False,
        },
        "integrity": integrity or _full_stack_benchmark_integrity(),
        "caveats": [
            "Benchmark kontrol sözleşmesi doğrulanamadı; üç-modül çalıştırma ve sonuç iddiaları kapalıdır."
        ],
        "evidence": [],
    }


def _full_stack_benchmark_blocker_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for profile in (640, 960):
        labels[f"profile:{profile}:authorization_plan:missing"] = (
            f"{profile}: launch-authorized full-stack planı eksik"
        )
        labels[f"profile:{profile}:pose:engine_pin:missing"] = (
            f"{profile}: Pose TensorRT engine pini eksik"
        )
        labels[f"profile:{profile}:pose:infer_config_pin:missing"] = (
            f"{profile}: Pose DeepStream infer config pini eksik"
        )
        labels[f"profile:{profile}:ppe:engine_pin:missing"] = (
            f"{profile}: PPE TensorRT engine pini eksik"
        )
        labels[f"profile:{profile}:ppe:infer_config_pin:missing"] = (
            f"{profile}: PPE DeepStream infer config pini eksik"
        )
    labels["runtime:fusion_capability:not_runtime_ready"] = (
        "R3 fusion capability henüz runtime_ready değil"
    )
    return labels


def _full_stack_benchmark_plan_semantics_valid(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    try:
        projected = dict(plan)
        claimed_fingerprint = projected.pop("fingerprint_sha256")
        execution = plan["execution"]
        topology = plan["topology"]
        runtime = plan["runtime"]
        image = runtime["image"]
        sources = plan["sources"]
        profiles = plan["profiles"]
        metrics = plan["metrics_contract"]
        context = plan["performance_context"]
        baseline = context["person_only_baseline"]
        estimate = context["full_stack_estimate"]
        readiness = plan["readiness"]
        if (
            plan["schema_version"]
            != "deepsafe.deepstream-full-stack-benchmark-plan/v1"
            or plan["plan_id"]
            != "ds9-three-module-12-source-640-960-300s-r1"
            or plan["deterministic"] is not True
            or claimed_fingerprint != FULL_STACK_BENCHMARK_PLAN_FINGERPRINT
            or _canonical_sha256(projected) != claimed_fingerprint
            or plan["state"] != "blocked_missing_runtime_artifacts"
            or execution["execution_mode"] != "foreground_only"
            or execution["profile_order"] != [640, 960]
            or execution["separate_process_per_profile"] is not True
            or execution["simulated_streams"] != 12
            or execution["warmup_seconds"] != 15
            or execution["measurement_seconds"] != 300
            or execution["gpu_lease"]["required"] is not True
            or execution["gpu_lease"]["owner_kind"] != "capacity_5min"
            or execution["gpu_lease"]["held_for_process_lifetime"] is not True
            or execution["gpu_lease"]["heartbeat_required"] is not True
            or execution["gpu_lease"]["background_execution_allowed"] is not False
            or topology["modules"] != ["person", "pose", "ppe"]
            or topology["all_modules_enabled_together"] is not True
            or topology["parallel_pattern"]
            != "nvidia_parallel_inference_nvdsmetamux"
            or topology["nvdsmetamux_required"] is not True
            or len(sources) != 12
            or [source["source_id"] for source in sources] != list(range(12))
            or len({source["video_type"] for source in sources}) != 12
            or len({source["plan_uri"] for source in sources}) != 12
            or len({source["media_pin"]["sha256"] for source in sources}) != 12
            or not {"medium_close", "overhead_security_camera"}.issubset(
                {
                    view
                    for source in sources
                    for view in source["view_types"]
                }
            )
            or runtime["deepstream_version"] != "9.0.0"
            or runtime["tensorrt_version"] != "10.14.1.48"
            or image["image_id"] != FULL_STACK_BENCHMARK_EXPECTED_IMAGE_ID
            or image["parser_sha256"]
            != FULL_STACK_BENCHMARK_EXPECTED_PARSER_SHA256
            or runtime["fusion"]["publication_id"]
            != "deepsafe-fusion-ds9-9946965e-r3"
            or [profile["model_input"] for profile in profiles] != [640, 960]
        ):
            return False

        derived_blockers: list[str] = []
        for profile in profiles:
            profile_id = profile["model_input"]
            if profile["simulated_streams"] != 12:
                return False
            if profile["authorization_plan_pin"] is None:
                derived_blockers.append(
                    f"profile:{profile_id}:authorization_plan:missing"
                )
            modules = profile["modules"]
            if [module["role"] for module in modules] != [
                "person",
                "pose",
                "ppe",
            ]:
                return False
            for module in modules:
                role = module["role"]
                expected_gie = {"person": 1, "pose": 2, "ppe": 3}[role]
                if module["gie_unique_id"] != expected_gie:
                    return False
                for field in ("engine_pin", "infer_config_pin"):
                    if module[field] is None:
                        derived_blockers.append(
                            f"profile:{profile_id}:{role}:{field}:missing"
                        )
                expected_status = (
                    "ready"
                    if module["engine_pin"] is not None
                    and module["infer_config_pin"] is not None
                    and (
                        role == "person"
                        or (
                            module["postprocess_library_pin"] is not None
                            and module["association_contract_pin"] is not None
                        )
                    )
                    else "pending"
                )
                if module["status"] != expected_status:
                    return False
        if runtime["fusion"]["capability_status"] != "runtime_ready":
            derived_blockers.append(
                "runtime:fusion_capability:not_runtime_ready"
            )
        if runtime["qualification"]["status"] != "production_ready":
            derived_blockers.append(
                "runtime:qualification:not_production_ready"
            )
        derived_blockers = sorted(set(derived_blockers))
        if (
            readiness["blockers"] != derived_blockers
            or readiness["execution_ready"] is not False
            or len(derived_blockers) != 11
            or set(derived_blockers)
            != set(_full_stack_benchmark_blocker_labels())
        ):
            return False

        baseline_profiles = baseline["profiles"]
        estimate_profiles = estimate["profiles"]
        return bool(
            baseline["classification"] == "person_only_measured_baseline"
            and baseline["eligible_as_full_stack_result"] is False
            and baseline["eligible_as_capacity_threshold"] is False
            and baseline_profiles
            == [
                {
                    "model_input": 640,
                    "aggregate_mean_fps": 464.733,
                    "per_stream_mean_fps": 38.729,
                },
                {
                    "model_input": 960,
                    "aggregate_mean_fps": 305.799,
                    "per_stream_mean_fps": 25.484,
                },
            ]
            and estimate["classification"] == "estimate_not_measured"
            and estimate["eligible_as_result"] is False
            and estimate["eligible_as_acceptance_evidence"] is False
            and estimate_profiles
            == [
                {
                    "model_input": 640,
                    "aggregate_fps_range": [90.0, 190.0],
                    "per_stream_fps_range": [7.5, 15.833],
                },
                {
                    "model_input": 960,
                    "aggregate_fps_range": [55.0, 125.0],
                    "per_stream_fps_range": [4.583, 10.417],
                },
            ]
            and metrics["throughput"]["required_statistics"]
            == ["mean", "p05"]
            and metrics["latency"]["minimum_per_source_coverage"] == 0.95
            and metrics["metadata_fusion"][
                "minimum_per_source_coverage"
            ]
            == 0.95
            and len(metrics["output_evidence_roles"]) == 12
            and plan["safety"]
            == {
                "docker_called": False,
                "gpu_queried": False,
                "gpu_process_started": False,
                "inference_started": False,
            }
        )
    except (KeyError, TypeError, ValueError, RecursionError):
        return False


def _full_stack_benchmark_receipt_schema_valid(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    try:
        properties = schema["properties"]
        run_schema = schema["$defs"]["run"]
        run_properties = run_schema["properties"]
        return bool(
            schema["$schema"]
            == "https://json-schema.org/draft/2020-12/schema"
            and schema["$id"]
            == (
                "https://deepsafe.local/schemas/"
                "deepstream-full-stack-benchmark-receipt-v1.schema.json"
            )
            and properties["schema_version"]["const"]
            == "deepsafe.deepstream-full-stack-benchmark-receipt/v1"
            and properties["state"]["const"] == "measurement_complete"
            and properties["result_kind"]["const"] == "measured_full_stack"
            and run_properties["deepstream_version"]["const"] == "9.0.0"
            and run_properties["modules_enabled_together"]["const"]
            == ["person", "pose", "ppe"]
            and run_properties["source_count"]["const"] == 12
            and run_properties["measurement"]["properties"][
                "steady_state_seconds"
            ]["const"]
            == 300
            and run_properties["evidence_pins"]["minItems"] == 12
            and run_properties["evidence_pins"]["maxItems"] == 12
        )
    except (KeyError, TypeError, ValueError):
        return False


def _full_stack_benchmark(reader: ArtifactReader) -> dict[str, Any]:
    """Project an exact-pinned, non-executable three-module benchmark plan."""

    integrity = _full_stack_benchmark_integrity()
    parsed: dict[str, dict[str, Any] | None] = {}
    reads: dict[str, WorkspacePinRead] = {}
    for key in ("plan", "plan_schema", "receipt_schema"):
        pin = FULL_STACK_BENCHMARK_ADMIN_PINS[key]
        result, value = _workspace_pin_json(
            reader,
            pin,
            expected_path=pin["path"],
            maximum_bytes=FULL_STACK_BENCHMARK_MAX_BYTES,
        )
        reads[key] = result
        parsed[key] = value
        integrity[f"{key}_exact_pin_verified"] = result.available
    validator_pin = FULL_STACK_BENCHMARK_ADMIN_PINS["validator"]
    validator_read = _read_workspace_pin(
        reader,
        validator_pin,
        expected_path=validator_pin["path"],
        maximum_bytes=FULL_STACK_BENCHMARK_MAX_BYTES,
        collect=False,
    )
    reads["validator"] = validator_read
    integrity["validator_exact_pin_verified"] = validator_read.available
    if any(not result.available for result in reads.values()):
        key, result = next(
            (key, result)
            for key, result in reads.items()
            if not result.available
        )
        return _full_stack_benchmark_unavailable(
            f"{key}_{result.state}", integrity=integrity
        )
    plan = parsed["plan"]
    plan_schema = parsed["plan_schema"]
    receipt_schema = parsed["receipt_schema"]
    if plan is None or plan_schema is None or receipt_schema is None:
        return _full_stack_benchmark_unavailable(
            "json_contract_invalid", integrity=integrity
        )
    plan_schema_identity = bool(
        plan_schema.get("$schema")
        == "https://json-schema.org/draft/2020-12/schema"
        and plan_schema.get("$id")
        == (
            "https://deepsafe.local/schemas/"
            "deepstream-full-stack-benchmark-plan-v1.schema.json"
        )
    )
    try:
        _validate_schema_node(plan, plan_schema, plan_schema)
    except (TypeError, ValueError, RecursionError):
        schema_replayed = False
    else:
        schema_replayed = plan_schema_identity
    integrity["plan_schema_replayed"] = schema_replayed
    projected = dict(plan)
    claimed_fingerprint = projected.pop("fingerprint_sha256", None)
    integrity["plan_fingerprint_replayed"] = bool(
        claimed_fingerprint == FULL_STACK_BENCHMARK_PLAN_FINGERPRINT
        and _canonical_sha256(projected) == claimed_fingerprint
    )
    integrity["plan_semantics_replayed"] = (
        _full_stack_benchmark_plan_semantics_valid(plan)
    )
    integrity["receipt_contract_replayed"] = (
        _full_stack_benchmark_receipt_schema_valid(receipt_schema)
    )
    if not all(integrity.values()):
        return _full_stack_benchmark_unavailable(
            "exact_pin_schema_or_semantic_replay_invalid",
            integrity=integrity,
        )

    baseline_by_profile = {
        item["model_input"]: item
        for item in plan["performance_context"]["person_only_baseline"][
            "profiles"
        ]
    }
    estimate_by_profile = {
        item["model_input"]: item
        for item in plan["performance_context"]["full_stack_estimate"][
            "profiles"
        ]
    }
    profiles: dict[str, Any] = {}
    for profile in plan["profiles"]:
        profile_id = profile["model_input"]
        modules = {module["role"]: module for module in profile["modules"]}
        baseline = baseline_by_profile[profile_id]
        estimate = estimate_by_profile[profile_id]
        profiles[str(profile_id)] = {
            "person_engine_config_ready": modules["person"]["status"]
            == "ready",
            "pose_engine_config_ready": modules["pose"]["status"]
            == "ready",
            "ppe_engine_config_ready": modules["ppe"]["status"]
            == "ready",
            "authorization_plan_ready": profile["authorization_plan_pin"]
            is not None,
            "person_only_baseline": {
                "classification": "person_only_measured_baseline",
                "aggregate_mean_fps": baseline["aggregate_mean_fps"],
                "per_stream_mean_fps": baseline["per_stream_mean_fps"],
                "eligible_as_full_stack_result": False,
            },
            "full_stack_estimate": {
                "classification": "estimate_not_measured",
                "aggregate_fps_range": list(
                    estimate["aggregate_fps_range"]
                ),
                "per_stream_fps_range": list(
                    estimate["per_stream_fps_range"]
                ),
                "eligible_as_result": False,
            },
            "measurement": {
                "executed": False,
                "result_available": False,
            },
        }
    blocker_labels = _full_stack_benchmark_blocker_labels()
    blockers = [
        {"label": blocker_labels[code]}
        for code in plan["readiness"]["blockers"]
    ]
    return {
        "label": "Üç modül 12-kamera / 5-dakika benchmark planı",
        "available": True,
        "state": "blocked_missing_runtime_artifacts",
        "reason": (
            "pose_ppe_engines_configs_authorization_and_fusion_pending"
        ),
        "ready": False,
        "execution_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "scope": {
            "profiles": [640, 960],
            "separate_runs": True,
            "simulated_streams": 12,
            "distinct_sources": 12,
            "distinct_video_types": 12,
            "warmup_seconds_per_run": 15,
            "measurement_seconds_per_run": 300,
            "execution_mode": "foreground_only",
            "minimum_per_source_coverage": 0.95,
            "minimum_output_fps_per_source": 25.0,
        },
        "runtime": {
            "deepstream": "9.0.0",
            "tensorrt": "10.14.1.48",
            "image": "exact_pinned_ds9_image",
            "parser": "exact_pinned_yolo_parser",
            "fusion": "R3_exact_pinned_publication",
        },
        "profiles": profiles,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "measurement": {
            "executed": False,
            "full_stack_result_available": False,
            "receipt_expected_result_kind": "measured_full_stack",
            "required_output_evidence_roles": 12,
            "live_large_artifact_replay_performed_by_admin": False,
        },
        "integrity": integrity,
        "caveats": [
            "Person-only ölçümler full-stack sonucu veya kapasite kabulü değildir.",
            "Full-stack aralıkları estimate_not_measured sınıfındadır; ölçülmüş sonuç değildir.",
            "Admin yalnız küçük kontrol artefaktlarını replay eder; video, engine veya GPU çalıştırmaz.",
            "Bu kart product-readiness kampanyasını veya mevcut kanıtları değiştirmez.",
        ],
        "evidence": [],
    }


def _product_acceptance_policy(reader: ArtifactReader) -> dict[str, Any]:
    """Project the immutable owner-approved thresholds, never an outcome.

    The policy reader verifies the exact reviewed fingerprint and its pinned
    schema through symlink-free file descriptors.  This projection deliberately
    contains no mutable result fields and cannot turn policy presence into a
    passing product gate.
    """

    unavailable = {
        "label": "Sahip onaylı üç modül kabul eşikleri",
        "available": False,
        "state": "artifact_error",
        "reason": "owner_acceptance_policy_invalid_or_missing",
        "acceptance_state": "not_evaluated",
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "policy": None,
        "scope": {},
        "quality_thresholds": {},
        "capacity": {},
        "cadence": {},
        "endurance": {},
        "caveats": [
            "Kabul eşiği sözleşmesi doğrulanamadı; bütün ürün kapıları kapalı kalır.",
        ],
        "evidence": [],
    }
    try:
        value = load_approved_policy(reader.workspace_root)
    except (AcceptancePolicyError, OSError, RuntimeError, ValueError):
        return unavailable

    scope = value["scope"]
    quality = value["quality_thresholds"]
    person = quality["person"]
    distance = quality["exact_25m_person"]
    pose = quality["pose"]
    ppe = quality["ppe"]
    capacity = value["capacity_thresholds"]
    cadence = value["execution_cadence"]
    endurance = value["endurance_thresholds"]
    approval = value["approval"]
    immutability = value["pre_run_immutability"]

    return {
        "label": "Sahip onaylı üç modül kabul eşikleri",
        "available": True,
        "state": "approved_not_evaluated",
        "reason": None,
        "acceptance_state": "not_evaluated",
        "ready": False,
        "final_claim_allowed": False,
        "does_not_imply_product_readiness": True,
        "read_only": True,
        "execution_actions_available": False,
        "policy": {
            "policy_id": value["policy_id"],
            "status": value["status"],
            "approved_at_utc": approval["approval_recorded_at_utc"],
            "effective_at_utc": immutability[
                "effective_for_measurement_runs_started_at_or_after_utc"
            ],
            "pre_run_fingerprint_sha256": value[
                "pre_run_fingerprint_sha256"
            ],
            "policy_alone_can_pass_any_gate": False,
        },
        "scope": {
            "scene_types": scope["minimum_distinct_video_types"],
            "required_view_types": list(scope["required_view_types"]),
            "model_input_sizes": list(scope["profiles"]),
            "simulated_streams": scope["camera_count"],
            "duration_seconds_per_run": scope[
                "performance_measurement_seconds"
            ],
            "required_modules": list(scope["required_modules"]),
            "modules_must_run_together": scope["modules_must_run_together"],
            "deepstream_version": scope["deepstream_version"],
        },
        "quality_thresholds": {
            "person": {
                "precision": person["precision"]["threshold"],
                "recall": person["recall"]["threshold"],
                "f1": person["f1"]["threshold"],
                "large_person_recall": person["large_person_recall"][
                    "threshold"
                ],
            },
            "exact_25m_person_recall": distance["recall"]["threshold"],
            "pose_pck_at_0_2": pose["threshold"],
            "ppe": {
                "precision": ppe["precision"]["threshold"],
                "violation_recall": ppe["violation_recall"]["threshold"],
                "maximum_alert_latency_seconds": ppe[
                    "maximum_alert_latency_seconds"
                ],
                "maximum_false_alarms_per_camera_hour": ppe[
                    "maximum_false_alarms_per_camera_hour"
                ],
            },
        },
        "capacity": {
            "camera_count": capacity["camera_count"],
            "minimum_output_fps_per_camera": capacity[
                "minimum_output_fps_per_camera"
            ],
            "measurement_seconds": capacity["measurement_seconds"],
            "each_camera_must_pass": capacity["scope"]
            == "each_camera_must_individually_pass",
            "cross_camera_average_substitution_allowed": capacity[
                "cross_camera_average_substitution_allowed"
            ],
        },
        "cadence": {
            "person_maximum_skipped_frames": cadence["person"][
                "maximum_consecutive_skipped_decoded_frames"
            ],
            "pose_maximum_skipped_frames": cadence["pose"][
                "maximum_consecutive_skipped_decoded_frames"
            ],
            "ppe_maximum_skipped_frames": cadence["ppe"][
                "maximum_consecutive_skipped_decoded_frames"
            ],
        },
        "endurance": {
            "segments": endurance["segments"],
            "segment_seconds": endurance["segment_seconds"],
            "total_seconds": endurance["total_seconds"],
            "minimum_p05_baseline_fraction": endurance[
                "throughput_retention"
            ]["minimum_fraction_of_final_baseline"],
            "maximum_fault_counts": dict(
                endurance["runtime_fault_maximum_counts"]
            ),
        },
        "caveats": [
            "Bu kart yalnız run öncesi eşik sözleşmesini doğrular; hiçbir kabul kapısını tek başına geçirmez.",
            "Eski kişi-only ölçümleri final üç-modül kabul kanıtı değildir.",
        ],
        "evidence": [],
    }


def load_validation_status() -> dict[str, Any]:
    """Return compact projections only; raw artifact fields never pass through."""

    reader = ArtifactReader()
    finalization_bundle = _finalization_bundle(reader)
    product_finalization_v2 = _product_finalization_v2(reader)
    campaigns = {
        "deepstream91_static_qualification": load_deepstream91_static_status(),
        "deepstream91_native_build": load_deepstream91_native_status(),
        "deepstream91_engine_builder_r1c3": load_ds91_engine_builder_r1c3_status(),
        "deepstream91_full_stack_preflight_r1": load_ds91_preflight_r1_status(),
        "deepstream91_full_stack_preflight_r2": load_ds91_preflight_r2_status(),
        "driver595_maintenance_r4": load_driver595_r4_status(),
        "driver595_live_qualification_r7": load_driver595_r7_status(),
        "product_acceptance_policy": _product_acceptance_policy(reader),
        "person_model_upgrade": _person_model_upgrade_readiness(reader),
        "person_rtdetrv4_tensorrt_r14i": load_person_r14i_status(),
        "pose_model_readiness": _pose_readiness(reader),
        "pose_mmpose_yoloxpose_tensorrt_r13i": load_pose_r13i_status(),
        "ppe_seed_readiness": _ppe_seed_readiness(reader),
        "ppe_five_class_quarantine": _ppe_five_class_readiness(reader),
        "ppe_public_source_quarantine": _ppe_lo_cpped_source_quarantine(reader),
        "ppe_construction_ppe_quarantine": load_construction_ppe_status(),
        "ppe_safetyvision_challenger": load_safetyvision_challenger_status(),
        "ppe_safetyvision_phase_b_a32": load_ppe_a32_status(),
        "gpu_lease_v5": load_gpu_lease_v5_status(),
        "objective_completion": _objective_completion(reader),
        "product_readiness": _product_readiness(reader),
        "full_stack_benchmark": _full_stack_benchmark(reader),
        "campaign_report": _campaign_report(reader),
        "person_detection_quality": _person_detection_quality(reader),
        "ppe_video_source_registry": _ppe_video_source_registry(reader),
        "gpu_reentry": _gpu_reentry(reader),
        "scene_benchmark": _scene_benchmark(reader),
        "caviar": _caviar(reader),
        "rlivit": _rlivit(reader),
        "loaf_deepstream": _loaf_deepstream(reader),
        "loaf_distance_bins": _loaf_distance_bins(reader),
        "site_distance_25m": _site_distance_25m(reader),
        "open_video_review": _open_video_review(reader),
        "endurance": _endurance(reader),
    }
    campaigns["campaign_report"]["finalized"] = bool(
        finalization_bundle["committed"]
        and campaigns["campaign_report"].get("reason") != "stale_lineage"
    )
    for key in ("objective_completion", "product_readiness"):
        campaigns[key]["finalized"] = bool(
            (
                finalization_bundle["committed"]
                or product_finalization_v2["committed"]
            )
            and campaigns[key].get("reason") != "stale_lineage"
        )
    return {
        "schema_version": "deepsafe.admin-validation/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "execution_actions_available": False,
        "max_artifact_bytes": reader.max_bytes,
        "finalization_bundle": finalization_bundle,
        "product_finalization_v2": product_finalization_v2,
        "campaigns": campaigns,
    }


def load_validation_artifact(key: str) -> ValidationArtifact:
    """Load one allow-listed evidence artifact after the same safety checks."""

    reader = ArtifactReader()
    try:
        result = reader.read(key)
    except KeyError as exc:
        raise ValidationArtifactError("unknown_artifact") from exc
    if not result.spec.raw_download_allowed:
        raise ValidationArtifactError("projection_only")
    if not result.available or result.content is None:
        raise ValidationArtifactError(result.state)
    filename = Path(result.relative_path).name
    return ValidationArtifact(result.content, result.spec.media_type, filename)
