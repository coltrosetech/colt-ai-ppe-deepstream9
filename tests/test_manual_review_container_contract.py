import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


MANUAL_ENV_KEYS = {
    "DEEPSAFE_MANUAL_REVIEW_SOURCE",
    "DEEPSAFE_MANUAL_REVIEW_ASSET_INDEX",
    "DEEPSAFE_MANUAL_REVIEW_ASSET_ROOT",
    "DEEPSAFE_MANUAL_REVIEW_MAX_ASSET_BYTES",
}

SITE_DISTANCE_ENV = {
    "DEEPSAFE_VALIDATION_WORKSPACE_ROOT": "/workspace",
    "DEEPSAFE_VALIDATION_SCHEMA_ROOT": "/app/validation/schemas",
}

READ_ONLY_REVIEW_MOUNTS = {
    "./validation/results:/workspace/validation-results:ro",
    "./validation/open_video_review:/workspace/manual-review-source:ro",
    "./validation/contact-sheets:/workspace/contact-sheets:ro",
    "./validation:/workspace/validation:ro",
}

GPU_BUILD_ARG_ENV = {
    "DEEPSTREAM_BASE_REF",
    "DEEPSTREAM_BASE_DIGEST",
    "DEEPSTREAM_YOLO_PARSER_SHA256",
    "DEEPSAFE_RUNTIME_CONTROLLER_SHA256",
    "DEEPSAFE_RUNTIME_CONTROL_MANIFEST_SHA256",
    "DEEPSAFE_DOCKERIGNORE_SHA256",
}


def test_admin_and_deepstream_share_manual_review_runtime_contract():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    admin = services["admin"]
    deepstream = services["deepstream"]

    assert {key: admin["environment"][key] for key in MANUAL_ENV_KEYS} == {
        key: deepstream["environment"][key] for key in MANUAL_ENV_KEYS
    }
    assert READ_ONLY_REVIEW_MOUNTS <= set(admin["volumes"])
    assert READ_ONLY_REVIEW_MOUNTS <= set(deepstream["volumes"])
    assert {key: admin["environment"][key] for key in SITE_DISTANCE_ENV} == SITE_DISTANCE_ENV
    assert {key: deepstream["environment"][key] for key in SITE_DISTANCE_ENV} == SITE_DISTANCE_ENV


def test_both_admin_serving_images_copy_only_required_validation_contract_module():
    common_required = {
        "COPY validation/__init__.py /app/validation/__init__.py",
        "COPY validation/open_video_manual_review.py /app/validation/open_video_manual_review.py",
    }
    required_by_image = {
        Path("admin/Dockerfile"): {
            "COPY validation/strict_json.py /app/validation/strict_json.py",
            "COPY validation/site_distance_evaluation.py /app/validation/site_distance_evaluation.py",
            "COPY validation/site_distance_readiness_v2.py /app/validation/site_distance_readiness_v2.py",
            "COPY validation/site_distance_evaluator_v2.py /app/validation/site_distance_evaluator_v2.py",
            "COPY validation/schemas /app/validation/schemas",
        },
        Path("deepstream/Dockerfile"): {
            "COPY validation/schemas/site-distance-evaluation-plan-v1.schema.json /app/validation/schemas/site-distance-evaluation-plan-v1.schema.json",
            "COPY validation/schemas/distance-validation-v1.schema.json /app/validation/schemas/distance-validation-v1.schema.json",
            "COPY validation/schemas/validation-campaign-report-v1.schema.json /app/validation/schemas/validation-campaign-report-v1.schema.json",
        },
    }
    required_schemas = {
        "site-distance-evaluation-plan-v1.schema.json",
        "distance-validation-v1.schema.json",
        "validation-campaign-report-v1.schema.json",
    }
    assert all((Path("validation/schemas") / name).is_file() for name in required_schemas)

    for dockerfile, image_required in required_by_image.items():
        lines = {
            line.strip()
            for line in dockerfile.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert common_required | image_required <= lines
        copy_sources = {
            parts[1].rstrip("/")
            for line in lines
            if line.startswith("COPY ") and len(parts := line.split()) >= 3
        }
        assert "validation" not in copy_sources


def test_admin_is_loopback_only_and_gpu_build_pins_do_not_block_its_compose_parse():
    compose_text = Path("docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)
    admin = compose["services"]["admin"]
    deepstream = compose["services"]["deepstream"]

    assert admin["ports"] == ["127.0.0.1:8080:8080"]
    assert deepstream["profiles"] == ["gpu"]
    assert set(deepstream["build"]["args"]) == GPU_BUILD_ARG_ENV
    for key in GPU_BUILD_ARG_ENV:
        assert deepstream["build"]["args"][key] == f"${{{key}:-}}"
        assert f"${{{key}:?" not in compose_text

    admin_dockerfile = Path("admin/Dockerfile").read_text(encoding="utf-8")
    deepstream_dockerfile = Path("deepstream/Dockerfile").read_text(encoding="utf-8")
    # Uvicorn listens on the container interface. The admin-only deployment's
    # localhost boundary is the explicit host-side Compose publication above,
    # not a change to either image's immutable CMD contract.
    assert '"--host", "0.0.0.0"' in admin_dockerfile
    assert '"--host", "0.0.0.0"' in deepstream_dockerfile


def test_env_example_lists_every_fail_closed_gpu_build_pin():
    keys = {
        line.partition("=")[0]
        for line in Path(".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    assert GPU_BUILD_ARG_ENV <= keys


def test_compose_cli_resolves_admin_without_gpu_environment_and_keeps_loopback():
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker CLI is unavailable")
    version = subprocess.run(
        [docker, "compose", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode != 0:
        pytest.skip("docker compose plugin is unavailable")

    environment = os.environ.copy()
    environment.pop("COMPOSE_PROFILES", None)
    environment.pop("COMPOSE_FILE", None)
    for key in GPU_BUILD_ARG_ENV:
        environment.pop(key, None)
    result = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            "/dev/null",
            "-f",
            "docker-compose.yml",
            "config",
            "--format",
            "json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert set(resolved["services"]) == {"admin"}
    assert resolved["services"]["admin"]["ports"] == [
        {
            "mode": "ingress",
            "host_ip": "127.0.0.1",
            "target": 8080,
            "published": "8080",
            "protocol": "tcp",
        }
    ]
