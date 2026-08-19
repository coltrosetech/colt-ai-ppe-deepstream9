from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def artifact_pin(
    results: Path,
    relative: str,
    evidence_id: str,
    *,
    campaign_evidence: bool = False,
) -> dict[str, Any]:
    path = results / relative
    content = path.read_bytes()
    row: dict[str, Any] = {
        "id": evidence_id,
        "path": f"validation/results/{relative}",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    if campaign_evidence:
        value = json.loads(content)
        row.update(
            {
                "media_type": "application/json",
                "state": "ok",
                "schema_version": value["schema_version"],
            }
        )
    return row


def write_admin_endurance_lineage(
    results: Path,
    *,
    config_fingerprint: str = "a" * 64,
    static_input_fingerprint: str = "b" * 64,
    checkpoint_config_fingerprint: str | None = None,
    checkpoint_static_input_fingerprint: str | None = None,
    status_config_fingerprint: str | None = None,
    status_static_input_fingerprint: str | None = None,
    state: str = "running",
    validated_seconds: int = 0,
) -> dict[str, dict[str, Any]]:
    campaign_name = "deepstream9-12-camera-seven-day"
    target_seconds = 7 * 24 * 60 * 60
    segment_seconds = 6 * 60 * 60
    throughput_floor = {"status": "verified"}
    power_safety = {"operating_policy_mode": "workstation_managed"}
    input_pins: list[dict[str, Any]] = []
    complete = state == "complete"
    updated_at_utc = (
        "2026-07-24T00:00:00+00:00"
        if complete
        else "2026-07-17T00:00:01+00:00"
    )
    finished_at_utc = updated_at_utc if complete else None
    segments: list[dict[str, Any]] = []
    if complete:
        for index in range(28):
            profile = 640 if index % 2 == 0 else 960
            segment_id = f"segment-{index:03d}-{profile}"
            segments.append(
                {
                    "index": index,
                    "segment_id": segment_id,
                    "profile": profile,
                    "campaign_day": index * segment_seconds // 86400 + 1,
                    "duration_seconds": segment_seconds,
                    "status": "healthy",
                    "validated_seconds": segment_seconds,
                    "attempts": [{"attempt": 1}],
                    "attempt_receipts": [
                        {
                            "path": (
                                "validation/results/endurance/current/segments/"
                                f"{segment_id}/attempt-01/attempt-receipt.json"
                            ),
                            "size_bytes": 1,
                            "sha256": "e" * 64,
                        }
                    ],
                }
            )
    resolved = {
        "schema_version": "deepsafe.endurance-campaign/v1",
        "name": campaign_name,
        "duration_seconds": target_seconds,
        "config_fingerprint": config_fingerprint,
        "static_input_fingerprint": static_input_fingerprint,
        "throughput_floor": throughput_floor,
        "power_safety": power_safety,
        "input_pins": input_pins,
    }
    identity = {
        "state": state,
        "dry_run": False,
        "campaign_name": campaign_name,
        "throughput_floor": throughput_floor,
        "power_safety_policy": power_safety,
        "updated_at_utc": updated_at_utc,
        "started_at_utc": "2026-07-17T00:00:00+00:00",
        "finished_at_utc": finished_at_utc,
        "target_validated_seconds": target_seconds,
        "validated_seconds": validated_seconds,
        "active": None,
        "unexpected_restarts": 0,
        "orphan_recoveries": 0,
        "campaign_health_gates": [],
        "input_pins": input_pins,
    }
    checkpoint = {
        "schema_version": "deepsafe.endurance-checkpoint/v1",
        **identity,
        "config_fingerprint": (
            checkpoint_config_fingerprint or config_fingerprint
        ),
        "static_input_fingerprint": (
            checkpoint_static_input_fingerprint or static_input_fingerprint
        ),
        "segments": segments,
    }
    status = {
        "schema_version": "deepsafe.endurance-status/v1",
        **identity,
        "config_fingerprint": status_config_fingerprint or config_fingerprint,
        "static_input_fingerprint": (
            status_static_input_fingerprint or static_input_fingerprint
        ),
        "available": True,
        "progress_fraction": validated_seconds / target_seconds,
        "segments": {
            "total": len(segments),
            "status_counts": {"healthy": len(segments)} if complete else {},
        },
    }
    relatives = {
        "endurance_campaign_resolved": "endurance/current/campaign-resolved.json",
        "endurance_checkpoint": "endurance/current/checkpoint.json",
        "endurance_status": "endurance/current/status.json",
    }
    for relative, value in zip(
        relatives.values(), (resolved, checkpoint, status), strict=True
    ):
        write_json(results / relative, value)
    return {
        evidence_id: artifact_pin(
            results, relative, evidence_id, campaign_evidence=True
        )
        for evidence_id, relative in relatives.items()
    }
