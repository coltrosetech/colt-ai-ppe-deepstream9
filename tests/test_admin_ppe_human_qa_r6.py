from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker

import admin.ppe_human_qa as ppe_module
from admin.app import app
from admin.ppe_human_qa import (
    AUTHORIZATION_EFFECT,
    PACKET_RECEIPT_FILE,
    PACKET_RECEIPT_SHA256,
    POLICY_NAMES,
    AuditState,
    PpeHumanQaConfig,
    PpeHumanQaError,
    PpeHumanQaService,
)


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "validation/results/ppe/human-qa/mendeley-ppe-four-class-r6"
SCHEMAS = ROOT / "validation/schemas"
TOKEN = "ppe-r6-admin-test-token"
FIRST_SAMPLE = "ppe-r6-helmet-boundary-retained-0001"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


@pytest.fixture
def ppe_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    data = tmp_path / "data"
    monkeypatch.setenv("DEEPSAFE_ADMIN_TOKEN", TOKEN)
    monkeypatch.setenv("DEEPSAFE_DATA", str(data))
    monkeypatch.setenv("DEEPSAFE_PPE_HUMAN_QA_PACKET", str(PACKET))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(SCHEMAS))
    monkeypatch.setenv("DEEPSAFE_PPE_HUMAN_QA_MAX_TILE_BYTES", "4194304")
    return {"data": data}


def _headers(**extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"} | extra


def _sample_body(**updates: str) -> dict[str, str]:
    body = {
        "decision": "approve_sample_semantics",
        "reason_code": "visually_correct",
        "notes": "",
    }
    body.update(updates)
    return body


def test_ppe_r6_every_api_is_fail_closed_authenticated(ppe_env, monkeypatch):
    endpoints = [
        ("GET", "/api/ppe-human-qa-r6/queue"),
        ("GET", f"/api/ppe-human-qa-r6/samples/{FIRST_SAMPLE}"),
        ("GET", f"/api/ppe-human-qa-r6/tiles/{FIRST_SAMPLE}"),
        ("PUT", f"/api/ppe-human-qa-r6/samples/{FIRST_SAMPLE}"),
        ("PUT", f"/api/ppe-human-qa-r6/policies/{POLICY_NAMES[0]}"),
        ("GET", "/api/ppe-human-qa-r6/export"),
    ]
    with TestClient(app) as client:
        for method, path in endpoints:
            assert client.request(method, path).status_code == 401
        assert (
            client.get(
                "/api/ppe-human-qa-r6/queue",
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        for unsafe in ("", "change-me"):
            monkeypatch.setenv("DEEPSAFE_ADMIN_TOKEN", unsafe)
            assert (
                client.get(
                    "/api/ppe-human-qa-r6/queue",
                    headers={"Authorization": f"Bearer {unsafe}"},
                ).status_code
                == 503
            )


def test_queue_is_exact_pinned_filtered_and_omits_raw_payload_links(ppe_env):
    with TestClient(app) as client:
        response = client.get(
            "/api/ppe-human-qa-r6/queue?limit=3&role=calibration&status=pending",
            headers=_headers(),
        )
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        payload = response.json()
        assert payload["packet_integrity"] == (
            "exact_pins_verified_metadata_assets_verified_on_access"
        )
        assert payload["progress"] == {
            "state": "in_progress",
            "expected_samples": 718,
            "decided_samples": 0,
            "pending_samples": 718,
            "sample_decision_counts": {},
            "expected_policies": 6,
            "decided_policies": 0,
            "pending_policies": 6,
            "completed_at": None,
            "training_authorized": False,
            "production_ready": False,
            "new_exact_training_authorization_required": True,
            "authorization_effect": AUTHORIZATION_EFFECT,
        }
        assert len(payload["items"]) == 3
        assert all(item["role"] == "calibration" for item in payload["items"])
        assert len(payload["policies"]) == 6
        serialized = json.dumps(payload, sort_keys=True)
        assert "development_holdout" not in serialized
        assert "candidate_label" not in serialized
        assert "source_label" not in serialized
        assert "images/train" not in serialized
        assert str(PACKET) not in serialized
        assert "sha256" not in serialized
        assert payload["audit"] == {
            "append_only": True,
            "chain_verified": True,
            "event_count": 0,
            "chain_head_present": False,
        }


def test_tile_service_only_serves_sample_bound_exact_png(ppe_env):
    expected = PACKET / "tiles/ppe-r6-helmet-boundary-retained-0001.png"
    with TestClient(app) as client:
        response = client.get(
            f"/api/ppe-human-qa-r6/tiles/{FIRST_SAMPLE}", headers=_headers()
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["content-security-policy"] == "default-src 'none'; sandbox"
        assert response.content == expected.read_bytes()
        assert hashlib.sha256(response.content).hexdigest() == (
            "243ad1806d5071a3b785a6aec0b60f8a7aea8c231a1e922f6927b02ee5292b69"
        )
        assert (
            client.get(
                "/api/ppe-human-qa-r6/tiles/..%2Freceipt.json", headers=_headers()
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/api/ppe-human-qa-r6/tiles/ppe-r6-not-real-0001",
                headers=_headers(),
            ).status_code
            == 404
        )


def test_sample_put_uses_optimistic_revision_locks_identity_and_strict_export(ppe_env):
    path = f"/api/ppe-human-qa-r6/samples/{FIRST_SAMPLE}"
    with TestClient(app) as client:
        detail = client.get(path, headers=_headers())
        assert detail.status_code == 200
        assert detail.headers["etag"] == '"ppe-r6-r0"'
        assert detail.json()["training_authorized"] is False

        missing_match = client.put(
            path,
            headers=_headers(**{"X-Reviewer-ID": "reviewer-a"}),
            json=_sample_body(),
        )
        assert missing_match.status_code == 428
        saved = client.put(
            path,
            headers=_headers(
                **{
                    "X-Reviewer-ID": "reviewer-a",
                    "If-Match": '"ppe-r6-r0"',
                }
            ),
            json=_sample_body(),
        )
        assert saved.status_code == 200
        assert saved.headers["etag"] == '"ppe-r6-r1"'
        assert saved.json()["training_authorized"] is False
        assert saved.json()["production_ready"] is False

        stale = client.put(
            path,
            headers=_headers(
                **{
                    "X-Reviewer-ID": "reviewer-a",
                    "If-Match": '"ppe-r6-r0"',
                }
            ),
            json=_sample_body(),
        )
        assert stale.status_code == 412
        wrong_reviewer = client.put(
            path,
            headers=_headers(
                **{
                    "X-Reviewer-ID": "reviewer-b",
                    "If-Match": '"ppe-r6-r1"',
                }
            ),
            json=_sample_body(),
        )
        assert wrong_reviewer.status_code == 409

        export = client.get("/api/ppe-human-qa-r6/export", headers=_headers())
        assert export.status_code == 200
        value = export.json()
        schema = json.loads(
            (SCHEMAS / "ppe-human-qa-adjudication-r6-v1.schema.json").read_text()
        )
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).validate(value)
        assert value["reviewer_identity"] == "reviewer-a"
        assert value["completion"]["decided_samples"] == 1
        assert value["training_authorized"] is False
        assert value["production_ready"] is False
        assert value["authorization_effect"] == AUTHORIZATION_EFFECT


def test_concurrent_same_revision_has_exactly_one_append_only_winner(ppe_env):
    service = PpeHumanQaService(PpeHumanQaConfig.from_env())

    def write() -> str:
        try:
            service.put_sample(
                FIRST_SAMPLE,
                expected_revision=0,
                reviewer_identity="reviewer-a",
                decision="approve_sample_semantics",
                reason_code="visually_correct",
                notes="",
            )
            return "saved"
        except PpeHumanQaError as exc:
            return exc.state

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = sorted(pool.map(lambda _: write(), range(2)))
    assert results == ["saved", "stale_revision"]
    queue = service.queue(limit=1)
    assert queue["audit"]["event_count"] == 1
    assert queue["progress"]["decided_samples"] == 1


def test_policy_put_is_revisioned_and_never_authorizes_training(ppe_env):
    policy = POLICY_NAMES[0]
    with TestClient(app) as client:
        before = client.get("/api/ppe-human-qa-r6/export", headers=_headers())
        assert before.status_code == 409
        saved = client.put(
            f"/api/ppe-human-qa-r6/policies/{policy}",
            headers=_headers(
                **{
                    "X-Reviewer-ID": "reviewer-a",
                    "If-Match": '"ppe-r6-r0"',
                }
            ),
            json={"decision": "reject", "notes": "Policy basis needs remediation."},
        )
        assert saved.status_code == 200
        assert saved.headers["etag"] == '"ppe-r6-r1"'
        assert saved.json()["training_authorized"] is False
        queue = client.get("/api/ppe-human-qa-r6/queue", headers=_headers()).json()
        assert queue["progress"]["decided_policies"] == 1
        assert queue["progress"]["decided_samples"] == 0
        assert queue["progress"]["training_authorized"] is False


def test_audit_rows_are_append_only_and_hash_tamper_fails_closed(ppe_env):
    path = f"/api/ppe-human-qa-r6/samples/{FIRST_SAMPLE}"
    with TestClient(app) as client:
        assert (
            client.put(
                path,
                headers=_headers(
                    **{
                        "X-Reviewer-ID": "reviewer-a",
                        "If-Match": '"ppe-r6-r0"',
                    }
                ),
                json=_sample_body(),
            ).status_code
            == 200
        )
        database = ppe_env["data"] / "ppe-human-qa-r6/adjudication.sqlite3"
        connection = sqlite3.connect(database)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM audit_events")
        connection.rollback()
        connection.execute("DROP TRIGGER ppe_r6_audit_no_update")
        connection.execute(
            "UPDATE audit_events SET event_json = ? WHERE sequence = 1", ("{}",)
        )
        connection.commit()
        connection.close()
        assert (
            client.get("/api/ppe-human-qa-r6/queue", headers=_headers()).status_code
            == 503
        )


def test_receipt_or_schema_drift_fails_before_any_tile_or_raw_payload_open(
    ppe_env, tmp_path, monkeypatch
):
    bad_packet = tmp_path / "bad-packet"
    bad_packet.mkdir()
    receipt = (PACKET / "receipt.json").read_bytes() + b" "
    (bad_packet / "receipt.json").write_bytes(receipt)
    monkeypatch.setenv("DEEPSAFE_PPE_HUMAN_QA_PACKET", str(bad_packet))
    with TestClient(app) as client:
        assert (
            client.get("/api/ppe-human-qa-r6/queue", headers=_headers()).status_code
            == 503
        )

    monkeypatch.setenv("DEEPSAFE_PPE_HUMAN_QA_PACKET", str(PACKET))
    bad_schemas = tmp_path / "bad-schemas"
    bad_schemas.mkdir()
    for name in (
        "ppe-human-qa-adjudication-r6-v1.schema.json",
        "ppe-human-qa-packet-receipt-r6-v1.schema.json",
        "ppe-human-qa-sample-r6-v1.schema.json",
    ):
        shutil.copyfile(SCHEMAS / name, bad_schemas / name)
    with (bad_schemas / "ppe-human-qa-adjudication-r6-v1.schema.json").open("ab") as handle:
        handle.write(b" ")
    monkeypatch.setenv("DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(bad_schemas))
    with TestClient(app) as client:
        assert (
            client.get("/api/ppe-human-qa-r6/queue", headers=_headers()).status_code
            == 503
        )


def test_development_holdout_reference_is_rejected_before_visual_access(
    ppe_env, tmp_path, monkeypatch
):
    bad_packet = tmp_path / "excluded-packet"
    bad_packet.mkdir()
    rows = [json.loads(line) for line in (PACKET / "samples.jsonl").read_text().splitlines()]
    rows[0]["source"]["image"]["path"] = rows[0]["source"]["image"][
        "path"
    ].replace("/train/", "/development_holdout/")
    samples_payload = b"".join(_canonical(row) + b"\n" for row in rows)
    samples_pin = {
        "bytes": len(samples_payload),
        "sha256": hashlib.sha256(samples_payload).hexdigest(),
    }
    (bad_packet / "samples.jsonl").write_bytes(samples_payload)

    receipt = json.loads((PACKET / "receipt.json").read_text())
    receipt["selection"]["samples"] = {"path": "samples.jsonl", **samples_pin}
    receipt.pop("receipt_sha256")
    receipt_sha = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt["receipt_sha256"] = receipt_sha
    receipt_payload = _canonical(receipt) + b"\n"
    (bad_packet / "receipt.json").write_bytes(receipt_payload)

    monkeypatch.setattr(ppe_module, "SAMPLES_FILE", samples_pin)
    monkeypatch.setattr(ppe_module, "PACKET_RECEIPT_SHA256", receipt_sha)
    monkeypatch.setattr(
        ppe_module,
        "PACKET_RECEIPT_FILE",
        {
            "bytes": len(receipt_payload),
            "sha256": hashlib.sha256(receipt_payload).hexdigest(),
        },
    )
    service = PpeHumanQaService(
        PpeHumanQaConfig(
            packet_root=bad_packet,
            schema_root=SCHEMAS,
            data_dir=ppe_env["data"] / "excluded",
            max_tile_bytes=4 * 1024 * 1024,
        )
    )
    with pytest.raises(PpeHumanQaError) as caught:
        service.queue()
    assert caught.value.state == "excluded_payload_reference"
    assert not (bad_packet / "artifact-manifest.jsonl").exists()


def test_audit_store_unavailable_and_parent_or_database_symlinks_fail_closed(
    ppe_env, tmp_path, monkeypatch
):
    with TestClient(app) as client:
        not_a_directory = tmp_path / "not-a-directory"
        not_a_directory.write_text("blocked")
        monkeypatch.setenv("DEEPSAFE_DATA", str(not_a_directory))
        assert (
            client.get("/api/ppe-human-qa-r6/queue", headers=_headers()).status_code
            == 503
        )

        target_root = tmp_path / "target-root"
        target_root.mkdir()
        linked_root = tmp_path / "linked-root"
        linked_root.symlink_to(target_root, target_is_directory=True)
        monkeypatch.setenv("DEEPSAFE_DATA", str(linked_root))
        assert (
            client.get("/api/ppe-human-qa-r6/queue", headers=_headers()).status_code
            == 503
        )

        ordinary_root = tmp_path / "ordinary-root"
        audit_dir = ordinary_root / "ppe-human-qa-r6"
        audit_dir.mkdir(parents=True)
        target_db = tmp_path / "target.sqlite3"
        target_db.write_bytes(b"")
        (audit_dir / "adjudication.sqlite3").symlink_to(target_db)
        monkeypatch.setenv("DEEPSAFE_DATA", str(ordinary_root))
        assert (
            client.get("/api/ppe-human-qa-r6/queue", headers=_headers()).status_code
            == 503
        )
        assert target_db.read_bytes() == b""


def test_actual_complete_audit_is_immutable_and_export_remains_non_authorizing(ppe_env):
    service = PpeHumanQaService(PpeHumanQaConfig.from_env())
    context = service._context()
    connection = service._connect()
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
    events = []
    previous = None
    sequence = 0
    rows = []
    for sample in context.samples:
        reason = (
            "unknown_must_not_be_negative_ground_truth"
            if sample["candidate_state"] == "zero_label_unknown_context"
            else "visually_correct"
        )
        rows.append(
            (
                "sample",
                sample["sample_id"],
                {
                    "sample_id": sample["sample_id"],
                    "sample_row_sha256": context.sample_hashes[sample["sample_id"]],
                    "decision": "approve_sample_semantics",
                    "reason_code": reason,
                    "notes": "",
                    "decided_at": now,
                },
            )
        )
    for policy in POLICY_NAMES:
        rows.append(
            (
                "policy",
                policy,
                {
                    "decision": "approve",
                    "notes": "Human policy decision only.",
                    "decided_at": now,
                },
            )
        )
    for entity_type, entity_id, decision_row in rows:
        sequence += 1
        event = {
            "schema_version": ppe_module.AUDIT_SCHEMA_VERSION,
            "sequence": sequence,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "entity_revision": 1,
            "recorded_at": now,
            "reviewer_identity": "reviewer-a",
            "packet_receipt_sha256": PACKET_RECEIPT_SHA256,
            "packet_file_sha256": PACKET_RECEIPT_FILE["sha256"],
            "decision_row": decision_row,
            "decision_row_sha256": hashlib.sha256(_canonical(decision_row)).hexdigest(),
            "previous_event_sha256": previous,
        }
        event["event_sha256"] = hashlib.sha256(_canonical(event)).hexdigest()
        events.append(
            (
                sequence,
                entity_type,
                entity_id,
                1,
                now,
                _canonical(event).decode(),
                event["event_sha256"],
                previous,
            )
        )
        previous = event["event_sha256"]
    connection.executemany(
        "INSERT INTO audit_events "
        "(sequence, entity_type, entity_id, entity_revision, recorded_at, "
        "event_json, event_sha256, previous_event_sha256) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        events,
    )
    connection.commit()
    connection.close()

    with TestClient(app) as client:
        queue = client.get("/api/ppe-human-qa-r6/queue", headers=_headers())
        assert queue.status_code == 200
        assert queue.json()["progress"]["state"] == "complete"
        update = client.put(
            f"/api/ppe-human-qa-r6/samples/{FIRST_SAMPLE}",
            headers=_headers(
                **{
                    "X-Reviewer-ID": "reviewer-a",
                    "If-Match": '"ppe-r6-r1"',
                }
            ),
            json=_sample_body(),
        )
        assert update.status_code == 409
        export = client.get("/api/ppe-human-qa-r6/export", headers=_headers())
        assert export.status_code == 200
        value = export.json()
        assert value["state"] == "complete"
        assert value["completion"]["decided_samples"] == 718
        assert value["completion"]["all_policies_decided"] is True
        assert value["training_authorized"] is False
        assert value["production_ready"] is False
        assert value["authorization_effect"] == AUTHORIZATION_EFFECT


def test_complete_export_still_requires_new_exact_training_authorization(ppe_env):
    service = PpeHumanQaService(
        PpeHumanQaConfig(
            packet_root=PACKET,
            schema_root=SCHEMAS,
            data_dir=ppe_env["data"] / "synthetic-complete",
            max_tile_bytes=4 * 1024 * 1024,
        )
    )
    context = service._context()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    heads = {}
    for sample in context.samples:
        reason = (
            "unknown_must_not_be_negative_ground_truth"
            if sample["candidate_state"] == "zero_label_unknown_context"
            else "visually_correct"
        )
        row = {
            "sample_id": sample["sample_id"],
            "sample_row_sha256": context.sample_hashes[sample["sample_id"]],
            "decision": "approve_sample_semantics",
            "reason_code": reason,
            "notes": "",
            "decided_at": now,
        }
        heads[("sample", sample["sample_id"])] = {"decision_row": row}
    for policy in POLICY_NAMES:
        heads[("policy", policy)] = {
            "decision_row": {
                "decision": "approve",
                "notes": "Human policy decision only.",
                "decided_at": now,
            }
        }
    value = service._build_export(
        context,
        AuditState(
            heads=heads,
            event_count=724,
            reviewer_identity="reviewer-a",
            complete=True,
            completed_at=now,
            chain_head_sha256="0" * 64,
        ),
    )
    assert value["state"] == "complete"
    assert value["completion"]["decided_samples"] == 718
    assert value["completion"]["all_policies_decided"] is True
    assert value["training_authorized"] is False
    assert value["production_ready"] is False
    assert value["authorization_effect"] == AUTHORIZATION_EFFECT


def test_admin_ui_exposes_ppe_queue_filters_policies_and_guardrails():
    html = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    for marker in (
        'id="ppeHumanQaR6Section"',
        'id="ppeQaQueue"',
        'id="ppeQaCategory"',
        'id="ppeQaPolicies"',
        "loadPpeHumanQa(true)",
        "savePpeHumanQaSample()",
        "savePpeHumanQaPolicy",
        "/api/ppe-human-qa-r6/export",
        "yeni bir exact eğitim yetkilendirmesi gerekir",
    ):
        assert marker in html


def test_container_contract_includes_runtime_dependency_and_read_only_packet_mount():
    requirements = (ROOT / "admin/requirements.txt").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()
    dockerfile = (ROOT / "admin/Dockerfile").read_text()
    assert "jsonschema==4.26.0" in requirements
    assert (
        "DEEPSAFE_PPE_HUMAN_QA_PACKET: "
        "/workspace/validation-results/ppe/human-qa/mendeley-ppe-four-class-r6"
        in compose
    )
    assert "./validation/results:/workspace/validation-results:ro" in compose
    assert "COPY admin /app/admin" in dockerfile
    assert "COPY validation/schemas /app/validation/schemas" in dockerfile
