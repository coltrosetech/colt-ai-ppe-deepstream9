import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app, pipeline
from admin.manual_review import ManualReviewError, ManualReviewService
from tests.test_campaign_report import _write_verified_synthetic_distance_v2_final
from validation import report_campaign as campaign


TOKEN = "manual-review-test-token"
PNG = b"\x89PNG\r\n\x1a\nfixture"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source(index: int, *, sensitive: bool = False) -> dict:
    source_id = f"source-{index:02d}"
    observation = {
        "person_expected": "not_reviewed" if sensitive else "yes",
        "visible_person_count_range": {
            "min": None if sensitive else 1,
            "max": None if sensitive else 1,
        },
        "scorable_person_count_range": {
            "min": None if sensitive else 1,
            "max": None if sensitive else 1,
        },
        "scale_classes": [] if sensitive else ["medium"],
        "dominant_scale": None if sensitive else "medium",
        "occlusion": "unknown" if sensitive else "none",
        "top_view": False,
        "high_oblique": not sensitive,
        "medium_close": not sensitive,
        "partial_body_only": False,
        "ambiguity": [],
        "notes": [],
    }
    return {
        "schema_version": "deepsafe.open-video-source-frame-review/v1",
        "record_id": source_id,
        "scene_id": "sensitive-scene" if sensitive else f"scene-{index:02d}",
        "sensitive": sensitive,
        "frame": {"index": index + 10, "timestamp_seconds": float(index)},
        "window": {
            "segment_label": "closed" if sensitive else "person-visible",
            "segment_role": "closed_review" if sensitive else "person_visible",
            "negative_window": False,
        },
        "observation": observation,
    }


def _asset_id(decision_id: str, kind: str) -> str:
    return "a_" + hashlib.sha256(f"{decision_id}:{kind}".encode()).hexdigest()


@pytest.fixture
def manual_env(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    root.mkdir()
    contents = {
        "source_image": PNG,
        "overlay_image": PNG,
        "review_report": b'{"ok":true}\n',
        "predictions": b'{"frame":1}\n',
    }
    media_types = {
        "source_image": "image/png",
        "overlay_image": "image/png",
        "review_report": "application/json",
        "predictions": "application/x-ndjson",
    }
    paths = {}
    for kind, content in contents.items():
        suffix = ".png" if kind.endswith("image") else (".json" if kind == "review_report" else ".jsonl")
        path = root / f"shared-{kind}{suffix}"
        path.write_bytes(content)
        paths[kind] = path

    ordinary = [_source(index) for index in range(21)]
    sensitive = _source(99, sensitive=True)
    source_path = tmp_path / "sources.jsonl"
    source_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in [*ordinary, sensitive]) + "\n",
        encoding="utf-8",
    )

    assets = {}
    decisions = []
    for source in ordinary:
        for profile in (640, 960):
            decision_id = f"{source['record_id']}:{profile}"
            evidence = {}
            for kind, content in contents.items():
                asset_id = _asset_id(decision_id, kind)
                evidence[kind] = asset_id
                assets[asset_id] = {
                    "decision_id": decision_id,
                    "source_review_id": source["record_id"],
                    "scene_id": source["scene_id"],
                    "frame_index": source["frame"]["index"],
                    "model_input": profile,
                    "kind": kind,
                    "relative_path": paths[kind].relative_to(root).as_posix(),
                    "sha256": _sha(content),
                    "media_type": media_types[kind],
                    "size_bytes": len(content),
                }
            decisions.append(
                {
                    "decision_id": decision_id,
                    "source_review_id": source["record_id"],
                    "scene_id": source["scene_id"],
                    "frame_index": source["frame"]["index"],
                    "model_input": profile,
                    "source_observation": source["observation"],
                    "evidence": evidence,
                }
            )
    index_path = tmp_path / "index.json"

    def write_index():
        index_path.write_text(
            json.dumps(
                {
                    "schema_version": "deepsafe.open-video-manual-assets/v1",
                    "status": "complete",
                    "bundle_id": "b_" + "b" * 64,
                    "decision_count": 42,
                    "asset_count": 168,
                    "profiles": [640, 960],
                    "review_confidence_threshold": 0.25,
                    "sensitive_media_included": False,
                    "source_records_sha256": _sha(source_path.read_bytes()),
                    "campaign_plan_sha256": "c" * 64,
                    "decisions": decisions,
                    "assets": assets,
                    "input_provenance": {"videos": {}, "jobs": []},
                    "metric_guardrail": "Sampled decisions are not dense ground truth or dataset-level metrics.",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    write_index()
    data = tmp_path / "data"
    monkeypatch.setenv("DEEPSAFE_ADMIN_TOKEN", TOKEN)
    monkeypatch.setenv("DEEPSAFE_DATA", str(data))
    monkeypatch.setenv("DEEPSAFE_MANUAL_REVIEW_SOURCE", str(source_path))
    monkeypatch.setenv("DEEPSAFE_MANUAL_REVIEW_ASSET_INDEX", str(index_path))
    monkeypatch.setenv("DEEPSAFE_MANUAL_REVIEW_ASSET_ROOT", str(root))
    monkeypatch.setenv("DEEPSAFE_MANUAL_REVIEW_MAX_ASSET_BYTES", "1048576")
    return {
        "root": root,
        "data": data,
        "source_path": source_path,
        "index_path": index_path,
        "assets": assets,
        "decisions": decisions,
        "paths": paths,
        "write_index": write_index,
    }


def _headers(**extra):
    return {"Authorization": f"Bearer {TOKEN}"} | extra


def _reviewed_body(**updates):
    body = {
        "status": "reviewed",
        "detection_count_reviewed": 1,
        "visible_person_count_confirmed": 1,
        "scorable_person_count_confirmed": 1,
        "true_positive_count": 1,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "ignored_detection_count": 0,
        "unscorable_visible_person_count": 0,
        "reasons": [],
        "reviewer_type": "human_with_ai_assist",
    }
    body.update(updates)
    return body


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_review_auth_is_fail_closed_for_default_and_required_for_every_method(manual_env, monkeypatch):
    endpoints = [
        ("GET", "/api/manual-review/queue"),
        ("GET", "/api/manual-review/decisions/source-00:640"),
        ("GET", "/api/manual-review/assets/0123456789abcdef"),
        ("PUT", "/api/manual-review/decisions/source-00:640"),
    ]
    with TestClient(app) as client:
        for method, path in endpoints:
            assert client.request(method, path).status_code == 401
        assert client.get("/api/manual-review/queue", headers={"Authorization": "Bearer wrong"}).status_code == 401

        for unsafe in ("", "change-me"):
            monkeypatch.setenv("DEEPSAFE_ADMIN_TOKEN", unsafe)
            for method, path in endpoints:
                assert client.request(method, path, headers={"Authorization": f"Bearer {unsafe}"}).status_code == 503


def test_queue_and_detail_omit_sensitive_rows_and_internal_paths_or_hashes(manual_env):
    with TestClient(app) as client:
        response = client.get("/api/manual-review/queue", headers=_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["source_count"] == 21
        assert payload["decision_count"] == 42
        assert payload["status_counts"]["pending_review"] == 42
        assert all(
            {asset["kind"] for asset in profile["assets"]}
            == {"source_image", "overlay_image"}
            for item in payload["items"]
            for profile in item["profiles"]
        )
        serialized = json.dumps(payload)
        assert "sensitive-scene" not in serialized
        assert str(manual_env["root"]) not in serialized
        forbidden = ("path", "sha", "hash", "command", "uri")
        assert not any(any(part in key.lower() for part in forbidden) for key in _all_keys(payload))

        detail = client.get(
            "/api/manual-review/decisions/source-00:640", headers=_headers()
        )
        assert detail.status_code == 200
        assert detail.headers["etag"] == '"mr-r0"'
        assert len(detail.json()["comparison_profiles"]) == 2
        assert client.get(
            "/api/manual-review/decisions/source-99:640", headers=_headers()
        ).status_code == 404


def test_index_is_cryptographically_bound_to_the_read_only_source_file(manual_env):
    original = manual_env["source_path"].read_text(encoding="utf-8")
    manual_env["source_path"].write_text(original + "\n", encoding="utf-8")
    with TestClient(app) as client:
        response = client.get("/api/manual-review/queue", headers=_headers())
    assert response.status_code == 503
    assert str(manual_env["source_path"]) not in response.text


@pytest.mark.parametrize(
    "surface",
    ["source_duplicate", "source_nonfinite", "index_duplicate", "index_nonfinite"],
)
def test_queue_rejects_ambiguous_or_nonfinite_source_and_index_json(
    manual_env, surface
):
    if surface.startswith("source"):
        path = manual_env["source_path"]
        before = path.read_text(encoding="utf-8")
        if surface == "source_duplicate":
            after = before.replace(
                '"person_expected": "yes"',
                '"person_expected": "no", "person_expected": "yes"',
                1,
            )
        else:
            after = before.replace('"timestamp_seconds": 0.0', '"timestamp_seconds": NaN', 1)
    else:
        path = manual_env["index_path"]
        before = path.read_text(encoding="utf-8")
        if surface == "index_duplicate":
            after = before.replace(
                '"status": "complete"',
                '"status": "incomplete", "status": "complete"',
                1,
            )
        else:
            after = before.replace(
                '"review_confidence_threshold": 0.25',
                '"review_confidence_threshold": Infinity',
                1,
            )
    assert after != before
    path.write_text(after, encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/api/manual-review/queue", headers=_headers())
    assert response.status_code == 503
    assert str(path) not in response.text


@pytest.mark.parametrize(
    "raw",
    [
        '{"outer":{"value":1,"value":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
    ],
)
def test_decision_store_json_loader_rejects_duplicate_and_nonfinite_values(raw):
    with pytest.raises(ManualReviewError) as captured:
        ManualReviewService._load_json(raw, "invalid_decision_store")
    assert captured.value.state == "invalid_decision_store"
    assert captured.value.status_code == 503


def test_asset_delivery_is_opaque_bounded_and_nosniff(manual_env):
    decision = manual_env["decisions"][0]
    asset_id = decision["evidence"]["source_image"]
    with TestClient(app) as client:
        response = client.get(f"/api/manual-review/assets/{asset_id}", headers=_headers())
        assert response.status_code == 200
        assert response.content == PNG
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-content-type-options"] == "nosniff"
        report_id = decision["evidence"]["review_report"]
        assert client.get(f"/api/manual-review/assets/{report_id}", headers=_headers()).status_code == 404
        assert client.get("/api/manual-review/assets/..%2Fsecret", headers=_headers()).status_code == 404


@pytest.mark.parametrize("failure", ["traversal", "symlink", "hash", "signature", "oversize"])
def test_asset_failures_are_rejected_without_leaking_files(manual_env, tmp_path, failure):
    decision = manual_env["decisions"][0]
    asset_id = decision["evidence"]["overlay_image"]
    asset = manual_env["assets"][asset_id]
    if failure == "traversal":
        asset["relative_path"] = "../secret.png"
    elif failure == "symlink":
        outside = tmp_path / "outside.png"
        outside.write_bytes(PNG)
        link = manual_env["root"] / "linked.png"
        link.symlink_to(outside)
        asset.update(relative_path="linked.png", sha256=_sha(PNG), size_bytes=len(PNG))
    elif failure == "hash":
        asset["sha256"] = "0" * 64
    elif failure == "signature":
        bad = b"not-a-png"
        manual_env["paths"]["overlay_image"].write_bytes(bad)
        asset.update(sha256=_sha(bad), size_bytes=len(bad))
    else:
        asset["size_bytes"] = 1_048_577
    manual_env["write_index"]()

    with TestClient(app) as client:
        response = client.get(f"/api/manual-review/assets/{asset_id}", headers=_headers())
    assert response.status_code in {404, 503}
    assert str(tmp_path) not in response.text


def test_put_requires_revision_enforces_reconciliation_and_source_ranges(manual_env):
    path = "/api/manual-review/decisions/source-00:640"
    base_headers = _headers(**{"X-Reviewer-ID": "operator-1"})
    with TestClient(app) as client:
        assert client.put(path, headers=base_headers, json=_reviewed_body()).status_code == 428
        invalid = _reviewed_body(false_positive_count=1)
        response = client.put(
            path,
            headers=base_headers | {"If-Match": '"mr-r0"'},
            json=invalid,
        )
        assert response.status_code == 422
        outside = _reviewed_body(
            detection_count_reviewed=2,
            visible_person_count_confirmed=2,
            scorable_person_count_confirmed=2,
            true_positive_count=2,
        )
        assert client.put(
            path,
            headers=base_headers | {"If-Match": '"mr-r0"'},
            json=outside,
        ).status_code == 422
        ambiguous = _reviewed_body(
            status="ambiguous",
            detection_count_reviewed=None,
            visible_person_count_confirmed=None,
            scorable_person_count_confirmed=None,
            true_positive_count=None,
            false_positive_count=None,
            false_negative_count=None,
            ignored_detection_count=None,
            unscorable_visible_person_count=None,
            reasons=[],
        )
        assert client.put(
            path,
            headers=base_headers | {"If-Match": '"mr-r0"'},
            json=ambiguous,
        ).status_code == 422


def test_put_accepts_counts_inside_ai_supported_expanded_source_range(
    manual_env,
):
    sources = [
        json.loads(line)
        for line in manual_env["source_path"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    source = next(row for row in sources if row["record_id"] == "source-00")
    source["observation"]["visible_person_count_range"] = {"min": 4, "max": 5}
    source["observation"]["scorable_person_count_range"] = {"min": 3, "max": 3}
    manual_env["source_path"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in sources) + "\n",
        encoding="utf-8",
    )
    for decision in manual_env["decisions"]:
        if decision["source_review_id"] == source["record_id"]:
            decision["source_observation"] = source["observation"]
    manual_env["write_index"]()

    body = _reviewed_body(
        detection_count_reviewed=3,
        visible_person_count_confirmed=4,
        scorable_person_count_confirmed=3,
        true_positive_count=3,
        unscorable_visible_person_count=1,
    )
    with TestClient(app) as client:
        response = client.put(
            "/api/manual-review/decisions/source-00:640",
            headers=_headers(
                **{"X-Reviewer-ID": "operator-1", "If-Match": '"mr-r0"'}
            ),
            json=body,
        )

    assert response.status_code == 200


def test_valid_put_binds_server_fields_exports_jsonl_and_keeps_audit_immutable(manual_env):
    path = "/api/manual-review/decisions/source-00:640"
    headers = _headers(**{"X-Reviewer-ID": "operator-1", "If-Match": '"mr-r0"'})
    with TestClient(app) as client:
        response = client.put(path, headers=headers, json=_reviewed_body())
        assert response.status_code == 200
        assert response.headers["etag"] == '"mr-r1"'
        selected = response.json()["selected"]
        assert response.json()["persistence"] == {
            "decision_committed": True,
            "exports_current": True,
        }
        assert selected["revision"] == 1
        assert "review" not in selected
        assert "overlay_evidence" not in selected

        stale = client.put(path, headers=headers, json=_reviewed_body())
        assert stale.status_code == 409

    export_dir = manual_env["data"] / "manual-review"
    decisions = [json.loads(line) for line in (export_dir / "overlay-decisions-v1.jsonl").read_text().splitlines()]
    assert len(decisions) == 42
    stored = next(row for row in decisions if row["decision_id"] == "source-00:640")
    assert stored["source_review_id"] == "source-00"
    assert stored["scene_id"] == "scene-00"
    assert stored["frame_index"] == 10
    assert stored["model_input"] == 640
    assert stored["review"]["reviewer_id"] == "operator-1"
    assert stored["review"]["reviewed_at"].endswith("+00:00")
    assert stored["overlay_evidence"]["overlay_image_sha256"] == _sha(PNG)

    events = [json.loads(line) for line in (export_dir / "audit-events-v1.jsonl").read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["revision"] == 1
    assert events[0]["previous_event_sha256"] is None
    database = export_dir / "manual-review.sqlite3"
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE audit_events SET recorded_at = 'tampered'")


def _restore_audit_triggers(connection):
    connection.executescript(
        """
        CREATE TRIGGER audit_events_no_update
        BEFORE UPDATE ON audit_events BEGIN
            SELECT RAISE(ABORT, 'audit events are immutable');
        END;
        CREATE TRIGGER audit_events_no_delete
        BEFORE DELETE ON audit_events BEGIN
            SELECT RAISE(ABORT, 'audit events are immutable');
        END;
        """
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "event_json",
        "malformed_event_json",
        "binary_event_json",
        "event_sha256",
        "previous_event_sha256",
        "audit_revision",
        "current_revision",
        "current_row_sha256",
        "current_row_json_serialization",
        "orphan_current",
    ],
)
def test_audit_or_current_tampering_fails_closed_before_read_or_write(
    manual_env, tamper
):
    path = "/api/manual-review/decisions/source-00:640"
    write_headers = _headers(
        **{"X-Reviewer-ID": "operator-1", "If-Match": '"mr-r0"'}
    )
    with TestClient(app) as client:
        assert client.put(path, headers=write_headers, json=_reviewed_body()).status_code == 200

    database = manual_env["data"] / "manual-review" / "manual-review.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER audit_events_no_update")
        connection.execute("DROP TRIGGER audit_events_no_delete")
        if tamper == "event_json":
            raw = connection.execute(
                "SELECT event_json FROM audit_events WHERE decision_id = ?",
                ("source-00:640",),
            ).fetchone()[0]
            event = json.loads(raw)
            event["reviewer_id"] = "tampered-operator"
            connection.execute(
                "UPDATE audit_events SET event_json = ? WHERE decision_id = ?",
                (
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "source-00:640",
                ),
            )
        elif tamper == "malformed_event_json":
            connection.execute(
                "UPDATE audit_events SET event_json = 7 WHERE decision_id = ?",
                ("source-00:640",),
            )
        elif tamper == "binary_event_json":
            connection.execute(
                "UPDATE audit_events SET event_json = ? WHERE decision_id = ?",
                (sqlite3.Binary(b"\xff\xfe"), "source-00:640"),
            )
        elif tamper == "event_sha256":
            connection.execute(
                "UPDATE audit_events SET event_sha256 = ? WHERE decision_id = ?",
                ("0" * 64, "source-00:640"),
            )
        elif tamper == "previous_event_sha256":
            connection.execute(
                "UPDATE audit_events SET previous_event_sha256 = ? WHERE decision_id = ?",
                ("0" * 64, "source-00:640"),
            )
        elif tamper == "audit_revision":
            connection.execute(
                "UPDATE audit_events SET revision = 5 WHERE decision_id = ?",
                ("source-00:640",),
            )
        elif tamper == "current_revision":
            connection.execute(
                "UPDATE current_decisions SET revision = 5 WHERE decision_id = ?",
                ("source-00:640",),
            )
        elif tamper == "current_row_sha256":
            connection.execute(
                "UPDATE current_decisions SET row_sha256 = ? WHERE decision_id = ?",
                ("0" * 64, "source-00:640"),
            )
        elif tamper == "current_row_json_serialization":
            raw = connection.execute(
                "SELECT row_json FROM current_decisions WHERE decision_id = ?",
                ("source-00:640",),
            ).fetchone()[0]
            connection.execute(
                "UPDATE current_decisions SET row_json = ? WHERE decision_id = ?",
                (" " + raw, "source-00:640"),
            )
        else:
            connection.execute(
                "DELETE FROM audit_events WHERE decision_id = ?",
                ("source-00:640",),
            )
        _restore_audit_triggers(connection)
        connection.commit()
        before = connection.execute(
            "SELECT revision FROM current_decisions WHERE decision_id = ?",
            ("source-00:640",),
        ).fetchone()[0]
        before_count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    with TestClient(app) as client:
        assert client.get("/api/manual-review/queue", headers=_headers()).status_code == 503
        assert client.get(path, headers=_headers()).status_code == 503
        attempted = client.put(
            path,
            headers=_headers(
                **{"X-Reviewer-ID": "operator-2", "If-Match": '"mr-r1"'}
            ),
            json=_reviewed_body(),
        )
        assert attempted.status_code == 503

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT revision FROM current_decisions WHERE decision_id = ?",
            ("source-00:640",),
        ).fetchone()[0] == before
        assert connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == before_count


def test_two_revision_audit_chain_is_canonical_and_linked(manual_env):
    path = "/api/manual-review/decisions/source-00:640"
    with TestClient(app) as client:
        first = client.put(
            path,
            headers=_headers(
                **{"X-Reviewer-ID": "operator-1", "If-Match": '"mr-r0"'}
            ),
            json=_reviewed_body(),
        )
        assert first.status_code == 200
        second = client.put(
            path,
            headers=_headers(
                **{"X-Reviewer-ID": "operator-2", "If-Match": '"mr-r1"'}
            ),
            json=_reviewed_body(reasons=["second inspection"]),
        )
        assert second.status_code == 200
        assert second.headers["etag"] == '"mr-r2"'

    database = manual_env["data"] / "manual-review" / "manual-review.sqlite3"
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT event_json, event_sha256, previous_event_sha256 "
            "FROM audit_events WHERE decision_id = ? ORDER BY revision",
            ("source-00:640",),
        ).fetchall()
    assert len(rows) == 2
    previous = None
    for raw, stored_sha, stored_previous in rows:
        event = json.loads(raw)
        claimed = event.pop("event_sha256")
        calculated = _sha(
            json.dumps(
                event,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        assert claimed == stored_sha == calculated
        assert event["previous_event_sha256"] == stored_previous == previous
        previous = claimed


def test_paired_current_and_audit_deletion_is_caught_by_genesis_ledger(manual_env):
    path = "/api/manual-review/decisions/source-00:640"
    with TestClient(app) as client:
        assert client.put(
            path,
            headers=_headers(
                **{"X-Reviewer-ID": "operator-1", "If-Match": '"mr-r0"'}
            ),
            json=_reviewed_body(),
        ).status_code == 200

    database = manual_env["data"] / "manual-review" / "manual-review.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER audit_events_no_update")
        connection.execute("DROP TRIGGER audit_events_no_delete")
        connection.execute(
            "DELETE FROM audit_events WHERE decision_id = ?", ("source-00:640",)
        )
        connection.execute(
            "DELETE FROM current_decisions WHERE decision_id = ?", ("source-00:640",)
        )
        _restore_audit_triggers(connection)
        connection.commit()

    with TestClient(app) as client:
        assert client.get("/api/manual-review/queue", headers=_headers()).status_code == 503
        assert client.put(
            path,
            headers=_headers(
                **{"X-Reviewer-ID": "operator-2", "If-Match": '"mr-r0"'}
            ),
            json=_reviewed_body(),
        ).status_code == 503


def test_inert_audit_trigger_contract_is_rejected(manual_env):
    with TestClient(app) as client:
        assert client.put(
            "/api/manual-review/decisions/source-00:640",
            headers=_headers(
                **{"X-Reviewer-ID": "operator-1", "If-Match": '"mr-r0"'}
            ),
            json=_reviewed_body(),
        ).status_code == 200

    database = manual_env["data"] / "manual-review" / "manual-review.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER audit_events_no_update")
        connection.execute(
            """CREATE TRIGGER audit_events_no_update
               BEFORE UPDATE ON audit_events WHEN 0 BEGIN
                   SELECT RAISE(ABORT, 'audit events are immutable');
               END"""
        )
        connection.commit()
    with TestClient(app) as client:
        assert client.get("/api/manual-review/queue", headers=_headers()).status_code == 503


def test_export_failure_reports_committed_sqlite_state_without_false_503(
    manual_env, monkeypatch
):
    def fail_export(*_args, **_kwargs):
        raise ManualReviewError("decision_export_unavailable")

    monkeypatch.setattr(
        ManualReviewService, "_atomic_jsonl", staticmethod(fail_export)
    )
    path = "/api/manual-review/decisions/source-00:640"
    with TestClient(app) as client:
        response = client.put(
            path,
            headers=_headers(
                **{"X-Reviewer-ID": "operator-1", "If-Match": '"mr-r0"'}
            ),
            json=_reviewed_body(),
        )
        assert response.status_code == 200
        assert response.headers["etag"] == '"mr-r1"'
        assert response.json()["persistence"] == {
            "decision_committed": True,
            "exports_current": False,
        }
        assert client.get(path, headers=_headers()).status_code == 200

    database = manual_env["data"] / "manual-review" / "manual-review.sqlite3"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT revision FROM current_decisions WHERE decision_id = ?",
            ("source-00:640",),
        ).fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 1


def test_two_writers_with_same_etag_cannot_both_commit(manual_env):
    path = "/api/manual-review/decisions/source-00:640"
    headers = _headers(**{"X-Reviewer-ID": "operator", "If-Match": '"mr-r0"'})

    def submit():
        with TestClient(app) as client:
            return client.put(path, headers=headers, json=_reviewed_body()).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = sorted(pool.map(lambda _value: submit(), range(2)))
    assert statuses == [200, 409]


def test_manual_review_routes_have_no_gpu_action(manual_env, monkeypatch):
    def forbidden_start(*_args, **_kwargs):
        raise AssertionError("manual review must not start the pipeline")

    monkeypatch.setattr(pipeline, "start", forbidden_start)
    review_routes = [route for route in app.routes if route.path.startswith("/api/manual-review")]
    assert all("POST" not in (route.methods or set()) for route in review_routes)
    with TestClient(app) as client:
        assert client.get("/api/manual-review/queue", headers=_headers()).status_code == 200


def test_admin_html_uses_safe_dom_and_session_scoped_token():
    with TestClient(app) as client:
        html = client.get("/").text
    assert 'id="manualReviewSection"' in html
    assert "sessionStorage" in html
    assert "localStorage" not in html
    assert "sources.innerHTML" not in html
    assert "textContent=source.name+' — '+source.uri" in html
    assert "saved.persistence.exports_current===false" in html
    assert "Bu alan pipeline veya GPU işi başlatmaz" in html
    assert "review-evidence-pair" in html
    assert "['source_image','Hash-bağlı ham kaynak kare']" in html
    assert "['overlay_image','Hash-bağlı model overlay']" in html
    assert "loadProtectedImage(img,asset)" in html


def test_loaf_dataset_rights_are_projected_as_bounded_booleans(tmp_path, monkeypatch):
    report_dir = tmp_path / "campaign-report"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.validation-campaign-report/v1",
                "generated_at_utc": "2026-07-16T06:00:00+00:00",
                "decision": {
                    "status": "blocked_by_hardware",
                    "accepted": False,
                    "final_claim_allowed": False,
                },
                "requirement_summary": {"state_counts": {}},
                "requirements": [],
                "campaigns": {
                    "loaf_preparation": {
                        "state": "prepared_not_evaluated",
                        "acceptance_effect": "none",
                        "can_satisfy_calibrated_25m_detection": False,
                        "splits": {},
                        "dataset_rights": {
                            "license_status": "unverified",
                            "internal_research_validation_only": True,
                            "model_training_allowed": False,
                            "redistribution_allowed": False,
                            "written_rights_clearance_required": True,
                            "guardrail_consistent": True,
                            "private_rights_path": "/private/rights.pdf",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text("# bounded report\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/validation")
    rights = response.json()["campaigns"]["campaign_report"]["loaf_preparation"][
        "dataset_rights"
    ]
    assert rights == {
        "license_status": "unverified",
        "internal_research_validation_only": True,
        "model_training_allowed": False,
        "redistribution_allowed": False,
        "written_rights_clearance_required": True,
        "guardrail_consistent": True,
    }
    assert "/private/rights.pdf" not in response.text


def _site_plan():
    return {
        "schema_version": "deepsafe.site-distance-evaluation-plan/v1",
        "status": "waiting_for_inputs",
        "dry_run": True,
        "gpu_or_docker_executed": False,
        "final_evaluation_written": False,
        "required_profiles": [640, 960],
        "distance_bin_m": [20, 25],
        "boundary": "lower_inclusive_upper_exclusive",
        "inputs": {
            key: {"path": f"/host/private/{key}.json", "present": False}
            for key in (
                "calibration",
                "ground_truth",
                "acceptance",
                "profile_640",
                "profile_960",
            )
        },
        "output": "/host/private/evaluation.json",
        "acceptance_policy": "no default threshold; final evaluation is written only when the owner-approved artifact and both completed profiles pass",
        "loaf_policy": "LOAF is auxiliary evidence and cannot substitute for deployment-camera calibration",
    }


def _site_pin(root: Path, path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": len(content),
        "sha256": _sha(content),
    }


def _site_write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _site_input_fingerprint(pins: list[dict], config: dict) -> str:
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
    return _sha(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    )


def _site_evaluation(root: Path):
    artifacts = {
        "distance_25m_implementation": _site_write(
            root / "validation/site_distance_evaluation.py",
            (PROJECT_ROOT / "validation/site_distance_evaluation.py").read_bytes(),
        ),
        "distance_25m_calibration": _site_write(
            root / "site/calibration.json", b'{"calibration":"verified"}\n'
        ),
        "distance_25m_calibration_verification": _site_write(
            root / "site/calibration-verification.md", b"verified survey\n"
        ),
        "distance_25m_ground_truth": _site_write(
            root / "site/ground-truth.json", b'{"ground_truth":"complete"}\n'
        ),
        "distance_25m_source_asset": _site_write(
            root / "site/source.mp4", b"site-source-video"
        ),
        "distance_25m_annotation_document": _site_write(
            root / "site/annotation-qa.md", b"annotation verified\n"
        ),
        "distance_25m_acceptance": _site_write(
            root / "site/acceptance.json", b'{"criterion":"owner-recall-v1"}\n'
        ),
        "distance_25m_acceptance_approval": _site_write(
            root / "site/acceptance-approval.md", b"owner approved\n"
        ),
        "distance_25m_profile_640_manifest": _site_write(
            root / "site/profiles/640/run-manifest.json", b'{"status":"complete"}\n'
        ),
        "distance_25m_profile_640_predictions": _site_write(
            root / "site/profiles/640/predictions.jsonl", b'{"frame":0}\n'
        ),
        "distance_25m_profile_960_manifest": _site_write(
            root / "site/profiles/960/run-manifest.json", b'{"status":"complete"}\n'
        ),
        "distance_25m_profile_960_predictions": _site_write(
            root / "site/profiles/960/predictions.jsonl", b'{"frame":0}\n'
        ),
    }
    pins = {key: _site_pin(root, path) for key, path in artifacts.items()}

    def profile(model_input):
        prefix = f"distance_25m_profile_{model_input}"
        return {
            "status": "complete",
            "model_input": model_input,
            "model_id": f"person-{model_input}",
            "completion_manifest": pins[f"{prefix}_manifest"],
            "predictions": pins[f"{prefix}_predictions"],
            "frame_records": 2,
            "ground_truth_instances": 2,
            "metrics": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.666667,
                "ap_101_point": 0.5,
                "ignored_predictions": 0,
                "predictions_excluded_outside_calibrated_band": 0,
            },
        }

    config = {
        "iou_threshold": 0.5,
        "confidence_threshold": 0.25,
        "distance_point": "bbox_bottom_center_ground_contact",
        "metric_geometry": "axis_aligned_bbox_iou",
        "out_of_band_policy": "GT outside 20<=d<25m is ignore; predictions outside the calibrated 20<=d<25m ROI are excluded",
    }
    input_pins = [
        pin
        for key, pin in pins.items()
        if key != "distance_25m_implementation"
    ]
    input_pins = sorted(input_pins, key=lambda pin: (pin["path"], pin["sha256"]))
    evaluation = {
        "schema_version": "deepsafe.distance-validation/v1",
        "status": "complete",
        "evidence_kind": "deployment_site_calibrated_ground_plane_person_detection",
        "generated_at": "2026-07-16T06:00:00+00:00",
        "implementation": pins["distance_25m_implementation"],
        "calibration": {
            "status": "verified",
            "model": "planar_homography_image_to_ground",
            "calibration_id": "cal-1",
            "site_id": "site-1",
            "camera_id": "camera-1",
            "camera_configuration_id": "camera-config-1",
            "artifact": pins["distance_25m_calibration"],
            "verification_document": pins[
                "distance_25m_calibration_verification"
            ],
        },
        "ground_truth": {
            "status": "complete",
            "evidence_kind": "deployment_site_calibrated_person_ground_truth",
            "dataset_id": "site-gt-1",
            "sequence_id": "site-sequence-1",
            "artifact": pins["distance_25m_ground_truth"],
            "source_asset": pins["distance_25m_source_asset"],
            "frames": 2,
            "all_person_instances": 2,
            "ground_truth_instances_20_25m": 2,
            "annotation_document": pins["distance_25m_annotation_document"],
        },
        "distance_unit": "m",
        "distance_bin_m": [20, 25],
        "boundary": "lower_inclusive_upper_exclusive",
        "evaluation_config": config,
        "profiles": {"640": profile(640), "960": profile(960)},
        "acceptance": {
            "status": "pass",
            "criterion": "owner-approved owner-recall-v1: recall >= 0.5 for each profile",
            "criterion_id": "owner-recall-v1",
            "minimum_ground_truth_instances_per_profile": 2,
            "minimum_ground_truth_status": "pass",
            "artifact": pins["distance_25m_acceptance"],
            "approval_document": pins["distance_25m_acceptance_approval"],
            "rules": [
                {
                    "metric": "recall",
                    "operator": "gte",
                    "threshold": 0.5,
                    "applies_to": "each_profile",
                    "profile_values": {"640": 0.5, "960": 0.5},
                    "status": "pass",
                }
            ],
        },
        "integrity": {
            "frame_set_status": "exact_for_ground_truth_and_both_profiles",
            "frame_key_sha256": "c" * 64,
            "profile_pair_status": "complete_640_and_960",
            "input_fingerprint_sha256": _site_input_fingerprint(
                input_pins, config
            ),
            "input_artifacts": input_pins,
        },
        "loaf_evidence": {
            "used": False,
            "role": "auxiliary_only_not_deployment_site_calibration",
            "can_substitute_for_site_calibration": False,
        },
        "gpu_or_docker_executed_by_evaluator": False,
    }
    return evaluation, artifacts


def _write_site_report(root: Path, evaluation_content: bytes, evaluation: dict):
    profiles = {
        key: {
            "ground_truth_instances": value["ground_truth_instances"],
            "frame_records": value["frame_records"],
            "model_id": value["model_id"],
            "metrics": value["metrics"],
        }
        for key, value in evaluation["profiles"].items()
    }
    report = json.loads(
        (PROJECT_ROOT / "validation/results/campaign-report/report.json").read_text()
    )
    distance = report["campaigns"]["distance_25m"]
    pin_media = {
        "distance_25m_implementation": "text/x-python",
        "distance_25m_calibration": "application/json",
        "distance_25m_calibration_verification": "text/markdown",
        "distance_25m_ground_truth": "application/json",
        "distance_25m_source_asset": "application/octet-stream",
        "distance_25m_annotation_document": "text/markdown",
        "distance_25m_acceptance": "application/json",
        "distance_25m_acceptance_approval": "text/markdown",
        "distance_25m_profile_640_manifest": "application/json",
        "distance_25m_profile_640_predictions": "application/x-ndjson",
        "distance_25m_profile_960_manifest": "application/json",
        "distance_25m_profile_960_predictions": "application/x-ndjson",
    }
    evaluation_pins = {
        "distance_25m_implementation": evaluation["implementation"],
        "distance_25m_calibration": evaluation["calibration"]["artifact"],
        "distance_25m_calibration_verification": evaluation["calibration"][
            "verification_document"
        ],
        "distance_25m_ground_truth": evaluation["ground_truth"]["artifact"],
        "distance_25m_source_asset": evaluation["ground_truth"]["source_asset"],
        "distance_25m_annotation_document": evaluation["ground_truth"][
            "annotation_document"
        ],
        "distance_25m_acceptance": evaluation["acceptance"]["artifact"],
        "distance_25m_acceptance_approval": evaluation["acceptance"][
            "approval_document"
        ],
        "distance_25m_profile_640_manifest": evaluation["profiles"]["640"][
            "completion_manifest"
        ],
        "distance_25m_profile_640_predictions": evaluation["profiles"]["640"][
            "predictions"
        ],
        "distance_25m_profile_960_manifest": evaluation["profiles"]["960"][
            "completion_manifest"
        ],
        "distance_25m_profile_960_predictions": evaluation["profiles"]["960"][
            "predictions"
        ],
    }
    expected_ids = {"distance_25m_evaluation", *evaluation_pins}
    distance.update(
        {
            "state": "proven",
            "accepted": True,
            "reasons": [],
            "schema_contract_valid": True,
            "pin_matrix_valid": True,
            "independent_cpu_recomputation_valid": True,
            "criterion": evaluation["acceptance"]["criterion"],
            "profiles": profiles,
            "evidence_ids": sorted(expected_ids),
        }
    )
    report["evidence"] = [
        row for row in report["evidence"] if row.get("id") not in expected_ids
    ]
    report["evidence"].append(
        {
            "id": "distance_25m_evaluation",
            "state": "ok",
            "path": "distance-25m/evaluation.json",
            "media_type": "application/json",
            "sha256": _sha(evaluation_content),
            "size_bytes": len(evaluation_content),
            "schema_version": "deepsafe.distance-validation/v1",
        }
    )
    for evidence_id, pin in evaluation_pins.items():
        report["evidence"].append(
            {
                "id": evidence_id,
                "state": "ok",
                "path": pin["path"],
                "media_type": pin_media[evidence_id],
                "sha256": pin["sha256"],
                "size_bytes": pin["bytes"],
            }
        )
    report["evidence"].sort(key=lambda row: row["id"])
    report_dir = root / "campaign-report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_site_25m_plan_waiting_never_becomes_proven_and_hides_raw_inputs(tmp_path, monkeypatch):
    result_dir = tmp_path / "distance-25m"
    result_dir.mkdir()
    (result_dir / "evaluation-plan.json").write_text(
        json.dumps(_site_plan()), encoding="utf-8"
    )
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/validation")
        direct = client.get(
            "/api/validation", params={"artifact": "site_distance_plan"}
        )
    card = response.json()["campaigns"]["site_distance_25m"]
    assert card["state"] == "waiting_for_inputs"
    assert card["proven"] is False
    assert card["accepted"] is False
    assert card["final_evaluation_present"] is False
    assert card["input_readiness"] == {
        "ready": 0,
        "required": 5,
        "plan_contract_valid": True,
    }
    assert "/host/private" not in response.text
    assert "private_command" not in response.text
    assert direct.status_code == 404


def test_site_25m_plan_with_extra_command_fails_strict_schema_without_leak(
    tmp_path, monkeypatch
):
    result_dir = tmp_path / "distance-25m"
    result_dir.mkdir()
    plan = _site_plan()
    plan["private_command"] = ["never", "project"]
    (result_dir / "evaluation-plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/validation")
    card = response.json()["campaigns"]["site_distance_25m"]
    assert card["state"] == "artifact_error"
    assert card["input_readiness"]["plan_contract_valid"] is False
    assert "private_command" not in response.text
    assert "never" not in response.text


def test_site_25m_final_without_bound_campaign_recomputation_stays_unproven(tmp_path, monkeypatch):
    result_dir = tmp_path / "distance-25m"
    result_dir.mkdir()
    evaluation, _ = _site_evaluation(tmp_path)
    (result_dir / "evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(tmp_path))
    with TestClient(app) as client:
        card = client.get("/api/validation").json()["campaigns"][
            "site_distance_25m"
        ]
    assert card["state"] == "unproven"
    assert card["final_evaluation_present"] is True
    assert card["proven"] is False
    assert card["profiles"] == {}


def test_site_25m_requires_exact_final_sha_and_independent_report_proof(tmp_path, monkeypatch):
    _, documents = _write_verified_synthetic_distance_v2_final(tmp_path)
    production = {
        "workspace_scoped_inputs": True,
        "fixture_markers_absent": True,
        "minimum_source_bytes": campaign.DISTANCE_V2_MINIMUM_SOURCE_BYTES,
        "minimum_per_bin_instances": campaign.DISTANCE_V2_MINIMUM_BIN_INSTANCES,
        "minimum_per_bin_independent_events": campaign.DISTANCE_V2_MINIMUM_BIN_EVENTS,
        "minimum_per_bin_unambiguous_events": (
            campaign.DISTANCE_V2_MINIMUM_UNAMBIGUOUS_EVENTS
        ),
        "minimum_endpoint_independent_events": (
            campaign.DISTANCE_V2_MINIMUM_ENDPOINT_EVENTS
        ),
        "minimum_exact_25m_instances": (
            campaign.DISTANCE_V2_MINIMUM_EXACT_25_INSTANCES
        ),
    }
    monkeypatch.setattr(
        campaign,
        "_distance_v2_production_contract",
        lambda payload, root: (True, production, documents),
    )
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    report_path = tmp_path / "validation/results/campaign-report/report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        admin_validation,
        "_site_distance_v2_production_contract",
        lambda reader, payload: (True, production, documents),
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_ROOT", str(tmp_path / "validation/results")
    )
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT",
        str(campaign.PROJECT_ROOT / "validation/schemas"),
    )
    with TestClient(app) as client:
        response = client.get("/api/validation")
        direct = client.get(
            "/api/validation", params={"artifact": "site_distance_evaluation_v2"}
        )
    card = response.json()["campaigns"]["site_distance_25m"]
    assert card["state"] == "proven"
    assert card["proven"] is True
    assert card["accepted"] is True
    assert card["calibration_verified"] is True
    assert card["profiles"]["640"]["metrics"]["recall"] == 1.0
    assert card["criterion_id"] == "owner_approved_paired_rules_v2"
    assert card["exact_25m_instances"] == 1
    assert "/host/private" not in response.text
    assert direct.status_code == 404

    malformed_report = json.loads(report_path.read_text(encoding="utf-8"))
    malformed_report["campaigns"]["distance_25m"]["profiles"]["640"][
        "metrics"
    ] = []
    report_path.write_text(json.dumps(malformed_report), encoding="utf-8")
    with TestClient(app) as client:
        malformed = client.get("/api/validation")
    assert malformed.status_code == 200
    malformed_card = malformed.json()["campaigns"]["site_distance_25m"]
    assert malformed_card["proven"] is False
    assert malformed_card["verification"]["live_semantic_replay"] is True
    assert malformed_card["verification"]["campaign_report_hash_binding"] is False

    # Preserve the receipt's semantic JSON while changing its exact file bytes.
    # The live replay still succeeds, but the stale report SHA must close the gate.
    report_path.write_text(json.dumps(report), encoding="utf-8")
    final_path = (
        tmp_path
        / "validation/results/distance-25m/evaluation-final-v2-001.json"
    )
    final_path.write_bytes(final_path.read_bytes() + b"\n")
    with TestClient(app) as client:
        stale = client.get("/api/validation").json()["campaigns"][
            "site_distance_25m"
        ]
    assert stale["state"] == "unproven"
    assert stale["proven"] is False
    assert stale["verification"]["live_semantic_replay"] is True
    assert stale["verification"]["campaign_report_hash_binding"] is False


@pytest.mark.parametrize(
    "tamper",
    (
        "live_prediction",
        "report_pin_sha",
        "report_pin_path",
        "report_evidence_ids",
        "report_extra_command",
        "final_fingerprint",
        "final_schema",
    ),
)
def test_site_25m_live_pin_and_recomputation_bindings_fail_closed(
    tmp_path, monkeypatch, tamper
):
    result_dir = tmp_path / "distance-25m"
    result_dir.mkdir()
    evaluation, artifacts = _site_evaluation(tmp_path)
    evaluation_path = result_dir / "evaluation.json"
    evaluation_content = json.dumps(evaluation).encode()
    evaluation_path.write_bytes(evaluation_content)
    _write_site_report(tmp_path, evaluation_content, evaluation)
    report_path = tmp_path / "campaign-report/report.json"

    if tamper == "live_prediction":
        artifacts["distance_25m_profile_640_predictions"].write_bytes(
            b'{"frame":"tampered"}\n'
        )
    else:
        report = json.loads(report_path.read_text())
        if tamper == "report_pin_sha":
            row = next(
                item
                for item in report["evidence"]
                if item["id"] == "distance_25m_profile_640_predictions"
            )
            row["sha256"] = "0" * 64
        elif tamper == "report_pin_path":
            duplicate = _site_write(
                tmp_path / "site/profiles/640/predictions-copy.jsonl",
                artifacts["distance_25m_profile_640_predictions"].read_bytes(),
            )
            row = next(
                item
                for item in report["evidence"]
                if item["id"] == "distance_25m_profile_640_predictions"
            )
            row["path"] = str(duplicate.relative_to(tmp_path))
        elif tamper == "report_evidence_ids":
            report["campaigns"]["distance_25m"]["evidence_ids"].remove(
                "distance_25m_profile_640_predictions"
            )
        elif tamper == "report_extra_command":
            report["private_command"] = ["must", "not", "project"]
        elif tamper == "final_fingerprint":
            evaluation["integrity"]["input_fingerprint_sha256"] = "0" * 64
            evaluation_content = json.dumps(evaluation).encode()
            evaluation_path.write_bytes(evaluation_content)
            row = next(
                item
                for item in report["evidence"]
                if item["id"] == "distance_25m_evaluation"
            )
            row["sha256"] = _sha(evaluation_content)
            row["size_bytes"] = len(evaluation_content)
        elif tamper == "final_schema":
            evaluation["integrity"].pop("input_fingerprint_sha256")
            evaluation_content = json.dumps(evaluation).encode()
            evaluation_path.write_bytes(evaluation_content)
            row = next(
                item
                for item in report["evidence"]
                if item["id"] == "distance_25m_evaluation"
            )
            row["sha256"] = _sha(evaluation_content)
            row["size_bytes"] = len(evaluation_content)
        report_path.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(tmp_path))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/validation")
    card = response.json()["campaigns"]["site_distance_25m"]
    assert card["state"] == "unproven"
    assert card["proven"] is False
    assert card["accepted"] is False
    assert card["profiles"] == {}
    assert "private_command" not in response.text
    assert "must" not in response.text
