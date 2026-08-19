from __future__ import annotations

from pathlib import Path

from admin import gpu_lease_v5_status as lease_status
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_lease_v5_card_replays_exact_closed_contract() -> None:
    card = lease_status.load_gpu_lease_v5_status()
    assert card["state"] == "contract_exact_host_tool_pins_stale"
    assert card["available"] is True
    assert card["contract_verified"] is True
    assert card["host_tools_current"] is False
    assert card["current_host_replay_eligible"] is False
    assert card["current_host_replay_blocker"] == "trusted executable hash differs before activation"
    assert card["host_tools"]["docker_cli"]["matches_contract"] is False
    assert card["host_tools"]["nvidia_smi"]["matches_contract"] is False
    assert card["contract_fingerprint_sha256"] == lease_status.FINGERPRINT
    assert card["live_plan_published"] is False
    assert card["activation_receipt_published"] is False
    assert card["execution_authorized"] is False
    assert card["gpu_or_docker_called_during_validation"] is False
    assert card["read_only"] is True
    assert card["execution_actions_available"] is False
    assert card["production_ready"] is False
    assert card["tests"] == {
        "focused_passed": 26,
        "regression_passed": 164,
        "failed": 0,
        "independent_review": "pass",
        "scope": "frozen_acceptance_baseline_not_current_host_replay",
        "p0": 0,
        "p1": 0,
        "p2": 0,
    }


def test_gpu_lease_v5_card_fails_closed_on_pin_drift(monkeypatch) -> None:
    pins = dict(lease_status.PINS)
    relative = "validation/contracts/gpu-lease-v5.json"
    pins[relative] = (pins[relative][0], "0" * 64)
    monkeypatch.setattr(lease_status, "PINS", pins)
    card = lease_status.load_gpu_lease_v5_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["contract_verified"] is False
    assert card["execution_authorized"] is False


def test_validation_payload_and_ui_expose_closed_gpu_plan_gate() -> None:
    payload = load_validation_status()
    card = payload["campaigns"]["gpu_lease_v5"]
    assert card["state"] == "contract_exact_host_tool_pins_stale"
    assert card["host_tools_current"] is False
    assert payload["read_only"] is True
    assert payload["execution_actions_available"] is False
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "gpu_lease_v5" in page
    assert "campaign.execution_authorized" in page
    assert "fetch('/api/gpu-lease/v5/run')" not in page


def test_admin_mounts_gpu_lease_controls_and_results_read_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "./validation/results:/workspace/validation-results:ro" in compose
    assert "./validation:/workspace/validation:ro" in compose
    assert "/usr/bin/docker:/host-tools/docker:ro" in compose
    assert "/usr/bin/nvidia-smi:/host-tools/nvidia-smi:ro" in compose


def test_gpu_lease_v5_card_keeps_execution_closed_when_host_tools_match(monkeypatch) -> None:
    monkeypatch.setattr(
        lease_status,
        "_host_tool_status",
        lambda contract: {
            "current": True,
            "tools": {
                "docker_cli": {"matches_contract": True},
                "nvidia_smi": {"matches_contract": True},
            },
        },
    )
    card = lease_status.load_gpu_lease_v5_status()
    assert card["state"] == "contract_verified_no_live_plan"
    assert card["host_tools_current"] is True
    assert card["current_host_replay_eligible"] is True
    assert card["current_host_replay_blocker"] is None
    assert card["execution_authorized"] is False
