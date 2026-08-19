"""Read-only admin projection for the DeepStream 9.1 static preflight."""

from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validation.deepstream91_qualification import (
    DEFAULT_CONTRACT,
    DEFAULT_DRIVER_FILE,
    DEFAULT_OS_RELEASE,
    QualificationError,
    build_report,
    load_contract,
    read_driver_version,
    read_os_release,
    validate_report,
)


def load_deepstream91_static_status(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    driver_file: Path | None = None,
    os_release_path: Path | None = None,
    machine: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Project compact evidence without exposing any execution action."""

    driver_path = driver_file or Path(
        os.getenv("DEEPSAFE_DS91_HOST_DRIVER_FILE", str(DEFAULT_DRIVER_FILE))
    )
    release_path = os_release_path or Path(
        os.getenv("DEEPSAFE_DS91_HOST_OS_RELEASE", str(DEFAULT_OS_RELEASE))
    )
    try:
        contract = load_contract(contract_path)
        os_release = read_os_release(release_path)
        report = build_report(
            contract,
            driver_version=read_driver_version(driver_path),
            os_id=os_release["ID"],
            os_version=os_release["VERSION_ID"],
            machine=machine or platform.machine(),
            generated_at_utc=(
                generated_at_utc
                or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            ),
        )
        validate_report(report, contract)
    except (OSError, QualificationError, ValueError, TypeError, AttributeError):
        return {
            "label": "DeepStream 9.1 statik yeterlilik",
            "state": "unavailable_integrity_error",
            "reason": "static_qualification_evidence_unavailable",
            "available": False,
            "static_prerequisites_met": False,
            "live_probe_performed": False,
            "production_ready": False,
            "read_only": True,
            "execution_actions_available": False,
            "caveats": [
                "Statik kanıt okunamadı; canlı runtime veya ürün hazırlığı iddia edilemez."
            ],
            "evidence": [],
        }

    static_met = report["status"] == "static_prerequisites_met_live_probe_not_run"
    return {
        "label": "DeepStream 9.1 statik yeterlilik",
        "state": report["status"],
        "reason": (
            report["blockers"][0]
            if report["blockers"]
            else "live_runtime_probe_not_run"
        ),
        "available": True,
        "static_prerequisites_met": static_met,
        "live_probe_performed": False,
        "production_ready": False,
        "read_only": True,
        "execution_actions_available": False,
        "contract_fingerprint_sha256": report["contract_fingerprint_sha256"],
        "runtime": {
            "deepstream": report["target"]["deepstream"],
            "cuda": report["target"]["cuda"],
            "tensorrt": report["target"]["tensorrt"],
            "image": report["target"]["image_ref"],
            "image_index_digest": report["target"]["image_index_digest"],
            "linux_amd64_manifest_digest": report["target"][
                "linux_amd64_manifest_digest"
            ],
        },
        "host": dict(report["observed_host"]),
        "required_host": dict(report["required_host"]),
        "checks": dict(report["checks"]),
        "blockers": list(report["blockers"]),
        "actions_performed": dict(report["actions_performed"]),
        "caveats": [
            "Bu kart yalnız CPU-only statik önkoşulları gösterir.",
            "Canlı image, ABI, TensorRT, DeepStream ve GPU smoke ayrı kapılardır.",
        ],
        "evidence": [],
    }
