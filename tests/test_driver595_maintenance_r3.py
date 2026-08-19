from __future__ import annotations

from validation import driver595_maintenance_r3 as r3


def _fixture(*, extra: tuple[str, ...] = ()) -> str:
    lines = ["NOTE: This is only a simulation!"]
    lines.extend(f"Inst {package} (1 test [amd64])" for package in sorted(r3.EXPECTED_INSTALL))
    lines.extend(f"Remv {package} [1]" for package in sorted(r3.EXPECTED_REMOVE))
    lines.extend(extra)
    lines.append("0 upgraded, 22 newly installed, 16 to remove and 147 not upgraded.")
    return "\n".join(lines)


def test_exact_simulation_shape_passes() -> None:
    report = r3.parse_apt_simulation(_fixture())
    assert report["status"] == "pass"
    assert report["missing_install"] == []
    assert report["unexpected_install"] == []
    assert report["missing_remove"] == []
    assert report["unexpected_remove"] == []


def test_dkms_or_foreign_kernel_install_is_blocked() -> None:
    report = r3.parse_apt_simulation(
        _fixture(extra=("Inst nvidia-dkms-595 (1 test [amd64])",))
    )
    assert report["status"] == "blocked"
    assert report["checks"]["no_595_dkms_install"] is False
    assert report["unexpected_install"] == ["nvidia-dkms-595"]


def test_missing_old_driver_removal_is_blocked() -> None:
    output = _fixture().replace("Remv nvidia-driver-590 [1]\n", "")
    report = r3.parse_apt_simulation(output)
    assert report["status"] == "blocked"
    assert report["missing_remove"] == ["nvidia-driver-590"]


def test_install_request_is_exact_and_explicitly_forbids_dkms() -> None:
    assert len(r3.EXPECTED_INSTALL) == 22
    assert len(r3.EXPECTED_REMOVE) == 16
    assert f"nvidia-driver-595={r3.TARGET_DRIVER_DEB}" in r3.INSTALL_REQUEST
    assert (
        f"linux-modules-nvidia-595-generic-hwe-24.04={r3.TARGET_KERNEL_DEB}"
        in r3.INSTALL_REQUEST
    )
    assert (
        f"linux-modules-nvidia-595-{r3.FALLBACK_KERNEL}={r3.FALLBACK_KERNEL_DEB}"
        in r3.INSTALL_REQUEST
    )
    assert "nvidia-dkms-590-" in r3.INSTALL_REQUEST
    assert "nvidia-dkms-595-" in r3.INSTALL_REQUEST


def test_helper_has_no_apply_mode() -> None:
    assert "apply" not in {"simulate", "verify-pre-reboot", "verify-post-reboot", "print-install"}
