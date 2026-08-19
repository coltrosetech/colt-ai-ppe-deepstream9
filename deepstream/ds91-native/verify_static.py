#!/usr/bin/env python3
"""Fail-closed static verifier for the DeepStream 9.1 native build path."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCKERFILE = HERE / "Dockerfile"
BUILD_SCRIPT = HERE / "build.sh"
CONTRACT = HERE / "native-build-contract-v1.json"

BASE_DIGEST = "sha256:f6fa0247da9290979cbb05749e7da9435d089c93db7c4dcfe85ba2488b5f4994"
BASE_REF = f"nvcr.io/nvidia/deepstream@{BASE_DIGEST}"


def fail(message: str) -> None:
    raise AssertionError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        fail(f"{label}: expected exactly one occurrence, found {count}")


def main() -> int:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    if contract["schema_version"] != "deepsafe.ds91-native-build/v1":
        fail("unexpected contract schema")
    if contract["base"]["reference"] != BASE_REF:
        fail("contract base reference drift")
    if contract["base"]["digest"] != BASE_DIGEST:
        fail("contract base digest drift")
    if contract["base"]["platform"] != "linux/amd64":
        fail("contract platform drift")

    if dockerfile.count(f"FROM {BASE_REF} AS ") != 2:
        fail("build and runtime stages must hard-code the exact base reference")
    if re.search(r"^FROM\s+\$", dockerfile, flags=re.MULTILINE):
        fail("base image selection must not be build-argument overridable")

    for context_name in contract["named_contexts"]:
        marker = f"COPY --from={context_name} / "
        require_once(dockerfile, marker, f"named context {context_name}")
        marker = f"--build-context {context_name}="
        require_once(build_script, marker, f"build script context {context_name}")

    required_build_flags = (
        "--network=none",
        "--platform=linux/amd64",
        "--load",
        "--provenance=false",
    )
    for flag in required_build_flags:
        if flag not in build_script:
            fail(f"missing fail-closed build flag: {flag}")

    forbidden = {
        "remote URL": r"https?://",
        "curl": r"\bcurl\b",
        "wget": r"\bwget\b",
        "package manager": r"\b(?:apt|apt-get|dnf|yum|apk)\b",
        "git clone": r"\bgit\s+clone\b",
        "pip install": r"\bpip(?:3)?\s+install\b",
        "nvidia-smi": r"\bnvidia-smi\b",
        "trtexec": r"\btrtexec\b",
        "GPU device request": r"(?:--gpus|NVIDIA_VISIBLE_DEVICES)",
    }
    scanned = dockerfile + "\n" + build_script
    for label, pattern in forbidden.items():
        if re.search(pattern, scanned, flags=re.IGNORECASE):
            fail(f"forbidden {label} in executable build surface")

    for relative, expected in contract["legacy_files_not_modified"].items():
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            fail(f"frozen legacy file changed: {relative}: {actual}")

    for context_name, relative in contract["named_contexts"].items():
        path = ROOT / relative
        if not path.is_dir():
            fail(f"missing named context {context_name}: {relative}")

    print("DeepStream 9.1 native static contract: PASS")
    print(f"base={BASE_REF}")
    print("platform=linux/amd64 network=none gpu=false inference=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"DeepStream 9.1 native static contract: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
