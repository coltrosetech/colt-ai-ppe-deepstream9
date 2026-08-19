from __future__ import annotations

import base64
import hashlib
import io
import json
import stat
import zipfile
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import ppe_dataset.acquisition as acquisition
from ppe_dataset.acquisition import (
    ZipLimits,
    acquire_pinned_seed_asset,
    inspect_seed_zip,
)
from ppe_dataset.cli import main


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        url: str = "https://cdn.example.invalid/seed",
    ) -> None:
        self.stream = io.BytesIO(body)
        self.status = status
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value
        self.url = url

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self.url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class InterruptingResponse(FakeResponse):
    def __init__(self, body: bytes, split: int, **kwargs: object) -> None:
        super().__init__(body, **kwargs)
        self.split = split
        self.calls = 0

    def read(self, _size: int = -1) -> bytes:
        self.calls += 1
        if self.calls == 1:
            return self.stream.read(self.split)
        raise URLError("fixture connection cut")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


PNG_BLACK = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAD0lEQVQIHWNkAANGBjAAAAAjAAMz85CnAAAAAElFTkSuQmCC"
)
PNG_WHITE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVQIHWP8//8/AwMD4////xkYGAAy8AX9wEV12gAAAABJRU5ErkJggg=="
)


def write_manifest(
    tmp_path: Path,
    artifact_bytes: bytes,
    *,
    filename: str = "seed.zip",
    classes: list[str] | None = None,
    declared_images: int | None = None,
    declared_width: int | None = None,
    declared_height: int | None = None,
    include_artifact: bool = True,
) -> Path:
    artifact = None
    if include_artifact:
        artifact = {
            "filename": filename,
            "url": "https://example.invalid/seed",
            "local_path": f"raw/{filename}",
            "bytes": len(artifact_bytes),
            "sha256": sha256(artifact_bytes),
        }
    declared: dict[str, object] = {"classes": classes or ["helmet", "no_helmet"]}
    if declared_images is not None:
        declared["images"] = declared_images
    if declared_width is not None:
        declared["image_width"] = declared_width
    if declared_height is not None:
        declared["image_height"] = declared_height
    manifest = {
        "schema_version": "deepsafe.ppe-training-seed-sources/v1",
        "sources": [
            {
                "id": "fixture-seed",
                "artifact": artifact,
                "download_all_url": "https://example.invalid/unpinned.zip",
                "declared_content": declared,
                "eligibility": {
                    "download": True,
                    "quarantine_inspection": True,
                    "training": False,
                    "blockers": ["archive_structure_and_label_semantics_not_inspected", "provenance_incomplete"],
                },
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def make_zip(entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in entries:
            archive.writestr(name, body)
    return stream.getvalue()


def valid_zip(*, duplicate_images: bool = False, bad_label: bool = False) -> bytes:
    yaml = b"train: train/images\nval: valid/images\nnc: 2\nnames: [helmet, no_helmet]\n"
    entries: list[tuple[str | zipfile.ZipInfo, bytes]] = [
        ("data.yaml", yaml),
        ("train/images/a.png", PNG_BLACK),
        (
            "train/labels/a.txt",
            b"0 1.2 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n"
            if bad_label
            else b"0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n",
        ),
    ]
    entries.extend(
        [
            ("valid/images/b.png", PNG_BLACK if duplicate_images else PNG_WHITE),
            ("valid/labels/b.txt", b"1 0.5 0.5 0.2 0.2\n"),
        ]
    )
    return make_zip(entries)


def yolo_zip_with_config(
    yaml_body: bytes,
    *,
    yaml_path: str = "data.yaml",
    train_image: bytes = PNG_BLACK,
    validation_image: bytes = PNG_WHITE,
) -> bytes:
    return make_zip(
        [
            (yaml_path, yaml_body),
            ("train/images/a.png", train_image),
            ("train/labels/a.txt", b"0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n"),
            ("valid/images/b.png", validation_image),
            ("valid/labels/b.txt", b"1 0.5 0.5 0.2 0.2\n"),
        ]
    )


def materialized_archive(tmp_path: Path, body: bytes, *, images: int = 1) -> tuple[Path, Path]:
    manifest = write_manifest(
        tmp_path,
        body,
        classes=["helmet", "no_helmet"],
        declared_images=images,
    )
    archive = tmp_path / "raw" / "seed.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(body)
    return manifest, archive


def test_acquire_streams_and_atomically_publishes_pinned_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"PK\x03\x04fixture-pinned-zip"
    manifest = write_manifest(tmp_path, body)
    response = FakeResponse(
        body,
        headers={
            "Content-Type": "application/zip",
            "Content-Length": str(len(body)),
            "ETag": '"fixture-v1"',
        },
    )
    monkeypatch.setattr(acquisition, "urlopen", lambda *_args, **_kwargs: response)

    receipt = acquire_pinned_seed_asset(
        manifest,
        "fixture-seed",
        "artifact",
        root=tmp_path,
    )

    destination = tmp_path / "raw" / "seed.zip"
    assert receipt["accepted"] is True
    assert receipt["training_eligible"] is False
    assert destination.read_bytes() == body
    assert destination.stat().st_nlink == 1
    assert destination.stat().st_mode & 0o777 == 0o440
    assert not list(destination.parent.glob(".*.tmp"))


@pytest.mark.parametrize("hazard", ["duplicate_key", "manifest_symlink"])
def test_manifest_snapshot_is_strict_and_never_reaches_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hazard: str,
) -> None:
    body = b"PK\x03\x04expected"
    manifest = write_manifest(tmp_path, body)
    if hazard == "duplicate_key":
        source = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
        manifest.write_text(
            '{"sources":[],"sources":' + json.dumps(source) + "}",
            encoding="utf-8",
        )
    else:
        target = tmp_path / "manifest-target.json"
        manifest.rename(target)
        manifest.symlink_to(target.name)

    monkeypatch.setattr(
        acquisition,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network forbidden")),
    )
    receipt = acquire_pinned_seed_asset(manifest, "fixture-seed", "artifact", root=tmp_path)
    assert receipt["accepted"] is False
    assert receipt["failure"]["code"] in {"invalid_strict_json", "unsafe_json_file"}


def test_signed_url_is_redacted_from_receipt_and_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"PK\x03\x04" + b"x" * 64
    manifest = write_manifest(tmp_path, body)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["sources"][0]["artifact"]["url"] = "https://example.invalid/seed?token=do-not-store"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        acquisition,
        "urlopen",
        lambda *_args, **_kwargs: InterruptingResponse(
            body,
            12,
            headers={"Content-Length": str(len(body)), "ETag": '"signed-v1"'},
        ),
    )

    receipt = acquire_pinned_seed_asset(
        manifest,
        "fixture-seed",
        "artifact",
        root=tmp_path,
        resume=True,
    )
    identity = (tmp_path / "raw" / "seed.zip.partial.identity.json").read_text(encoding="utf-8")
    assert receipt["request"]["url"] == "https://example.invalid/seed"
    assert "do-not-store" not in json.dumps(receipt)
    assert "do-not-store" not in identity
    assert "requested_url_sha256" in identity


def test_existing_destination_is_never_overwritten_or_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"PK\x03\x04expected"
    manifest = write_manifest(tmp_path, body)
    destination = tmp_path / "raw" / "seed.zip"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"owner-data")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network must not be called")

    monkeypatch.setattr(acquisition, "urlopen", forbidden)
    receipt = acquire_pinned_seed_asset(manifest, "fixture-seed", "artifact", root=tmp_path)
    assert receipt["accepted"] is False
    assert receipt["failure"]["code"] == "destination_exists"
    assert destination.read_bytes() == b"owner-data"


def test_atomic_publish_is_bound_to_open_fd_not_replaceable_temp_name(tmp_path: Path) -> None:
    original = b"verified-open-fd-content"
    staging = tmp_path / "staging.partial"
    staging.write_bytes(original)
    fd = acquisition._safe_open_existing_partial(staging)
    try:
        staging.unlink()
        staging.write_bytes(b"same-user-name-replacement")
        destination = tmp_path / "published.bin"
        acquisition._copy_verified_fd_and_publish(
            fd,
            destination,
            expected_bytes=len(original),
            expected_sha256=sha256(original),
        )
    finally:
        acquisition.os.close(fd)
    assert destination.read_bytes() == original
    assert staging.read_bytes() == b"same-user-name-replacement"


def test_atomic_publish_rejects_parent_directory_identity_swap(tmp_path: Path) -> None:
    source = tmp_path / "source.partial"
    source.write_bytes(b"verified")
    source_fd = acquisition._safe_open_existing_partial(source)
    parent = tmp_path / "publish"
    parent.mkdir()
    expected_parent = acquisition._directory_identity(parent)
    moved_parent = tmp_path / "publish-moved"
    parent.rename(moved_parent)
    parent.mkdir()
    try:
        with pytest.raises(acquisition.SeedContractError) as raised:
            acquisition._copy_verified_fd_and_publish(
                source_fd,
                parent / "artifact.bin",
                expected_bytes=len(b"verified"),
                expected_sha256=sha256(b"verified"),
                expected_parent_identity=expected_parent,
            )
    finally:
        acquisition.os.close(source_fd)
    assert raised.value.code == "destination_parent_changed"
    assert not (parent / "artifact.bin").exists()
    assert not (moved_parent / "artifact.bin").exists()


def test_json_proxy_payload_is_evidenced_and_never_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = b"PK\x03\x04real-archive"
    proxy_error = json.dumps({"error": "upstream unavailable", "status": 502}).encode()
    proxy_error += b" " * (395 - len(proxy_error))
    manifest = write_manifest(tmp_path, expected)
    response = FakeResponse(
        proxy_error,
        headers={"Content-Type": "application/json", "Content-Length": str(len(proxy_error))},
    )
    monkeypatch.setattr(acquisition, "urlopen", lambda *_args, **_kwargs: response)

    receipt = acquire_pinned_seed_asset(manifest, "fixture-seed", "artifact", root=tmp_path)
    assert receipt["failure"]["code"] == "proxy_json_error"
    evidence = receipt["failure"]["details"]["response_payload"]
    assert evidence["kind"] == "json"
    assert evidence["captured_bytes"] == 395
    assert evidence["json"]["status"] == 502
    assert not (tmp_path / "raw" / "seed.zip").exists()
    assert not (tmp_path / "raw" / "seed.zip.partial").exists()


def test_oversized_first_chunk_cannot_leave_a_resumable_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"PK\x03\x04short"
    oversized = expected + b"-unexpected-tail"
    manifest = write_manifest(tmp_path, expected)
    monkeypatch.setattr(
        acquisition,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            oversized,
            headers={"Content-Length": str(len(oversized)), "ETag": '"wrong-body"'},
        ),
    )
    receipt = acquire_pinned_seed_asset(
        manifest,
        "fixture-seed",
        "artifact",
        root=tmp_path,
        resume=True,
    )
    assert receipt["failure"]["code"] == "response_too_large"
    assert not (tmp_path / "raw" / "seed.zip").exists()
    assert not (tmp_path / "raw" / "seed.zip.partial").exists()
    assert not (tmp_path / "raw" / "seed.zip.partial.identity.json").exists()


def test_http_502_body_is_recorded_and_no_file_survives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = b"PK\x03\x04real-archive"
    manifest = write_manifest(tmp_path, expected)
    headers = Message()
    headers["Content-Type"] = "text/plain"
    error = HTTPError(
        "https://example.invalid/seed",
        502,
        "Bad Gateway",
        headers,
        io.BytesIO(b"upstream Mendeley public API unavailable"),
    )
    monkeypatch.setattr(acquisition, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))

    receipt = acquire_pinned_seed_asset(manifest, "fixture-seed", "artifact", root=tmp_path)
    assert receipt["failure"]["code"] == "http_error"
    assert receipt["http"]["status"] == 502
    assert "Mendeley" in receipt["failure"]["details"]["response_payload"]["preview"]
    assert not (tmp_path / "raw" / "seed.zip").exists()


def test_unpinned_archive_is_rejected_before_network_but_pinned_yaml_can_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml_body = b"nc: 1\nnames: [helmet]\n"
    manifest_document = {
        "sources": [
            {
                "id": "five-class",
                "archive": None,
                "download_all_url": "https://example.invalid/unpinned.zip",
                "data_yaml": {
                    "url": "https://example.invalid/data.yaml",
                    "bytes": len(yaml_body),
                    "sha256": sha256(yaml_body),
                },
                "eligibility": {
                    "download": True,
                    "quarantine_inspection": True,
                    "training": False,
                    "blockers": ["unpinned_archive"],
                },
            }
        ]
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
    calls: list[str] = []

    def fake_urlopen(request: object, **_kwargs: object) -> FakeResponse:
        calls.append(getattr(request, "full_url"))
        return FakeResponse(yaml_body, headers={"Content-Length": str(len(yaml_body))})

    monkeypatch.setattr(acquisition, "urlopen", fake_urlopen)
    rejected = acquire_pinned_seed_asset(
        manifest,
        "five-class",
        "archive",
        root=tmp_path,
        destination=Path("raw/five-class.zip"),
    )
    assert rejected["failure"]["code"] == "asset_unpinned"
    assert calls == []

    accepted = acquire_pinned_seed_asset(
        manifest,
        "five-class",
        "data_yaml",
        root=tmp_path,
        destination=Path("raw/data.yaml"),
    )
    assert accepted["accepted"] is True
    assert (tmp_path / "raw" / "data.yaml").read_bytes() == yaml_body
    assert calls == ["https://example.invalid/data.yaml"]


def test_resume_requires_validator_and_only_appends_matching_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"PK\x03\x04" + b"a" * 100
    split = 37
    manifest = write_manifest(tmp_path, body)
    first = InterruptingResponse(
        body,
        split,
        headers={"Content-Length": str(len(body)), "ETag": '"fixture-v1"'},
    )
    responses: list[object] = [first]

    def first_urlopen(*_args: object, **_kwargs: object) -> object:
        return responses.pop(0)

    monkeypatch.setattr(acquisition, "urlopen", first_urlopen)
    interrupted = acquire_pinned_seed_asset(
        manifest,
        "fixture-seed",
        "artifact",
        root=tmp_path,
        resume=True,
    )
    assert interrupted["accepted"] is False
    assert interrupted["failure"]["code"] == "io_or_network_error"
    assert interrupted["resume_state"]["partial_retained"] is True
    assert interrupted["resume_state"]["partial_bytes"] == split
    partial = tmp_path / "raw" / "seed.zip.partial"
    assert partial.read_bytes() == body[:split]

    def resumed_urlopen(request: object, **_kwargs: object) -> FakeResponse:
        assert request.get_header("Range") == f"bytes={split}-"
        assert request.get_header("If-range") == '"fixture-v1"'
        return FakeResponse(
            body[split:],
            status=206,
            headers={
                "Content-Length": str(len(body) - split),
                "Content-Range": f"bytes {split}-{len(body) - 1}/{len(body)}",
                "ETag": '"fixture-v1"',
            },
        )

    monkeypatch.setattr(acquisition, "urlopen", resumed_urlopen)
    completed = acquire_pinned_seed_asset(
        manifest,
        "fixture-seed",
        "artifact",
        root=tmp_path,
        resume=True,
    )
    assert completed["accepted"] is True
    assert completed["resume_state"]["offset_before_request"] == split
    assert completed["resume_state"]["partial_retained"] is False
    assert (tmp_path / "raw" / "seed.zip").read_bytes() == body
    assert not partial.exists()
    assert not (tmp_path / "raw" / "seed.zip.partial.identity.json").exists()


def test_safe_yolo_zip_passes_quarantine_but_never_training_eligibility(tmp_path: Path) -> None:
    body = valid_zip(duplicate_images=True)
    manifest, archive = materialized_archive(tmp_path, body, images=2)
    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)

    assert receipt["structural_pass"] is True
    assert receipt["accepted_to_quarantine"] is True
    assert receipt["training_eligible"] is False
    assert "provenance_review_required" in receipt["eligibility_blockers"]
    assert receipt["yolo"]["image_count"] == 2
    assert receipt["yolo"]["label_file_count"] == 2
    assert len(receipt["yolo"]["exact_duplicate_image_groups"]) == 1
    assert len(receipt["exact_duplicate_hash_ledger"]["members"]) == 5
    assert archive.is_file()


def test_nested_yaml_references_resolve_lexically_inside_archive(tmp_path: Path) -> None:
    body = yolo_zip_with_config(
        b"train: ../train/images\nval: ../valid/images\nnc: 2\nnames: [helmet, no_helmet]\n",
        yaml_path="configs/data.yaml",
    )
    manifest, _archive = materialized_archive(tmp_path, body, images=2)
    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)

    assert receipt["accepted_to_quarantine"] is True
    path_gate = next(
        gate for gate in receipt["gates"] if gate["id"] == "yolo_split_paths_resolve_in_archive"
    )
    assert path_gate["passed"] is True
    config = path_gate["details"]["configs"][0]
    assert config["splits"]["train"][0]["resolved_directory"] == "train/images"
    assert config["splits"]["val"][0]["resolved_directory"] == "valid/images"


@pytest.mark.parametrize(
    "yaml_body,error_fragment",
    [
        (
            b"train: ../train/images\nval: valid/images\nnc: 2\nnames: [helmet, no_helmet]\n",
            "archive kokunden disari",
        ),
        (
            b"train: train/images\nval: missing/images\nnc: 2\nnames: [helmet, no_helmet]\n",
            "yok veya goruntu icermiyor",
        ),
        (
            b"train: train/images\nval: train/images\nnc: 2\nnames: [helmet, no_helmet]\n",
            "ayrik degil",
        ),
        (
            b"train: https://example.invalid/images\nval: valid/images\nnc: 2\nnames: [helmet, no_helmet]\n",
            "URL/URI",
        ),
    ],
)
def test_yolo_split_paths_must_be_safe_existing_disjoint_archive_directories(
    tmp_path: Path,
    yaml_body: bytes,
    error_fragment: str,
) -> None:
    body = yolo_zip_with_config(yaml_body)
    manifest, _archive = materialized_archive(tmp_path, body, images=2)
    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)

    assert receipt["accepted_to_quarantine"] is False
    gate = next(
        gate for gate in receipt["gates"] if gate["id"] == "yolo_split_paths_resolve_in_archive"
    )
    assert gate["passed"] is False
    assert error_fragment in gate["details"]["errors"]["items"][0]["error"]


def test_duplicate_yaml_keys_are_rejected_instead_of_last_value_winning(tmp_path: Path) -> None:
    body = yolo_zip_with_config(
        b"train: train/images\ntrain: valid/images\nval: valid/images\nnc: 2\nnames: [helmet, no_helmet]\n"
    )
    manifest, _archive = materialized_archive(tmp_path, body, images=2)
    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)

    assert receipt["accepted_to_quarantine"] is False
    gate = next(gate for gate in receipt["gates"] if gate["id"] == "valid_yolo_yaml")
    assert gate["passed"] is False
    assert "Duplicate YAML mapping key" in gate["details"]["errors"]["items"][0]["error"]


def test_images_are_decoded_not_counted_by_extension_only(tmp_path: Path) -> None:
    body = yolo_zip_with_config(
        b"train: train/images\nval: valid/images\nnc: 2\nnames: [helmet, no_helmet]\n",
        train_image=b"not-a-png-despite-the-extension",
    )
    manifest, _archive = materialized_archive(tmp_path, body, images=2)
    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)

    assert receipt["accepted_to_quarantine"] is False
    gate = next(gate for gate in receipt["gates"] if gate["id"] == "image_payloads_decodable")
    assert gate["passed"] is False
    assert gate["details"]["decoded_count"] == 1
    assert "PNG signature/IHDR" in gate["details"]["issues"]["items"][0]["error"]


def test_image_pixel_limit_is_checked_before_decode_acceptance(tmp_path: Path) -> None:
    body = valid_zip()
    manifest, _archive = materialized_archive(tmp_path, body, images=2)
    receipt = inspect_seed_zip(
        manifest,
        "fixture-seed",
        "artifact",
        root=tmp_path,
        limits=ZipLimits(max_image_pixels=1),
    )

    assert receipt["accepted_to_quarantine"] is False
    gate = next(gate for gate in receipt["gates"] if gate["id"] == "image_payloads_decodable")
    assert gate["passed"] is False
    assert gate["details"]["decoded_count"] == 0
    assert "piksel limiti" in gate["details"]["issues"]["items"][0]["error"]


def test_declared_fixed_image_dimensions_are_verified_against_decoded_payloads(
    tmp_path: Path,
) -> None:
    body = valid_zip()
    manifest = write_manifest(
        tmp_path,
        body,
        classes=["helmet", "no_helmet"],
        declared_images=2,
        declared_width=640,
        declared_height=640,
    )
    archive = tmp_path / "raw" / "seed.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(body)

    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)

    assert receipt["accepted_to_quarantine"] is False
    gate = next(
        gate
        for gate in receipt["gates"]
        if gate["id"] == "declared_image_dimensions_match"
    )
    assert gate["passed"] is False
    assert gate["details"]["declared"] == {"width": 640, "height": 640}
    assert gate["details"]["decoded_count"] == 2
    assert gate["details"]["observed_dimensions"] == [
        {"width": 2, "height": 2, "count": 2}
    ]
    assert gate["details"]["mismatches"]["count"] == 2


def test_declared_fixed_image_dimensions_pass_when_every_decode_matches(
    tmp_path: Path,
) -> None:
    body = valid_zip()
    manifest = write_manifest(
        tmp_path,
        body,
        classes=["helmet", "no_helmet"],
        declared_images=2,
        declared_width=2,
        declared_height=2,
    )
    archive = tmp_path / "raw" / "seed.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(body)

    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)

    gate = next(
        gate
        for gate in receipt["gates"]
        if gate["id"] == "declared_image_dimensions_match"
    )
    assert gate["passed"] is True
    assert gate["details"]["mismatches"]["count"] == 0
    assert receipt["accepted_to_quarantine"] is True


def test_archive_path_swap_cannot_change_the_fd_bound_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = valid_zip()
    manifest, archive = materialized_archive(tmp_path, body, images=2)
    real_zipfile = zipfile.ZipFile
    swapped = False

    def swapping_zipfile(file: object, *args: object, **kwargs: object) -> zipfile.ZipFile:
        nonlocal swapped
        if not swapped:
            swapped = True
            archive.rename(archive.with_suffix(".original"))
            archive.write_bytes(body)
        return real_zipfile(file, *args, **kwargs)

    monkeypatch.setattr(acquisition.zipfile, "ZipFile", swapping_zipfile)
    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)
    assert receipt["accepted_to_quarantine"] is False
    assert receipt["failure"]["code"] == "archive_changed"


@pytest.mark.parametrize("hazard", ["traversal", "case_collision", "symlink"])
def test_zip_metadata_hazards_fail_before_member_decompression(tmp_path: Path, hazard: str) -> None:
    entries: list[tuple[str | zipfile.ZipInfo, bytes]] = [
        ("data.yaml", b"nc: 1\nnames: [helmet]\n"),
        ("train/images/a.jpg", b"image"),
        ("train/labels/a.txt", b"0 0.5 0.5 0.2 0.2\n"),
    ]
    if hazard == "traversal":
        entries.append(("../escape.txt", b"never extract"))
    elif hazard == "case_collision":
        entries.extend([("extra/A.txt", b"a"), ("extra/a.txt", b"b")])
    else:
        link = zipfile.ZipInfo("unsafe-link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        entries.append((link, b"../../outside"))
    body = make_zip(entries)
    manifest, _archive = materialized_archive(tmp_path, body, images=1)

    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)
    assert receipt["accepted_to_quarantine"] is False
    assert receipt["failure"]["code"] == "unsafe_zip_metadata"
    assert receipt["exact_duplicate_hash_ledger"]["members"] == []
    assert not (tmp_path / "escape.txt").exists()


def test_malformed_bbox_and_compression_bomb_limits_fail_closed(tmp_path: Path) -> None:
    malformed = valid_zip(bad_label=True)
    manifest, _archive = materialized_archive(tmp_path, malformed, images=2)
    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)
    assert receipt["accepted_to_quarantine"] is False
    assert "valid_yolo_detection_labels" in {
        gate["id"] for gate in receipt["gates"] if not gate["passed"]
    }
    assert receipt["yolo"]["malformed_labels"]["items"][0]["code"] == "bbox_out_of_range"

    bombish = make_zip(
        [
            ("data.yaml", b"nc: 1\nnames: [helmet]\n"),
            ("train/images/a.jpg", b"A" * 20_000),
            ("train/labels/a.txt", b"0 0.5 0.5 0.2 0.2\n"),
        ]
    )
    bomb_dir = tmp_path / "bomb"
    bomb_dir.mkdir()
    bomb_manifest, _bomb_archive = materialized_archive(bomb_dir, bombish, images=1)
    limited = inspect_seed_zip(
        bomb_manifest,
        "fixture-seed",
        "artifact",
        root=bomb_dir,
        limits=ZipLimits(
            max_member_compression_ratio=2.0,
            max_total_compression_ratio=2.0,
        ),
    )
    assert limited["accepted_to_quarantine"] is False
    assert limited["failure"]["code"] == "unsafe_zip_metadata"
    assert limited["exact_duplicate_hash_ledger"]["members"] == []


def test_cli_writes_immutable_machine_readable_quarantine_receipt(tmp_path: Path) -> None:
    body = valid_zip()
    manifest, _archive = materialized_archive(tmp_path, body, images=2)
    receipt_path = tmp_path / "receipt.json"
    args = [
        "inspect-seed",
        "--manifest",
        str(manifest),
        "--source-id",
        "fixture-seed",
        "--root",
        str(tmp_path),
        "--receipt",
        str(receipt_path),
    ]
    assert main(args) == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["accepted_to_quarantine"] is True
    assert receipt_path.stat().st_mode & 0o777 == 0o440
    before = receipt_path.read_bytes()
    assert main(args) == 2
    assert receipt_path.read_bytes() == before


def test_receipt_self_hash_replay_and_external_pin_fail_closed(tmp_path: Path) -> None:
    body = valid_zip()
    manifest, _archive = materialized_archive(tmp_path, body, images=2)
    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)
    receipt_path = tmp_path / "receipt.json"
    acquisition.write_receipt_no_overwrite(receipt_path, receipt)
    original_fingerprint = receipt["receipt_sha256"]

    verified = acquisition.verify_seed_receipt_file(
        receipt_path,
        expected_receipt_sha256=original_fingerprint,
    )
    assert verified["valid"] is True
    assert verified["external_pin_verified"] is True
    assert verified["json_schema_validation_performed"] is False
    assert verified["semantic_replay_performed"] is False
    assert verified["authenticity_signature_verified"] is False

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["source_id"] = "tampered-source"
    receipt_path.chmod(0o600)
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    receipt_path.chmod(0o440)
    with pytest.raises(acquisition.SeedContractError) as mismatch:
        acquisition.verify_seed_receipt_file(receipt_path)
    assert mismatch.value.code == "receipt_self_hash_mismatch"

    tampered.pop("receipt_sha256")
    acquisition.seal_receipt(tampered)
    receipt_path.chmod(0o600)
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    receipt_path.chmod(0o440)
    assert acquisition.verify_seed_receipt_file(receipt_path)["valid"] is True
    with pytest.raises(acquisition.SeedContractError) as external_mismatch:
        acquisition.verify_seed_receipt_file(
            receipt_path,
            expected_receipt_sha256=original_fingerprint,
        )
    assert external_mismatch.value.code == "receipt_expected_sha256_mismatch"


def test_cli_verifies_receipt_against_explicit_fingerprint(tmp_path: Path) -> None:
    body = valid_zip()
    manifest, _archive = materialized_archive(tmp_path, body, images=2)
    receipt = inspect_seed_zip(manifest, "fixture-seed", "artifact", root=tmp_path)
    receipt_path = tmp_path / "receipt.json"
    acquisition.write_receipt_no_overwrite(receipt_path, receipt)
    output = tmp_path / "verification.json"

    assert main(
        [
            "verify-seed-receipt",
            "--receipt",
            str(receipt_path),
            "--expected-receipt-sha256",
            receipt["receipt_sha256"],
            "--output",
            str(output),
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["external_pin_verified"] is True


def test_receipts_validate_against_published_json_schemas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    body = b"PK\x03\x04fixture"
    manifest = write_manifest(tmp_path, body)
    monkeypatch.setattr(
        acquisition,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(body, headers={"Content-Length": str(len(body))}),
    )
    acquisition_receipt = acquire_pinned_seed_asset(
        manifest,
        "fixture-seed",
        "artifact",
        root=tmp_path,
    )
    acquisition_schema = json.loads(
        Path("ppe_dataset/schemas/ppe-seed-acquisition-receipt-v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(acquisition_schema).validate(acquisition_receipt)

    zip_dir = tmp_path / "inspect"
    zip_dir.mkdir()
    zip_body = valid_zip()
    inspect_manifest, _archive = materialized_archive(zip_dir, zip_body, images=2)
    quarantine_receipt = inspect_seed_zip(
        inspect_manifest,
        "fixture-seed",
        "artifact",
        root=zip_dir,
    )
    quarantine_schema = json.loads(
        Path("ppe_dataset/schemas/ppe-seed-quarantine-receipt-v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(quarantine_schema).validate(quarantine_receipt)
