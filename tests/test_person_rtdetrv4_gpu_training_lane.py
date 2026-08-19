from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import jsonschema
import pytest

from validation import person_rtdetrv4_gpu_training as host
from validation import person_rtdetrv4_gpu_training_plan_r9 as plan_r9
from validation import person_rtdetrv4_gpu_training_plan_r10 as plan_r10


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "models/person/training-lanes/rtdetrv4-s-r-livit-person-r1-gpu-v1"


def _load_container_runner():
    path = LANE / "container_runner.py"
    spec = importlib.util.spec_from_file_location("deepsafe_test_container_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dependency_compatibility():
    path = LANE / "dependency_compatibility.py"
    spec = importlib.util.spec_from_file_location("deepsafe_test_dependency_compatibility", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runtime_inventory():
    path = LANE / "runtime_inventory.py"
    spec = importlib.util.spec_from_file_location("deepsafe_test_runtime_inventory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


container = _load_container_runner()
compatibility_verifier = _load_dependency_compatibility()
runtime_inventory = _load_runtime_inventory()


def _normalize_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _FakeLabels:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class _FakeDevice:
    def __init__(self, kind: str, index: int | None = None) -> None:
        self.type = kind
        self.index = index

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class _FakeTensor:
    def __init__(self, shape: list[int]) -> None:
        self.shape = shape
        self.dtype = "torch.float32"
        self.device = _FakeDevice("cpu")


class HybridEncoder:
    def __init__(self) -> None:
        self.eval_spatial_size = [640, 640]
        self.use_encoder_idx = [2]
        self.hidden_dim = 256
        self.feat_strides = [8, 16, 32]
        self.pos_embed2 = _FakeTensor([1, 400, 256])
        self._buffers: dict[str, _FakeTensor] = {}
        self._non_persistent_buffers_set: set[str] = set()

    def register_buffer(
        self, name: str, value: _FakeTensor, *, persistent: bool
    ) -> None:
        self._buffers[name] = value
        setattr(self, name, value)
        if not persistent:
            self._non_persistent_buffers_set.add(name)


class _FakePositionModel:
    def __init__(self) -> None:
        self.encoder = HybridEncoder()

    def to(self, device: _FakeDevice) -> "_FakePositionModel":
        target_index = 0 if device.index is None else device.index
        for value in self.encoder._buffers.values():
            value.device = _FakeDevice(device.type, target_index)
        return self


class _FakeTorch:
    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, _FakeTensor)


def test_eval_position_repair_registers_nonpersistent_buffer_and_follows_cuda() -> None:
    model = _FakePositionModel()
    receipt = container.register_hybrid_encoder_eval_position_buffers(
        torch=_FakeTorch, model=model
    )
    assert receipt["policy_id"] == (
        "rtdetrv4-hybrid-encoder-eval-position-buffer-v1"
    )
    assert receipt["state_dict_semantics_changed"] is False
    assert receipt["buffers"] == [
        {
            "name": "pos_embed2",
            "encoder_index": 2,
            "feature_stride": 32,
            "shape": [1, 400, 256],
            "dtype": "torch.float32",
            "origin_device": "cpu",
            "persistent": False,
        }
    ]
    assert model.encoder._buffers["pos_embed2"] is model.encoder.pos_embed2
    assert "pos_embed2" in model.encoder._non_persistent_buffers_set

    model.to(_FakeDevice("cuda"))
    moved = container.verify_hybrid_encoder_eval_position_buffers(
        torch=_FakeTorch,
        model=model,
        expected_device=_FakeDevice("cuda"),
    )
    assert moved == {
        "expected_device": "cuda:0",
        "buffers": [{"name": "pos_embed2", "device": "cuda:0"}],
    }


def test_eval_position_repair_rejects_shape_drift_before_registration() -> None:
    model = _FakePositionModel()
    model.encoder.pos_embed2.shape = [1, 399, 256]
    with pytest.raises(container.LaneError, match="cached tensor shape mismatch"):
        container.register_hybrid_encoder_eval_position_buffers(
            torch=_FakeTorch, model=model
        )
    assert model.encoder._buffers == {}


class _FakeSmokeLoader:
    batch_size = 1
    shuffle = False

    def __init__(self, target_counts: list[int]) -> None:
        self.batches = [
            (
                f"samples-{index}",
                [
                    {
                        "labels": _FakeLabels(count),
                        "image_id": _FakeScalar(100 + index),
                        "idx": _FakeScalar(index),
                    }
                ],
            )
            for index, count in enumerate(target_counts)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def test_smoke_selector_skips_leading_negative_train_samples_deterministically() -> None:
    samples, targets, audit = container.select_annotated_smoke_batch(
        _FakeSmokeLoader([0, 0, 3, 1])
    )
    assert samples == "samples-2"
    assert targets[0]["image_id"].item() == 102
    assert audit == {
        "strategy": "first_annotated_train_batch_in_deterministic_coco_order",
        "source_split": "train",
        "batch_size": 1,
        "shuffle": False,
        "scan_limit_batches": 4,
        "available_batches": 4,
        "batches_examined": 3,
        "leading_negative_batches_skipped": 2,
        "selected_zero_based_batch_index": 2,
        "selected_dataset_index": 2,
        "selected_image_id": 102,
        "selected_target_count": 3,
        "official_test_opened": False,
        "test_unseen_opened": False,
    }


def test_smoke_selector_fails_closed_when_all_train_samples_are_negative() -> None:
    with pytest.raises(container.LaneError, match="no annotated sample within 3 deterministic batches"):
        container.select_annotated_smoke_batch(_FakeSmokeLoader([0, 0, 0]))


def test_smoke_selector_fails_closed_if_loader_order_is_not_coco_order() -> None:
    loader = _FakeSmokeLoader([0, 2])
    loader.batches[1][1][0]["idx"] = _FakeScalar(7)
    with pytest.raises(container.LaneError, match="order differs from deterministic COCO order"):
        container.select_annotated_smoke_batch(loader)


def test_pinned_train_split_starts_with_negative_then_annotated_real_sample() -> None:
    document = json.loads(
        (ROOT / "data/derived/r-livit/person-rtdetrv4-coco-v1/annotations/instances_train.json")
        .read_text(encoding="utf-8")
    )
    counts: dict[int, int] = {}
    for annotation in document["annotations"]:
        image_id = int(annotation["image_id"])
        counts[image_id] = counts.get(image_id, 0) + 1
    assert [(row["id"], row["file_name"]) for row in document["images"][:2]] == [
        (2, "000/02.png"),
        (6, "000/06.png"),
    ]
    assert counts.get(2, 0) == 0
    assert counts[6] == 2


def test_bounded_determinism_accepts_only_exact_grid_sampler_warning() -> None:
    with container.bounded_nondeterminism_audit() as audit:
        container.warnings.warn(
            container.GRID_SAMPLER_BACKWARD_WARNING,
            UserWarning,
        )
    assert audit["policy_id"] == "rtdetrv4-grid-sampler-backward-bounded-v2"
    assert audit["accepted_warning_count"] == 1
    assert audit["accepted_operators"] == ["grid_sampler_2d_backward_cuda"]
    assert audit["accepted_variants"] == ["exact_base"]
    assert audit["accepted_message_sha256_counts"] == {
        container.GRID_SAMPLER_BACKWARD_WARNING_SHA256: 1
    }
    assert audit["known_limitation_observed"] is True
    assert audit["unexpected_nondeterminism_warning_count"] == 0
    assert audit["bitwise_reproducibility_claimed"] is False


def test_bounded_determinism_accepts_exact_pinned_runtime_suffix() -> None:
    with container.bounded_nondeterminism_audit() as audit:
        container.warnings.warn(
            container.GRID_SAMPLER_BACKWARD_RUNTIME_WARNING,
            UserWarning,
        )
    assert audit["accepted_warning_count"] == 1
    assert audit["accepted_operators"] == ["grid_sampler_2d_backward_cuda"]
    assert audit["accepted_variants"] == ["exact_pytorch_context_cpp_186"]
    assert audit["accepted_message_sha256_counts"] == {
        container.GRID_SAMPLER_BACKWARD_RUNTIME_WARNING_SHA256: 1
    }
    assert audit["allowlist"][1]["internal_suffix_sha256"] == (
        container.GRID_SAMPLER_BACKWARD_INTERNAL_SUFFIX_SHA256
    )


def test_bounded_determinism_rejects_unexpected_nondeterministic_operator() -> None:
    unexpected = (
        "some_new_backward_cuda does not have a deterministic implementation, "
        "but deterministic algorithms are enabled"
    )
    with pytest.raises(
        container.LaneError,
        match="unexpected nondeterministic operation warning rejected",
    ):
        with container.bounded_nondeterminism_audit():
            container.warnings.warn(unexpected, UserWarning)


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("grid_sampler_2d_backward_cuda warning wording changed", UserWarning),
        (container.GRID_SAMPLER_BACKWARD_WARNING, RuntimeWarning),
        (
            container.GRID_SAMPLER_BACKWARD_WARNING
            + " (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/Context.cpp:185.)",
            UserWarning,
        ),
        (
            container.GRID_SAMPLER_BACKWARD_WARNING
            + " (Triggered internally at /tmp/pytorch/aten/src/ATen/Context.cpp:186.)",
            UserWarning,
        ),
        (container.GRID_SAMPLER_BACKWARD_RUNTIME_WARNING + " ", UserWarning),
        (container.GRID_SAMPLER_BACKWARD_RUNTIME_WARNING, RuntimeWarning),
    ],
)
def test_bounded_determinism_rejects_changed_grid_warning_tuple(
    message: str, category: type[Warning]
) -> None:
    with pytest.raises(
        container.LaneError,
        match="unexpected nondeterministic operation warning rejected",
    ):
        with container.bounded_nondeterminism_audit():
            container.warnings.warn(message, category)


def test_bounded_determinism_no_warning_is_valid_and_disclosed() -> None:
    with container.bounded_nondeterminism_audit() as audit:
        pass
    assert audit["accepted_warning_count"] == 0
    assert audit["accepted_operators"] == []
    assert audit["accepted_message_sha256_counts"] == {}
    assert audit["accepted_variants"] == []
    assert audit["known_limitation_observed"] is False
    assert audit["unexpected_nondeterminism_warning_count"] == 0
    assert audit["unexpected_nondeterminism_fail_closed"] is True


def _valid_amp_overflow_attempt_record(attempt: int = 1) -> dict:
    scale_before = container._expected_smoke_amp_scale(attempt)
    return {
        "attempt": attempt,
        "outcome": "overflow_backoff",
        "gradients": {
            "nonfinite_gradient_tensor_count": 2,
            "all_gradients_finite": False,
        },
        "found_inf": {
            "overflow_detected": True,
            "found_inf_total": 1.0,
        },
        "scaler": {
            "scale_before": scale_before,
            "scale_after": scale_before * container.SMOKE_AMP_BACKOFF_FACTOR,
            "state_before_sha256": "e" * 64,
            "state_after_sha256": "f" * 64,
        },
        "state_integrity": {
            "model_before_sha256": "a" * 64,
            "model_after_step_pre_rollback_sha256": "a" * 64,
            "model_after_step_sha256": "a" * 64,
            "model_state_changed_before_rollback": False,
            "model_state_rollback_executed": True,
            "optimizer_before_sha256": "b" * 64,
            "optimizer_after_step_sha256": "b" * 64,
            "ema_before_sha256": "c" * 64,
            "ema_after_step_sha256": "c" * 64,
            "input_before_sha256": "d" * 64,
            "input_after_sha256": "d" * 64,
            "optimizer_step_skipped": True,
            "ema_update_executed": False,
            "ema_updates_before": 0,
            "ema_updates_after": 0,
        },
    }


def _valid_amp_finite_attempt_record(attempt: int = 1) -> dict:
    scale = container._expected_smoke_amp_scale(attempt)
    return {
        "attempt": attempt,
        "outcome": "finite_optimizer_ema_update",
        "gradients": {
            "gradient_tensor_count": 4,
            "finite_gradient_tensor_count": 4,
            "nonfinite_gradient_tensor_count": 0,
            "finite_gradient_l2_squared": 3.25,
            "all_gradients_finite": True,
            "finite_nonzero_gradient": True,
        },
        "found_inf": {
            "overflow_detected": False,
            "found_inf_total": 0.0,
        },
        "scaler": {
            "scale_before": scale,
            "scale_after": scale,
            "state_before_sha256": "e" * 64,
            "state_after_sha256": "f" * 64,
        },
        "state_integrity": {
            "model_before_sha256": "a" * 64,
            "model_after_step_sha256": "1" * 64,
            "optimizer_before_sha256": "b" * 64,
            "optimizer_after_step_sha256": "2" * 64,
            "ema_before_sha256": "c" * 64,
            "ema_after_step_sha256": "3" * 64,
            "input_before_sha256": "d" * 64,
            "input_after_sha256": "d" * 64,
            "optimizer_step_skipped": False,
            "ema_update_executed": True,
            "ema_updates_before": 7,
            "ema_updates_after": 8,
        },
    }


def test_amp_overflow_attempt_accepts_exact_backoff_and_unchanged_state() -> None:
    container._validate_overflow_attempt_record(
        _valid_amp_overflow_attempt_record()
    )


@pytest.mark.parametrize(
    ("section", "key", "mutated"),
    [
        ("state_integrity", "model_after_step_sha256", "e" * 64),
        ("state_integrity", "model_after_step_pre_rollback_sha256", "e" * 63),
        ("state_integrity", "model_state_changed_before_rollback", True),
        ("state_integrity", "model_state_rollback_executed", False),
        ("state_integrity", "optimizer_after_step_sha256", "e" * 64),
        ("state_integrity", "ema_after_step_sha256", "e" * 64),
        ("state_integrity", "input_after_sha256", "e" * 64),
        ("state_integrity", "optimizer_step_skipped", False),
        ("state_integrity", "ema_update_executed", True),
        ("scaler", "scale_after", 32767.0),
        ("found_inf", "found_inf_total", 0.0),
        ("found_inf", "overflow_detected", False),
        ("gradients", "nonfinite_gradient_tensor_count", 0),
        ("state_integrity", "ema_updates_after", 1),
        ("scaler", "state_after_sha256", "e" * 64),
    ],
)
def test_amp_overflow_attempt_rejects_state_scale_and_found_inf_mutations(
    section: str, key: str, mutated: object
) -> None:
    record = _valid_amp_overflow_attempt_record()
    record[section][key] = mutated
    with pytest.raises(
        container.LaneError,
        match="overflow attempt violated scale/state integrity",
    ):
        container._validate_overflow_attempt_record(record)


@pytest.mark.parametrize("attempt", range(1, container.SMOKE_AMP_MAX_ATTEMPTS + 1))
def test_amp_finite_attempt_accepts_every_bounded_backoff_position(
    attempt: int,
) -> None:
    container._validate_finite_attempt_record(
        _valid_amp_finite_attempt_record(attempt)
    )


@pytest.mark.parametrize(
    ("section", "key", "mutated"),
    [
        ("state_integrity", "model_after_step_sha256", "a" * 64),
        ("state_integrity", "optimizer_after_step_sha256", "b" * 64),
        ("state_integrity", "ema_after_step_sha256", "c" * 64),
        ("state_integrity", "input_after_sha256", "f" * 64),
        ("state_integrity", "optimizer_step_skipped", True),
        ("state_integrity", "ema_update_executed", False),
        ("state_integrity", "ema_updates_after", 7),
        ("scaler", "scale_after", 32768.0),
        ("scaler", "state_after_sha256", "e" * 64),
        ("found_inf", "found_inf_total", 1.0),
        ("found_inf", "overflow_detected", True),
        ("gradients", "nonfinite_gradient_tensor_count", 1),
        ("gradients", "finite_nonzero_gradient", False),
    ],
)
def test_amp_finite_attempt_rejects_state_scale_and_gradient_mutations(
    section: str, key: str, mutated: object
) -> None:
    record = _valid_amp_finite_attempt_record()
    record[section][key] = mutated
    with pytest.raises(
        container.LaneError,
        match="finite AMP attempt violated optimizer/EMA/scaler integrity",
    ):
        container._validate_finite_attempt_record(record)


def test_amp_scale_schedule_is_exact_from_attempt_one_through_sixteen() -> None:
    assert [
        container._expected_smoke_amp_scale(attempt)
        for attempt in range(1, container.SMOKE_AMP_MAX_ATTEMPTS + 1)
    ] == [65536.0, 32768.0, 16384.0, 8192.0, 4096.0, 2048.0, 1024.0,
          512.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0]
    with pytest.raises(container.LaneError, match="outside the bounded schedule"):
        container._expected_smoke_amp_scale(0)
    with pytest.raises(container.LaneError, match="outside the bounded schedule"):
        container._expected_smoke_amp_scale(17)


def test_amp_backoff_limit_is_exact_and_persistent_overflow_fails_closed() -> None:
    container._validate_overflow_attempt_record(
        _valid_amp_overflow_attempt_record(container.SMOKE_AMP_MAX_ATTEMPTS)
    )
    container._require_amp_attempt_remaining(container.SMOKE_AMP_MAX_ATTEMPTS - 1)
    with pytest.raises(
        container.LaneError,
        match="overflow persisted through 16 attempts",
    ):
        container._require_amp_attempt_remaining(container.SMOKE_AMP_MAX_ATTEMPTS)


def test_amp_warning_delta_rejects_count_hash_inconsistency() -> None:
    before = {
        "accepted_warning_count": 0,
        "accepted_message_sha256_counts": {},
    }
    after = {
        "accepted_warning_count": 1,
        "accepted_message_sha256_counts": {},
    }
    with pytest.raises(container.LaneError, match="count/message delta mismatch"):
        container._warning_audit_delta(before, after)


def _host_amp_attempt(record: dict) -> dict:
    result = json.loads(json.dumps(record))
    result.update(
        {
            "rng_seed_replayed": 20260718,
            "forward": {
                "outputs_finite": True,
                "losses_finite": True,
                "total_loss": 4.5,
                "losses": {"loss_bbox": 1.25, "loss_vfl": 3.25},
            },
            "warning_audit_delta": {
                "accepted_warning_count": 0,
                "accepted_message_sha256_counts": {},
            },
        }
    )
    result["found_inf"].update(
        {
            "reader": "GradScaler._found_inf_per_device",
            "devices": [
                {
                    "device": "cuda:0",
                    "found_inf": result["found_inf"]["found_inf_total"],
                }
            ],
        }
    )
    result["scaler"]["backoff_factor"] = 0.5
    return result


def _valid_host_amp_execution(attempt_count: int = 2) -> dict:
    attempts = [
        _host_amp_attempt(_valid_amp_overflow_attempt_record(attempt))
        for attempt in range(1, attempt_count)
    ]
    finite = _host_amp_attempt(_valid_amp_finite_attempt_record(attempt_count))
    for attempt in attempts:
        attempt["state_integrity"]["ema_updates_before"] = 0
        attempt["state_integrity"]["ema_updates_after"] = 0
    finite["state_integrity"]["ema_updates_before"] = 0
    finite["state_integrity"]["ema_updates_after"] = 1
    attempts.append(finite)
    first_state = attempts[0]["state_integrity"]
    final_state = attempts[-1]["state_integrity"]
    return {
        "backward": {
            "executed": True,
            "finite_gradient_tensor_count": 4,
            "nonfinite_gradient_tensor_count": 0,
            "gradient_l2_norm": 2.5,
        },
        "quality_metric_computed": False,
        "full_run_resumable_checkpoint": False,
        "checkpoint": {
            "path": "/output/checkpoints/smoke-one-step.pth",
            "bytes": len(b"checkpoint"),
            "sha256": host.hashlib.sha256(b"checkpoint").hexdigest(),
        },
        "optimizer": {
            "step_count": 1,
            "model_state_sha256_before": first_state["model_before_sha256"],
            "model_state_sha256_after": final_state["model_after_step_sha256"],
            "model_state_changed": True,
            "optimizer_state_sha256_before": first_state[
                "optimizer_before_sha256"
            ],
            "optimizer_state_sha256_after": final_state[
                "optimizer_after_step_sha256"
            ],
            "optimizer_state_changed": True,
        },
        "ema": {
            "enabled": True,
            "update_count": 1,
            "state_sha256_before": first_state["ema_before_sha256"],
            "state_sha256_after": final_state["ema_after_step_sha256"],
            "state_changed": True,
        },
        "amp": {
            "enabled": True,
            "policy_id": host.R9_SMOKE_AMP_POLICY_ID,
            "contract": {
                "type": "GradScaler",
                "enabled": True,
                "initial_scale": 65536.0,
                "backoff_factor": 0.5,
                "growth_factor": 2.0,
                "growth_interval": 2000,
                "max_attempts": 16,
            },
            "attempt_count": attempt_count,
            "overflow_backoff_count": attempt_count - 1,
            "attempts": attempts,
            "event_evidence": "events.jsonl:smoke_amp_attempt_completed",
            "scaler_updated": True,
            "persistent_nonfinite_fail_closed": True,
            "bitwise_reproducibility_claimed": False,
        },
    }


def _write_host_amp_events(path: Path, attempts: list[dict]) -> None:
    rows = [
        {
            "schema_version": "deepsafe.person-rtdetrv4-training-event/v1",
            "event": "run_started",
            "created_at_utc": "2026-07-18T02:00:00Z",
        }
    ]
    rows.extend(
        {
            "schema_version": "deepsafe.person-rtdetrv4-training-event/v1",
            "event": "smoke_amp_attempt_completed",
            "created_at_utc": f"2026-07-18T02:00:{index:02d}Z",
            **attempt,
        }
        for index, attempt in enumerate(attempts, start=1)
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_host_smoke_checkpoint(root: Path) -> None:
    checkpoint = root / "checkpoints/smoke-one-step.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")


def test_host_accepts_exact_amp_attempts_and_matching_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    execution = _valid_host_amp_execution()
    _write_host_smoke_checkpoint(tmp_path)
    _write_host_amp_events(tmp_path / "events.jsonl", execution["amp"]["attempts"])
    monkeypatch.setattr(host, "_regular", lambda path: path.lstat())
    host._validate_smoke_amp_receipt(
        execution,
        {"accepted_warning_count": 0, "accepted_message_sha256_counts": {}},
        tmp_path,
    )


@pytest.mark.parametrize(
    ("attempt_index", "section", "key", "mutated", "message"),
    [
        (0, "state_integrity", "model_after_step_sha256", "f" * 64,
         "overflow attempt mutated protected state"),
        (0, "scaler", "scale_after", 123.0, "overflow attempt mutated protected state"),
        (1, "scaler", "scale_after", 16384.0,
         "final finite update evidence mismatch"),
        (1, "state_integrity", "ema_updates_after", 0,
         "final finite update evidence mismatch"),
    ],
)
def test_host_rejects_amp_state_and_scale_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attempt_index: int,
    section: str,
    key: str,
    mutated: object,
    message: str,
) -> None:
    execution = _valid_host_amp_execution()
    execution["amp"]["attempts"][attempt_index][section][key] = mutated
    _write_host_smoke_checkpoint(tmp_path)
    _write_host_amp_events(tmp_path / "events.jsonl", execution["amp"]["attempts"])
    monkeypatch.setattr(host, "_regular", lambda path: path.lstat())
    with pytest.raises(host.GpuLaneError, match=message):
        host._validate_smoke_amp_receipt(
            execution,
            {"accepted_warning_count": 0, "accepted_message_sha256_counts": {}},
            tmp_path,
        )


def test_host_rejects_amp_event_receipt_divergence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    execution = _valid_host_amp_execution()
    _write_host_smoke_checkpoint(tmp_path)
    events = json.loads(json.dumps(execution["amp"]["attempts"]))
    events[-1]["forward"]["total_loss"] = 9.0
    _write_host_amp_events(tmp_path / "events.jsonl", events)
    monkeypatch.setattr(host, "_regular", lambda path: path.lstat())
    with pytest.raises(host.GpuLaneError, match="events do not exactly match"):
        host._validate_smoke_amp_receipt(
            execution,
            {"accepted_warning_count": 0, "accepted_message_sha256_counts": {}},
            tmp_path,
        )


def test_execution_plan_is_canonical_schema_valid_and_all_sources_are_pinned() -> None:
    plan = host.load_plan()
    assert plan["status"] == "ready_not_executed"
    assert plan["execution"] == {
        "performed_during_preparation": False,
        "docker_build_executed": False,
        "gpu_queried": False,
        "gpu_workload_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
    }
    assert plan["base_image"]["resolved_image_id"] == host.BASE_IMAGE_ID
    assert plan["hardware"]["gpu_uuid"] == host.GPU_UUID
    assert plan["model"]["loaded_state_sha256"] == container.EXPECTED_LOADED_STATE_SHA256
    assert len(plan["source_pins"]) >= 29
    assert plan["runtime_isolation"]["global_gpu_lease_contract"] == (
        host.gpu_lease_contract_projection()
    )
    assert plan["build"]["revision"] == 10
    assert plan["build"]["default_attempt_id"] == "eval-device-r10-001"
    assert plan["build"]["default_image_reference"] == "deepsafe-rtdetrv4-person:r10"
    assert plan["build"]["parent_r1"]["plan"]["sha256"] == (
        "822404eff4b05a5c357ce2772f3f572966cef66aae9d20d0651a65b9263e0352"
    )
    assert plan["build"]["parent_r1"]["failed_build_receipt"]["status"] == "failed"
    assert plan["build"]["parent_r2"]["plan"]["sha256"] == (
        "4ed41978a1e4fdd36ec8f6d9577c3de8d306c5af8ad7be92d27342f3b6ed3291"
    )
    assert plan["build"]["parent_r2"]["failure_class"] == "PIP_CHECK_SPIN_CLICK_CONFLICT"
    assert plan["build"]["parent_r3"]["plan"]["sha256"] == (
        "f55f5331921ea92b6336a168114f1d2a62561d4f63e81396894a0041433e1827"
    )
    assert plan["build"]["parent_r3"]["failure_class"] == (
        "PIP_CHECK_HUGGINGFACE_HUB_CLICK_CONFLICT"
    )
    assert plan["build"]["parent_r4"]["plan"]["sha256"] == (
        "663d25b9a67fa7b13173b5480fb8361949cddae915a35297e40e54bbb327d0cc"
    )
    assert plan["build"]["parent_r4"]["failure_class"] == (
        "PREFLIGHT_SHADOWED_DISTRIBUTIONS_TREATED_AS_ACTIVE"
    )
    assert plan["build"]["parent_r5"]["successful_build_receipt"]["status"] == "passed"
    assert plan["build"]["parent_r5"]["smoke_failure_host_receipt"]["status"] == "failed"
    assert plan["build"]["parent_r5"]["failed_run_id"] == "smoke-one-step-001"
    assert plan["build"]["parent_r6"]["successful_build_receipt"]["status"] == "passed"
    assert plan["build"]["parent_r6"]["smoke_failure_host_receipt"]["status"] == "failed"
    assert plan["build"]["parent_r6"]["failed_run_id"] == "smoke-one-step-002"
    assert plan["build"]["parent_r7"]["successful_build_receipt"]["status"] == "passed"
    assert plan["build"]["parent_r7"]["smoke_failure_host_receipt"]["status"] == "failed"
    assert plan["build"]["parent_r7"]["failed_run_id"] == "smoke-one-step-003"
    assert plan["build"]["parent_r8"]["successful_build_receipt"]["status"] == "passed"
    assert plan["build"]["parent_r8"]["smoke_failure_host_receipt"]["status"] == "failed"
    assert plan["build"]["parent_r8"]["failed_run_id"] == "smoke-one-step-004"
    assert plan["build"]["parent_r8"]["failure_class"] == (
        "NONFINITE_GRADIENT_AFTER_UNSCALE_WITHOUT_FOUND_INF_AUDIT"
    )
    assert plan["build"]["parent_r8"]["amp_overflow_confirmed"] is False
    assert plan["build"]["parent_r9"]["successful_build_receipt"]["status"] == "passed"
    assert plan["build"]["parent_r9"]["successful_smoke_host_receipt"]["status"] == "passed"
    assert plan["build"]["parent_r9"]["baseline_failure_host_receipt"]["status"] == "failed"
    assert plan["build"]["parent_r9"]["successful_smoke_run_id"] == "smoke-one-step-005"
    assert plan["build"]["parent_r9"]["failed_baseline_run_id"] == "baseline-eval-001"
    assert plan["build"]["runtime_smoke_repair"] == {
        "failed_run_id": "smoke-one-step-001",
        "next_run_id": "smoke-one-step-002",
        "failure_class": "LEADING_NEGATIVE_TRAIN_SAMPLE",
        "root_cause": (
            "shuffle=false batch_size=1 selected train COCO index 0 image_id=2 with zero "
            "annotations; first annotated train sample is index 1 image_id=6"
        ),
        "selector": "first_annotated_train_batch_in_deterministic_coco_order",
        "scan_limit_batches": 1524,
        "batch_size": 1,
        "train_only": True,
        "selection_receipt_required": True,
        "all_negative_fail_closed": True,
        "official_test_opened": False,
        "test_unseen_opened": False,
    }
    assert plan["build"]["runtime_determinism_repair"] == {
        "failed_run_id": "smoke-one-step-002",
        "next_run_id": "smoke-one-step-003",
        "failure_class": "PYTORCH_CUDA_GRID_SAMPLER_BACKWARD_NOT_DETERMINISTIC",
        "root_cause": (
            "PyTorch 2.13 CUDA grid_sampler_2d_backward_cuda has no deterministic "
            "implementation for the RT-DETRv4 deformable sampler backward path"
        ),
        "policy_id": "rtdetrv4-grid-sampler-backward-bounded-v1",
        "allowed_operator": "grid_sampler_2d_backward_cuda",
        "exact_warning_category": "UserWarning",
        "exact_warning_message_sha256": (
            "53f9cc720ad81b25018a49407268276b4f04c1cd871cfe48137d65188ef7daf7"
        ),
        "torch_deterministic_algorithms": True,
        "torch_deterministic_warn_only": True,
        "unexpected_nondeterminism_fail_closed": True,
        "seed": 20260718,
        "cublas_workspace_config": ":4096:8",
        "cudnn_deterministic": True,
        "bitwise_reproducibility_claimed": False,
        "warning_audit_receipt_required": True,
        "official_test_opened": False,
        "test_unseen_opened": False,
    }
    assert plan["build"]["runtime_warning_serialization_repair"] == {
        "failed_run_id": "smoke-one-step-003",
        "next_run_id": "smoke-one-step-004",
        "failure_class": "PINNED_PYTORCH_INTERNAL_WARNING_SUFFIX_NOT_ALLOWLISTED",
        "root_cause": (
            "PyTorch 2.13 appended its pinned Context.cpp:186 internal source suffix "
            "to the exact allowlisted base grid-sampler warning"
        ),
        "previous_policy_id": "rtdetrv4-grid-sampler-backward-bounded-v1",
        "policy_id": "rtdetrv4-grid-sampler-backward-bounded-v2",
        "allowed_operator": "grid_sampler_2d_backward_cuda",
        "exact_warning_category": "UserWarning",
        "exact_base_warning_message_sha256": (
            "53f9cc720ad81b25018a49407268276b4f04c1cd871cfe48137d65188ef7daf7"
        ),
        "exact_internal_suffix_sha256": (
            "02b1582c2cc2d39f00079065d1a1021d1056a0df10b886e2f4c2f24854d91e21"
        ),
        "exact_runtime_warning_message_sha256": (
            "fb657a375d48f1d665cb07b8748eb3d0a7e41f4c3bba05919014ca5389ce04e6"
        ),
        "allowed_message_sha256s": [
            "53f9cc720ad81b25018a49407268276b4f04c1cd871cfe48137d65188ef7daf7",
            "fb657a375d48f1d665cb07b8748eb3d0a7e41f4c3bba05919014ca5389ce04e6",
        ],
        "allowlist_variant_count": 2,
        "suffix_path_line_mutation_fail_closed": True,
        "torch_deterministic_algorithms": True,
        "torch_deterministic_warn_only": True,
        "unexpected_nondeterminism_fail_closed": True,
        "seed": 20260718,
        "cublas_workspace_config": ":4096:8",
        "cudnn_deterministic": True,
        "bitwise_reproducibility_claimed": False,
        "warning_audit_receipt_required": True,
        "official_test_opened": False,
        "test_unseen_opened": False,
    }
    amp_repair = plan["build"]["smoke_amp_overflow_repair"]
    assert amp_repair["failed_run_id"] == "smoke-one-step-004"
    assert amp_repair["next_run_id"] == "smoke-one-step-005"
    assert amp_repair["overflow_hypothesis_status"] == (
        "consistent_but_not_confirmed_by_r8_evidence"
    )
    assert amp_repair["r8_amp_overflow_confirmed"] is False
    assert amp_repair["policy_id"] == container.SMOKE_AMP_POLICY_ID
    assert amp_repair["grad_scaler_contract"] == {
        "type": "GradScaler",
        "enabled": True,
        "initial_scale": 65536.0,
        "backoff_factor": 0.5,
        "growth_factor": 2.0,
        "growth_interval": 2000,
        "max_attempts": 16,
    }
    assert amp_repair["scale_before_schedule"] == [
        container._expected_smoke_amp_scale(attempt)
        for attempt in range(1, container.SMOKE_AMP_MAX_ATTEMPTS + 1)
    ]
    assert amp_repair["sixteenth_overflow_scale_after"] == 1.0
    assert amp_repair["overflow_attempt_requires_unchanged"] == [
        "model",
        "optimizer",
        "ema",
        "input",
    ]
    assert amp_repair["overflow_model_state_transaction_rollback"] is True
    assert amp_repair["overflow_pre_rollback_model_hash_recorded"] is True
    assert amp_repair["overflow_optimizer_step_skipped"] is True
    assert amp_repair["overflow_ema_update_executed"] is False
    assert amp_repair["final_finite_nonzero_gradients_required"] is True
    assert amp_repair["final_scale_unchanged_due_growth_interval"] is True
    assert amp_repair["persistent_overflow_fail_closed"] is True
    assert amp_repair["bitwise_reproducibility_claimed"] is False
    eval_repair = plan["build"]["eval_position_device_repair"]
    assert eval_repair["prior_failed_run_id"] == "baseline-eval-001"
    assert eval_repair["next_smoke_run_id"] == "smoke-one-step-006"
    assert eval_repair["next_baseline_run_id"] == "baseline-eval-002"
    assert eval_repair["policy_id"] == (
        "rtdetrv4-hybrid-encoder-eval-position-buffer-v1"
    )
    assert eval_repair["registration"] == "torch.nn.Module.register_buffer"
    assert eval_repair["persistent"] is False
    assert eval_repair["state_dict_semantics_changed"] is False
    assert eval_repair["baseline_model_covered"] is True
    assert eval_repair["full_run_model_covered"] is True
    assert eval_repair["full_run_ema_covered"] is True
    assert eval_repair[
        "ema_materialized_from_remapped_cpu_model_before_registration"
    ] is True
    assert plan["modes"]["smoke_one_step"]["next_run_id"] == "smoke-one-step-006"
    assert plan["modes"]["baseline_eval"]["next_run_id"] == "baseline-eval-002"
    assert plan["build"]["dependency_layer_cache"][
        "dynamic_build_args_declared_after_dependency_run"
    ] is True
    assert plan["training_recipe"]["determinism"]["bitwise_reproducibility_claimed"] is False
    assert len(plan["training_recipe"]["determinism"]["allowlist"]) == 2


def test_r9_plan_publisher_is_cpu_only_deterministic_and_does_not_publish_on_build() -> None:
    before = plan_r9.OUTPUT.read_bytes()
    first = plan_r9.build_plan("2026-07-18T01:00:00Z")
    second = plan_r9.build_plan("2026-07-18T01:00:00Z")
    assert first == second
    assert plan_r9.common.canonical_fingerprint(first) == first["fingerprint_sha256"]
    assert first["execution"] == {
        "performed_during_preparation": False,
        "docker_build_executed": False,
        "gpu_queried": False,
        "gpu_workload_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
    }
    assert plan_r9.OUTPUT.read_bytes() == before
    source = Path(plan_r9.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "nvidia-smi" not in source
    assert first["build"]["immutable_revision_plan"] == {
        "path": (
            "models/person/training-lanes/"
            "rtdetrv4-s-r-livit-person-r1-gpu-v1/execution-plan-r9.json"
        ),
        "mode": "0440",
        "publication": "fsynced_temp_then_atomic_no_overwrite_hardlink",
        "overwrite_allowed": False,
        "byte_equal_to_current_plan_required": True,
        "current_plan_mode": "0664",
    }


def test_r9_publisher_atomically_creates_equal_current_and_immutable_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = tmp_path / "execution-plan.json"
    archive = tmp_path / "execution-plan-r9.json"
    current.write_bytes(b"old-current\n")
    current.chmod(0o664)
    monkeypatch.setattr(plan_r9, "OUTPUT", current)
    monkeypatch.setattr(plan_r9, "ARCHIVE", archive)

    def local_pin(path: Path) -> dict:
        payload = path.read_bytes()
        return {
            "path": path.name,
            "bytes": len(payload),
            "sha256": host.hashlib.sha256(payload).hexdigest(),
        }

    monkeypatch.setattr(plan_r9.common, "pin", local_pin)
    plan = {"schema_version": "test/v1", "fingerprint_sha256": "a" * 64}
    publication = plan_r9.publish(plan)
    expected = plan_r9.plan_payload(plan)
    assert current.read_bytes() == expected
    assert archive.read_bytes() == expected
    assert current.stat().st_mode & 0o777 == 0o664
    assert archive.stat().st_mode & 0o777 == 0o440
    assert publication["byte_equal"] is True
    assert publication["current_plan"]["sha256"] == publication[
        "immutable_revision_plan"
    ]["sha256"]

    before_current = current.read_bytes()
    before_archive = archive.read_bytes()
    with pytest.raises(plan_r9.PlanError, match="already exists; overwrite refused"):
        plan_r9.publish({"changed": True})
    assert current.read_bytes() == before_current
    assert archive.read_bytes() == before_archive


def test_r9_publisher_refuses_preexisting_archive_symlink_without_touching_current(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = tmp_path / "execution-plan.json"
    archive = tmp_path / "execution-plan-r9.json"
    target = tmp_path / "attacker-target.json"
    current.write_bytes(b"old-current\n")
    target.write_bytes(b"do-not-touch\n")
    archive.symlink_to(target.name)
    monkeypatch.setattr(plan_r9, "OUTPUT", current)
    monkeypatch.setattr(plan_r9, "ARCHIVE", archive)
    with pytest.raises(plan_r9.PlanError, match="already exists; overwrite refused"):
        plan_r9.publish({"changed": True})
    assert current.read_bytes() == b"old-current\n"
    assert archive.is_symlink()
    assert target.read_bytes() == b"do-not-touch\n"


def test_r9_publisher_rolls_back_its_archive_link_if_current_replace_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = tmp_path / "execution-plan.json"
    archive = tmp_path / "execution-plan-r9.json"
    current.write_bytes(b"old-current\n")
    monkeypatch.setattr(plan_r9, "OUTPUT", current)
    monkeypatch.setattr(plan_r9, "ARCHIVE", archive)

    def fail_replace(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(plan_r9.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        plan_r9.publish({"changed": True})
    assert current.read_bytes() == b"old-current\n"
    assert not archive.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["execution-plan.json"]


def test_checked_in_r9_archive_is_frozen_equal_and_no_overwrite() -> None:
    current = plan_r9.OUTPUT
    archive = plan_r9.ARCHIVE
    assert current.stat().st_mode & 0o777 == 0o664
    assert archive.stat().st_mode & 0o777 == 0o440
    before_current = current.read_bytes()
    before_archive = archive.read_bytes()
    plan = json.loads(archive.read_text(encoding="utf-8"))
    assert plan["build"]["revision"] == 9
    assert plan["fingerprint_sha256"] == host.R9_PLAN_FINGERPRINT
    with pytest.raises(plan_r9.PlanError, match="already exists; overwrite refused"):
        plan_r9.publish(plan)
    assert current.read_bytes() == before_current
    assert archive.read_bytes() == before_archive


def test_r10_plan_publisher_is_cpu_only_deterministic_and_does_not_publish_on_build() -> None:
    before = plan_r10.OUTPUT.read_bytes()
    first = plan_r10.build_plan("2026-07-18T03:05:00Z")
    second = plan_r10.build_plan("2026-07-18T03:05:00Z")
    assert first == second
    assert plan_r10.common.canonical_fingerprint(first) == first["fingerprint_sha256"]
    assert first["build"]["revision"] == 10
    assert first["build"]["parent_r9"]["successful_smoke_run_id"] == (
        "smoke-one-step-005"
    )
    assert first["build"]["parent_r9"]["failed_baseline_run_id"] == (
        "baseline-eval-001"
    )
    assert first["execution"] == {
        "performed_during_preparation": False,
        "docker_build_executed": False,
        "gpu_queried": False,
        "gpu_workload_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
    }
    assert plan_r10.OUTPUT.read_bytes() == before
    source = Path(plan_r10.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "nvidia-smi" not in source


def test_r10_lineage_rejects_tampered_r9_baseline_evidence_pin() -> None:
    plan = plan_r10.build_plan("2026-07-18T03:05:00Z")
    plan["build"]["parent_r9"]["baseline_failure_container_receipt"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(host.GpuLaneError, match="lineage pin mismatch"):
        host._validate_prior_build_lineage(plan)


def test_all_four_gpu_lane_schemas_are_valid_draft_2020_12() -> None:
    for path in host.SCHEMA_PATHS.values():
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)


def test_dependency_lock_has_45_unique_exact_hash_locked_wheels() -> None:
    text = (LANE / "requirements.lock").read_text(encoding="utf-8")
    rows = re.findall(
        r"(?m)^([A-Za-z0-9_.-]+)==([^\s\\]+) \\\n+\s+--hash=sha256:([0-9a-f]{64})$",
        text,
    )
    assert len(rows) == 45
    assert len({name.lower() for name, _, _ in rows}) == 45
    assert len({digest for _, _, digest in rows}) == 45
    assert not re.search(r"(?m)^[A-Za-z0-9_.-]+\s*(?:>=|<=|~=|>|<)", text)
    contract = json.loads((LANE / "dependency-contract.json").read_text(encoding="utf-8"))
    locked = {_normalize_package(name): version for name, version, _ in rows}
    expected = {
        _normalize_package(name): version
        for name, version in contract["hash_locked_packages"].items()
    }
    assert locked == expected
    assert contract["digest_pinned_base_packages"] == {
        "torch": "2.13.0+cu130",
        "torchvision": "0.28.0+cu130",
    }
    override = contract["installation"]["pep668_override"]
    assert override["enabled"] is True
    assert override["scope"] == "digest_pinned_derived_container_image_build_only"
    assert override["host_python_mutated"] is False
    assert "--no-deps" in override["controls"]
    compatibility = contract["base_image_compatibility"]
    removed = compatibility["removed_base_distribution"]
    assert removed["name"] == "spin"
    assert removed["version"] == "0.18"
    assert removed["reverse_active_requires_dist_edges_must_equal"] == 0
    assert compatibility["click"]["selected_pin"] == "click==8.4.2"
    assert compatibility["click"]["wheel_sha256"] == (
        "e6f9f66136c816745b9d65817da91d61d957fb16e02e4dcd0552553c5a197b76"
    )
    critical_edges = compatibility["selected_official_wheel_active_requires_dist"]
    assert critical_edges["huggingface-hub==1.24.0"][0] == "click<9.0.0,>=8.4.2"
    assert len(critical_edges["transformers==5.14.1"]) == 9
    assert len(critical_edges["accelerate==1.14.0"]) == 7
    assert len(critical_edges["huggingface-hub==1.24.0"]) == 9
    assert all(compatibility["verification"].values())
    metadata_selection = compatibility["installed_metadata_selection"]
    assert metadata_selection["expected_container_search_path"] == [
        "/opt/deepsafe/lane",
        "/usr/lib/python312.zip",
        "/usr/lib/python3.12",
        "/usr/lib/python3.12/lib-dynload",
        "/usr/local/lib/python3.12/dist-packages",
        "/usr/lib/python3/dist-packages",
    ]
    shadowed = metadata_selection["expected_preflight_shadowed_distributions"]
    assert sorted(shadowed) == ["pip", "setuptools", "wheel"]
    assert shadowed["pip"]["effective"]["version"] == "26.1.2"
    assert shadowed["pip"]["shadowed"][0]["version"] == "24.0"
    assert shadowed["setuptools"]["effective"]["version"] == "81.0.0"
    assert shadowed["setuptools"]["effective"]["metadata_path"].endswith(
        "setuptools-81.0.0.dist-info"
    )
    assert metadata_selection["same_precedence_duplicates_allowed"] is False
    assert locked["click"] == "8.4.2"


def test_dependency_verifier_lock_and_hub_click_closure_are_fail_closed() -> None:
    locked = compatibility_verifier.load_lock()
    assert len(locked) == 45
    assert locked["click"] == {
        "name": "click",
        "normalized_name": "click",
        "version": "8.4.2",
        "sha256": "e6f9f66136c816745b9d65817da91d61d957fb16e02e4dcd0552553c5a197b76",
    }
    records = {
        "click": {
            "name": "click",
            "version": "8.4.2",
            "requires": [],
        },
        "huggingface-hub": {
            "name": "huggingface-hub",
            "version": "1.24.0",
            "requires": ["click<9.0.0,>=8.4.2"],
        },
    }
    edges, unsatisfied, skipped = compatibility_verifier.active_edges(
        records, require_closed=True
    )
    assert len(edges) == 1
    assert unsatisfied == []
    assert skipped == 0

    records["click"]["version"] = "8.3.3"
    _, unsatisfied, _ = compatibility_verifier.active_edges(records, require_closed=False)
    assert unsatisfied == [
        {
            "source": "huggingface-hub",
            "source_version": "1.24.0",
            "requirement": "click<9.0.0,>=8.4.2",
            "target": "click",
            "target_version": "8.3.3",
            "reason": "version_mismatch",
        }
    ]
    with pytest.raises(compatibility_verifier.CompatibilityError, match="unsatisfied"):
        compatibility_verifier.active_edges(records, require_closed=True)


def test_dependency_compatibility_receipt_is_create_once_and_mode_0444(
    tmp_path: Path,
) -> None:
    receipt = {
        "schema_version": compatibility_verifier.SCHEMA_VERSION,
        "phase": "preflight",
        "status": "passed",
    }
    receipt["fingerprint_sha256"] = compatibility_verifier.canonical_fingerprint(receipt)
    destination = tmp_path / "dependency-preflight.json"
    compatibility_verifier.write_receipt(destination, receipt)
    assert destination.stat().st_mode & 0o777 == 0o444
    with pytest.raises(compatibility_verifier.CompatibilityError, match="overwrite"):
        compatibility_verifier.write_receipt(destination, receipt)


class _FakeDistribution:
    def __init__(
        self, *, name: str, version: str, root: Path, metadata_directory: str
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.requires: list[str] = []
        self._root = root
        self._path = root / metadata_directory

    def locate_file(self, value: str) -> Path:
        return self._root / value


def _patch_fake_distributions(
    monkeypatch: pytest.MonkeyPatch,
    *, search_path: list[Path],
    rows: list[_FakeDistribution],
    effective: dict[str, _FakeDistribution],
) -> None:
    monkeypatch.setattr(
        compatibility_verifier.sys, "path", [path.as_posix() for path in search_path]
    )
    monkeypatch.setattr(
        compatibility_verifier.importlib.metadata, "distributions", lambda: iter(rows)
    )
    monkeypatch.setattr(
        compatibility_verifier.importlib.metadata,
        "distribution",
        lambda name: effective[_normalize_package(name)],
    )


def test_effective_installed_records_use_sys_path_and_ledger_shadowed_copies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lane = tmp_path / "lane"
    local = tmp_path / "local"
    distro = tmp_path / "distro"
    for path in (lane, local, distro):
        path.mkdir()
    pip_local = _FakeDistribution(
        name="pip", version="26.1.2", root=local, metadata_directory="pip-26.1.2.dist-info"
    )
    pip_distro = _FakeDistribution(
        name="pip", version="24.0", root=distro, metadata_directory="pip-24.0.dist-info"
    )
    wheel = _FakeDistribution(
        name="wheel", version="0.47.0", root=local, metadata_directory="wheel-0.47.0.dist-info"
    )
    _patch_fake_distributions(
        monkeypatch,
        search_path=[lane, local, distro],
        rows=[pip_distro, wheel, pip_local],
        effective={"pip": pip_local, "wheel": wheel},
    )
    records, selection = compatibility_verifier.installed_records()
    assert records["pip"]["version"] == "26.1.2"
    assert selection["candidate_distribution_count"] == 3
    assert selection["effective_distribution_count"] == 2
    assert selection["shadowed_distribution_name_count"] == 1
    assert selection["shadowed_distribution_copy_count"] == 1
    assert selection["shadowed_distributions"][0]["effective"]["search_path_index"] == 1
    assert selection["shadowed_distributions"][0]["shadowed"][0]["search_path_index"] == 2


def test_digest_base_shadow_fixture_matches_exact_selection_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = compatibility_verifier.load_metadata_selection_contract()
    search_path = [Path(root) for root in contract["expected_container_search_path"]]
    rows: list[_FakeDistribution] = []
    effective: dict[str, _FakeDistribution] = {}
    for name, expected in contract["expected_preflight_shadowed_distributions"].items():
        effective_pin = expected["effective"]
        selected = _FakeDistribution(
            name=name,
            version=effective_pin["version"],
            root=Path(effective_pin["search_root"]),
            metadata_directory=Path(effective_pin["metadata_path"]).name,
        )
        rows.append(selected)
        effective[name] = selected
        for shadow_pin in expected["shadowed"]:
            rows.append(
                _FakeDistribution(
                    name=name,
                    version=shadow_pin["version"],
                    root=Path(shadow_pin["search_root"]),
                    metadata_directory=Path(shadow_pin["metadata_path"]).name,
                )
            )
    _patch_fake_distributions(
        monkeypatch,
        search_path=search_path,
        rows=list(reversed(rows)),
        effective=effective,
    )
    records, selection = compatibility_verifier.installed_records()
    assert {name: records[name]["version"] for name in sorted(records)} == {
        "pip": "26.1.2",
        "setuptools": "81.0.0",
        "wheel": "0.47.0",
    }
    verified = compatibility_verifier.validate_metadata_selection(
        selection, allow_additional_shadowed=False
    )
    assert verified["search_path_contract_verified"] is True


def test_effective_installed_records_reject_same_precedence_ambiguity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "site"
    root.mkdir()
    first = _FakeDistribution(
        name="pip", version="26.1.2", root=root, metadata_directory="pip-26.1.2.dist-info"
    )
    second = _FakeDistribution(
        name="pip", version="24.0", root=root, metadata_directory="pip-24.0.dist-info"
    )
    _patch_fake_distributions(
        monkeypatch,
        search_path=[root],
        rows=[first, second],
        effective={"pip": first},
    )
    with pytest.raises(compatibility_verifier.CompatibilityError, match="ambiguous"):
        compatibility_verifier.installed_records()


def test_effective_installed_records_reject_importlib_precedence_disagreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local = tmp_path / "local"
    distro = tmp_path / "distro"
    local.mkdir()
    distro.mkdir()
    first = _FakeDistribution(
        name="pip", version="26.1.2", root=local, metadata_directory="pip-26.1.2.dist-info"
    )
    second = _FakeDistribution(
        name="pip", version="24.0", root=distro, metadata_directory="pip-24.0.dist-info"
    )
    _patch_fake_distributions(
        monkeypatch,
        search_path=[local, distro],
        rows=[first, second],
        effective={"pip": second},
    )
    with pytest.raises(compatibility_verifier.CompatibilityError, match="disagrees"):
        compatibility_verifier.installed_records()


def test_effective_installed_records_reject_distribution_outside_sys_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    search_root = tmp_path / "search"
    unknown_root = tmp_path / "unknown"
    search_root.mkdir()
    unknown_root.mkdir()
    row = _FakeDistribution(
        name="pip", version="26.1.2", root=unknown_root, metadata_directory="pip.dist-info"
    )
    _patch_fake_distributions(
        monkeypatch,
        search_path=[search_root],
        rows=[row],
        effective={"pip": row},
    )
    with pytest.raises(compatibility_verifier.CompatibilityError, match="absent from sys.path"):
        compatibility_verifier.installed_records()


def test_metadata_selection_contract_rejects_unexpected_preflight_shadow_name() -> None:
    contract = compatibility_verifier.load_metadata_selection_contract()
    search_path = [
        {"index": index, "raw": root, "absolute": root, "exists": True}
        for index, root in enumerate(contract["expected_container_search_path"])
    ]
    rows = []
    for name, expected in contract["expected_preflight_shadowed_distributions"].items():
        effective = expected["effective"]
        rows.append(
            {
                "name": name,
                "effective": {
                    "name": name,
                    "metadata_path": f"{effective['search_root']}/{name}-effective.dist-info",
                    **effective,
                },
                "shadowed": [
                    {
                        "name": name,
                        "metadata_path": f"{row['search_root']}/{name}-shadowed.dist-info",
                        **row,
                    }
                    for row in expected["shadowed"]
                ],
            }
        )
    observed = {"policy": contract["policy"], "search_path": search_path, "shadowed_distributions": rows}
    result = compatibility_verifier.validate_metadata_selection(
        observed, allow_additional_shadowed=False
    )
    assert result["expected_shadowed_selections_verified"] is True
    observed["shadowed_distributions"].append({
        "name": "extra",
        "effective": {
            "name": "extra",
            "version": "2",
            "metadata_path": "/usr/local/lib/python3.12/dist-packages/extra-2.dist-info",
            "search_root": "/usr/local/lib/python3.12/dist-packages",
            "search_path_index": 4,
        },
        "shadowed": [{
            "name": "extra",
            "version": "1",
            "metadata_path": "/usr/lib/python3/dist-packages/extra-1.dist-info",
            "search_root": "/usr/lib/python3/dist-packages",
            "search_path_index": 5,
        }],
    })
    with pytest.raises(compatibility_verifier.CompatibilityError, match="names differ"):
        compatibility_verifier.validate_metadata_selection(
            observed, allow_additional_shadowed=False
        )
    assert compatibility_verifier.validate_metadata_selection(
        observed, allow_additional_shadowed=True
    )["additional_shadowed_names_allowed"] is True


def test_runtime_inventory_replays_effective_and_shadowed_metadata_selection() -> None:
    contract = json.loads((LANE / "dependency-contract.json").read_text(encoding="utf-8"))
    expected = contract["base_image_compatibility"]["installed_metadata_selection"]
    search_path = [
        {"index": index, "raw": root, "absolute": root, "exists": True}
        for index, root in enumerate(expected["expected_container_search_path"])
    ]
    shadowed = []
    for name, row in expected["expected_preflight_shadowed_distributions"].items():
        shadowed.append(
            {
                "name": name,
                "effective": {"name": name, **row["effective"]},
                "shadowed": [{"name": name, **copy} for copy in row["shadowed"]],
            }
        )
    selection = {
        "policy": expected["policy"],
        "search_path": search_path,
        "search_path_sha256": runtime_inventory._canonical_sha256(search_path),
        "shadowed_distribution_name_count": 3,
        "shadowed_distribution_copy_count": 3,
        "shadowed_distributions": shadowed,
    }
    preflight = {
        "base_environment": {
            "installed_metadata_selection": selection,
            "installed_metadata_selection_verification": {
                "search_path_contract_verified": True,
                "expected_shadowed_selections_verified": True,
                "additional_shadowed_names_allowed": False,
            },
        }
    }
    postflight = {
        "installed_environment": {
            "installed_metadata_selection": json.loads(json.dumps(selection)),
            "installed_metadata_selection_verification": {
                "search_path_contract_verified": True,
                "expected_shadowed_selections_verified": True,
                "additional_shadowed_names_allowed": True,
            },
        }
    }
    replay = runtime_inventory._verify_metadata_selection_receipts(
        preflight, postflight, contract
    )
    assert replay["verified"] is True
    postflight["installed_environment"]["installed_metadata_selection"][
        "shadowed_distributions"
    ][0]["effective"]["metadata_path"] = "/wrong"
    with pytest.raises(runtime_inventory.InventoryError, match="selection mismatch"):
        runtime_inventory._verify_metadata_selection_receipts(
            preflight, postflight, contract
        )


def test_dockerfile_uses_exact_base_named_context_and_hash_only_install() -> None:
    text = (LANE / "Dockerfile").read_text(encoding="utf-8")
    assert f"FROM {host.BASE_REFERENCE}" in text
    assert "pip install --break-system-packages --require-hashes" in text
    assert "--require-hashes --no-deps --only-binary=:all:" in text
    assert "COPY --from=rtdetr engine" in text
    assert "COPY --from=rtdetr configs" in text
    assert "COPY --from=rtdetr LICENSE" in text
    assert "COPY --from=rtdetr ." not in text
    assert "runtime_inventory.py --output /opt/deepsafe/runtime-inventory.json" in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    expected_order = [
        "pip download --require-hashes",
        "dependency_compatibility.py --phase preflight",
        "pip uninstall --break-system-packages --yes spin",
        "pip install --break-system-packages --require-hashes",
        "python -m pip check",
        "dependency_compatibility.py --phase postflight",
        "runtime_inventory.py --output /opt/deepsafe/runtime-inventory.json",
        "rm -rf /tmp/deepsafe-locked-wheels",
    ]
    offsets = [text.index(marker) for marker in expected_order]
    assert offsets == sorted(offsets)
    assert "--no-index --find-links=/tmp/deepsafe-locked-wheels" in text
    assert "io.deepsafe.dependency-compatibility.sha256" in text
    assert "io.deepsafe.runtime-inventory.sha256" in text
    assert text.index("dependency_compatibility.py --phase postflight") < text.index(
        "ENV PYTHONPATH=/opt/deepsafe/third_party/RT-DETRv4"
    )


def test_docker_dependency_run_cache_key_is_independent_of_r9_dynamic_args() -> None:
    text = (LANE / "Dockerfile").read_text(encoding="utf-8")
    dependency_copy = text.index(
        "COPY requirements.lock dependency-contract.json dependency_compatibility.py runtime_inventory.py ./"
    )
    dependency_run = text.index('RUN test -n "${LOCK_SHA256}"', dependency_copy)
    dependency_run_end = text.index("rm -rf /tmp/deepsafe-locked-wheels", dependency_run)
    upstream_copy = text.index("COPY --from=rtdetr engine")
    dependency_section = text[dependency_copy:dependency_run_end]
    assert "${LOCK_SHA256}" in dependency_section
    assert "${COMPATIBILITY_SHA256}" in dependency_section
    assert "${INVENTORY_SHA256}" in dependency_section
    assert "${PLAN_FINGERPRINT}" not in dependency_section
    assert "${RUNNER_SHA256}" not in dependency_section
    assert "${CONFIG_SHA256}" not in dependency_section

    dynamic_plan_arg = text.index("ARG PLAN_FINGERPRINT")
    dynamic_runner_arg = text.index("ARG RUNNER_SHA256")
    dynamic_config_arg = text.index("ARG CONFIG_SHA256")
    assert dependency_run_end < dynamic_plan_arg < upstream_copy
    assert dependency_run_end < dynamic_runner_arg < upstream_copy
    assert dependency_run_end < dynamic_config_arg < upstream_copy

    dynamic_copy = text.index(
        "COPY train.container.yml container_runner.py execution-plan.json ./"
    )
    dynamic_checks = text.index('RUN test -n "${PLAN_FINGERPRINT}"', dynamic_copy)
    dynamic_labels = text.index("io.deepsafe.plan.fingerprint", dynamic_copy)
    assert dependency_copy < dependency_run < dependency_run_end
    assert dependency_run_end < upstream_copy < dynamic_copy < dynamic_checks < dynamic_labels
    assert 'sha256sum container_runner.py' in text[dynamic_checks:dynamic_labels]
    assert 'sha256sum train.container.yml' in text[dynamic_checks:dynamic_labels]
    assert "fingerprint_sha256" in text[dynamic_checks:dynamic_labels]


def test_build_command_binds_all_source_hash_labels_without_gpu() -> None:
    plan = host.load_plan()
    command = host.render_build_command(
        plan=plan,
        image_ref=host.DEFAULT_IMAGE_REF,
        upstream_context=Path("/tmp/exact-git-archive-context"),
    )
    rendered = "\n".join(command)
    assert command[:3] == ["docker", "buildx", "build"]
    assert "--pull=false" in command
    assert "--network=default" in command
    assert f"PLAN_FINGERPRINT={plan['fingerprint_sha256']}" in command
    assert f"RUNNER_SHA256={host.file_sha256(host.RUNNER)}" in command
    assert f"LOCK_SHA256={host.file_sha256(host.LOCK)}" in command
    assert f"CONFIG_SHA256={host.file_sha256(host.CONTAINER_CONFIG)}" in command
    assert f"COMPATIBILITY_SHA256={host.file_sha256(host.DEPENDENCY_COMPATIBILITY)}" in command
    assert f"INVENTORY_SHA256={host.file_sha256(host.RUNTIME_INVENTORY)}" in command
    assert "--gpus" not in command
    assert "docker run" not in rendered


@pytest.mark.parametrize(
    ("mode", "flag"),
    [
        ("smoke_one_step", "--smoke-one-step"),
        ("baseline_eval", "--baseline-eval"),
        ("full_run", "--full-run"),
    ],
)
def test_runtime_commands_are_separate_gpu_uuid_pinned_and_read_only(
    mode: str, flag: str
) -> None:
    output = host.RUNS_ROOT / "render-only-does-not-exist"
    command = host.render_run_command(
        mode=mode,
        run_id="render-only-001",
        plan_fingerprint=host.load_plan()["fingerprint_sha256"],
        resolved_image_id="sha256:" + "a" * 64,
        output_dir=output,
    )
    assert command[0:2] == ["docker", "run"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert f"--gpus=device={host.GPU_UUID}" in command
    assert f"--env=CUDA_VISIBLE_DEVICES={host.GPU_UUID}" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert flag in command
    assert sum(item in {"--smoke-one-step", "--baseline-eval", "--full-run"} for item in command) == 1
    dataset_mount = next(item for item in command if "dst=/inputs/rlivit" in item)
    checkpoint_mount = next(item for item in command if "dst=/inputs/checkpoint/" in item)
    output_mount = next(item for item in command if "dst=/output" in item)
    assert ",readonly," in dataset_mount
    assert ",readonly," in checkpoint_mount
    assert "readonly" not in output_mount

    leased = host.render_gpu_lease_command(command)
    assert leased[:4] == [
        host.sys.executable,
        "-m",
        "validation.gpu_lease",
        "run",
    ]
    assert leased[leased.index("--owner-kind") + 1] == "person_training"
    assert leased[leased.index("--gpu-index") + 1] == "0"
    assert leased[leased.index("--ttl-seconds") + 1] == "30"
    separator = leased.index("--")
    assert leased[separator + 1 :] == command
    assert host.gpu_lease_command_sha256(command)


def test_cli_without_execute_is_inert_and_does_not_call_build_or_run(monkeypatch, capsys) -> None:
    monkeypatch.setattr(host, "execute_build", lambda **_: pytest.fail("build executed"))
    monkeypatch.setattr(host, "execute_run", lambda **_: pytest.fail("run executed"))
    assert host.main(["--smoke-one-step", "--run-id", "render-only-002"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["status"] == "rendered_not_executed"
    assert rendered["execution_performed"] is False


def test_executed_build_requires_unique_attempt_id() -> None:
    with pytest.raises(SystemExit):
        host.parse_args(
            [
                "--build-image",
                "--execute",
                "--accept-plan-fingerprint",
                "a" * 64,
            ]
        )
    parsed = host.parse_args(
        [
            "--build-image",
            "--execute",
            "--accept-plan-fingerprint",
            "a" * 64,
            "--build-attempt-id",
            "pep668-r2-test",
        ]
    )
    assert parsed.build_attempt_id == "pep668-r2-test"


def test_build_attempt_directory_is_create_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(host, "ROOT", tmp_path)
    monkeypatch.setattr(host, "BUILD_ATTEMPTS_ROOT", tmp_path / "attempts")
    created = host._create_build_attempt_directory("attempt-r2-001")
    assert created.is_dir()
    with pytest.raises(host.GpuLaneError, match="already exists"):
        host._create_build_attempt_directory("attempt-r2-001")


def test_evidence_freeze_is_durable_mode_0440_and_no_overwrite(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(host, "ROOT", tmp_path)
    evidence = tmp_path / "build-receipt.json"
    host.atomic_write(evidence, b"{}\n")
    host._freeze_published_file(evidence)
    assert evidence.stat().st_mode & 0o777 == 0o440
    with pytest.raises(host.GpuLaneError, match="already exists"):
        host.atomic_write(evidence, b'{"changed":true}\n')


@pytest.mark.parametrize(
    "parent_name", ["parent_r1", "parent_r2", "parent_r3", "parent_r4"]
)
def test_failed_build_evidence_is_frozen_and_exactly_pinned(parent_name: str) -> None:
    plan = host.load_plan()
    parent = plan["build"][parent_name]
    for key in ("plan", "failed_build_receipt", "failed_build_log"):
        pin = parent[key]
        path = ROOT / pin["path"]
        assert host.file_pin(path) == {name: pin[name] for name in ("path", "bytes", "sha256")}
        assert path.stat().st_mode & 0o777 == 0o440


def test_r6_grid_sampler_failure_lineage_is_frozen_and_exactly_pinned() -> None:
    parent = host.load_plan()["build"]["parent_r6"]
    for key in (
        "plan",
        "successful_build_receipt",
        "successful_build_log",
        "smoke_failure_host_receipt",
        "smoke_failure_container_receipt",
        "smoke_failure_log",
        "smoke_failure_events",
    ):
        pin = parent[key]
        path = ROOT / pin["path"]
        assert host.file_pin(path) == {
            name: pin[name] for name in ("path", "bytes", "sha256")
        }
        assert path.stat().st_mode & 0o777 == 0o440
    assert parent["optimizer_step_executed"] is False
    assert parent["annotated_sample_selection_reconstructed_from_pinned_r6_code_and_coco"] == {
        "dataset_index": 1,
        "image_id": 6,
        "target_count": 2,
        "direct_failure_receipt_field": False,
    }


def test_r7_runtime_warning_failure_lineage_is_frozen_and_exactly_pinned() -> None:
    parent = host.load_plan()["build"]["parent_r7"]
    for key in (
        "plan",
        "successful_build_receipt",
        "successful_build_log",
        "smoke_failure_host_receipt",
        "smoke_failure_container_receipt",
        "smoke_failure_log",
        "smoke_failure_events",
    ):
        pin = parent[key]
        path = ROOT / pin["path"]
        assert host.file_pin(path) == {
            name: pin[name] for name in ("path", "bytes", "sha256")
        }
        assert path.stat().st_mode & 0o777 == 0o440
    assert parent["optimizer_step_executed"] is False
    assert parent["observed_warning_category"] == "UserWarning"
    assert parent["observed_base_message_sha256"] == (
        container.GRID_SAMPLER_BACKWARD_WARNING_SHA256
    )
    assert parent["observed_internal_suffix_sha256"] == (
        container.GRID_SAMPLER_BACKWARD_INTERNAL_SUFFIX_SHA256
    )
    assert parent["observed_full_message_sha256"] == (
        container.GRID_SAMPLER_BACKWARD_RUNTIME_WARNING_SHA256
    )


def test_execute_rejects_wrong_plan_fingerprint_before_docker(monkeypatch) -> None:
    monkeypatch.setattr(host, "execute_run", lambda **_: pytest.fail("run executed"))
    with pytest.raises(host.GpuLaneError, match="fingerprint"):
        host.main(
            [
                "--smoke-one-step",
                "--execute",
                "--run-id",
                "blocked-001",
                "--accept-plan-fingerprint",
                "0" * 64,
            ]
        )


def test_container_cli_requires_exactly_one_mode_and_resume_is_full_only() -> None:
    common = [
        "--run-id", "test-run-001",
        "--plan-fingerprint", "a" * 64,
        "--resolved-image-id", "sha256:" + "b" * 64,
        "--gpu-uuid", container.EXPECTED_GPU_UUID,
    ]
    assert container.parse_args(["--smoke-one-step", *common]).smoke_one_step is True
    with pytest.raises(SystemExit):
        container.parse_args(["--smoke-one-step", "--baseline-eval", *common])
    with pytest.raises(SystemExit):
        container.parse_args(
            [
                "--baseline-eval",
                *common,
                "--resume", "/inputs/resume/checkpoint.pth",
                "--resume-sha256", "c" * 64,
            ]
        )


def test_baseline_source_has_no_optimizer_or_backward_path() -> None:
    source = (LANE / "container_runner.py").read_text(encoding="utf-8")
    baseline = source.split("def run_baseline_eval(", 1)[1].split("def _plain_metrics", 1)[0]
    assert ".optimizer" not in baseline
    assert ".backward(" not in baseline
    assert "len(data_loader.dataset) != 384" in baseline
    assert "person_operating_point" in baseline
    assert "coco-eval-raw.json" in baseline


def test_full_run_source_has_epoch_boundary_resume_and_60_epoch_completion() -> None:
    source = (LANE / "container_runner.py").read_text(encoding="utf-8")
    full = source.split("def run_full_training(", 1)[1].split("def output_artifact_pins", 1)[0]
    assert "train_one_epoch(" in full
    assert "evaluate(" in full
    assert "range(start_epoch, TOTAL_EPOCHS)" in full
    assert "epoch_completed=epoch" in full
    assert "epoch_records[-1][\"epoch\"] != TOTAL_EPOCHS - 1" in full
    assert "strict=True" in source.split("def load_resume_checkpoint(", 1)[1].split("def run_smoke_one_step", 1)[0]


def test_operating_point_counts_true_false_positive_and_false_negative() -> None:
    val = {
        "annotations": [
            {"image_id": 1, "bbox": [0, 0, 10, 10]},
            {"image_id": 1, "bbox": [20, 20, 10, 10]},
        ]
    }
    detections = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [40, 40, 5, 5], "score": 0.8},
    ]
    metric = container.person_operating_point(val, detections)
    assert (metric["true_positive"], metric["false_positive"], metric["false_negative"]) == (1, 1, 1)
    assert metric["precision"] == pytest.approx(0.5)
    assert metric["recall"] == pytest.approx(0.5)


def test_atomic_publication_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "receipt.json"
    container._atomic_write(destination, b"first\n")
    with pytest.raises(container.LaneError, match="already exists"):
        container._atomic_write(destination, b"second\n")
    assert destination.read_bytes() == b"first\n"


def test_read_only_mutation_probe_does_not_change_input(tmp_path: Path) -> None:
    target = tmp_path / "input.json"
    target.write_bytes(b"immutable\n")
    target.chmod(0o440)
    before = target.read_bytes()
    try:
        result = container._prove_read_only(target)
        assert result in {"EACCES", "EPERM", "EROFS"}
        assert target.read_bytes() == before
    finally:
        target.chmod(0o600)


def test_workspace_path_guard_rejects_symlink_components(tmp_path: Path, monkeypatch) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(host, "ROOT", tmp_path)
    with pytest.raises(host.GpuLaneError, match="symlink"):
        host._inside_root(link / "artifact.json", must_exist=False)


def test_output_pin_rejects_symlink_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(host.GpuLaneError, match="regular non-symlink"):
        host._artifact_pin(link, tmp_path)
