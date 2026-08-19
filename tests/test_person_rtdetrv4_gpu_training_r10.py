from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from validation import person_rtdetrv4_gpu_training as host
from validation import person_rtdetrv4_gpu_training_plan_r10 as r10


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "models/person/training-lanes/rtdetrv4-s-r-livit-person-r1-gpu-v1"


def _load_container_runner():
    path = LANE / "container_runner.py"
    spec = importlib.util.spec_from_file_location(
        "deepsafe_test_r10_container_runner", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


container = _load_container_runner()


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
    def __init__(
        self,
        *,
        indices: list[int] | None = None,
        fail_registration_for: str | None = None,
    ) -> None:
        self.eval_spatial_size = [640, 640]
        self.use_encoder_idx = [1, 2] if indices is None else indices
        self.hidden_dim = 256
        self.feat_strides = [8, 16, 32]
        self.pos_embed1 = _FakeTensor([1, 1600, 256])
        self.pos_embed2 = _FakeTensor([1, 400, 256])
        self._buffers: dict[str, _FakeTensor] = {}
        self._non_persistent_buffers_set: set[str] = set()
        self.fail_registration_for = fail_registration_for

    def register_buffer(
        self, name: str, value: _FakeTensor, *, persistent: bool
    ) -> None:
        if name == self.fail_registration_for:
            raise RuntimeError("injected registration failure")
        self._buffers[name] = value
        setattr(self, name, value)
        if not persistent:
            self._non_persistent_buffers_set.add(name)


class _FakeModel:
    def __init__(self, encoder: HybridEncoder) -> None:
        self.encoder = encoder


class _FakeTorch:
    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, _FakeTensor)


def test_r10_replays_frozen_r9_lineage_and_builds_a_schema_valid_plan() -> None:
    r10.validate_r9_evidence()
    plan = r10.build_plan("2026-07-18T03:30:00Z")
    host.validate_schema(plan)

    assert plan["fingerprint_sha256"] == host.canonical_fingerprint(plan)
    assert plan["build"]["revision"] == 10
    assert plan["build"]["parent_r9"]["resolved_image_id"] == r10.R9_IMAGE_ID
    repair = plan["build"]["eval_position_device_repair"]
    assert repair["policy_id"] == r10.POLICY_ID
    assert repair["registered_before_model_to"] is True
    assert repair["cuda_device_verified_after_model_to"] is True
    assert repair["state_dict_semantics_changed"] is False
    assert repair["baseline_model_covered"] is True
    assert repair["full_run_model_covered"] is True
    assert repair["full_run_ema_covered"] is True
    assert (
        repair["ema_materialized_from_remapped_cpu_model_before_registration"]
        is True
    )
    assert plan["execution"] == {
        "performed_during_preparation": False,
        "docker_build_executed": False,
        "gpu_queried": False,
        "gpu_workload_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
    }
    for path in r10.EXPECTED_FROZEN_SHA256:
        assert path.lstat().st_mode & 0o777 == 0o440


def test_eval_position_preflight_does_not_partially_mutate_multiple_indices() -> None:
    encoder = HybridEncoder()
    first = encoder.pos_embed1
    second = encoder.pos_embed2
    encoder.pos_embed2.shape = [1, 399, 256]

    with pytest.raises(container.LaneError, match="cached tensor shape mismatch"):
        container.register_hybrid_encoder_eval_position_buffers(
            torch=_FakeTorch, model=_FakeModel(encoder)
        )

    assert encoder._buffers == {}
    assert encoder._non_persistent_buffers_set == set()
    assert encoder.pos_embed1 is first
    assert encoder.pos_embed2 is second


def test_eval_position_registration_failure_rolls_back_plain_attributes() -> None:
    encoder = HybridEncoder(fail_registration_for="pos_embed2")
    first = encoder.pos_embed1
    second = encoder.pos_embed2

    with pytest.raises(
        container.LaneError, match="registration raised unexpectedly"
    ):
        container.register_hybrid_encoder_eval_position_buffers(
            torch=_FakeTorch, model=_FakeModel(encoder)
        )

    assert encoder._buffers == {}
    assert encoder._non_persistent_buffers_set == set()
    assert encoder.pos_embed1 is first
    assert encoder.pos_embed2 is second


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("use_encoder_idx", [True], "use_encoder_idx is invalid"),
        ("hidden_dim", 256.0, "hidden_dim is invalid"),
        ("feat_strides", [8, 16, 32.0], "stride is invalid"),
    ],
)
def test_eval_position_repair_rejects_numeric_type_coercion(
    field: str, value: object, message: str
) -> None:
    encoder = HybridEncoder(indices=[2])
    setattr(encoder, field, value)

    with pytest.raises(container.LaneError, match=message):
        container.register_hybrid_encoder_eval_position_buffers(
            torch=_FakeTorch, model=_FakeModel(encoder)
        )
    assert encoder._buffers == {}


def _assert_anchors_are_ordered(source: str, anchors: list[str]) -> None:
    cursor = -1
    for anchor in anchors:
        position = source.find(anchor, cursor + 1)
        assert position >= 0, f"missing call-order anchor: {anchor}"
        assert position > cursor
        cursor = position


def test_yamlconfig_and_model_ema_materialization_order_is_preserved() -> None:
    baseline = inspect.getsource(container.run_baseline_eval)
    _assert_anchors_are_ordered(
        baseline,
        [
            "model = config.model",
            "remap = load_ema_person_remap(model, torch)",
            "eval_position_buffers = register_hybrid_encoder_eval_position_buffers(",
            "model.to(device).eval()",
            "verify_hybrid_encoder_eval_position_buffers(",
        ],
    )

    full = inspect.getsource(container.run_full_training)
    _assert_anchors_are_ordered(
        full,
        [
            "model = config.model",
            "remap = load_ema_person_remap(model, torch)",
            "ema = config.ema",
            "model_eval_position_buffers = register_hybrid_encoder_eval_position_buffers(",
            "ema_eval_position_buffers = register_hybrid_encoder_eval_position_buffers(",
            "model.to(device)",
            "model_eval_position_buffers[\"after_model_to\"]",
            "optimizer = config.optimizer",
            "ema = ema.to(device)",
            "ema_eval_position_buffers[\"after_model_to\"]",
        ],
    )

    yaml_config = (
        ROOT / "third_party/RT-DETRv4/engine/core/yaml_config.py"
    ).read_text(encoding="utf-8")
    model_ema = (
        ROOT / "third_party/RT-DETRv4/engine/optim/ema.py"
    ).read_text(encoding="utf-8")
    assert "self._ema = create('ema', self.global_cfg, model=self.model)" in yaml_config
    assert "self.module = deepcopy(dist_utils.de_parallel(model)).eval()" in model_ema
