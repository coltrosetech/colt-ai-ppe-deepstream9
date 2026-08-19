from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "models/person/training-plans/rtdetrv4-s-r-livit-person-r1.json"
CONFIG = ROOT / "models/person/training-plans/rtdetrv4-s-r-livit-person-r1.yml"
DATA = ROOT / "data/derived/r-livit/person-rtdetrv4-coco-v1"
PREP_RECEIPT = DATA / "preparation-receipt.json"
DRY_RECEIPT = ROOT / "validation/results/person/training/rtdetrv4-s-r-livit-person-r1-cpu-dry-run.json"
PREP_SCRIPT = ROOT / "validation/person_rtdetrv4_finetune_prepare.py"
DRY_SCRIPT = ROOT / "validation/person_rtdetrv4_finetune_dryrun.py"

SCHEMA_PAIRS = (
    (
        ROOT / "validation/schemas/person-rtdetrv4-finetune-plan-v1.schema.json",
        PLAN,
    ),
    (
        ROOT / "validation/schemas/person-rtdetrv4-finetune-preparation-receipt-v1.schema.json",
        PREP_RECEIPT,
    ),
    (
        ROOT / "validation/schemas/person-rtdetrv4-finetune-cpu-dry-run-receipt-v1.schema.json",
        DRY_RECEIPT,
    ),
)


def load_json(path: Path) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    assert isinstance(value, dict)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(document: dict) -> str:
    unsigned = copy.deepcopy(document)
    unsigned.pop("fingerprint_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(("schema_path", "document_path"), SCHEMA_PAIRS)
def test_documents_validate_against_draft_2020_12_schema(schema_path: Path, document_path: Path) -> None:
    schema = load_json(schema_path)
    document = load_json(document_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    assert list(validator.iter_errors(document)) == []


@pytest.mark.parametrize("document_path", (PLAN, DATA / "manifest.json", PREP_RECEIPT, DRY_RECEIPT))
def test_canonical_fingerprints_are_valid(document_path: Path) -> None:
    document = load_json(document_path)
    assert document["fingerprint_sha256"] == fingerprint(document)


def test_plan_exactly_pins_source_upstream_checkpoint_config_and_closed_gates() -> None:
    plan = load_json(PLAN)
    assert plan["fingerprint_sha256"] == "cbbfd5a7df5991a5ed8ebdbbdf2744b2393a48ce47683f7b181fd063e3b8d74b"
    assert sha256(PLAN) == "3041d00094faffa394785fbbb3efbb51185d80add07d2911d19069c443237cf6"
    assert plan["inputs"]["source_dataset"]["manifest_fingerprint_sha256"] == (
        "7704a207b56f0562c940549938ef4e7d078ab7b26c1baa66b4f1e09326a973fc"
    )
    assert plan["inputs"]["source_dataset"]["split_plan_fingerprint_sha256"] == (
        "fc3a9df8c28481aba6a833c469396c870a0268f1779490ab6b2021502f1a2b99"
    )
    assert plan["inputs"]["upstream"]["commit"] == "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"
    assert plan["inputs"]["upstream"]["official_config"]["sha256"] == (
        "45cf2abdc91e2a83b2d759b7c49526880d12a70ee44c8cdd8674dd604985bbe0"
    )
    assert plan["inputs"]["checkpoint"]["file"]["sha256"] == (
        "238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee"
    )
    assert plan["model_contract"]["derived_config"]["sha256"] == sha256(CONFIG)
    assert plan["conversion"]["converter"]["sha256"] == sha256(PREP_SCRIPT)
    assert plan["gates"] == {
        "download_authorized": False,
        "gpu_authorized": False,
        "quality_claim_authorized": False,
        "training_authorized": False,
    }
    assert plan["commands"]["future_training"] is None


def test_plan_schema_fails_closed_on_authorization_or_test_role_mutation() -> None:
    schema = load_json(SCHEMA_PAIRS[0][0])
    validator = Draft202012Validator(schema)
    plan = load_json(PLAN)

    mutated = copy.deepcopy(plan)
    mutated["gates"]["training_authorized"] = True
    assert list(validator.iter_errors(mutated))

    mutated = copy.deepcopy(plan)
    mutated["inputs"]["source_dataset"]["allowed_roles"].append("test")
    assert list(validator.iter_errors(mutated))

    mutated = copy.deepcopy(plan)
    mutated["resolution_contract"]["train_at_960"] = True
    assert list(validator.iter_errors(mutated))


@pytest.mark.parametrize(
    ("split", "expected_images", "expected_annotations", "expected_empty"),
    (("train", 1524, 13396, 55), ("val", 384, 3256, 8)),
)
def test_coco_counts_categories_bboxes_and_image_reuse(
    split: str, expected_images: int, expected_annotations: int, expected_empty: int
) -> None:
    document = load_json(DATA / f"annotations/instances_{split}.json")
    assert document["categories"] == [{"id": 1, "name": "person", "supercategory": "person"}]
    assert len(document["images"]) == expected_images
    assert len(document["annotations"]) == expected_annotations
    image_ids = {image["id"] for image in document["images"]}
    assert len(image_ids) == expected_images
    image_by_id = {image["id"]: image for image in document["images"]}
    annotated_image_ids = set()
    annotation_ids = set()
    source_root = ROOT / f"data/derived/r-livit/person-finetune-v1/images/{split}"

    for image in document["images"]:
        assert image["file_name"].count("/") == 1
        assert "test" not in image["file_name"].lower()
        assert "unseen" not in image["file_name"].lower()
        source = source_root / image["file_name"]
        assert source.is_symlink()
        assert source.resolve(strict=True).is_file()

    for annotation in document["annotations"]:
        assert annotation["id"] not in annotation_ids
        annotation_ids.add(annotation["id"])
        assert annotation["image_id"] in image_ids
        annotated_image_ids.add(annotation["image_id"])
        assert annotation["category_id"] == 1
        assert annotation["iscrowd"] == 0
        x, y, width, height = annotation["bbox"]
        assert all(math.isfinite(value) for value in (x, y, width, height, annotation["area"]))
        assert x >= 0 and y >= 0 and width > 0 and height > 0 and annotation["area"] > 0
        image = image_by_id[annotation["image_id"]]
        assert x + width <= image["width"] + 1e-5
        assert y + height <= image["height"] + 1e-5
        assert annotation["area"] == pytest.approx(width * height, rel=1e-9, abs=1e-4)
    assert len(document["images"]) - len(annotated_image_ids) == expected_empty


def test_group_safe_conversion_never_opens_official_test_or_unseen() -> None:
    train = load_json(DATA / "annotations/instances_train.json")
    val = load_json(DATA / "annotations/instances_val.json")
    source_manifest = load_json(ROOT / "data/derived/r-livit/person-finetune-v1/manifest.json")
    official_test_sequences = set(source_manifest["splits"]["official_test_exclusion"]["sequences"])

    train_sequences = {image["deepsafe_sequence_id"] for image in train["images"]}
    val_sequences = {image["deepsafe_sequence_id"] for image in val["images"]}
    train_groups = {image["deepsafe_capture_group_id"] for image in train["images"]}
    val_groups = {image["deepsafe_capture_group_id"] for image in val["images"]}
    assert not train_sequences & val_sequences
    assert not train_groups & val_groups
    assert not official_test_sequences & (train_sequences | val_sequences)

    receipt = load_json(PREP_RECEIPT)
    assert receipt["source_verification"]["official_test_opened"] is False
    assert receipt["source_verification"]["test_unseen_opened"] is False
    assert receipt["source_verification"]["official_test_output_frames"] == 0


def test_derived_root_contains_metadata_only_and_no_images() -> None:
    relative_files = {
        path.relative_to(DATA).as_posix()
        for path in DATA.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert relative_files == {
        "annotations/instances_train.json",
        "annotations/instances_val.json",
        "manifest.json",
        "preparation-receipt.json",
    }
    assert not any(path.is_symlink() for path in DATA.rglob("*"))
    assert not (DATA / "images").exists()


def test_preparation_receipt_pins_exact_outputs_and_is_no_execution() -> None:
    receipt = load_json(PREP_RECEIPT)
    for pin in receipt["outputs"].values():
        path = DATA / pin["path"]
        assert path.stat().st_size == pin["bytes"]
        assert sha256(path) == pin["sha256"]
    assert receipt["counts"]["total_images"] == 1908
    assert receipt["counts"]["total_annotations"] == 16652
    assert receipt["invariants"]["train_val_sequence_overlap"] == 0
    assert receipt["invariants"]["train_val_capture_group_overlap"] == 0
    assert all(value is False for value in receipt["gates"].values())
    assert all(value is False for value in receipt["execution"].values())


def test_cpu_receipt_proves_explicit_ema_person_remap_and_finite_shapes() -> None:
    receipt = load_json(DRY_RECEIPT)
    load = receipt["checkpoint_load"]
    assert load["source"] == "ema.module"
    assert load["weights_only"] is True
    assert load["strict_model_state_load"] is True
    assert load["model_state_tensor_count"] == 796
    assert load["exact_shape_tensor_count"] == 787
    assert load["remapped_tensor_count"] == 9
    assert load["source_class_row"] == 0
    assert load["source_class_name"] == "COCO person"
    assert load["denoising_padding_source_row"] == 80
    assert load["random_classification_parameters_remaining"] == 0
    assert len(load["remapped_keys"]) == len(set(load["remapped_keys"])) == 9

    forward = receipt["structural_forward"]
    assert forward["sample_role"] == "group_safe_validation_not_official_test_or_unseen"
    assert forward["input_shape"] == [1, 3, 640, 640]
    assert forward["target_internal_class_ids"] == [0]
    assert forward["pred_logits_shape"] == [1, 300, 1]
    assert forward["pred_boxes_shape"] == [1, 300, 4]
    assert forward["output_tensors_finite"] is True
    assert forward["loss_tensors_finite"] is True
    assert forward["loss_keys"]
    assert forward["backward_executed"] is False
    assert forward["optimizer_constructed"] is False
    assert forward["quality_metric_computed"] is False


def test_cpu_receipt_pins_harness_config_and_cpu_only_runtime() -> None:
    receipt = load_json(DRY_RECEIPT)
    for field, path in (("harness", DRY_SCRIPT), ("derived_config", CONFIG), ("checkpoint", ROOT / receipt["checkpoint"]["path"])):
        pin = receipt[field]
        assert pin["bytes"] == path.stat().st_size
        assert pin["sha256"] == sha256(path)
    runtime = receipt["runtime"]
    assert runtime["device"] == "cpu"
    assert runtime["torch"].endswith("+cpu")
    assert runtime["torch_cuda_build"] is None
    assert runtime["cuda_available"] is False
    assert runtime["cuda_device_count"] == 0
    assert receipt["execution"]["real_sample_structural_forward_executed"] is True
    assert receipt["execution"]["loss_forward_executed"] is True
    for key in (
        "network_used",
        "download_executed",
        "gpu_exposed",
        "gpu_executed",
        "backward_executed",
        "optimizer_constructed",
        "optimizer_step_executed",
        "training_executed",
        "quality_evaluation_executed",
    ):
        assert receipt["execution"][key] is False
    assert all(value is False for value in receipt["gates"].values())


def test_resolution_decision_is_fixed_640_training_and_dual_deployment_only() -> None:
    plan = load_json(PLAN)
    receipt = load_json(DRY_RECEIPT)
    assert plan["resolution_contract"]["training"] == [640, 640]
    assert plan["resolution_contract"]["validation"] == [640, 640]
    assert plan["resolution_contract"]["deployment_profiles"] == [[640, 640], [960, 960]]
    assert plan["resolution_contract"]["train_at_960"] is False
    assert receipt["resolution_decision"]["profile_960_executed"] is False
    assert receipt["resolution_decision"]["profile_960_role"] == "future_deployment_export_and_benchmark_only"


def test_preparation_rerun_is_deterministic_and_does_not_rewrite() -> None:
    spec = importlib.util.spec_from_file_location("person_rtdetrv4_finetune_prepare", PREP_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = {path: (path.stat().st_mtime_ns, sha256(path)) for path in DATA.rglob("*.json")}
    result = module.prepare()
    after = {path: (path.stat().st_mtime_ns, sha256(path)) for path in DATA.rglob("*.json")}
    assert before == after
    assert set(result["write_actions"].values()) == {"verified_existing_identical"}
    assert result["training_executed"] is False
    assert result["gpu_executed"] is False
    assert result["download_executed"] is False


def test_scripts_have_no_network_subprocess_or_training_api() -> None:
    forbidden_imports = {"requests", "urllib", "socket", "subprocess"}
    for path in (PREP_SCRIPT, DRY_SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
        assert not imports & forbidden_imports
        assert "backward" not in calls
        assert "step" not in calls
        assert "train_one_epoch" not in calls
