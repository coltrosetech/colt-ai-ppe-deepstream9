from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

from validation import product_finalization_v2 as finalizer
from validation import product_finalization_v2_contract as contract


ROOT = Path(__file__).resolve().parents[1]


def test_contract_is_the_single_fixed_input_authority() -> None:
    assert finalizer.FIXED_INPUT_PATHS is contract.FIXED_INPUT_PATHS
    assert len(contract.FIXED_INPUT_PATHS) == 30
    assert len(set(contract.FIXED_INPUT_PATHS)) == 30
    assert contract.FIXED_INPUT_PATH_STRINGS == tuple(
        path.as_posix() for path in contract.FIXED_INPUT_PATHS
    )
    assert Path("validation/product_finalization_v2_contract.py") in (
        contract.FIXED_INPUT_PATHS
    )


def test_contract_module_is_stdlib_only() -> None:
    tree = ast.parse(
        (ROOT / "validation/product_finalization_v2_contract.py").read_text(
            encoding="utf-8"
        )
    )
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported == {"pathlib"}


def test_admin_imports_contract_not_execution_finalizer() -> None:
    source = (ROOT / "admin/validation.py").read_text(encoding="utf-8")
    assert "from validation.product_finalization_v2_contract import" in source
    assert "from validation.product_finalization_v2 import FIXED_INPUT_PATHS" not in source


def test_admin_dockerfile_copies_contract_only_module() -> None:
    dockerfile = (ROOT / "admin/Dockerfile").read_text(encoding="utf-8")
    assert (
        "COPY validation/product_finalization_v2_contract.py "
        "/app/validation/product_finalization_v2_contract.py"
    ) in dockerfile
    assert "COPY validation/product_finalization_v2.py" not in dockerfile


def test_container_shaped_contract_import_has_no_transitive_dependency(
    tmp_path: Path,
) -> None:
    app = tmp_path / "app"
    validation = app / "validation"
    validation.mkdir(parents=True)
    shutil.copy2(ROOT / "validation/__init__.py", validation / "__init__.py")
    shutil.copy2(
        ROOT / "validation/product_finalization_v2_contract.py",
        validation / "product_finalization_v2_contract.py",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(app)!r}); "
                "from validation.product_finalization_v2_contract import "
                "FIXED_INPUT_PATH_STRINGS; "
                "assert len(FIXED_INPUT_PATH_STRINGS) == 30; "
                "print(FIXED_INPUT_PATH_STRINGS[16])"
            ),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={},
    )
    assert completed.stdout.strip() == "validation/product_finalization_v2_contract.py"

