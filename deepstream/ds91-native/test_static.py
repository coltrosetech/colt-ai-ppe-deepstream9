#!/usr/bin/env python3
"""Small regression tests for the DS9.1 native build policy verifier."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent


class NativeBuildPolicyTests(unittest.TestCase):
    def test_contract_is_valid_json(self) -> None:
        contract = json.loads(
            (HERE / "native-build-contract-v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(contract["schema_version"], "deepsafe.ds91-native-build/v1")
        self.assertFalse(contract["build_policy"]["gpu_devices"])
        self.assertFalse(contract["build_policy"]["inference"])
        self.assertEqual(contract["build_policy"]["network"], "none")
        self.assertEqual(len(contract["named_contexts"]), 7)
        self.assertEqual(len(contract["components"]), 5)

    def test_static_verifier_passes(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HERE / "verify_static.py")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("static contract: PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()

