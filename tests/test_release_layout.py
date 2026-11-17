from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseLayoutTests(unittest.TestCase):
    def test_runtime_closure_has_eleven_models(self) -> None:
        payload = json.loads((ROOT / "configs/runtime-closure.json").read_text())
        self.assertEqual(payload["checkpoint_count"], 11)
        self.assertEqual(payload["config_count"], 11)
        self.assertEqual(len(payload["models"]), 11)

    def test_exact_inference_sources_are_present(self) -> None:
        required = {
            "container_inference.py",
            "e577_e570_axial_window_pruning.py",
            "e578_container_entrypoint.py",
            "e580_container_entrypoint.py",
            "model.py",
            "predict.py",
            "refiner.py",
        }
        observed = {path.name for path in (ROOT / "src/champion").glob("*.py")}
        self.assertLessEqual(required, observed)

    def test_final_image_sources_match_recorded_hashes(self) -> None:
        closure = json.loads(
            (ROOT / "configs/inference-source-closure.json").read_text(
                encoding="utf-8-sig"
            )
        )
        self.assertEqual(closure["file_count"], 26)
        for row in closure["files"]:
            path = ROOT / "src/champion" / row["path"]
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"]
            )


if __name__ == "__main__":
    unittest.main()
