"""Verify a downloaded model directory against the final-container manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_root))
    from champion.predict import validate_model_dir

    validate_model_dir(args.model_dir, verify_hashes=True)
    print("All 11 checkpoints and 11 configuration files match the final image.")


if __name__ == "__main__":
    main()
