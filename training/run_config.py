"""Translate a recorded training JSON file into the corresponding CLI command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


# These fields are recorded in the effective checkpoint configuration but are
# derived by the trainers rather than accepted as command-line options.
IGNORED_KEYS = {"input_channels", "train_subjects", "val_subjects", "world_size"}


def as_cli(config: dict[str, Any]) -> list[str]:
    command: list[str] = []
    for key, value in config.items():
        if key in IGNORED_KEYS or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            command.extend([flag, *(str(item) for item in value)])
        else:
            command.extend([flag, str(value)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--refiner", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    module = "champion.train_refiner" if args.refiner else "champion.train"
    command = [sys.executable, "-m", module, *as_cli(config)]
    print(subprocess.list2cmdline(command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
