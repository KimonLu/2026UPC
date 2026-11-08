"""Paths shared by the section runners."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("UPC_OUTPUT_ROOT", REPO_ROOT)).resolve()


def data_dir(part: int) -> Path:
    return OUTPUT_ROOT / "data" / f"part{part:02d}"


def figures_dir(part: int) -> Path:
    return OUTPUT_ROOT / "figures" / f"part{part:02d}"
