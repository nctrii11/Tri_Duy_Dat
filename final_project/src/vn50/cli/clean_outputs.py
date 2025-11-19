"""CLI: Remove all derived outputs so the pipeline can start fresh."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_project_root(base_path: Path) -> None:
    """Refuse to run if mandatory project folders are missing."""
    required = ["data", "reports"]
    missing = [name for name in required if not (base_path / name).exists()]
    if missing:
        logger.error(
            "Cannot run cleanup: missing required directories at %s: %s",
            base_path,
            ", ".join(missing),
        )
        raise SystemExit(1)


def _delete_children(target_dir: Path) -> int:
    """Delete all immediate children of target_dir and return the number of entries removed."""
    removed = 0
    for child in target_dir.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                shutil.rmtree(child)
            removed += 1
        except Exception as exc:  # pragma: no cover - log unexpected issues
            logger.error("Failed to delete %s: %s", child, exc)
    return removed


def _clean_directories(directories: Iterable[Path]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for directory in directories:
        if not directory.exists():
            logger.warning("Directory not found: %s (skipping)", directory)
            summary[str(directory)] = -1
            continue
        removed = _delete_children(directory)
        logger.info("Cleaned %s (removed %d items)", directory, removed)
        summary[str(directory)] = removed
    return summary


def _print_summary(summary: dict[str, int]) -> None:
    lines = ["Cleanup summary:"]
    for directory, count in summary.items():
        if count < 0:
            lines.append(f"- {directory}: missing (skipped)")
        else:
            lines.append(f"- {directory}: {count} items removed")
    print("\n".join(lines))


def main_app() -> None:
    base_path = Path.cwd()
    _ensure_project_root(base_path)

    targets = [
        base_path / "data" / "interim",
        base_path / "data" / "processed",
        base_path / "reports" / "artifacts",
        base_path / "reports" / "logs",
    ]

    summary = _clean_directories(targets)
    _print_summary(summary)


if __name__ == "__main__":
    main_app()

