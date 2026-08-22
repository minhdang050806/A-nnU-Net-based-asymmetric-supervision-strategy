#!/usr/bin/env python3
"""Install the AURA nnU-Net extensions into an nnU-Net v2 installation.

nnU-Net discovers trainers with ``recursive_find_python_class`` over the
installed ``nnunetv2`` package directory, so custom trainers and losses have to
live inside that package rather than beside it on ``sys.path``. This script
copies the files under ``nnunet_extensions/nnunetv2/`` into the target
installation, preserving their relative paths.

One upstream file is overwritten: ``nnunetv2/run/load_pretrained_weights.py``.
The replacement tolerates optional auxiliary segmentation heads that are absent
from a single-head pretrained checkpoint, which is what lets AURA initialize
from the OS50 checkpoint. A ``.orig`` backup is kept for every overwritten
file.

Usage:
    python scripts/install_extensions.py                  # auto-detect nnunetv2
    python scripts/install_extensions.py --nnunet-dir ../nnUNet
    python scripts/install_extensions.py --check          # verify only
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "nnunet_extensions" / "nnunetv2"

# Files that do not exist upstream and are purely additive.
NEW_FILES = (
    "training/loss/aura_loss.py",
    "training/nnUNetTrainer/variants/loss/nnUNetTrainerAURA_v0.py",
    "training/nnUNetTrainer/variants/loss/nnUNetTrainerDiceTopK10_OS50_LS005.py",
)

# Files that replace an upstream module. A .orig backup is written before the
# first overwrite so the installation can be reverted.
PATCHED_FILES = ("run/load_pretrained_weights.py",)

ALL_FILES = NEW_FILES + PATCHED_FILES


def locate_nnunetv2(explicit: Path | None) -> Path:
    """Return the ``nnunetv2`` package directory to install into."""
    if explicit is not None:
        candidate = explicit.resolve()
        if candidate.name != "nnunetv2":
            candidate = candidate / "nnunetv2"
        if not (candidate / "run" / "run_training.py").is_file():
            raise SystemExit(f"Not an nnunetv2 package directory: {candidate}")
        return candidate

    try:
        import nnunetv2
    except ImportError:
        raise SystemExit(
            "nnunetv2 is not importable. Install nnU-Net v2 first, or pass "
            "--nnunet-dir pointing at a checkout."
        )
    return Path(nnunetv2.__path__[0]).resolve()


def report(target: Path) -> int:
    """Print the install state of every managed file; return an exit code."""
    missing = 0
    for relative in ALL_FILES:
        source = SOURCE_ROOT / relative
        destination = target / relative
        if not destination.is_file():
            state = "MISSING"
            missing += 1
        elif filecmp.cmp(source, destination, shallow=False):
            state = "ok"
        else:
            state = "DIFFERS"
            missing += 1
        print(f"  [{state:>7}] {relative}")
    return 1 if missing else 0


def install(target: Path, force: bool) -> None:
    for relative in ALL_FILES:
        source = SOURCE_ROOT / relative
        destination = target / relative
        if not source.is_file():
            raise SystemExit(f"Missing extension source: {source}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if filecmp.cmp(source, destination, shallow=False):
                print(f"  [unchanged] {relative}")
                continue
            if relative in PATCHED_FILES:
                backup = destination.with_suffix(destination.suffix + ".orig")
                if not backup.exists():
                    shutil.copy2(destination, backup)
                    print(f"  [backup   ] {backup.relative_to(target)}")
            elif not force:
                raise SystemExit(
                    f"{relative} already exists with different content. "
                    f"Re-run with --force to overwrite."
                )
        shutil.copy2(source, destination)
        print(f"  [installed] {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--nnunet-dir",
        type=Path,
        default=None,
        help="nnU-Net checkout or nnunetv2 package directory "
             "(default: the importable nnunetv2 package)",
    )
    parser.add_argument("--check", action="store_true",
                        help="Report install state without copying anything")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite additive files that already differ")
    args = parser.parse_args()

    target = locate_nnunetv2(args.nnunet_dir)
    print(f"nnunetv2 package: {target}")

    if args.check:
        sys.exit(report(target))

    install(target, force=args.force)
    print(
        "\nDone. Verify with:\n"
        "  python -c \"from nnunetv2.training.loss.aura_loss import AURAAsymmetricLoss; print('ok')\""
    )


if __name__ == "__main__":
    main()
