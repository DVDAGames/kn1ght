#!/usr/bin/env python3
"""
Upload kn1ght-bullet artifacts and training checkpoints to HuggingFace Hub.

Uploads:
  dist/kn1ght-bullet/  → root of the HF model repo
  .data/models/        → checkpoints/<phase>/ prefix in the same repo

Checkpoint phases and their source directories:
  pretrain  →  .data/models/kn1ght-small/
  sft       →  .data/models/kn1ght-sft/ … kn1ght-sft-v5/
  dpo       →  .data/models/kn1ght-dpo/

By default only the sft and dpo phases are uploaded (pretrain has ~100 files).
Pass --phases pretrain sft dpo to include all phases, or --all-checkpoints to
upload every .pt file within the selected phases (default: latest only).

Usage:
  uv run python scripts/upload.py
  uv run python scripts/upload.py --no-checkpoints
  uv run python scripts/upload.py --phases dpo
  uv run python scripts/upload.py --phases pretrain sft dpo --all-checkpoints
  uv run python scripts/upload.py --repo InterwebAlchemy/kn1ght-bullet
  uv run python scripts/upload.py --dry-run
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPO_ID   = "InterwebAlchemy/kn1ght-bullet"
DIST_DIR  = ROOT / "dist" / "kn1ght-bullet"
MODELS_DIR = ROOT / ".data" / "models"

# Maps CLI phase name → list of local directories (in order)
PHASE_DIRS: dict[str, list[Path]] = {
    "pretrain": [MODELS_DIR / "kn1ght-small"],
    "sft": [
        MODELS_DIR / "kn1ght-sft",
        MODELS_DIR / "kn1ght-sft-v2",
        MODELS_DIR / "kn1ght-sft-v3",
        MODELS_DIR / "kn1ght-sft-v4",
        MODELS_DIR / "kn1ght-sft-v5",
    ],
    "dpo": [MODELS_DIR / "kn1ght-dpo"],
}

# Maps each local directory to its path prefix inside the HF repo
def hf_prefix(phase: str, local_dir: Path) -> str:
    """Return the checkpoints/<phase>/... prefix for a given local directory."""
    if phase == "sft":
        # kn1ght-sft → v1, kn1ght-sft-v2 → v2, …
        name = local_dir.name
        version = "v1" if name == "kn1ght-sft" else name.split("-v")[-1]
        return f"checkpoints/sft/v{version.lstrip('v')}"
    return f"checkpoints/{phase}"


def collect_checkpoints(
    phases: list[str],
    all_checkpoints: bool,
) -> list[tuple[Path, str]]:
    """
    Return a list of (local_path, hf_path) pairs for the requested phases.

    When all_checkpoints is False, only ckpt_latest.pt is included from each
    directory (plus all non-latest checkpoints from dpo, which are few and
    represent meaningful training milestones).
    """
    pairs: list[tuple[Path, str]] = []

    for phase in phases:
        dirs = PHASE_DIRS.get(phase, [])
        for local_dir in dirs:
            if not local_dir.exists():
                print(f"  Warning: {local_dir} does not exist — skipping")
                continue

            prefix = hf_prefix(phase, local_dir)
            ckpts = sorted(local_dir.glob("ckpt_*.pt"))

            if all_checkpoints or phase == "dpo":
                selected = ckpts
            else:
                # Latest only — prefer the explicitly named ckpt_latest.pt
                latest = local_dir / "ckpt_latest.pt"
                selected = [latest] if latest.exists() else ckpts[-1:]

            for ckpt in selected:
                pairs.append((ckpt, f"{prefix}/{ckpt.name}"))

    return pairs


def upload_dist(api, repo_id: str, dry_run: bool) -> None:
    """Upload the full dist/kn1ght-bullet/ directory to the HF repo root."""
    if not DIST_DIR.exists():
        print(f"Error: {DIST_DIR} does not exist — run export.py first")
        sys.exit(1)

    print(f"\nUploading model artifacts: {DIST_DIR} → {repo_id}/")
    if dry_run:
        for f in sorted(DIST_DIR.rglob("*")):
            if f.is_file():
                rel = f.relative_to(DIST_DIR)
                print(f"  [dry-run] {rel}")
        return

    api.upload_folder(
        folder_path=str(DIST_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model artifacts",
    )
    print("  Done.")


def upload_checkpoints(
    api,
    repo_id: str,
    pairs: list[tuple[Path, str]],
    dry_run: bool,
) -> None:
    """Upload checkpoint files to the HF repo under checkpoints/."""
    if not pairs:
        print("\nNo checkpoints selected.")
        return

    total = len(pairs)
    print(f"\nUploading {total} checkpoint(s)...")

    for i, (local_path, hf_path) in enumerate(pairs, 1):
        size_mb = local_path.stat().st_size / 1e6
        print(f"  [{i}/{total}] {local_path.name}  ({size_mb:.0f} MB)  → {hf_path}")
        if dry_run:
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add checkpoint {hf_path}",
        )

    if not dry_run:
        print("  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload kn1ght-bullet to HuggingFace Hub")
    parser.add_argument(
        "--repo", default=REPO_ID,
        help=f"HuggingFace repo ID (default: {REPO_ID})",
    )
    parser.add_argument(
        "--phases", nargs="+", default=["sft", "dpo"],
        choices=list(PHASE_DIRS.keys()),
        metavar="PHASE",
        help="Checkpoint phases to upload: pretrain sft dpo (default: sft dpo)",
    )
    parser.add_argument(
        "--all-checkpoints", action="store_true",
        help="Upload every .pt file in the selected phases (default: latest only)",
    )
    parser.add_argument(
        "--no-checkpoints", action="store_true",
        help="Skip checkpoint upload entirely",
    )
    parser.add_argument(
        "--no-dist", action="store_true",
        help="Skip uploading dist/kn1ght-bullet/ model artifacts",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("  uv add huggingface_hub  or  pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    if not args.dry_run:
        # Verify authentication before doing any work
        try:
            user = api.whoami()
            print(f"Authenticated as: {user['name']}")
        except Exception:
            print("Error: not authenticated. Run:  huggingface-cli login")
            sys.exit(1)

    if not args.no_dist:
        upload_dist(api, args.repo, args.dry_run)

    if not args.no_checkpoints:
        pairs = collect_checkpoints(args.phases, args.all_checkpoints)
        upload_checkpoints(api, args.repo, pairs, args.dry_run)

    if args.dry_run:
        print("\n[dry-run] No files were uploaded.")
    else:
        print(f"\nAll uploads complete. View at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
