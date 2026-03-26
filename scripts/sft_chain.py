#!/usr/bin/env python3
"""
sft_chain.py — Chained legality-filtered SFT rounds for kn1ght models.

Runs finetune.py repeatedly, feeding each round's output checkpoint into the
next, stopping when legality pass rate improvement drops below a threshold or
a maximum number of rounds is reached.

Each round writes to its own versioned directory (.data/models/sft/<model>-vN/)
so no checkpoint is ever overwritten.  A JSON summary is appended after every
round so you have something to review without grepping logs.

Usage (wrap with caffeinate to prevent sleep):
    caffeinate -is uv run python scripts/sft_chain.py
    caffeinate -is uv run python scripts/sft_chain.py --model kn1ght-blitz
    caffeinate -is uv run python scripts/sft_chain.py --model kn1ght-blitz --start-round 2 \\
        --checkpoint .data/models/sft/kn1ght-blitz-v1/ckpt_latest.pt

Stopping criteria (any one triggers a stop):
    - Pass rate gain < --min-improvement pp (default 3.0)
    - Pass rate already >= --target-rate % (default 80.0)
    - --max-rounds rounds completed
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / ".data" / "models"


# ── Output parsing ─────────────────────────────────────────────────────────────

def _parse_metrics(output: str) -> tuple[float | None, float | None]:
    """Extract (pass_rate_pct, best_val_loss) from finetune.py stdout."""
    pass_rate = None
    val_loss = None

    m = re.search(r"\((\d+\.\d+)% pass rate\)", output)
    if m:
        pass_rate = float(m.group(1))

    m = re.search(r"Best val loss:\s*(\d+\.\d+)", output)
    if m:
        val_loss = float(m.group(1))

    return pass_rate, val_loss


# ── Single round ───────────────────────────────────────────────────────────────

def run_round(
    round_num: int,
    checkpoint: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[float | None, float | None, int]:
    """Run one finetune.py round.  Returns (pass_rate, val_loss, returncode)."""

    log_path = ROOT / ".data" / f"{args.model}_sft_v{round_num}.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python", "scripts/finetune.py",
        "--checkpoint",      str(checkpoint),
        "--output-dir",      str(output_dir),
        "--iters",           str(args.iters),
        "--batch-size",      str(args.batch_size),
        "--n-per-opening",   str(args.n_per_opening),
        "--openings-repeat", str(args.openings_repeat),
    ]

    print(f"\n{'='*70}")
    print(f"  Round {round_num}  |  checkpoint: {checkpoint}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Log:         {log_path}")
    print(f"{'='*70}\n", flush=True)

    accumulated: list[str] = []

    with open(log_path, "w") as log_file:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=ROOT,
        ) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                accumulated.append(line)

    full_output = "".join(accumulated)
    pass_rate, val_loss = _parse_metrics(full_output)
    return pass_rate, val_loss, proc.returncode


# ── Summary log ───────────────────────────────────────────────────────────────

def _load_summary(path: Path) -> list[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _save_summary(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def _print_summary_table(records: list[dict]) -> None:
    print("\n── SFT chain progress ──────────────────────────────────────────────")
    print(f"{'Round':>6}  {'Pass rate':>10}  {'Δ pp':>7}  {'Val loss':>10}  {'Status'}")
    print(f"{'─'*6}  {'─'*10}  {'─'*7}  {'─'*10}  {'─'*20}")
    for r in records:
        pr  = f"{r['pass_rate']:.1f}%" if r["pass_rate"] is not None else "—"
        vl  = f"{r['val_loss']:.4f}"  if r["val_loss"]  is not None else "—"
        d   = f"+{r['delta_pp']:.1f}" if r["delta_pp"]  is not None else "—"
        print(f"{r['round']:>6}  {pr:>10}  {d:>7}  {vl:>10}  {r['status']}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Chain multiple SFT rounds until pass rate converges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",          default="kn1ght-blitz",
                   help="Model name; versioned output dirs are sft/<model>-vN/")
    p.add_argument("--checkpoint",     default=None,
                   help="Initial checkpoint (default: pre-training/<model>/ckpt_latest.pt)")
    p.add_argument("--start-round",    type=int, default=1,
                   help="Round number to begin at (use >1 to resume a chain)")
    p.add_argument("--max-rounds",     type=int, default=10,
                   help="Hard cap on rounds regardless of improvement (default 10)")
    p.add_argument("--min-improvement",type=float, default=3.0,
                   help="Stop when pass rate gain is below this many pp (default 3.0)")
    p.add_argument("--target-rate",    type=float, default=80.0,
                   help="Stop when pass rate reaches this %% (default 80.0)")
    # finetune.py pass-throughs
    p.add_argument("--iters",          type=int, default=5_000)
    p.add_argument("--batch-size",     type=int, default=32,
                   help="Use 32 for blitz on MPS (larger sizes cause slow memory path)")
    p.add_argument("--n-per-opening",  type=int, default=5)
    p.add_argument("--openings-repeat",type=int, default=20)
    args = p.parse_args()

    summary_path = ROOT / ".data" / f"{args.model}_sft_chain.json"
    records = _load_summary(summary_path)

    # Resolve the first checkpoint
    if args.checkpoint:
        current_ckpt = Path(args.checkpoint)
    elif args.start_round > 1:
        # Resuming: use the previous round's output
        prev_dir = MODELS_DIR / "sft" / f"{args.model}-v{args.start_round - 1}"
        current_ckpt = prev_dir / "ckpt_latest.pt"
    else:
        current_ckpt = MODELS_DIR / "pre-training" / args.model / "ckpt_latest.pt"

    if not current_ckpt.exists():
        sys.exit(f"Checkpoint not found: {current_ckpt}")

    prev_pass_rate: float | None = None
    # Seed prev_pass_rate from the last completed round in an existing summary
    if records:
        last = records[-1]
        if last.get("pass_rate") is not None:
            prev_pass_rate = last["pass_rate"]

    print(f"\nkn1ght SFT chain — {args.model}")
    print(f"  Starting round : {args.start_round}")
    print(f"  Max rounds     : {args.max_rounds}")
    print(f"  Min improvement: {args.min_improvement} pp")
    print(f"  Target rate    : {args.target_rate}%")
    print(f"  Iters/round    : {args.iters}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Initial ckpt   : {current_ckpt}")
    print(f"  Summary log    : {summary_path}")

    for round_num in range(args.start_round, args.start_round + args.max_rounds):
        output_dir = MODELS_DIR / "sft" / f"{args.model}-v{round_num}"

        pass_rate, val_loss, returncode = run_round(
            round_num, current_ckpt, output_dir, args
        )

        delta_pp = (pass_rate - prev_pass_rate) if (pass_rate is not None and prev_pass_rate is not None) else None
        status = "completed"

        # Determine stop reason (evaluated after recording this round)
        stop_reason: str | None = None
        if returncode != 0:
            stop_reason = f"finetune.py exited with code {returncode}"
            status = "failed"
        elif pass_rate is None:
            stop_reason = "could not parse pass rate from output"
            status = "parse_error"
        elif pass_rate >= args.target_rate:
            stop_reason = f"pass rate {pass_rate:.1f}% >= target {args.target_rate:.1f}%"
        elif delta_pp is not None and delta_pp < args.min_improvement:
            stop_reason = f"improvement {delta_pp:+.1f} pp < threshold {args.min_improvement:.1f} pp"

        record = {
            "round":      round_num,
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(current_ckpt),
            "output_dir": str(output_dir),
            "pass_rate":  pass_rate,
            "val_loss":   val_loss,
            "delta_pp":   delta_pp,
            "status":     status,
            "stop_reason": stop_reason,
        }
        records.append(record)
        _save_summary(summary_path, records)
        _print_summary_table(records)

        if stop_reason:
            print(f"Stopping: {stop_reason}")
            break

        # Next round reads from this round's output
        current_ckpt = output_dir / "ckpt_latest.pt"
        prev_pass_rate = pass_rate
    else:
        print(f"Stopping: reached max-rounds ({args.max_rounds}).")

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
