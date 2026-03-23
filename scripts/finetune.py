#!/usr/bin/env python3
"""
KN1GHT Fine-tuning Script — Legality-Filtered SFT (Phase 2)

Builds on top of the pre-trained checkpoint from train.py:

  1. Load the pre-trained model
  2. Generate continuations from every opening prompt in CHESS_OPENINGS,
     plus truncated prefixes of those openings for shorter-context coverage
  3. Validate each continuation with python-chess — discard any game that
     contains an illegal move or is too short to be meaningful
  4. Fine-tune the model on the legal continuations at a lower learning rate
  5. Save to .data/models/kn1ght-sft/

The result is a model that retains the opening knowledge from pre-training
but has been nudged toward generating legal move sequences.

Usage:
    uv run python scripts/finetune.py
    uv run python scripts/finetune.py --checkpoint .data/models/kn1ght-small/ckpt_latest.pt
    uv run python scripts/finetune.py --n-per-opening 10 --iters 3000   # quick test
"""

import argparse
import io
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

# ── Import shared components from train.py ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    CHESS_OPENINGS,
    TOKENIZER_PATH,
    ChessGPT,
    ModelConfig,
    TokenStream,
    get_lr,
    get_turn_number_ids,
    select_device,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
PRETRAIN_DIR = ROOT / ".data" / "models" / "kn1ght-small"
SFT_OUTPUT_DIR = ROOT / ".data" / "models" / "kn1ght-sft"

G_START = "[g_start]"
G_END = "[g_end]"

# ── PGN Validation ────────────────────────────────────────────────────────────


def validate_pgn(text: str, min_half_moves: int = 6) -> bool:
    """Return True if every move in the PGN string is legal.

    Strips [g_start]/[g_end] special tokens, removes move numbers, and
    replays each move on a real chess.Board so that legality is checked
    against the actual game state rather than by pattern matching.
    """
    clean = text.replace(G_START, "").replace(G_END, "").strip()
    if not clean:
        return False

    # Drop result tokens that sometimes appear at the end
    clean = re.sub(r"\b(1-0|0-1|1/2-1/2|\*)\s*$", "", clean).strip()

    # Tokenise the PGN into individual move strings by removing move numbers
    # (handles both "1." and "1..." for black-first continuations)
    move_tokens = re.sub(r"\d+\.+\s*", " ", clean).split()
    move_tokens = [m.strip() for m in move_tokens if m.strip()]

    if len(move_tokens) < min_half_moves:
        return False

    board = chess.Board()
    for san in move_tokens:
        try:
            move = board.parse_san(san)
            board.push(move)
        except (
            chess.InvalidMoveError,
            chess.AmbiguousMoveError,
            chess.IllegalMoveError,
            ValueError,
        ):
            return False

    return True


# ── Prompt generation ─────────────────────────────────────────────────────────


def build_prompts(openings: list[tuple[str, str]]) -> list[str]:
    """Build a set of generation prompts from the opening list.

    For each full opening line we also generate a handful of truncated
    prefixes (at 2, 4, and 6 half-moves) so the model learns continuations
    from early in the opening, not just from the deepest position.
    """
    TRUNCATION_DEPTHS = (2, 4, 6)  # half-move counts to use as prompts

    prompts: set[str] = set()
    for _, pgn in openings:
        prompts.add(f"{G_START}{pgn}")

        parts = re.sub(r"\d+\.+\s*", " ", pgn).split()
        for k in TRUNCATION_DEPTHS:
            if k < len(parts):
                renumbered = _renumber(parts[:k])
                prompts.add(f"{G_START}{renumbered}")

    return sorted(prompts)


def _renumber(half_moves: list[str]) -> str:
    """Convert a flat list of half-moves back into numbered PGN notation."""
    out = []
    for i, move in enumerate(half_moves):
        if i % 2 == 0:
            out.append(f"{i // 2 + 1}.{move}")
        else:
            out.append(move)
    return " ".join(out)


# ── Generation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_continuations(
    model: ChessGPT,
    tokenizer: Tokenizer,
    prompts: list[str],
    device: torch.device,
    n_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    g_end_id: int,
) -> list[str]:
    """Generate `n_per_prompt` continuations for every prompt.

    Stops each generation early when the model produces [g_end], so sequences
    don't run past the natural end of a game.
    """
    model.eval()
    results: list[str] = []

    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        prompt_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        for _ in range(n_per_prompt):
            out = _generate_until_end(
                model, prompt_tensor, g_end_id, max_new_tokens, temperature, top_k
            )
            text = tokenizer.decode(out[0].tolist())
            results.append(text)

    return results


def _generate_until_end(
    model: ChessGPT,
    idx: torch.Tensor,
    g_end_id: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Like model.generate() but stops as soon as [g_end] is produced."""
    import torch.nn.functional as F

    block_size = model.cfg.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
        if next_tok.item() == g_end_id:
            break
    return idx


# ── HF data loader ────────────────────────────────────────────────────────────


def load_hf_games(hf_dataset: str, n: int) -> list[str]:
    """Stream N PGN strings from the HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets library not found — run: uv add datasets")

    print(f"Loading {n:,} HF games for anchoring …")
    ds = load_dataset(hf_dataset, split="train", streaming=True)
    games: list[str] = []
    for row in ds:
        pgn = row.get("pgn") or row.get("PGN") or row.get("text") or ""
        pgn = pgn.strip()
        if pgn:
            games.append(pgn)
        if len(games) >= n:
            break
    print(f"  Loaded {len(games):,} HF games")
    return games


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class FinetuneConfig:
    # Source
    checkpoint: str = str(PRETRAIN_DIR / "ckpt_latest.pt")
    output_dir: str = str(SFT_OUTPUT_DIR)
    # Generation
    n_per_opening: int = 5          # continuations generated per prompt
    temperature: float = 0.7        # lower = more conservative = more legal moves
    top_k: int = 40
    max_gen_tokens: int = 80        # shorter sequences fail later; keep them tight
    min_half_moves: int = 6         # minimum legal half-moves to keep a game
    # HF data mixing — prevents catastrophic forgetting of pre-training knowledge
    hf_dataset: str = "InterwebAlchemy/pgn-dataset-including-special-tokens"
    hf_mix_games: int = 10_000      # HF games blended into fine-tuning data
    # Fine-tuning
    batch_size: int = 32
    learning_rate: float = 1e-4     # lower than pre-training (3e-4)
    min_lr: float = 1e-5
    max_iters: int = 5_000
    warmup_iters: int = 200
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    turn_number_weight: float = 0.15
    openings_repeat: int = 5        # reduced — HF mixing handles the anchoring now
    # Eval / logging
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 50
    save_interval: int = 1000
    seed: int = 42


# ── Fine-tuning loop ──────────────────────────────────────────────────────────


def finetune(cfg: FinetuneConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = select_device()
    print(f"Device: {device}")

    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}\nRun train.py first.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    g_end_id = tokenizer.encode(G_END).ids[0]
    turn_ids = get_turn_number_ids(tokenizer)

    # ── Load pre-trained model ────────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = ChessGPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    pretrain_step = ckpt.get("step", "?")
    print(f"  Loaded (pre-training step {pretrain_step}, {model.num_params:,} params)")

    # ── Generate & validate continuations ─────────────────────────────────────
    prompts = build_prompts(CHESS_OPENINGS)
    print(f"\nGenerating {cfg.n_per_opening} continuations × {len(prompts)} prompts …")

    raw = generate_continuations(
        model, tokenizer, prompts, device,
        n_per_prompt=cfg.n_per_opening,
        max_new_tokens=cfg.max_gen_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        g_end_id=g_end_id,
    )

    legal, illegal = [], []
    for text in raw:
        (legal if validate_pgn(text, cfg.min_half_moves) else illegal).append(text)

    pct = 100 * len(legal) / len(raw) if raw else 0
    print(f"  Generated: {len(raw):,}  Legal: {len(legal):,}  Illegal: {len(illegal):,}  ({pct:.1f}% pass rate)")

    if len(legal) < 50:
        print("WARNING: very few legal games — consider lowering --min-half-moves or running more pre-training.")

    # ── Build datasets ────────────────────────────────────────────────────────
    random.shuffle(legal)
    n_val = max(20, len(legal) // 10)
    val_games = legal[:n_val]
    train_generated = legal[n_val:]

    # Load HF games to anchor the model — prevents forgetting pre-training knowledge
    hf_games = load_hf_games(cfg.hf_dataset, cfg.hf_mix_games)

    # Oversample opening lines, mix in generated legal games + HF anchor data
    opening_games = [pgn for _, pgn in CHESS_OPENINGS] * cfg.openings_repeat
    all_train = opening_games + train_generated + hf_games
    random.shuffle(all_train)

    print(f"  Fine-tune train: {len(all_train):,} games  ({len(opening_games):,} opening samples + {len(train_generated):,} generated + {len(hf_games):,} HF)")
    print(f"  Fine-tune val:   {len(val_games):,} games")

    ds_train = TokenStream(all_train, tokenizer, model_cfg.block_size, turn_ids, cfg.turn_number_weight)
    ds_val   = TokenStream(val_games,  tokenizer, model_cfg.block_size, turn_ids, cfg.turn_number_weight)

    if len(ds_train) == 0:
        sys.exit("Training dataset is empty — not enough legal games to fine-tune on.")

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({"model": asdict(model_cfg), "finetune": asdict(cfg)}, f, indent=2)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": cfg.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(0.9, 0.95),
    )

    # ── Fine-tuning loop ──────────────────────────────────────────────────────
    model.train()
    step = 0
    best_val_loss = float("inf")
    train_iter = iter(dl_train)
    t0 = time.time()

    print(f"\n── Fine-tuning ({cfg.max_iters} steps) ────────────────────────────────────")
    print(f"{'step':>7}  {'train_loss':>10}  {'val_loss':>10}  {'lr':>10}  {'ms/step':>8}")

    while step < cfg.max_iters:
        lr = get_lr(step, cfg.warmup_iters, cfg.max_iters, cfg.learning_rate, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            x, y, w = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train)
            x, y, w = next(train_iter)

        x, y, w = x.to(device), y.to(device), w.to(device)
        _, loss = model(x, targets=y, loss_weights=w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        step += 1

        if step % cfg.log_interval == 0:
            elapsed = (time.time() - t0) * 1000 / cfg.log_interval
            print(f"{step:>7}  {loss.item():>10.4f}  {'—':>10}  {lr:>10.2e}  {elapsed:>8.1f}", flush=True)
            t0 = time.time()

        if step % cfg.eval_interval == 0:
            val_loss = _evaluate(model, dl_val, device, cfg.eval_iters)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            print(f"{step:>7}  {'—':>10}  {val_loss:>10.4f}  {lr:>10.2e}  {'BEST' if is_best else '':>8}", flush=True)
            model.train()

        if step % cfg.save_interval == 0:
            _save(model, optimizer, model_cfg, cfg, step, best_val_loss, output_dir)
            _sample(model, tokenizer, device, model_cfg.block_size, g_end_id)
            model.train()

    print("\nFine-tuning complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    _save(model, optimizer, model_cfg, cfg, step, best_val_loss, output_dir)
    _sample(model, tokenizer, device, model_cfg.block_size, g_end_id)


# ── Helpers ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def _evaluate(model, dl, device, max_iters):
    model.eval()
    losses = []
    for i, (x, y, w) in enumerate(dl):
        if i >= max_iters:
            break
        x, y, w = x.to(device), y.to(device), w.to(device)
        _, loss = model(x, targets=y, loss_weights=w)
        losses.append(loss.item())
    return sum(losses) / len(losses) if losses else float("inf")


def _save(model, optimizer, model_cfg, cfg, step, val_loss, output_dir):
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "model_cfg": asdict(model_cfg),
    }
    path = output_dir / f"ckpt_{step:06d}.pt"
    torch.save(payload, path)
    torch.save(payload, output_dir / "ckpt_latest.pt")
    print(f"  ✓ Checkpoint saved → {path.name}")


@torch.no_grad()
def _sample(model, tokenizer, device, block_size, g_end_id):
    model.eval()
    prompts = [
        f"{G_START}1.e4 e5 2.Nf3",
        f"{G_START}1.d4 Nf6 2.c4",
        f"{G_START}1.e4 c5 2.Nf3",
    ]
    print("\n── Sample generation ──────────────────────────────────────────────")
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = _generate_until_end(model, idx, g_end_id, max_new_tokens=40, temperature=0.7, top_k=30)
        # Strip [g_start]/[g_end] from display
        text = tokenizer.decode(out[0].tolist()).replace(G_START, "").replace(G_END, "").strip()
        print(f"  {text}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="KN1GHT legality-filtered SFT fine-tuning")
    p.add_argument("--checkpoint", type=str, default=str(PRETRAIN_DIR / "ckpt_latest.pt"))
    p.add_argument("--output-dir", type=str, default=str(SFT_OUTPUT_DIR))
    p.add_argument("--n-per-opening", type=int, default=5, help="Continuations generated per prompt")
    p.add_argument("--min-half-moves", type=int, default=6, help="Min legal half-moves to keep a game")
    p.add_argument("--iters", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--openings-repeat", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = FinetuneConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        n_per_opening=args.n_per_opening,
        min_half_moves=args.min_half_moves,
        max_iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        openings_repeat=args.openings_repeat,
    )
    finetune(cfg)


if __name__ == "__main__":
    main()
