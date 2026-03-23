#!/usr/bin/env python3
"""
KN1GHT DPO Training Script — Move Quality Alignment (Phase 3)

Implements Direct Preference Optimization to improve the model's move quality
ranking. Uses Stockfish to evaluate positions from the opening list and build
preference pairs (chosen = good move, rejected = inaccuracy), then trains the
policy model against a frozen reference copy of the same checkpoint.

DPO loss:
    -log σ(β * ((log π_θ(chosen) - log π_ref(chosen))
                  - (log π_θ(rejected) - log π_ref(rejected))))

Reference: Rafailov et al. "Direct Preference Optimization: Your Language
           Model is Secretly a Reward Model", 2023.

Usage:
    uv run python scripts/dpo.py
    uv run python scripts/dpo.py --checkpoint .data/models/kn1ght-sft-v5/ckpt_005000.pt
    uv run python scripts/dpo.py --stockfish /usr/local/bin/stockfish
    uv run python scripts/dpo.py --pairs-cache .data/dpo_pairs.json   # reuse cached pairs
"""

import argparse
import copy
import json
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

# ── Import shared components from train.py ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    CHESS_OPENINGS,
    HF_DATASET,
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
SFT_DIR = ROOT / ".data" / "models" / "kn1ght-sft-v5"
DPO_OUTPUT_DIR = ROOT / ".data" / "models" / "kn1ght-dpo"
DEFAULT_PAIRS_CACHE = ROOT / ".data" / "dpo_pairs.json"

G_START = "[g_start]"

# ── PGN helpers ───────────────────────────────────────────────────────────────


def moves_to_pgn(half_moves: list[str]) -> str:
    """Convert a flat list of SAN half-moves to numbered PGN notation.

    E.g. ["e4", "e5", "Nf3"] → "1.e4 e5 2.Nf3"
    """
    parts = []
    for i, move in enumerate(half_moves):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.{move}")
        else:
            parts.append(move)
    return " ".join(parts)


def parse_pgn_moves(pgn: str) -> list[str]:
    """Extract a flat list of SAN half-moves from a PGN string."""
    clean = re.sub(r"\d+\.+\s*", " ", pgn)
    tokens = clean.split()
    return [t.strip() for t in tokens if t.strip()]


# ── Stockfish helpers ─────────────────────────────────────────────────────────


def find_stockfish(hint: Optional[str] = None) -> Optional[str]:
    """Return path to a Stockfish binary, or None if not found."""
    if hint and Path(hint).exists():
        return hint
    for name in ["stockfish", "stockfish-17", "stockfish-16", "stockfish-15"]:
        path = shutil.which(name)
        if path:
            return path
    # Common macOS Homebrew paths
    for candidate in [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


def centipawns(score: chess.engine.PovScore) -> Optional[int]:
    """Convert a PovScore to centipawns from the current side's perspective.

    Returns None for positions with forced mates (treat as ±10000 cp).
    """
    cp = score.relative.score(mate_score=10_000)
    return cp


# ── Pair generation ───────────────────────────────────────────────────────────


@dataclass
class Pair:
    context: str   # PGN string of moves played so far, e.g. "1.e4 e5 2.Nf3"
    chosen: str    # SAN of the better move
    rejected: str  # SAN of the worse move
    cp_chosen: int   # centipawn score of chosen move (higher = better)
    cp_rejected: int  # centipawn score of rejected move


def build_dpo_pairs(
    openings: list[tuple[str, str]],
    engine_path: str,
    depth: int = 15,
    multipv: int = 5,
    max_half_moves: int = 30,       # stop analysing after this many half-moves
    chosen_cp_loss: int = 30,       # a move is "chosen" if it loses <= this many cp
    rejected_cp_loss: int = 50,     # a move is "rejected" if it loses >= this many cp
    max_pairs_per_position: int = 3,
) -> list[Pair]:
    """Analyse opening positions with Stockfish and build preference pairs.

    For each position (up to max_half_moves deep in each opening line):
      - Run Stockfish multipv to get the top candidate moves with scores.
      - Chosen moves: within chosen_cp_loss of the best move.
      - Rejected moves: more than rejected_cp_loss worse than the best move.
      - Cross-product: every (chosen, rejected) combination.

    This generates training signal focused on opening quality — the phase
    where the tutoring app interacts with students most.
    """
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception as e:
        sys.exit(f"Could not open Stockfish at {engine_path!r}: {e}")

    pairs: list[Pair] = []
    seen_contexts: set[str] = set()  # deduplicate positions reached via multiple openings

    print(f"Analysing {len(openings)} opening lines (Stockfish depth={depth}, multipv={multipv}) …")

    for name, pgn in openings:
        board = chess.Board()
        half_moves = parse_pgn_moves(pgn)
        played: list[str] = []

        for i, san in enumerate(half_moves):
            if i >= max_half_moves:
                break

            # Build context PGN for the current position
            context_pgn = moves_to_pgn(played)
            context_key = board.fen()  # deduplicate by position, not by path
            if context_key in seen_contexts:
                # Still need to push the move before continuing
                try:
                    board.push_san(san)
                    played.append(san)
                except (chess.InvalidMoveError, chess.AmbiguousMoveError, chess.IllegalMoveError):
                    break
                continue
            seen_contexts.add(context_key)

            # Need at least 2 legal moves to form a pair
            if board.is_game_over() or len(list(board.legal_moves)) < 2:
                try:
                    board.push_san(san)
                    played.append(san)
                except (chess.InvalidMoveError, chess.AmbiguousMoveError, chess.IllegalMoveError):
                    break
                continue

            # Analyse with Stockfish
            try:
                infos = engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=min(multipv, len(list(board.legal_moves))),
                )
            except Exception:
                try:
                    board.push_san(san)
                    played.append(san)
                except (chess.InvalidMoveError, chess.AmbiguousMoveError, chess.IllegalMoveError):
                    break
                continue

            if not infos:
                try:
                    board.push_san(san)
                    played.append(san)
                except (chess.InvalidMoveError, chess.AmbiguousMoveError, chess.IllegalMoveError):
                    break
                continue

            # Score of the best move (in cp, from side-to-move perspective)
            best_cp = centipawns(infos[0]["score"])
            if best_cp is None:
                try:
                    board.push_san(san)
                    played.append(san)
                except (chess.InvalidMoveError, chess.AmbiguousMoveError, chess.IllegalMoveError):
                    break
                continue

            # Categorise each candidate move
            good_moves: list[tuple[str, int]] = []   # (san, cp)
            bad_moves: list[tuple[str, int]] = []

            for info in infos:
                if "pv" not in info or not info["pv"]:
                    continue
                move = info["pv"][0]
                move_san = board.san(move)
                move_cp = centipawns(info["score"])
                if move_cp is None:
                    continue
                loss = best_cp - move_cp  # >= 0; higher = worse
                if loss <= chosen_cp_loss:
                    good_moves.append((move_san, move_cp))
                elif loss >= rejected_cp_loss:
                    bad_moves.append((move_san, move_cp))

            # Build pairs (cross-product of good × bad)
            new_pairs = []
            for chosen_san, chosen_cp in good_moves:
                for rejected_san, rejected_cp in bad_moves:
                    new_pairs.append(Pair(
                        context=context_pgn,
                        chosen=chosen_san,
                        rejected=rejected_san,
                        cp_chosen=chosen_cp,
                        cp_rejected=rejected_cp,
                    ))

            # Limit pairs per position to avoid over-weighting the same position
            if new_pairs:
                random.shuffle(new_pairs)
                pairs.extend(new_pairs[:max_pairs_per_position])

            # Advance the board along the opening line
            try:
                board.push_san(san)
                played.append(san)
            except (chess.InvalidMoveError, chess.AmbiguousMoveError, chess.IllegalMoveError):
                break

    engine.quit()
    print(f"  Generated {len(pairs):,} preference pairs from {len(seen_contexts):,} unique positions")
    return pairs


def load_hf_games(hf_dataset: str, n: int) -> list[str]:
    """Stream N PGN strings from the HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets library not found — run: uv add datasets")
    print(f"Loading {n:,} HF games for SFT anchoring …")
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


def save_pairs(pairs: list[Pair], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(p) for p in pairs]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Pairs saved → {path}")


def load_pairs(path: Path) -> list[Pair]:
    with open(path) as f:
        data = json.load(f)
    pairs = [Pair(**d) for d in data]
    print(f"  Loaded {len(pairs):,} pairs from {path}")
    return pairs


# ── Dataset ───────────────────────────────────────────────────────────────────


class DPODataset(Dataset):
    """Dataset of tokenised preference pairs for DPO training.

    Each item encodes:
      chosen_ids   — context tokens + chosen move tokens
      chosen_move_len  — number of tokens that belong to the chosen move
      rejected_ids     — context tokens + rejected move tokens
      rejected_move_len — number of tokens that belong to the rejected move
    """

    def __init__(
        self,
        pairs: list[Pair],
        tokenizer: Tokenizer,
        block_size: int,
    ):
        self.items: list[dict] = []
        skipped = 0

        for pair in pairs:
            # Build the context PGN text
            ctx_text = f"{G_START}{pair.context}" if pair.context else G_START

            # Tokenise context alone to know where it ends
            ctx_ids = tokenizer.encode(ctx_text).ids

            # Tokenise context + move together so the BPE boundary at the junction
            # is handled correctly (a space before the move is part of PGN)
            sep = " " if pair.context else ""
            chosen_text = f"{ctx_text}{sep}{pair.chosen}"
            rejected_text = f"{ctx_text}{sep}{pair.rejected}"

            chosen_full = tokenizer.encode(chosen_text).ids
            rejected_full = tokenizer.encode(rejected_text).ids

            chosen_move_ids = chosen_full[len(ctx_ids):]
            rejected_move_ids = rejected_full[len(ctx_ids):]

            # Skip pairs where the move produced no new tokens (shouldn't happen)
            if not chosen_move_ids or not rejected_move_ids:
                skipped += 1
                continue

            # Truncate from the left if the sequence exceeds block_size
            def truncate(ids, move_len):
                if len(ids) > block_size:
                    ids = ids[-block_size:]
                # Ensure move tokens are still fully present after truncation
                if len(ids) < move_len:
                    return None, move_len
                return ids, min(move_len, len(ids))

            chosen_ids, c_mlen = truncate(chosen_full, len(chosen_move_ids))
            rejected_ids, r_mlen = truncate(rejected_full, len(rejected_move_ids))

            if chosen_ids is None or rejected_ids is None:
                skipped += 1
                continue

            self.items.append({
                "chosen_ids": chosen_ids,
                "chosen_move_len": c_mlen,
                "rejected_ids": rejected_ids,
                "rejected_move_len": r_mlen,
            })

        if skipped:
            print(f"  DPODataset: skipped {skipped} malformed pairs")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_pairs(batch: list[dict]):
    """Pad sequences to the longest in the batch and return tensors."""
    def pad(seqs: list[list[int]]):
        max_len = max(len(s) for s in seqs)
        padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
        lens = []
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            lens.append(len(s))
        return padded, lens

    chosen_padded, chosen_lens = pad([item["chosen_ids"] for item in batch])
    rejected_padded, rejected_lens = pad([item["rejected_ids"] for item in batch])
    chosen_move_lens = [item["chosen_move_len"] for item in batch]
    rejected_move_lens = [item["rejected_move_len"] for item in batch]

    return (
        chosen_padded, chosen_lens, chosen_move_lens,
        rejected_padded, rejected_lens, rejected_move_lens,
    )


# ── Log-probability extraction ────────────────────────────────────────────────


def move_log_probs(
    logits: torch.Tensor,        # [B, T, V] — model output
    ids: torch.Tensor,           # [B, T]    — input token ids (padded)
    seq_lens: list[int],         # actual (unpadded) sequence lengths
    move_lens: list[int],        # number of move tokens at the end of each seq
) -> torch.Tensor:               # [B] — log P(move | context) per item
    """Extract the sum of move-token log probs from a batch of forward passes.

    logits[b, t, :] is the predictive distribution over position t+1.
    Move tokens for item b occupy positions [L-m, ..., L-1] in the (unpadded)
    sequence, where L = seq_lens[b] and m = move_lens[b].  The log prob of
    move token i is taken from logits at position (L - m - 1 + i).
    """
    log_p = F.log_softmax(logits, dim=-1)  # [B, T, V]
    results = []
    for b in range(logits.shape[0]):
        L = seq_lens[b]
        m = move_lens[b]
        move_start = L - m

        # Positions in logits that predict the move tokens
        pred_pos = torch.arange(
            move_start - 1, move_start - 1 + m, device=logits.device
        )  # [m]
        # Token IDs at the move positions
        tok_ids = ids[b, move_start : move_start + m]  # [m]

        # Gather log probs at the move positions
        gathered = log_p[b, pred_pos, :].gather(1, tok_ids.unsqueeze(1)).squeeze(1)
        results.append(gathered.sum())

    return torch.stack(results)  # [B]


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class DPOConfig:
    # Source
    checkpoint: str = str(SFT_DIR / "ckpt_005000.pt")
    output_dir: str = str(DPO_OUTPUT_DIR)
    pairs_cache: str = str(DEFAULT_PAIRS_CACHE)
    # Stockfish
    stockfish: Optional[str] = None  # None = auto-detect
    engine_depth: int = 15
    engine_multipv: int = 5
    max_half_moves: int = 30          # positions to analyse per opening
    chosen_cp_loss: int = 30          # move is "good" if within this many cp of best
    rejected_cp_loss: int = 50        # move is "bad" if this many cp worse than best
    max_pairs_per_position: int = 3
    # DPO
    beta: float = 0.1                 # KL penalty coefficient; start conservative
    # SFT regularisation — prevents catastrophic forgetting of PGN generation quality
    hf_dataset: str = HF_DATASET
    hf_mix_games: int = 5_000         # HF games mixed in as SFT anchor signal
    sft_weight: float = 0.3           # weight of SFT loss relative to DPO loss
    turn_number_weight: float = 0.15  # de-emphasise move-number tokens (same as SFT)
    # Optimisation
    batch_size: int = 16
    learning_rate: float = 2e-5      # lower than original; DPO is sensitive to LR
    min_lr: float = 2e-6
    max_iters: int = 1_000            # converges fast; stop before over-optimising
    warmup_iters: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    # Eval / logging
    eval_interval: int = 100
    log_interval: int = 20
    save_interval: int = 250
    val_fraction: float = 0.1
    seed: int = 42


# ── Training ──────────────────────────────────────────────────────────────────


def train_dpo(cfg: DPOConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = select_device()
    print(f"Device: {device}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # ── Load policy model ─────────────────────────────────────────────────────
    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_cfg"])

    policy = ChessGPT(model_cfg).to(device)
    policy.load_state_dict(ckpt["model_state"])
    print(f"  Policy model: {policy.num_params:,} parameters")

    # ── Frozen reference model ─────────────────────────────────────────────────
    reference = copy.deepcopy(policy)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    print("  Reference model: frozen copy of policy initialisation")

    # ── Preference pairs ──────────────────────────────────────────────────────
    pairs_cache = Path(cfg.pairs_cache)
    if pairs_cache.exists():
        print(f"\nLoading cached pairs from {pairs_cache} …")
        pairs = load_pairs(pairs_cache)
    else:
        sf_path = find_stockfish(cfg.stockfish)
        if sf_path is None:
            sys.exit(
                "Stockfish not found. Install it (brew install stockfish) or pass --stockfish /path/to/binary."
            )
        print(f"\nStockfish: {sf_path}")
        pairs = build_dpo_pairs(
            CHESS_OPENINGS,
            engine_path=sf_path,
            depth=cfg.engine_depth,
            multipv=cfg.engine_multipv,
            max_half_moves=cfg.max_half_moves,
            chosen_cp_loss=cfg.chosen_cp_loss,
            rejected_cp_loss=cfg.rejected_cp_loss,
            max_pairs_per_position=cfg.max_pairs_per_position,
        )
        save_pairs(pairs, pairs_cache)

    if len(pairs) < 10:
        sys.exit(f"Too few pairs ({len(pairs)}) — check Stockfish settings and opening list.")

    # ── Split train / val ─────────────────────────────────────────────────────
    random.shuffle(pairs)
    n_val = max(10, int(len(pairs) * cfg.val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"  Train pairs: {len(train_pairs):,}   Val pairs: {len(val_pairs):,}")

    ds_train = DPODataset(train_pairs, tokenizer, model_cfg.block_size)
    ds_val = DPODataset(val_pairs, tokenizer, model_cfg.block_size)

    if len(ds_train) == 0:
        sys.exit("Training dataset is empty after tokenisation — check pairs.")

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pairs,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pairs,
    )

    # ── SFT anchor dataset (prevents catastrophic forgetting) ─────────────────
    # Mix cross-entropy loss on HF games into every DPO step, the same way
    # finetune.py does.  Without this the model forgets PGN structure rapidly.
    turn_ids = get_turn_number_ids(tokenizer)
    hf_games = load_hf_games(cfg.hf_dataset, cfg.hf_mix_games)
    ds_sft = TokenStream(hf_games, tokenizer, model_cfg.block_size, turn_ids, cfg.turn_number_weight)
    dl_sft = DataLoader(ds_sft, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    sft_iter = iter(dl_sft)
    print(f"  SFT anchor: {len(hf_games):,} HF games  ({len(ds_sft):,} windows)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    decay_params = [p for n, p in policy.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in policy.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    policy.train()
    step = 0
    best_val_acc = 0.0
    train_iter = iter(dl_train)
    t0 = time.time()

    print(f"\n── DPO Training ({cfg.max_iters} steps, β={cfg.beta}, sft_weight={cfg.sft_weight}) ────────────────")
    print(f"{'step':>7}  {'dpo_loss':>8}  {'sft_loss':>8}  {'reward_acc':>10}  {'val_acc':>8}  {'lr':>10}  {'ms/step':>8}")

    while step < cfg.max_iters:
        lr = get_lr(step, cfg.warmup_iters, cfg.max_iters, cfg.learning_rate, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train)
            batch = next(train_iter)

        (
            chosen_ids, chosen_lens, chosen_move_lens,
            rejected_ids, rejected_lens, rejected_move_lens,
        ) = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]

        # ── Policy log probs (need grad) ──────────────────────────────────────
        chosen_logits, _ = policy(chosen_ids)
        policy_chosen_lp = move_log_probs(chosen_logits, chosen_ids, chosen_lens, chosen_move_lens)

        rejected_logits, _ = policy(rejected_ids)
        policy_rejected_lp = move_log_probs(rejected_logits, rejected_ids, rejected_lens, rejected_move_lens)

        # ── Reference log probs (no grad) ─────────────────────────────────────
        with torch.no_grad():
            ref_chosen_logits, _ = reference(chosen_ids)
            ref_chosen_lp = move_log_probs(ref_chosen_logits, chosen_ids, chosen_lens, chosen_move_lens)

            ref_rejected_logits, _ = reference(rejected_ids)
            ref_rejected_lp = move_log_probs(ref_rejected_logits, rejected_ids, rejected_lens, rejected_move_lens)

        # ── DPO loss ──────────────────────────────────────────────────────────
        pi_logratios = policy_chosen_lp - policy_rejected_lp
        ref_logratios = ref_chosen_lp - ref_rejected_lp
        dpo_loss = -F.logsigmoid(cfg.beta * (pi_logratios - ref_logratios)).mean()

        # ── SFT anchor loss (prevents forgetting PGN generation quality) ──────
        try:
            sx, sy, sw = next(sft_iter)
        except StopIteration:
            sft_iter = iter(dl_sft)
            sx, sy, sw = next(sft_iter)
        sx, sy, sw = sx.to(device), sy.to(device), sw.to(device)
        _, sft_loss = policy(sx, targets=sy, loss_weights=sw)

        loss = dpo_loss + cfg.sft_weight * sft_loss

        # ── Reward accuracy (fraction of pairs correctly ordered) ─────────────
        with torch.no_grad():
            reward_chosen = cfg.beta * (policy_chosen_lp - ref_chosen_lp)
            reward_rejected = cfg.beta * (policy_rejected_lp - ref_rejected_lp)
            reward_acc = (reward_chosen > reward_rejected).float().mean().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()
        step += 1

        if step % cfg.log_interval == 0:
            elapsed = (time.time() - t0) * 1000 / cfg.log_interval
            print(
                f"{step:>7}  {dpo_loss.item():>8.4f}  {sft_loss.item():>8.4f}  {reward_acc:>10.3f}  {'—':>8}  {lr:>10.2e}  {elapsed:>8.1f}",
                flush=True,
            )
            t0 = time.time()

        if step % cfg.eval_interval == 0:
            val_acc, val_loss = _evaluate_dpo(policy, reference, dl_val, device, cfg.beta)
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            print(
                f"{step:>7}  {val_loss:>8.4f}  {'—':>10}  {val_acc:>8.3f}  {lr:>10.2e}  {'BEST' if is_best else '':>8}",
                flush=True,
            )
            policy.train()

        if step % cfg.save_interval == 0:
            _save(policy, optimizer, model_cfg, cfg, step, best_val_acc, output_dir)
            _sample_generation(policy, tokenizer, device, model_cfg.block_size)
            policy.train()

    print("\nDPO training complete.")
    print(f"Best val reward accuracy: {best_val_acc:.3f}")
    _save(policy, optimizer, model_cfg, cfg, step, best_val_acc, output_dir)
    _sample_generation(policy, tokenizer, device, model_cfg.block_size)


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def _evaluate_dpo(
    policy: ChessGPT,
    reference: ChessGPT,
    dl: DataLoader,
    device: torch.device,
    beta: float,
    max_batches: int = 30,
) -> tuple[float, float]:
    """Return (reward_accuracy, mean_dpo_loss) on the validation set."""
    policy.eval()
    accs, losses = [], []

    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        (
            chosen_ids, chosen_lens, chosen_move_lens,
            rejected_ids, rejected_lens, rejected_move_lens,
        ) = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]

        chosen_logits, _ = policy(chosen_ids)
        policy_chosen_lp = move_log_probs(chosen_logits, chosen_ids, chosen_lens, chosen_move_lens)

        rejected_logits, _ = policy(rejected_ids)
        policy_rejected_lp = move_log_probs(rejected_logits, rejected_ids, rejected_lens, rejected_move_lens)

        ref_chosen_logits, _ = reference(chosen_ids)
        ref_chosen_lp = move_log_probs(ref_chosen_logits, chosen_ids, chosen_lens, chosen_move_lens)

        ref_rejected_logits, _ = reference(rejected_ids)
        ref_rejected_lp = move_log_probs(ref_rejected_logits, rejected_ids, rejected_lens, rejected_move_lens)

        pi_logratios = policy_chosen_lp - policy_rejected_lp
        ref_logratios = ref_chosen_lp - ref_rejected_lp
        loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

        reward_chosen = beta * (policy_chosen_lp - ref_chosen_lp)
        reward_rejected = beta * (policy_rejected_lp - ref_rejected_lp)
        acc = (reward_chosen > reward_rejected).float().mean().item()

        accs.append(acc)
        losses.append(loss.item())

    return (
        sum(accs) / len(accs) if accs else 0.0,
        sum(losses) / len(losses) if losses else float("inf"),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _save(model, optimizer, model_cfg, cfg, step, best_val_acc, output_dir):
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "model_cfg": asdict(model_cfg),
    }
    path = output_dir / f"ckpt_{step:06d}.pt"
    torch.save(payload, path)
    torch.save(payload, output_dir / "ckpt_latest.pt")
    print(f"  ✓ Checkpoint saved → {path.name}")


@torch.no_grad()
def _sample_generation(model, tokenizer, device, block_size):
    """Generate a few sample continuations to sanity-check the model."""
    import torch.nn.functional as F

    model.eval()
    g_end_id = tokenizer.encode("[g_end]").ids[0]
    prompts = [
        f"{G_START}1.e4 e5 2.Nf3",
        f"{G_START}1.d4 Nf6 2.c4",
        f"{G_START}1.e4 c5 2.Nf3",
    ]
    print("\n── Sample generation ──────────────────────────────────────────────")
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(40):
            cond = idx[:, -block_size:]
            logits, _ = model(cond)
            logits = logits[:, -1, :] / 0.7
            v, _ = torch.topk(logits, min(30, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            if next_tok.item() == g_end_id:
                break
        text = tokenizer.decode(idx[0].tolist()).replace(G_START, "").replace("[g_end]", "").strip()
        print(f"  {text}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="KN1GHT DPO training — move quality alignment")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(SFT_DIR / "ckpt_005000.pt"),
        help="SFT checkpoint to start DPO from (also used as the reference model)",
    )
    p.add_argument("--output-dir", type=str, default=str(DPO_OUTPUT_DIR))
    p.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to Stockfish binary (auto-detected if omitted)",
    )
    p.add_argument(
        "--pairs-cache",
        type=str,
        default=str(DEFAULT_PAIRS_CACHE),
        help="JSON file to cache/load Stockfish preference pairs",
    )
    p.add_argument("--depth", type=int, default=15, help="Stockfish search depth")
    p.add_argument("--multipv", type=int, default=5, help="Stockfish multipv candidates")
    p.add_argument("--beta", type=float, default=0.1, help="DPO β coefficient")
    p.add_argument("--sft-weight", type=float, default=0.3, help="Weight of SFT anchor loss relative to DPO loss")
    p.add_argument("--iters", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument(
        "--chosen-cp-loss",
        type=int,
        default=30,
        help="Max centipawn loss for a move to be 'chosen'",
    )
    p.add_argument(
        "--rejected-cp-loss",
        type=int,
        default=50,
        help="Min centipawn loss for a move to be 'rejected'",
    )
    p.add_argument(
        "--rebuild-pairs",
        action="store_true",
        help="Ignore cached pairs and re-run Stockfish analysis",
    )
    return p.parse_args()


def main():
    args = parse_args()

    pairs_cache = Path(args.pairs_cache)
    if args.rebuild_pairs and pairs_cache.exists():
        pairs_cache.unlink()
        print(f"Removed existing pairs cache: {pairs_cache}")

    cfg = DPOConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        pairs_cache=args.pairs_cache,
        stockfish=args.stockfish,
        engine_depth=args.depth,
        engine_multipv=args.multipv,
        beta=args.beta,
        sft_weight=args.sft_weight,
        max_iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        chosen_cp_loss=args.chosen_cp_loss,
        rejected_cp_loss=args.rejected_cp_loss,
    )
    train_dpo(cfg)


if __name__ == "__main__":
    main()
