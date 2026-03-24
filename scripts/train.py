#!/usr/bin/env python3
"""
kn1ght Chess Language Model — Training Script

Architecture: GPT-style decoder-only transformer (kn1ght-small)
  - 4 layers, 4 heads, 256 embedding dim, 256 token context
  - ~4M parameters — fast to train, good for opening recognition

Key design decisions:
  1. Weighted cross-entropy loss that de-emphasises turn-number tokens
     (they're structurally predictable, so we reduce their gradient signal)
  2. Chess-openings oversampling — common ECO openings are repeated N times
     so the model internalises them before general game patterns
  3. Causal next-token prediction over a sliding window of concatenated games
  4. MPS / CUDA / CPU auto-detection

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --max-games 5000 --iters 5000   # quick smoke-test
    uv run python scripts/train.py --generate "1.e4 e5 2.Nf3"      # inference only
"""

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = ROOT / "src" / "tokenizer" / "kn1ght-tokenizer.json"
OUTPUT_DIR = ROOT / ".data" / "models" / "kn1ght-small"

# The special-tokens variant already has [g_start] / [g_end] wrapped around each game
HF_DATASET = "InterwebAlchemy/pgn-dataset-including-special-tokens"

# ─── Common Chess Openings ────────────────────────────────────────────────────
# ECO-coded openings in plain PGN (no metadata).  These will be oversampled
# during training so the model develops strong prior knowledge of opening lines.

CHESS_OPENINGS = [
    # ── Ruy Lopez ─────────────────────────────────────────────────────────────
    (
        "Ruy Lopez (Main Line)",
        "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O",
    ),
    (
        "Ruy Lopez (Berlin)",
        "1.e4 e5 2.Nf3 Nc6 3.Bb5 Nf6 4.O-O Nxe4 5.d4 Nd6 6.Bxc6 dxc6 7.dxe5 Nf5",
    ),
    (
        "Ruy Lopez (Marshall Attack)",
        "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O 9.h3 Na5 10.Bc2 c5 11.d4 Nd7",
    ),
    (
        "Ruy Lopez (Breyer)",
        "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O 9.h3 Nb8 10.d4 Nbd7",
    ),
    (
        "Ruy Lopez (Chigorin)",
        "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O 9.h3 Na5 10.Bc2 c5 11.d4 Qc7",
    ),
    (
        "Ruy Lopez (Archangel)",
        "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Bc5 6.c3 b5 7.Bb3 d6 8.d4 Bb6",
    ),
    (
        "Ruy Lopez (Exchange)",
        "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Bxc6 dxc6 5.O-O f6 6.d4 exd4 7.Nxd4 c5",
    ),
    (
        "Ruy Lopez (Anti-Marshall)",
        "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O 9.a4",
    ),
    # ── Italian / Open Games ──────────────────────────────────────────────────
    (
        "Italian (Giuoco Piano)",
        "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4 exd4 6.cxd4 Bb4+ 7.Nc3",
    ),
    (
        "Italian (Giuoco Pianissimo)",
        "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.d3 Nf6 5.Nc3 d6 6.O-O O-O",
    ),
    (
        "Italian (Modern)",
        "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.O-O Nf6 5.d3 O-O 6.Nc3 d6 7.h3",
    ),
    (
        "Italian (Two Knights)",
        "1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5 d5 5.exd5 Na5 6.Bb5+ c6 7.dxc6 bxc6",
    ),
    ("Evans Gambit", "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.b4 Bxb4 5.c3 Ba5 6.d4 exd4 7.O-O"),
    (
        "Max Lange Attack",
        "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.O-O Nf6 5.d4 exd4 6.e5 d5 7.exf6 dxc4 8.Re1+",
    ),
    (
        "Scotch Game",
        "1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Nxd4 Nf6 5.Nxc6 bxc6 6.e5 Qe7 7.Qe2 Nd5",
    ),
    ("Scotch Gambit", "1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Bc4 Bc5 5.c3 Nf6 6.cxd4 Bb4+"),
    ("Four Knights", "1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6 4.Bb5 Nd4 5.Ba4 Bc5 6.Nxe5 O-O"),
    (
        "Petroff Defense",
        "1.e4 e5 2.Nf3 Nf6 3.Nxe5 d6 4.Nf3 Nxe4 5.d4 d5 6.Bd3 Be7 7.O-O Nc6",
    ),
    (
        "Philidor Defense",
        "1.e4 e5 2.Nf3 d6 3.d4 Nf6 4.Nc3 Nbd7 5.Bc4 Be7 6.O-O O-O",
    ),
    (
        "King's Gambit Accepted",
        "1.e4 e5 2.f4 exf4 3.Nf3 g5 4.h4 g4 5.Ne5 Nf6 6.d4 d6 7.Nd3",
    ),
    (
        "King's Gambit Declined",
        "1.e4 e5 2.f4 Bc5 3.Nf3 d6 4.c3 Nf6 5.d4 exd4 6.cxd4 Bb6",
    ),
    (
        "King's Gambit (Falkbeer Counter)",
        "1.e4 e5 2.f4 d5 3.exd5 e4 4.d3 Nf6 5.dxe4 Nxe4 6.Nf3 Bc5",
    ),
    ("Vienna Game", "1.e4 e5 2.Nc3 Nf6 3.f4 d5 4.fxe5 Nxe4 5.Nf3 Be7 6.d4 Nxc3"),
    ("Danish Gambit", "1.e4 e5 2.d4 exd4 3.c3 dxc3 4.Bc4 cxb2 5.Bxb2 d5 6.Bxd5 Nf6"),
    # ── Sicilian ─────────────────────────────────────────────────────────────
    (
        "Sicilian Najdorf",
        "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 6.Bg5 e6 7.f4",
    ),
    (
        "Sicilian Najdorf (English Attack)",
        "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 6.Be3 e5 7.Nb3 Be6 8.f3",
    ),
    (
        "Sicilian Dragon",
        "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 g6 6.Be3 Bg7 7.f3 O-O",
    ),
    (
        "Sicilian Dragon (Yugoslav Attack)",
        "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 g6 6.Be3 Bg7 7.f3 O-O 8.Qd2 Nc6 9.O-O-O",
    ),
    (
        "Sicilian Sveshnikov",
        "1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 e5 6.Ndb5 d6 7.Bg5 a6 8.Na3 b5",
    ),
    (
        "Sicilian Scheveningen",
        "1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 d6 6.Be2 Be7 7.O-O O-O",
    ),
    (
        "Sicilian Keres Attack",
        "1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 d6 6.g4 Nc6 7.g5 Nd7",
    ),
    (
        "Sicilian Classical",
        "1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 d6 6.Bg5 e6 7.Qd2",
    ),
    (
        "Sicilian Richter-Rauzer",
        "1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 d6 6.Bg5 e6 7.Qd2 a6 8.O-O-O",
    ),
    (
        "Sicilian Taimanov",
        "1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 e6 5.Nc3 Qc7 6.Be2 a6 7.O-O Nf6",
    ),
    ("Sicilian Kan", "1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4 a6 5.Nc3 Qc7 6.Be2 Nf6 7.O-O"),
    (
        "Sicilian Accelerated Dragon",
        "1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 g6 5.Nc3 Bg7 6.Be3 Nf6",
    ),
    (
        "Alapin Sicilian",
        "1.e4 c5 2.c3 Nf6 3.e5 Nd5 4.d4 cxd4 5.Nf3 Nc6 6.Bc4 Nb6 7.Bb3",
    ),
    (
        "Smith-Morra Gambit",
        "1.e4 c5 2.d4 cxd4 3.c3 dxc3 4.Nxc3 Nc6 5.Nf3 d6 6.Bc4 e6 7.O-O",
    ),
    (
        "Grand Prix Attack",
        "1.e4 c5 2.Nc3 Nc6 3.f4 g6 4.Nf3 Bg7 5.Bc4 e6 6.f5 exf5 7.exf5",
    ),
    # ── French ───────────────────────────────────────────────────────────────
    ("French Winawer", "1.e4 e6 2.d4 d5 3.Nc3 Bb4 4.e5 c5 5.a3 Bxc3+ 6.bxc3 Ne7 7.Qg4"),
    (
        "French Classical",
        "1.e4 e6 2.d4 d5 3.Nc3 Nf6 4.Bg5 Be7 5.e5 Nfd7 6.Bxe7 Qxe7 7.f4",
    ),
    ("French Tarrasch", "1.e4 e6 2.d4 d5 3.Nd2 Nf6 4.e5 Nfd7 5.Bd3 c5 6.c3 Nc6 7.Ne2"),
    ("French Advance", "1.e4 e6 2.d4 d5 3.e5 c5 4.c3 Nc6 5.Nf3 Qb6 6.Bd3 cxd4 7.cxd4"),
    (
        "French Exchange",
        "1.e4 e6 2.d4 d5 3.exd5 exd5 4.Nf3 Nf6 5.Bd3 Bd6 6.O-O O-O 7.Bg5",
    ),
    (
        "French Rubinstein",
        "1.e4 e6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Nd7 5.Nf3 Ngf6 6.Nxf6+ Nxf6 7.Bd3 c5",
    ),
    (
        "French MacCutcheon",
        "1.e4 e6 2.d4 d5 3.Nc3 Nf6 4.Bg5 Bb4 5.e5 h6 6.Bd2 Bxc3 7.bxc3 Ne4",
    ),
    # ── Caro-Kann ────────────────────────────────────────────────────────────
    (
        "Caro-Kann Classical",
        "1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5 5.Ng3 Bg6 6.h4 h6 7.Nf3 Nd7",
    ),
    ("Caro-Kann Advance", "1.e4 c6 2.d4 d5 3.e5 Bf5 4.Nf3 e6 5.Be2 c5 6.O-O Nc6 7.c3"),
    (
        "Caro-Kann Panov",
        "1.e4 c6 2.d4 d5 3.exd5 cxd5 4.c4 Nf6 5.Nc3 e6 6.Nf3 Be7 7.cxd5",
    ),
    (
        "Caro-Kann Exchange",
        "1.e4 c6 2.d4 d5 3.exd5 cxd5 4.Bd3 Nc6 5.c3 Nf6 6.Bf4 Bg4 7.Qb3",
    ),
    (
        "Caro-Kann Two Knights",
        "1.e4 c6 2.Nc3 d5 3.Nf3 Bg4 4.h3 Bxf3 5.Qxf3 e6 6.d4 Nf6 7.Bd3",
    ),
    # ── Scandinavian ─────────────────────────────────────────────────────────
    (
        "Scandinavian Defense",
        "1.e4 d5 2.exd5 Qxd5 3.Nc3 Qa5 4.d4 Nf6 5.Nf3 Bf5 6.Bc4 e6",
    ),
    (
        "Scandinavian (Modern)",
        "1.e4 d5 2.exd5 Nf6 3.d4 Nxd5 4.Nf3 g6 5.c4 Nb6 6.Nc3 Bg7",
    ),
    # ── Pirc / Modern / Alekhine / Nimzowitsch ────────────────────────────────
    ("Pirc Defense", "1.e4 d6 2.d4 Nf6 3.Nc3 g6 4.Nf3 Bg7 5.Be2 O-O 6.O-O c6 7.a4"),
    ("Modern Defense", "1.e4 g6 2.d4 Bg7 3.Nc3 d6 4.Be3 a6 5.Qd2 b5 6.f3 Nd7"),
    ("Alekhine Defense", "1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.Nf3 Bg4 5.Be2 e6 6.O-O Be7 7.h3"),
    (
        "Nimzowitsch Defense",
        "1.e4 Nc6 2.d4 d5 3.Nc3 dxe4 4.d5 Ne5 5.Qd4 Ng6 6.Bc4",
    ),
    # ── Queen's Gambit ────────────────────────────────────────────────────────
    (
        "Queen's Gambit Accepted",
        "1.d4 d5 2.c4 dxc4 3.Nf3 Nf6 4.e3 e6 5.Bxc4 c5 6.O-O a6 7.Bb3",
    ),
    (
        "Queen's Gambit Declined",
        "1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3 Nbd7 7.Rc1",
    ),
    (
        "QGD Lasker",
        "1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3 Ne4 7.Bxe7 Qxe7 8.cxd5 Nxc3 9.bxc3 exd5",
    ),
    (
        "QGD Tartakower",
        "1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3 h6 7.Bh4 b6 8.cxd5 Nxd5 9.Bxe7 Qxe7",
    ),
    # ── Slav / Semi-Slav ─────────────────────────────────────────────────────
    ("Slav Defense", "1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 dxc4 5.a4 Bf5 6.e3 e6 7.Bxc4"),
    (
        "Semi-Slav (Meran)",
        "1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 e6 5.e3 Nbd7 6.Bd3 dxc4 7.Bxc4",
    ),
    (
        "Semi-Slav (Botvinnik)",
        "1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 e6 5.Bg5 dxc4 6.e4 b5 7.e5 h6 8.Bh4 g5",
    ),
    # ── Nimzo-Indian ─────────────────────────────────────────────────────────
    (
        "Nimzo-Indian (Rubinstein)",
        "1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.e3 O-O 5.Bd3 d5 6.Nf3 c5 7.O-O",
    ),
    (
        "Nimzo-Indian (4.Qc2)",
        "1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.Qc2 O-O 5.a3 Bxc3+ 6.Qxc3 b6 7.Bg5",
    ),
    (
        "Nimzo-Indian (Samisch)",
        "1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.a3 Bxc3+ 5.bxc3 O-O 6.f3 d5 7.cxd5 exd5",
    ),
    # ── Queen's Indian ────────────────────────────────────────────────────────
    (
        "Queen's Indian",
        "1.d4 Nf6 2.c4 e6 3.Nf3 b6 4.g3 Bb7 5.Bg2 Be7 6.O-O O-O 7.Nc3",
    ),
    (
        "Queen's Indian (Petrosian)",
        "1.d4 Nf6 2.c4 e6 3.Nf3 b6 4.a3 Bb7 5.Nc3 d5 6.cxd5 Nxd5 7.e3 Be7",
    ),
    # ── King's Indian ─────────────────────────────────────────────────────────
    (
        "King's Indian Defense",
        "1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5 7.O-O",
    ),
    (
        "King's Indian (Classical)",
        "1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5 7.O-O Nc6 8.d5 Ne7 9.Ne1 Nd7",
    ),
    (
        "King's Indian (Samisch)",
        "1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.f3 O-O 6.Be3 e5 7.d5",
    ),
    (
        "King's Indian (Averbakh)",
        "1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Be2 O-O 6.Bg5 c5 7.d5 e6",
    ),
    (
        "King's Indian (Four Pawns)",
        "1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.f4 O-O 6.Nf3 c5 7.d5 e6",
    ),
    # ── Grunfeld ─────────────────────────────────────────────────────────────
    (
        "Grunfeld Defense",
        "1.d4 Nf6 2.c4 g6 3.Nc3 d5 4.cxd5 Nxd5 5.e4 Nxc3 6.bxc3 Bg7 7.Bc4",
    ),
    (
        "Grunfeld Exchange",
        "1.d4 Nf6 2.c4 g6 3.Nc3 d5 4.cxd5 Nxd5 5.e4 Nxc3 6.bxc3 Bg7 7.Nf3",
    ),
    (
        "Grunfeld (Russian System)",
        "1.d4 Nf6 2.c4 g6 3.Nc3 d5 4.Nf3 Bg7 5.Qb3 dxc4 6.Qxc4 O-O 7.e4",
    ),
    # ── Benoni / Benko ────────────────────────────────────────────────────────
    (
        "Benoni Defense",
        "1.d4 Nf6 2.c4 c5 3.d5 e6 4.Nc3 exd5 5.cxd5 d6 6.e4 g6 7.Nf3 Bg7",
    ),
    (
        "Benko Gambit",
        "1.d4 Nf6 2.c4 c5 3.d5 b5 4.cxb5 a6 5.bxa6 Bxa6 6.Nc3 d6 7.Nf3 g6",
    ),
    (
        "Budapest Gambit",
        "1.d4 Nf6 2.c4 e5 3.dxe5 Ng4 4.Nf3 Nc6 5.Bf4 Bb4+ 6.Nc3 Qe7",
    ),
    # ── Other Indian / d4 systems ─────────────────────────────────────────────
    ("Bogo-Indian", "1.d4 Nf6 2.c4 e6 3.Nf3 Bb4+ 4.Nbd2 b6 5.g3 Bb7 6.Bg2 O-O 7.O-O"),
    (
        "Catalan Opening",
        "1.d4 Nf6 2.c4 e6 3.g3 d5 4.Bg2 Be7 5.Nf3 O-O 6.O-O dxc4 7.Qc2",
    ),
    (
        "Catalan (Closed)",
        "1.d4 Nf6 2.c4 e6 3.g3 d5 4.Bg2 Be7 5.Nf3 O-O 6.O-O c6 7.Qc2 Nbd7 8.Nbd2",
    ),
    ("London System", "1.d4 d5 2.Nf3 Nf6 3.Bf4 e6 4.e3 Bd6 5.Bg3 O-O 6.Nbd2 c5 7.c3"),
    (
        "Trompowsky Attack",
        "1.d4 Nf6 2.Bg5 Ne4 3.Bf4 d5 4.e3 c5 5.Bd3 Nf6 6.Nd2 Nc6 7.c3",
    ),
    (
        "Torre Attack",
        "1.d4 Nf6 2.Nf3 e6 3.Bg5 d5 4.Nbd2 Be7 5.e3 O-O 6.Bd3 Nbd7 7.O-O",
    ),
    (
        "Colle System",
        "1.d4 d5 2.Nf3 Nf6 3.e3 e6 4.Bd3 c5 5.c3 Nc6 6.Nbd2 Bd6 7.O-O O-O",
    ),
    (
        "Stonewall Attack",
        "1.d4 d5 2.e3 Nf6 3.Bd3 c5 4.c3 Nc6 5.f4 Bg4 6.Nf3 e6 7.O-O Bd6",
    ),
    (
        "Blackmar-Diemer Gambit",
        "1.d4 d5 2.e4 dxe4 3.Nc3 Nf6 4.f3 exf3 5.Nxf3 Bf5 6.Bc4 e6 7.O-O",
    ),
    # ── Dutch ─────────────────────────────────────────────────────────────────
    (
        "Dutch Defense (Leningrad)",
        "1.d4 f5 2.g3 Nf6 3.Bg2 g6 4.Nf3 Bg7 5.O-O O-O 6.c4 d6 7.Nc3",
    ),
    (
        "Dutch Stonewall",
        "1.d4 f5 2.Nf3 Nf6 3.g3 e6 4.Bg2 d5 5.c4 c6 6.O-O Bd6 7.b3",
    ),
    # ── Flank openings ───────────────────────────────────────────────────────
    (
        "English Opening",
        "1.c4 e5 2.Nc3 Nf6 3.Nf3 Nc6 4.g3 Bb4 5.Bg2 O-O 6.O-O e4 7.Ng5",
    ),
    (
        "English Symmetrical",
        "1.c4 c5 2.Nc3 Nc6 3.g3 g6 4.Bg2 Bg7 5.Nf3 Nf6 6.O-O O-O 7.d3",
    ),
    (
        "English (Four Knights)",
        "1.c4 e5 2.Nc3 Nf6 3.Nf3 Nc6 4.d4 exd4 5.Nxd4 Bb4 6.Nxc6 bxc6 7.e3 O-O",
    ),
    (
        "English (Reversed Sicilian)",
        "1.c4 e5 2.Nc3 Nf6 3.g3 d5 4.cxd5 Nxd5 5.Bg2 Nb6 6.Nf3 Nc6 7.O-O",
    ),
    ("Reti Opening", "1.Nf3 d5 2.g3 Nf6 3.Bg2 c6 4.O-O Bf5 5.d3 e6 6.Nbd2 Be7 7.b3"),
    (
        "King's Indian Attack",
        "1.Nf3 Nf6 2.g3 g6 3.Bg2 Bg7 4.O-O O-O 5.d3 d6 6.Nbd2 c5 7.e4",
    ),
    (
        "King's Indian Attack (vs French)",
        "1.Nf3 d5 2.g3 e6 3.Bg2 Nf6 4.O-O Be7 5.d3 c5 6.Nbd2 Nc6 7.e4",
    ),
    ("Bird's Opening", "1.f4 d5 2.Nf3 Nf6 3.e3 g6 4.Be2 Bg7 5.O-O O-O 6.d3 c5 7.Qe1"),
    (
        "From's Gambit",
        "1.f4 e5 2.fxe5 d6 3.exd6 Bxd6 4.Nf3 g5 5.g3 g4 6.Nh4",
    ),
    (
        "Larsen's Opening",
        "1.b3 e5 2.Bb2 Nc6 3.e3 d5 4.Bb5 Bd6 5.Nf3 Nge7 6.O-O O-O",
    ),
    # ── Short "seed" lines (reinforce very first moves) ───────────────────────
    ("Open Game", "1.e4 e5"),
    ("Sicilian", "1.e4 c5"),
    ("French", "1.e4 e6"),
    ("Caro-Kann", "1.e4 c6"),
    ("Scandinavian", "1.e4 d5"),
    ("Pirc/Modern", "1.e4 d6"),
    ("Alekhine", "1.e4 Nf6"),
    ("Nimzowitsch", "1.e4 Nc6"),
    ("Queen's Pawn", "1.d4 d5"),
    ("Indian Systems", "1.d4 Nf6"),
    ("Dutch", "1.d4 f5"),
    ("English", "1.c4"),
    ("Reti", "1.Nf3"),
    ("e4 e5 Nf3", "1.e4 e5 2.Nf3"),
    ("e4 e5 Nf3 Nc6", "1.e4 e5 2.Nf3 Nc6"),
    ("e4 e5 Nf3 Nc6 Bc4", "1.e4 e5 2.Nf3 Nc6 3.Bc4"),
    ("e4 e5 Nf3 Nc6 Bb5", "1.e4 e5 2.Nf3 Nc6 3.Bb5"),
    ("d4 d5 c4", "1.d4 d5 2.c4"),
    ("d4 d5 Nf3", "1.d4 d5 2.Nf3"),
    ("d4 Nf6 c4 g6", "1.d4 Nf6 2.c4 g6"),
    ("d4 Nf6 c4 e6 Nc3 Bb4", "1.d4 Nf6 2.c4 e6 3.Nc3 Bb4"),
    ("d4 Nf6 c4 c5", "1.d4 Nf6 2.c4 c5"),
]

# ─── Model ───────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        nh, hd = self.n_head, self.head_dim
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (hd**-0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ChessGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying
        self.tok_emb.weight = self.head.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
    ):
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Sequence {T} > block_size {self.cfg.block_size}"
        )
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            if loss_weights is not None:
                # Per-token weighted cross-entropy
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="none",
                )
                loss = (loss * loss_weights.view(-1)).mean()
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ─── Dataset ─────────────────────────────────────────────────────────────────


class TokenStream(Dataset):
    """Flat token stream built from a list of PGN strings.

    Each game is tokenised and appended to a single long array.  The dataset
    yields (x, y, weights) tuples of length `block_size` where:
      x  — input tokens
      y  — target tokens (x shifted by 1)
      w  — per-token loss weights (turn numbers get a lower weight)
    """

    def __init__(
        self,
        games: list[str],
        tokenizer: Tokenizer,
        block_size: int,
        turn_number_ids: set[int],
        turn_weight: float = 0.15,
        g_start_id: int = 0,
        g_end_id: int = 1,
    ):
        self.block_size = block_size
        self.turn_number_ids = turn_number_ids
        self.turn_weight = turn_weight

        # Tokenise and concatenate all games
        tokens: list[int] = []
        for pgn in games:
            # Wrap in special tokens if not already present
            text = pgn.strip()
            if not text.startswith("[g_start]"):
                text = f"[g_start]{text}[g_end]"
            ids = tokenizer.encode(text).ids
            tokens.extend(ids)

        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        # Build per-token weights
        w = torch.ones(self.block_size, dtype=torch.float)
        for i, tid in enumerate(y.tolist()):
            if tid in self.turn_number_ids:
                w[i] = self.turn_weight
        return x, y, w


# ─── Utilities ────────────────────────────────────────────────────────────────


def get_turn_number_ids(tokenizer: Tokenizer) -> set[int]:
    """Return the set of token IDs that represent move-number prefixes.

    We scan move numbers 1–300 in both ' N.' and 'N.' forms so the set is
    comprehensive without any hardcoding.
    """
    ids: set[int] = set()
    for n in range(1, 301):
        for prefix in (f"{n}.", f" {n}."):
            encoded = tokenizer.encode(prefix).ids
            # Only treat it as a turn-number token if it encodes to a single ID
            if len(encoded) == 1:
                ids.add(encoded[0])
    return ids


def get_lr(
    step: int, warmup: int, max_iters: int, max_lr: float, min_lr: float
) -> float:
    if step < warmup:
        return max_lr * step / warmup
    if step > max_iters:
        return min_lr
    decay = (step - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (max_lr - min_lr)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Training ─────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    # Data
    hf_dataset: str = HF_DATASET
    max_train_games: int = 100_000  # cap games loaded from HuggingFace
    openings_repeat: int = 10  # oversample openings N times
    # Model
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 256
    dropout: float = 0.1
    # Optimisation
    batch_size: int = 64
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    max_iters: int = 20_000
    warmup_iters: int = 500
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    # Chess-specific loss
    turn_number_weight: float = 0.15
    # Evaluation / logging
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 50
    save_interval: int = 2000
    # Output
    output_dir: str = str(OUTPUT_DIR)
    seed: int = 1997
    # Resume
    resume_from: Optional[str] = None      # path to a checkpoint to resume from


def load_pgn_games(cfg: TrainConfig, split: str = "train") -> list[str]:
    """Load PGN strings from the HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets library not found — run: uv add datasets")

    print(f"Loading {cfg.hf_dataset} ({split}) …")
    ds = load_dataset(cfg.hf_dataset, split=split, streaming=True)
    games: list[str] = []
    for row in ds:
        pgn = row.get("pgn") or row.get("PGN") or row.get("text") or ""
        pgn = pgn.strip()
        if pgn:
            games.append(pgn)
        if len(games) >= cfg.max_train_games:
            break
    print(f"  Loaded {len(games):,} games")
    return games


def build_datasets(
    cfg: TrainConfig, tokenizer: Tokenizer, turn_ids: set[int]
) -> tuple[TokenStream, TokenStream]:
    """Build train and validation TokenStream datasets."""
    train_games = load_pgn_games(cfg, split="train")

    # Try loading a dedicated validation split; fall back to carving from train.
    # Carve at most 10% of the available games (cap at 2000) so small runs still
    # have enough training data.
    try:
        val_games = load_pgn_games(cfg, split="validation")
    except Exception:
        n_val = min(2000, max(100, len(train_games) // 10))
        val_games = train_games[-n_val:]
        train_games = train_games[:-n_val]

    # Build opening games (repeated for oversampling)
    opening_games = [pgn for _, pgn in CHESS_OPENINGS] * cfg.openings_repeat
    random.shuffle(opening_games)
    all_train = opening_games + train_games

    print(
        f"  Train tokens from {len(all_train):,} games "
        f"({len(opening_games):,} opening samples + {len(train_games):,} HF games)"
    )

    ds_train = TokenStream(
        all_train, tokenizer, cfg.block_size, turn_ids, cfg.turn_number_weight
    )
    ds_val = TokenStream(
        val_games, tokenizer, cfg.block_size, turn_ids, cfg.turn_number_weight
    )
    return ds_train, ds_val


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = select_device()
    print(f"Device: {device}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    turn_ids = get_turn_number_ids(tokenizer)
    print(
        f"Turn-number token IDs: {len(turn_ids)} distinct tokens (weight={cfg.turn_number_weight})"
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    ds_train, ds_val = build_datasets(cfg, tokenizer, turn_ids)
    print(f"  Train windows: {len(ds_train):,}   Val windows: {len(ds_val):,}")

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )
    model = ChessGPT(model_cfg).to(device)
    print(f"Model parameters: {model.num_params:,}")

    # Save model config alongside checkpoint
    with open(output_dir / "config.json", "w") as f:
        json.dump({"model": asdict(model_cfg), "train": asdict(cfg)}, f, indent=2)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    step = 0
    best_val_loss = float("inf")
    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        if not ckpt_path.exists():
            sys.exit(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        step = ckpt["step"]
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from {ckpt_path.name} (step {step}, best val loss {best_val_loss:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    train_iter = iter(dl_train)
    t0 = time.time()

    print("\n── Training ──────────────────────────────────────────────────────")
    print(
        f"{'step':>7}  {'train_loss':>10}  {'val_loss':>10}  {'lr':>10}  {'ms/step':>8}"
    )

    while step < cfg.max_iters:
        # LR schedule
        lr = get_lr(
            step, cfg.warmup_iters, cfg.max_iters, cfg.learning_rate, cfg.min_lr
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Fetch next batch (cycle the iterator)
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
            print(
                f"{step:>7}  {loss.item():>10.4f}  {'—':>10}  {lr:>10.2e}  {elapsed:>8.1f}",
                flush=True,
            )
            t0 = time.time()

        if step % cfg.eval_interval == 0:
            val_loss = evaluate(model, dl_val, device, cfg.eval_iters)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            print(
                f"{step:>7}  {'—':>10}  {val_loss:>10.4f}  {lr:>10.2e}  {'BEST' if is_best else '':>8}",
                flush=True,
            )
            model.train()

        if step % cfg.save_interval == 0:
            ckpt_path = output_dir / f"ckpt_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "model_cfg": asdict(model_cfg),
                },
                ckpt_path,
            )
            # Also save "latest" pointer
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "model_cfg": asdict(model_cfg),
                },
                output_dir / "ckpt_latest.pt",
            )
            print(f"  ✓ Checkpoint saved → {ckpt_path.name}")
            _sample_generation(model, tokenizer, device, cfg.block_size)
            model.train()

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    _sample_generation(model, tokenizer, device, cfg.block_size)


@torch.no_grad()
def evaluate(
    model: ChessGPT, dl: DataLoader, device: torch.device, max_iters: int
) -> float:
    model.eval()
    losses = []
    for i, (x, y, w) in enumerate(dl):
        if i >= max_iters:
            break
        x, y, w = x.to(device), y.to(device), w.to(device)
        _, loss = model(x, targets=y, loss_weights=w)
        losses.append(loss.item())
    return sum(losses) / len(losses) if losses else float("inf")


@torch.no_grad()
def _sample_generation(
    model: ChessGPT, tokenizer: Tokenizer, device: torch.device, block_size: int
):
    model.eval()
    prompts = [
        "[g_start]1.e4 e5 2.Nf3",
        "[g_start]1.d4 Nf6 2.c4",
        "[g_start]1.e4 c5 2.Nf3",
    ]
    print("\n── Sample generation ──────────────────────────────────────────────")
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=20, temperature=0.7, top_k=30)
        text = tokenizer.decode(out[0].tolist())
        print(f"  {text}")
    print()


def generate_from_checkpoint(
    prompt: str,
    ckpt_path: Path,
    max_tokens: int = 30,
    temperature: float = 0.7,
    top_k: int = 40,
):
    """Load a checkpoint and generate from a prompt."""
    device = select_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = ChessGPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    text = prompt.strip()
    if not text.startswith("[g_start]"):
        text = f"[g_start]{text}"
    ids = tokenizer.encode(text).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k
    )
    return tokenizer.decode(out[0].tolist())


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="kn1ght chess LM training")
    p.add_argument(
        "--max-games", type=int, default=100_000, help="Games to load from HF dataset"
    )
    p.add_argument("--iters", type=int, default=20_000, help="Training iterations")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument(
        "--openings-repeat", type=int, default=10, help="Opening oversampling factor"
    )
    p.add_argument(
        "--turn-weight",
        type=float,
        default=0.15,
        help="Loss weight for turn-number tokens",
    )
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument(
        "--generate",
        type=str,
        default=None,
        metavar="PROMPT",
        help="Skip training, generate from latest checkpoint",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to load for --generate",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Resume training from this checkpoint (model + optimizer state + step count)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.generate is not None:
        output_dir = Path(args.output_dir)
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
        else:
            ckpt_path = output_dir / "ckpt_latest.pt"
        if not ckpt_path.exists():
            sys.exit(f"No checkpoint found at {ckpt_path}")
        result = generate_from_checkpoint(args.generate, ckpt_path)
        print(result)
        return

    cfg = TrainConfig(
        max_train_games=args.max_games,
        max_iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        openings_repeat=args.openings_repeat,
        turn_number_weight=args.turn_weight,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )
    train(cfg)


if __name__ == "__main__":
    main()
