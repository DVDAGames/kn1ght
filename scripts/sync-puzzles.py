#!/usr/bin/env python3
"""
Sync Lichess puzzle positions to .data/puzzles/pgn_puzzles.jsonl

Reads the pre-downloaded lichess_puzzles.csv.zst, fetches the source game PGN
from the Lichess API for each selected puzzle, reconstructs the PGN context up
to the puzzle position, and appends new records to pgn_puzzles.jsonl.

Already-fetched puzzle IDs are skipped automatically (idempotent).

Usage:
    uv run python scripts/sync-puzzles.py               # fetch 500 more
    uv run python scripts/sync-puzzles.py --count 2000  # fetch 2000 more
    uv run python scripts/sync-puzzles.py --count 2000 --token $LICHESS_API_TOKEN
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import chess
import chess.pgn
import io

ROOT = Path(__file__).resolve().parent.parent
PUZZLES_DIR = ROOT / ".data" / "puzzles"
CSV_PATH = PUZZLES_DIR / "lichess_puzzles.csv.zst"
OUT_FILE = PUZZLES_DIR / "pgn_puzzles.jsonl"


def uci_to_san(uci: str, fen: str) -> str | None:
    """Convert a UCI move string to SAN given the board FEN."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return board.san(move)
    except Exception:
        pass
    return None


def extract_game_id(game_url: str) -> str:
    """Extract the Lichess game ID from a GameUrl value.

    GameUrl values include optional /black or /white suffixes and #NN move
    anchors, e.g. 'lichess.org/AbCdEfGh/black#36'.
    The game ID is always the path segment at index 3.
    """
    return game_url.split("/")[3].split("#")[0]


def fetch_game_pgn(game_id: str, token: str = "") -> str | None:
    url = (
        f"https://lichess.org/game/export/{game_id}"
        "?moves=true&clocks=false&evals=false&opening=false"
    )
    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": "kn1ght-puzzle-builder/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8")
    except Exception as exc:
        print(f"    fetch error ({game_id}): {exc}")
        return None


def reconstruct_pgn_context(pgn_text: str, target_fen: str) -> str | None:
    """Walk the game PGN until the board matches target_fen; return moves so far."""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None
    except Exception:
        return None

    board = game.board()
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    node = game

    # Normalise FEN for comparison: strip move clocks (last two fields)
    def norm(fen: str) -> str:
        return " ".join(fen.split()[:4])

    while node.variations:
        node = node.variations[0]
        board.push(node.move)
        if norm(board.fen()) == norm(target_fen):
            # Build a minimal PGN string from the starting position up to here
            g = chess.pgn.Game()
            g.setup(chess.STARTING_FEN)
            n = g
            for move in board.move_stack:
                n = n.add_variation(move)
            return g.accept(exporter).strip()

    return None


def load_csv(csv_path: Path):
    """Load the puzzle CSV (zst or plain) and return a list of dicts."""
    import csv

    if csv_path.suffix == ".zst":
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        with open(csv_path, "rb") as fh:
            stream = dctx.stream_reader(fh)
            text = stream.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    p = argparse.ArgumentParser(description="Sync Lichess puzzle positions")
    p.add_argument("--count", type=int, default=500, help="Number of new puzzles to fetch")
    p.add_argument("--rating-min", type=int, default=1200)
    p.add_argument("--rating-max", type=int, default=1900)
    p.add_argument(
        "--sleep",
        type=float,
        default=0.7,
        help="Seconds to sleep between Lichess API requests (lower with --token)",
    )
    p.add_argument(
        "--token",
        type=str,
        default=os.environ.get("LICHESS_API_TOKEN", ""),
        help="Lichess OAuth token (increases rate limit ~9x)",
    )
    args = p.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"Puzzle CSV not found: {CSV_PATH}\nRun the build-puzzle-dataset notebook first to download it.")

    print(f"Loading puzzle CSV from {CSV_PATH} …")
    rows = load_csv(CSV_PATH)
    print(f"  {len(rows):,} total puzzles in CSV")

    # Load already-fetched IDs
    existing_ids: set[str] = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["puzzle_id"])
    print(f"  {len(existing_ids):,} puzzles already fetched")

    # Filter
    import random

    candidates = []
    for row in rows:
        pid = row.get("PuzzleId", "")
        if pid in existing_ids:
            continue
        try:
            rating = int(row["Rating"])
        except (KeyError, ValueError):
            continue
        if not (args.rating_min <= rating <= args.rating_max):
            continue
        themes = set(row.get("Themes", "").split())
        if "middlegame" not in themes:
            continue
        if themes & {"opening", "endgame"}:
            continue
        candidates.append(row)

    print(f"  {len(candidates):,} candidates after filtering")
    if not candidates:
        print("No candidates — nothing to fetch.")
        return

    random.shuffle(candidates)
    sample = candidates[: args.count]
    print(f"  Fetching {len(sample):,} new puzzles …")
    if args.token:
        print(f"  Using Lichess API token (higher rate limit; sleep={args.sleep}s)")
    else:
        print(f"  No token — unauthenticated (~100 req/min, sleep={args.sleep}s)")
        print("  Tip: set LICHESS_API_TOKEN or pass --token to speed this up ~9×")

    written = skipped = 0
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "a") as out_fh:
        for i, row in enumerate(sample):
            puzzle_id = row["PuzzleId"]
            fen = row["FEN"]
            moves_uci = row.get("Moves", "").split()
            game_url = row["GameUrl"]
            rating = int(row["Rating"])
            themes = row.get("Themes", "").split()

            print(f"[{i + 1}/{len(sample)}] {puzzle_id}  rating={rating}  {game_url}")

            if not moves_uci:
                print("    skip: no moves")
                skipped += 1
                continue

            best_uci = moves_uci[0]
            best_san = uci_to_san(best_uci, fen)
            if best_san is None:
                print(f"    skip: UCI→SAN failed ({best_uci})")
                skipped += 1
                continue

            game_id = extract_game_id(game_url)
            pgn_text = fetch_game_pgn(game_id, token=args.token)
            time.sleep(args.sleep)

            if pgn_text is None:
                skipped += 1
                continue

            pgn_context = reconstruct_pgn_context(pgn_text, fen)
            if pgn_context is None:
                print("    skip: FEN not found in game")
                skipped += 1
                continue

            record = {
                "puzzle_id": puzzle_id,
                "game_id": game_id,
                "rating": rating,
                "themes": themes,
                "pgn_context": pgn_context,
                "fen": fen,
                "best_move_uci": best_uci,
                "best_move_san": best_san,
            }
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()
            written += 1
            n_moves = len(pgn_context.split(".")) - 1
            print(f"    OK  context≈{n_moves} moves  answer={best_san}")

    total = len(existing_ids) + written
    print(f"\nDone. Written {written:,} new puzzles (skipped {skipped:,}). Total: {total:,}")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
