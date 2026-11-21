from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


DATASET_REPO = "Lichess/chess-position-evaluations"
DATASET_SPLIT = "train"
MANIFEST_SCHEMA = "localchessengine.nnue.fen_pool.manifest.v1"
CHECKSUM_OFFSET_BASIS = 14695981039346656037
CHECKSUM_PRIME = 1099511628211
MAX_U64 = (1 << 64) - 1


class FenPoolError(RuntimeError):
    pass


def _mix_byte(checksum: int, value: int) -> int:
    checksum ^= value & 0xFF
    checksum = (checksum * CHECKSUM_PRIME) & MAX_U64
    return checksum


def stable_u64(text: str) -> int:
    checksum = CHECKSUM_OFFSET_BASIS
    for byte in text.encode("utf-8"):
        checksum = _mix_byte(checksum, byte)
    return checksum


def parse_board(placement: str) -> dict[int, str]:
    board: dict[int, str] = {}
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise FenPoolError("FEN placement must have 8 ranks")
    white_kings = 0
    black_kings = 0
    for rank_from_top, rank_text in enumerate(ranks):
        rank = 7 - rank_from_top
        file_index = 0
        for symbol in rank_text:
            if "1" <= symbol <= "8":
                file_index += int(symbol)
                continue
            if symbol not in "PNBRQKpnbrqk":
                raise FenPoolError("FEN placement contains invalid piece")
            if file_index >= 8:
                raise FenPoolError("FEN rank overflows")
            board[(rank * 8) + file_index] = symbol
            white_kings += 1 if symbol == "K" else 0
            black_kings += 1 if symbol == "k" else 0
            file_index += 1
        if file_index != 8:
            raise FenPoolError("FEN rank is not 8 files wide")
    if white_kings != 1 or black_kings != 1:
        raise FenPoolError("FEN must contain exactly one king per side")
    return board


def square_index(text: str) -> int:
    if len(text) != 2 or text[0] < "a" or text[0] > "h" or text[1] < "1" or text[1] > "8":
        raise FenPoolError("invalid square")
    return (int(text[1]) - 1) * 8 + (ord(text[0]) - ord("a"))


def validate_castling_rights(text: str) -> None:
    if text == "-":
        return
    if not text or "-" in text:
        raise FenPoolError("castling rights must be '-' or a KQkq subset")
    order = "KQkq"
    cursor = -1
    for symbol in text:
        found = order.find(symbol)
        if found < 0 or found <= cursor:
            raise FenPoolError("castling rights must be unique and canonical")
        cursor = found


def validate_fen_state(fields: list[str], board: dict[int, str]) -> None:
    validate_castling_rights(fields[2])
    if sum(1 for piece in board.values() if piece.isupper()) > 16:
        raise FenPoolError("white cannot have more than 16 pieces")
    if sum(1 for piece in board.values() if piece.islower()) > 16:
        raise FenPoolError("black cannot have more than 16 pieces")
    for square, piece in board.items():
        rank = square // 8
        if piece.lower() == "p" and rank in {0, 7}:
            raise FenPoolError("pawns cannot appear on first or eighth rank")
    ep = fields[3]
    if ep != "-":
        ep_index = square_index(ep)
        if ep[1] not in {"3", "6"}:
            raise FenPoolError("en-passant target rank must be 3 or 6")
        if ep_index in board:
            raise FenPoolError("en-passant target square must be empty")


def canonical_fen(source_fen: str) -> str:
    fields = source_fen.strip().split()
    if len(fields) == 4:
        fields = [*fields, "0", "1"]
    if len(fields) != 6:
        raise FenPoolError("FEN must have 4 source fields or 6 full fields")
    if fields[1] not in {"w", "b"}:
        raise FenPoolError("FEN side-to-move must be w or b")
    if fields[4] and (not fields[4].isdigit() or int(fields[4]) < 0):
        raise FenPoolError("FEN halfmove clock is invalid")
    if fields[5] and (not fields[5].isdigit() or int(fields[5]) <= 0):
        raise FenPoolError("FEN fullmove number is invalid")
    board = parse_board(fields[0])
    validate_fen_state(fields, board)
    return " ".join(fields)


def hf_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FenPoolError(
            "missing optional dependency: install trainer with the fen-pool extra or install datasets"
        ) from exc

    dataset_kwargs: dict[str, Any] = {
        "path": DATASET_REPO,
        "split": args.split,
        "streaming": True,
    }
    if args.cache_dir:
        dataset_kwargs["cache_dir"] = str(args.cache_dir)
    dataset = load_dataset(**dataset_kwargs)
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer)
    try:
        yield from dataset
    finally:
        del dataset


@dataclass
class FenPoolStats:
    source_rows_seen: int = 0
    accepted: int = 0
    duplicate_fen: int = 0
    sampled_out_by_hash: int = 0
    filtered_mate: int = 0
    filtered_missing_fen: int = 0
    filtered_invalid_fen: int = 0
    filtered_source_depth: int = 0
    filtered_source_knodes: int = 0
    filtered_source_eval_out_of_range: int = 0


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_fen_row(
    row: dict[str, Any],
    *,
    min_source_depth: int,
    min_source_knodes: int,
    source_max_abs_cp: int,
    skip_mates: bool,
) -> tuple[str | None, str | None]:
    if row.get("mate") is not None and skip_mates:
        return None, "filtered_mate"
    source_fen = str(row.get("fen") or "").strip()
    if not source_fen:
        return None, "filtered_missing_fen"
    source_depth = _parse_int(row.get("depth"))
    source_knodes = _parse_int(row.get("knodes"))
    source_cp = row.get("cp")
    if source_depth < min_source_depth:
        return None, "filtered_source_depth"
    if source_knodes < min_source_knodes:
        return None, "filtered_source_knodes"
    if source_cp is not None and abs(_parse_int(source_cp)) > source_max_abs_cp:
        return None, "filtered_source_eval_out_of_range"
    try:
        return canonical_fen(source_fen), None
    except FenPoolError:
        return None, "filtered_invalid_fen"


def _source_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if args.input_jsonl is not None:
        with args.input_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield json.loads(text)
        return
    yield from hf_rows(args)


def write_fen_pool(args: argparse.Namespace) -> FenPoolStats:
    stats = FenPoolStats()
    seen: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in _source_rows(args):
            stats.source_rows_seen += 1
            if args.max_source_rows and stats.source_rows_seen > args.max_source_rows:
                break
            source_fen = str(row.get("fen") or "")
            row_hash = stable_u64(source_fen)
            if args.sample_mod > 1 and row_hash % args.sample_mod != args.sample_remainder:
                stats.sampled_out_by_hash += 1
                continue
            fen, reason = clean_fen_row(
                row,
                min_source_depth=args.min_source_depth,
                min_source_knodes=args.min_source_knodes,
                source_max_abs_cp=args.source_max_abs_cp,
                skip_mates=args.skip_mates,
            )
            if reason is not None:
                setattr(stats, reason, getattr(stats, reason) + 1)
                continue
            assert fen is not None
            if fen in seen:
                stats.duplicate_fen += 1
                continue
            seen.add(fen)
            if args.format == "jsonl":
                output.write(json.dumps({"fen": fen}, sort_keys=True) + "\n")
            elif args.format == "csv":
                if stats.accepted == 0:
                    output.write("fen\n")
                output.write('"' + fen.replace('"', '""') + '"\n')
            else:
                output.write(fen + "\n")
            stats.accepted += 1
            if stats.accepted >= args.positions:
                break
    return stats


def write_manifest(args: argparse.Namespace, stats: FenPoolStats, elapsed_seconds: float) -> None:
    if args.manifest is None:
        return
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": "trainer.fen_pool.manifest",
        "schema": MANIFEST_SCHEMA,
        "dataset_id": args.dataset_id,
        "date": args.date,
        "source_task": "TASK-034",
        "output": str(args.output),
        "format": args.format,
        "positions": stats.accepted,
        "external_position_source": True,
        "stockfish_labels_present": True,
        "stockfish_labels_used": False,
        "source": {
            "provider": "Hugging Face Hub" if args.input_jsonl is None else "local_jsonl_fixture",
            "dataset": DATASET_REPO if args.input_jsonl is None else str(args.input_jsonl),
            "split": args.split,
            "license": "cc0-1.0" if args.input_jsonl is None else "local-fixture",
            "policy": "FENs only; Stockfish cp/line are not training targets.",
        },
        "filters": {
            "min_source_depth": args.min_source_depth,
            "min_source_knodes": args.min_source_knodes,
            "source_max_abs_cp": args.source_max_abs_cp,
            "skip_mates": args.skip_mates,
            "sample_mod": args.sample_mod,
            "sample_remainder": args.sample_remainder,
            "shuffle_buffer": args.shuffle_buffer,
            "shuffle_seed": args.shuffle_seed,
        },
        "stats": asdict(stats),
        "elapsed_seconds": elapsed_seconds,
    }
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a deterministic Lichess/HF FEN pool for C++ LocalChessEngine relabeling."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--dataset-id", default="lce-task-034-lichess-fen-pool")
    parser.add_argument("--date", default="2026-05-12T00:00:00Z")
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument("--cache-dir", type=Path, default=Path(".deps/huggingface/datasets"))
    parser.add_argument("--positions", type=int, default=100_000)
    parser.add_argument("--format", choices=["fen", "jsonl", "csv"], default="fen")
    parser.add_argument("--min-source-depth", type=int, default=0)
    parser.add_argument("--min-source-knodes", type=int, default=0)
    parser.add_argument("--source-max-abs-cp", type=int, default=2500)
    parser.add_argument("--sample-mod", type=int, default=1)
    parser.add_argument("--sample-remainder", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=34034)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--skip-mates", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.positions <= 0:
        parser.error("--positions must be positive")
    if args.sample_mod <= 0:
        parser.error("--sample-mod must be positive")
    if args.sample_remainder < 0 or args.sample_remainder >= args.sample_mod:
        parser.error("--sample-remainder must be in [0, sample-mod)")
    if args.shuffle_buffer < 0:
        parser.error("--shuffle-buffer must be non-negative")
    if args.max_source_rows < 0:
        parser.error("--max-source-rows must be non-negative")
    if args.source_max_abs_cp <= 0:
        parser.error("--source-max-abs-cp must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    hard_exit = argv is None
    args = parse_args(sys.argv[1:] if argv is None else argv)
    start = time.perf_counter()
    stats = write_fen_pool(args)
    if stats.accepted < args.positions:
        print(
            f"error: only produced {stats.accepted} FENs; requested {args.positions}",
            file=sys.stderr,
        )
        return 1
    write_manifest(args, stats, time.perf_counter() - start)
    print(
        json.dumps(
            {
                "status": "pass",
                "dataset_id": args.dataset_id,
                "positions": stats.accepted,
                "source_rows_seen": stats.source_rows_seen,
                "stockfish_labels_used": False,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    if hard_exit:
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
