#!/usr/bin/env python3
"""Run LocalChessEngine Cute Chess matches and emit parseable evidence."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = REPO_ROOT / "build" / "release" / "engine" / "localchessengine"
DEFAULT_CUTECHESS = REPO_ROOT / ".deps" / "cutechess-build" / "cutechess-cli"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "build" / "matches"


@dataclass(frozen=True)
class SprtConfig:
    elo0: float
    elo1: float
    alpha: float = 0.05
    beta: float = 0.05


@dataclass(frozen=True)
class MatchPreset:
    time_control: str
    games: int
    rounds: int
    hash_mb: int
    threads: int
    maxmoves: int
    sprt: SprtConfig | None = None


PRESETS: dict[str, MatchPreset] = {
    "smoke": MatchPreset(time_control="1+0.01", games=2, rounds=1, hash_mb=16, threads=1, maxmoves=20),
    "stc": MatchPreset(time_control="10+0.1", games=2, rounds=1, hash_mb=8, threads=1, maxmoves=0),
    "ltc": MatchPreset(time_control="60+0.6", games=2, rounds=1, hash_mb=64, threads=1, maxmoves=0),
    "sprt-feature": MatchPreset(
        time_control="10+0.1",
        games=2,
        rounds=1,
        hash_mb=8,
        threads=1,
        maxmoves=0,
        sprt=SprtConfig(elo0=0.0, elo1=5.0),
    ),
    "sprt-small-nnue": MatchPreset(
        time_control="10+0.1",
        games=2,
        rounds=1,
        hash_mb=8,
        threads=1,
        maxmoves=0,
        sprt=SprtConfig(elo0=0.0, elo1=50.0),
    ),
    "sprt-regression": MatchPreset(
        time_control="10+0.1",
        games=2,
        rounds=1,
        hash_mb=8,
        threads=1,
        maxmoves=0,
        sprt=SprtConfig(elo0=-3.0, elo1=0.0),
    ),
}

FAILURE_PATTERNS = {
    "illegal move": re.compile(r"\billegal\s+move\b|\bmakes?\s+an\s+illegal\s+move\b", re.IGNORECASE),
    "crash": re.compile(r"\bcrash(?:ed|es|ing)?\b|\bterminated\s+unexpectedly\b", re.IGNORECASE),
    "disconnect": re.compile(r"\bdisconnect(?:ed|s|ion)?\b|\babandoned\b", re.IGNORECASE),
    "stalled connection": re.compile(r"\bstalled\s+connection\b|\bconnection\s+stalls?\b", re.IGNORECASE),
    "timeout": re.compile(r"\btime(?:d)?\s*out\b|\btimeout\b|\bdraw\s+by\s+timeout\b", re.IGNORECASE),
    "time forfeit": re.compile(
        r"\btime\s+forfeit\b|\bforfeit(?:s|ed)?\s+on\s+time\b|\bloses?\s+on\s+time\b",
        re.IGNORECASE,
    ),
    "unterminated game": re.compile(r"\bunterminated\b|\bincomplete\s+game\b|\bno\s+result\b", re.IGNORECASE),
    "invalid result claim": re.compile(r"\binvalid\s+result\s+claim\b", re.IGNORECASE),
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Cute Chess match with LocalChessEngine-compatible UCI engines. "
            "SPRT presets put the test engine first, matching Cute Chess's A-vs-B SPRT direction."
        )
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--cutechess", type=Path, default=DEFAULT_CUTECHESS)
    parser.add_argument("--base-engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--test-engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--base-name", default="LCE-Base")
    parser.add_argument("--test-name", default="LCE-Test")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--games", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--time-control")
    parser.add_argument("--timemargin-ms", type=int, default=250)
    parser.add_argument("--hash-mb", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--base-option",
        action="append",
        default=[],
        help="Extra UCI option for the base engine, formatted as 'Name=Value'. May be repeated.",
    )
    parser.add_argument(
        "--test-option",
        action="append",
        default=[],
        help="Extra UCI option for the test engine, formatted as 'Name=Value'. May be repeated.",
    )
    parser.add_argument(
        "--both-option",
        action="append",
        default=[],
        help="Extra UCI option for both engines, formatted as 'Name=Value'. May be repeated.",
    )
    parser.add_argument("--maxmoves", type=int)
    parser.add_argument("--seed", type=int, default=25025)
    parser.add_argument("--openings", type=Path, help="EPD/PGN opening file for Cute Chess.")
    parser.add_argument("--opening-format", choices=("epd", "pgn"))
    parser.add_argument("--opening-plies", type=positive_int)
    parser.add_argument("--opening-order", choices=("random", "sequential"), default="sequential")
    parser.add_argument(
        "--opening-policy",
        choices=("default", "encounter", "round"),
        default="default",
    )
    parser.add_argument(
        "--repeat",
        nargs="?",
        const=2,
        type=positive_int,
        help="Play each opening twice by default, or the provided repeat count.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Cute Chess protocol debug output.")
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def parse_float_token(token: str) -> float | str | None:
    lowered = token.lower()
    if lowered == "nan":
        return None
    if lowered in {"inf", "+inf", "-inf"}:
        return lowered
    return float(token)


def parse_score(log_text: str) -> dict[str, Any]:
    score_re = re.compile(
        r"Score of (?P<lhs>.+?) vs (?P<rhs>.+?): "
        r"(?P<wins>\d+) - (?P<losses>\d+) - (?P<draws>\d+)\s+"
        r"\[(?P<score>[0-9.]+)\]\s+(?P<games>\d+)"
    )
    matches = list(score_re.finditer(log_text))
    if not matches:
        return {}

    match = matches[-1]
    return {
        "score_line": match.group(0),
        "lhs": match.group("lhs"),
        "rhs": match.group("rhs"),
        "wins": int(match.group("wins")),
        "losses": int(match.group("losses")),
        "draws": int(match.group("draws")),
        "score": float(match.group("score")),
        "games_completed": int(match.group("games")),
    }


def parse_elo(log_text: str) -> dict[str, Any]:
    elo_re = re.compile(
        r"Elo difference:\s+(?P<diff>[-+]?\d+(?:\.\d+)?)\s+\+/-\s+"
        r"(?P<error>[-+]?\d+(?:\.\d+)?),\s+LOS:\s+(?P<los>[-+]?\d+(?:\.\d+)?|nan)\s+%,"
        r"\s+DrawRatio:\s+(?P<draw>[-+]?\d+(?:\.\d+)?|nan)\s+%",
        re.IGNORECASE,
    )
    matches = list(elo_re.finditer(log_text))
    if not matches:
        return {}

    match = matches[-1]
    return {
        "difference": float(match.group("diff")),
        "error": float(match.group("error")),
        "los_percent": parse_float_token(match.group("los")),
        "draw_ratio_percent": parse_float_token(match.group("draw")),
    }


def parse_sprt(log_text: str) -> dict[str, Any]:
    value = r"[-+]?(?:\d+(?:\.\d+)?|inf|nan)"
    sprt_re = re.compile(
        rf"SPRT:\s+llr\s+(?P<llr>{value})\s+\((?P<percent>{value})%\),\s+"
        rf"lbound\s+(?P<lower>{value}),\s+ubound\s+(?P<upper>{value})",
        re.IGNORECASE,
    )
    matches = list(sprt_re.finditer(log_text))
    if not matches:
        return {}

    match = matches[-1]
    return {
        "line": match.group(0),
        "llr": parse_float_token(match.group("llr")),
        "llr_percent": parse_float_token(match.group("percent")),
        "lower_bound": parse_float_token(match.group("lower")),
        "upper_bound": parse_float_token(match.group("upper")),
    }


def sprt_has_decision(sprt_output: dict[str, Any]) -> bool:
    llr = sprt_output.get("llr")
    lower = sprt_output.get("lower_bound")
    upper = sprt_output.get("upper_bound")
    if not all(isinstance(value, (int, float)) for value in (llr, lower, upper)):
        return False
    return llr <= lower or llr >= upper


def detect_failures(*texts: str) -> list[str]:
    searchable_text = "\n".join(text for text in texts if text)
    return [label for label, pattern in FAILURE_PATTERNS.items() if pattern.search(searchable_text)]


def read_text_if_present(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def cutechess_version(cutechess: Path) -> str:
    completed = subprocess.run(
        [str(cutechess), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def opening_format(path: Path, explicit_format: str | None) -> str:
    if explicit_format is not None:
        return explicit_format
    if path.suffix.lower() == ".epd":
        return "epd"
    return "pgn"


def build_command(args: argparse.Namespace, preset: MatchPreset, pgn_path: Path) -> list[str]:
    sprt = preset.sprt
    time_control = args.time_control or preset.time_control
    games = args.games if args.games is not None else preset.games
    rounds = args.rounds if args.rounds is not None else preset.rounds
    hash_mb = args.hash_mb if args.hash_mb is not None else preset.hash_mb
    threads = args.threads if args.threads is not None else preset.threads
    maxmoves = args.maxmoves if args.maxmoves is not None else preset.maxmoves
    base_options = [f"option.{option}" for option in [*args.both_option, *args.base_option]]
    test_options = [f"option.{option}" for option in [*args.both_option, *args.test_option]]

    command = [
        str(args.cutechess),
        "-engine",
        f"name={args.test_name}",
        f"cmd={args.test_engine}",
        f"option.Threads={threads}",
        f"option.Hash={hash_mb}",
        *test_options,
        "-engine",
        f"name={args.base_name}",
        f"cmd={args.base_engine}",
        f"option.Threads={threads}",
        f"option.Hash={hash_mb}",
        *base_options,
        "-each",
        "proto=uci",
        f"tc={time_control}",
        f"timemargin={args.timemargin_ms}",
        "-variant",
        "standard",
        "-games",
        str(games),
        "-rounds",
        str(rounds),
        "-concurrency",
        str(args.concurrency),
        "-srand",
        str(args.seed),
        "-event",
        f"LocalChessEngine {args.preset}",
        "-site",
        "local",
        "-pgnout",
        str(pgn_path),
    ]
    if args.openings is not None:
        openings = [
            f"file={args.openings}",
            f"format={opening_format(args.openings, args.opening_format)}",
            f"order={args.opening_order}",
            f"policy={args.opening_policy}",
        ]
        if args.opening_plies is not None:
            openings.append(f"plies={args.opening_plies}")
        command.extend(["-openings", *openings])
    if args.repeat is not None:
        command.append("-repeat")
        if args.repeat != 2:
            command.append(str(args.repeat))
    if maxmoves > 0:
        command.extend(["-maxmoves", str(maxmoves)])
    if sprt is not None:
        command.extend(
            [
                "-sprt",
                f"elo0={sprt.elo0:g}",
                f"elo1={sprt.elo1:g}",
                f"alpha={sprt.alpha:g}",
                f"beta={sprt.beta:g}",
            ]
        )
    if args.debug:
        command.append("-debug")
    return command


def run_match(args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    args.cutechess = require_file(args.cutechess, "cutechess-cli")
    args.base_engine = require_file(args.base_engine, "base engine")
    args.test_engine = require_file(args.test_engine, "test engine")
    if args.openings is not None:
        args.openings = require_file(args.openings, "opening file")

    output_dir = args.out_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp()}-{args.preset}"
    output_dir = output_dir if output_dir.is_absolute() else (REPO_ROOT / output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "cutechess.log"
    pgn_path = output_dir / "match.pgn"
    result_path = output_dir / "result.json"
    command = build_command(args, preset, pgn_path)

    completed = subprocess.run(
        command,
        check=False,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_text = completed.stdout
    log_header = [
        f"# date_utc: {datetime.now(timezone.utc).isoformat()}",
        f"# command: {shlex.join(command)}",
        "",
    ]
    log_path.write_text("\n".join(log_header) + log_text, encoding="utf-8")

    score = parse_score(log_text)
    elo = parse_elo(log_text)
    sprt_output = parse_sprt(log_text)
    pgn_text = read_text_if_present(pgn_path)
    failures = detect_failures(log_text, pgn_text)
    sprt_decided = sprt_has_decision(sprt_output)
    if sprt_decided:
        failures = [failure for failure in failures if failure != "unterminated game"]
    if completed.returncode != 0 and not (sprt_decided and not failures):
        failures.append(f"cutechess exited with {completed.returncode}")
    if not score:
        failures.append("missing final score line")
    if not pgn_path.is_file():
        failures.append("missing PGN output")

    sprt = preset.sprt
    result = {
        "event": "match.sprt.result",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "pass": len(failures) == 0,
        "failure_reasons": failures,
        "command": command,
        "cutechess": {
            "path": str(args.cutechess),
            "version": cutechess_version(args.cutechess),
        },
        "base": {
            "name": args.base_name,
            "engine": str(args.base_engine),
        },
        "test": {
            "name": args.test_name,
            "engine": str(args.test_engine),
        },
        "settings": {
            "time_control": args.time_control or preset.time_control,
            "timemargin_ms": args.timemargin_ms,
            "games": args.games if args.games is not None else preset.games,
            "rounds": args.rounds if args.rounds is not None else preset.rounds,
            "concurrency": args.concurrency,
            "threads": args.threads if args.threads is not None else preset.threads,
            "hash_mb": args.hash_mb if args.hash_mb is not None else preset.hash_mb,
            "base_options": args.base_option,
            "test_options": args.test_option,
            "both_options": args.both_option,
            "maxmoves": args.maxmoves if args.maxmoves is not None else preset.maxmoves,
            "seed": args.seed,
            "openings": None
            if args.openings is None
            else {
                "file": str(args.openings),
                "format": opening_format(args.openings, args.opening_format),
                "plies": args.opening_plies,
                "order": args.opening_order,
                "policy": args.opening_policy,
                "repeat": args.repeat,
            },
        },
        "match": score,
        "elo": elo,
        "sprt": {
            "enabled": sprt is not None,
            "bounds": None if sprt is None else [sprt.elo0, sprt.elo1],
            "alpha": None if sprt is None else sprt.alpha,
            "beta": None if sprt is None else sprt.beta,
            "output": sprt_output,
        },
        "artifacts": {
            "log": str(log_path),
            "pgn": str(pgn_path),
            "json": str(result_path),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {result_path}")
    print(f"wrote {log_path}")
    print(f"wrote {pgn_path}")
    if failures:
        for failure in failures:
            print(f"failure: {failure}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    try:
        return run_match(parse_args())
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
