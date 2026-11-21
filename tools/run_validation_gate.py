#!/usr/bin/env python3
"""Run LocalChessEngine candidate validation gates and write evidence reports."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ENGINE = REPO_ROOT / "build" / "release" / "engine" / "localchessengine"
DEFAULT_PERFT = REPO_ROOT / "build" / "release" / "localchessengine_perft"
DEFAULT_CUTECHESS = REPO_ROOT / ".deps" / "cutechess-build" / "cutechess-cli"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "build" / "validation"
DEFAULT_BENCH_NODES = 18417
OUTPUT_TAIL_LIMIT = 12_000
STATISTICAL_CHECK_NAMES = {"Strength promotion evidence", "Regression evidence"}


@dataclass(frozen=True)
class PerftReference:
    fen_id: str
    name: str
    fen: str
    nodes: dict[int, int]


CPW_PERFT_REFERENCES: tuple[PerftReference, ...] = (
    PerftReference(
        fen_id="cpw-initial",
        name="Initial position",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        nodes={0: 1, 1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609, 6: 119060324},
    ),
    PerftReference(
        fen_id="cpw-kiwipete",
        name="Position 2 Kiwipete",
        fen="r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w "
        "KQkq - 0 1",
        nodes={
            0: 1,
            1: 48,
            2: 2039,
            3: 97862,
            4: 4085603,
            5: 193690690,
            6: 8031647685,
        },
    ),
    PerftReference(
        fen_id="cpw-position-3",
        name="Position 3",
        fen="8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        nodes={0: 1, 1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624, 6: 11030083},
    ),
    PerftReference(
        fen_id="cpw-position-4",
        name="Position 4",
        fen="r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq "
        "- 0 1",
        nodes={0: 1, 1: 6, 2: 264, 3: 9467, 4: 422333, 5: 15833292, 6: 706045033},
    ),
    PerftReference(
        fen_id="cpw-position-5",
        name="Position 5",
        fen="rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        nodes={0: 1, 1: 44, 2: 1486, 3: 62379, 4: 2103487, 5: 89941194},
    ),
    PerftReference(
        fen_id="cpw-position-6",
        name="Position 6",
        fen="r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        nodes={
            0: 1,
            1: 46,
            2: 2079,
            3: 89890,
            4: 3894594,
            5: 164075551,
            6: 6923051137,
        },
    ),
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def output_tail(text: str) -> str:
    if len(text) <= OUTPUT_TAIL_LIMIT:
        return text
    return text[-OUTPUT_TAIL_LIMIT:]


def command_to_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(name: str, command: list[str], *, required: bool = True) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed_ms = round((time.monotonic() - started) * 1000)
    passed = completed.returncode == 0
    return {
        "name": name,
        "event": "validation.command",
        "command": command,
        "command_text": command_to_text(command),
        "required": required,
        "pass": passed,
        "returncode": completed.returncode,
        "elapsed_ms": elapsed_ms,
        "output_tail": output_tail(completed.stdout),
    }


def parse_bench_output(text: str) -> dict[str, int] | None:
    matches = list(re.finditer(r"(?P<nodes>\d+)\s+nodes\s+(?P<nps>\d+)\s+nps", text))
    if not matches:
        return None
    match = matches[-1]
    return {"nodes": int(match.group("nodes")), "nps": int(match.group("nps"))}


def parse_perft_nodes(text: str) -> int | None:
    match = re.search(r"^nodes:\s+(?P<nodes>\d+)\s*$", text, flags=re.MULTILINE)
    if match is None:
        return None
    return int(match.group("nodes"))


def sprt_bounds_type(sprt: dict[str, Any]) -> str:
    if not bool(sprt.get("enabled")):
        return "not_sprt"
    bounds = sprt.get("bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return "unknown"
    lower_elo, upper_elo = bounds
    if not isinstance(lower_elo, (int, float)) or not isinstance(upper_elo, (int, float)):
        return "unknown"
    return "gainer" if lower_elo + upper_elo >= 0 else "non_regression"


def classify_sprt(match_result: dict[str, Any], *, minimum_games: int) -> dict[str, Any]:
    match = match_result.get("match", {})
    games_completed = int(match.get("games_completed") or 0)
    sprt = match_result.get("sprt", {})
    output = sprt.get("output", {}) if isinstance(sprt, dict) else {}
    llr = output.get("llr") if isinstance(output, dict) else None
    lower = output.get("lower_bound") if isinstance(output, dict) else None
    upper = output.get("upper_bound") if isinstance(output, dict) else None
    enabled = bool(sprt.get("enabled")) if isinstance(sprt, dict) else False
    bounds_type = sprt_bounds_type(sprt) if isinstance(sprt, dict) else "unknown"

    classification = "not_sprt"
    if enabled and isinstance(llr, (int, float)) and isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        if llr <= lower:
            classification = "accept_h0"
        elif llr >= upper:
            classification = "accept_h1"
        else:
            classification = "inconclusive"
    elif enabled:
        classification = "missing_sprt_output"

    evidence_kind = "meaningful" if enabled and games_completed >= minimum_games else "wiring_smoke"
    promotes_strength = (
        evidence_kind == "meaningful"
        and bounds_type == "gainer"
        and classification == "accept_h1"
    )
    blocks_regression = (
        evidence_kind == "meaningful"
        and bounds_type == "non_regression"
        and classification == "accept_h1"
    )

    return {
        "event": "match.sprt.classification",
        "classification": classification,
        "bounds_type": bounds_type,
        "evidence_kind": evidence_kind,
        "minimum_games": minimum_games,
        "games_completed": games_completed,
        "promotes_strength": promotes_strength,
        "blocks_regression": blocks_regression,
        "llr": llr,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def profile_defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "perft_depth": 4,
            "quality_command": "debug_ctest",
            "match_preset": "smoke",
            "match_games": 2,
            "match_rounds": 1,
            "match_time_control": "1+0.01",
            "match_maxmoves": 20,
            "minimum_sprt_games": 400,
        }
    if profile == "candidate":
        return {
            "perft_depth": 4,
            "quality_command": "full_quality",
            "match_preset": "sprt-regression",
            "match_games": 2,
            "match_rounds": 1,
            "match_time_control": "1+0.01",
            "match_maxmoves": 8,
            "minimum_sprt_games": 400,
        }
    if profile == "extended":
        return {
            "perft_depth": 5,
            "quality_command": "full_quality",
            "match_preset": "sprt-regression",
            "match_games": 40,
            "match_rounds": 1,
            "match_time_control": "10+0.1",
            "match_maxmoves": 0,
            "minimum_sprt_games": 400,
        }
    raise ValueError(f"unknown profile: {profile}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "candidate", "extended"), default="smoke")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--base-engine", type=Path, default=DEFAULT_RELEASE_ENGINE)
    parser.add_argument("--test-engine", type=Path, default=DEFAULT_RELEASE_ENGINE)
    parser.add_argument("--perft-bin", type=Path, default=DEFAULT_PERFT)
    parser.add_argument("--cutechess", type=Path, default=DEFAULT_CUTECHESS)
    parser.add_argument("--perft-depth", type=int)
    parser.add_argument("--bench-nodes", type=int, default=DEFAULT_BENCH_NODES)
    parser.add_argument("--minimum-sprt-games", type=int)
    parser.add_argument("--match-games", type=int)
    parser.add_argument("--match-rounds", type=int)
    parser.add_argument("--match-time-control")
    parser.add_argument("--match-maxmoves", type=int)
    parser.add_argument("--skip-match", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str, failures: list[str]) -> bool:
    if path.is_file():
        return True
    failures.append(f"{label} not found: {path}")
    return False


def run_quality(profile: str, defaults: dict[str, Any]) -> list[dict[str, Any]]:
    del profile
    if defaults["quality_command"] == "full_quality":
        return [run_command("full local quality gate", ["bash", "tools/validate-quality.sh"])]
    return [
        run_command("match parser regression", ["python3", "tools/run_match_parser_test.py"]),
        run_command("validation gate parser regression", ["python3", "tools/run_validation_gate_test.py"]),
        run_command("debug configure", ["cmake", "--preset", "debug"]),
        run_command("debug build", ["cmake", "--build", "--preset", "debug"]),
        run_command("debug ctest", ["ctest", "--preset", "debug", "--output-on-failure"]),
    ]


def run_release_build() -> list[dict[str, Any]]:
    return [
        run_command("release configure", ["cmake", "--preset", "release"]),
        run_command("release build", ["cmake", "--build", "--preset", "release"]),
    ]


def run_bench(engine: Path, expected_nodes: int) -> tuple[dict[str, Any], dict[str, Any]]:
    command = run_command("release engine bench", [str(engine), "bench"])
    parsed = parse_bench_output(command["output_tail"])
    passed = command["pass"] and parsed is not None and parsed["nodes"] == expected_nodes
    bench = {
        "event": "engine.bench.result",
        "engine": str(engine),
        "expected_nodes": expected_nodes,
        "actual_nodes": None if parsed is None else parsed["nodes"],
        "nps": None if parsed is None else parsed["nps"],
        "pass": passed,
    }
    if not passed:
        command["pass"] = False
    return command, bench


def run_perft(perft_bin: Path, depth: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for reference in CPW_PERFT_REFERENCES:
        actual_depth = depth if depth in reference.nodes else max(
            (candidate_depth for candidate_depth in reference.nodes if candidate_depth <= depth),
            default=-1,
        )
        expected = reference.nodes.get(actual_depth)
        if expected is None:
            records.append(
                {
                    "event": "engine.perft.result",
                    "fen_id": reference.fen_id,
                    "name": reference.name,
                    "requested_depth": depth,
                    "depth": actual_depth,
                    "expected": None,
                    "actual": None,
                    "elapsed_ms": 0,
                    "pass": False,
                    "skipped": True,
                    "reason": "no published reference for requested depth",
                }
            )
            continue

        started = time.monotonic()
        completed = subprocess.run(
            [str(perft_bin), "perft", str(actual_depth), *reference.fen.split()],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed_ms = round((time.monotonic() - started) * 1000)
        actual = parse_perft_nodes(completed.stdout)
        passed = completed.returncode == 0 and actual == expected
        records.append(
            {
                "event": "engine.perft.result",
                "fen_id": reference.fen_id,
                "name": reference.name,
                "fen": reference.fen,
                "requested_depth": depth,
                "depth": actual_depth,
                "expected": expected,
                "actual": actual,
                "elapsed_ms": elapsed_ms,
                "pass": passed,
                "returncode": completed.returncode,
                "output_tail": output_tail(completed.stdout),
            }
        )
    return records


def run_match(args: argparse.Namespace, defaults: dict[str, Any], out_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if args.skip_match:
        return None, None, {
            "name": "cutechess match",
            "event": "validation.command",
            "command": [],
            "required": True,
            "pass": False,
            "returncode": 1,
            "elapsed_ms": 0,
            "output_tail": "match skipped by --skip-match; validation gate is not complete",
        }

    match_dir = out_dir / "match"
    command = [
        "python3",
        "tools/run_match.py",
        "--preset",
        defaults["match_preset"],
        "--base-engine",
        str(args.base_engine),
        "--test-engine",
        str(args.test_engine),
        "--games",
        str(args.match_games if args.match_games is not None else defaults["match_games"]),
        "--rounds",
        str(args.match_rounds if args.match_rounds is not None else defaults["match_rounds"]),
        "--time-control",
        args.match_time_control or defaults["match_time_control"],
        "--out-dir",
        str(match_dir),
    ]
    maxmoves = args.match_maxmoves if args.match_maxmoves is not None else defaults["match_maxmoves"]
    if maxmoves > 0:
        command.extend(["--maxmoves", str(maxmoves)])

    command_record = run_command("cutechess match", command)
    result_path = match_dir / "result.json"
    match_result = None
    classification = None
    if result_path.is_file():
        match_result = json.loads(result_path.read_text(encoding="utf-8"))
        minimum_games = (
            args.minimum_sprt_games
            if args.minimum_sprt_games is not None
            else defaults["minimum_sprt_games"]
        )
        classification = classify_sprt(match_result, minimum_games=minimum_games)
        if not match_result.get("pass", False):
            command_record["pass"] = False
    else:
        command_record["pass"] = False
    return match_result, classification, command_record


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def report_command_line(command: dict[str, Any]) -> str:
    status = "PASS" if command["pass"] else "FAIL"
    return f"- {status} `{command['name']}` ({command['elapsed_ms']} ms): `{command.get('command_text', '')}`"


def write_markdown_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Validation Gate Report",
        "",
        f"- Date UTC: {result['date_utc']}",
        f"- Profile: `{result['profile']}`",
        f"- Overall result: {'PASS' if result['pass'] else 'FAIL'}",
        f"- Report JSON: `{result['artifacts']['json']}`",
        "",
        "## Commands",
        "",
    ]
    lines.extend(report_command_line(command) for command in result["commands"])
    lines.extend(["", "## Bench", ""])
    bench = result["bench"]
    lines.append(
        f"- {'PASS' if bench['pass'] else 'FAIL'} expected `{bench['expected_nodes']}` nodes, "
        f"actual `{bench['actual_nodes']}`, nps `{bench['nps']}`."
    )
    lines.extend(["", "## Perft", ""])
    for record in result["perft"]:
        status = "PASS" if record["pass"] else "FAIL"
        expected = record.get("expected")
        actual = record.get("actual")
        requested = record.get("requested_depth", record["depth"])
        depth_text = f"requested `{requested}`, ran `{record['depth']}`"
        lines.append(
            f"- {status} `{record['fen_id']}` depth {depth_text} expected `{expected}` "
            f"actual `{actual}` elapsed `{record['elapsed_ms']}` ms."
        )
    lines.extend(["", "## Match Evidence", ""])
    match = result.get("match")
    classification = result.get("sprt_classification")
    if match is None:
        lines.append("- Match skipped.")
    else:
        lines.append(f"- Match pass: `{match.get('pass')}`; failures: `{match.get('failure_reasons')}`.")
        lines.append(
            f"- Evidence kind: `{classification['evidence_kind']}`; bounds type: "
            f"`{classification['bounds_type']}`; SPRT classification: "
            f"`{classification['classification']}`; games: `{classification['games_completed']}`."
        )
    lines.extend(["", "## Promotion Checklist", ""])
    for item in result["promotion_checklist"]:
        lines.append(f"- {'PASS' if item['pass'] else 'FAIL'} {item['name']}: {item['details']}")
    if result["failure_reasons"]:
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend(f"- {reason}" for reason in result["failure_reasons"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_promotion_checklist(result: dict[str, Any]) -> list[dict[str, Any]]:
    perft_pass = all(record.get("pass", False) for record in result["perft"])
    bench_pass = bool(result["bench"].get("pass", False))
    match = result.get("match")
    match_pass = match is not None and bool(match.get("pass", False))
    classification = result.get("sprt_classification")
    promotes_strength = bool(classification and classification.get("promotes_strength", False))
    blocks_regression = bool(classification and classification.get("blocks_regression", False))
    return [
        {
            "name": "Perft correctness gate",
            "pass": perft_pass,
            "details": "all requested CPW positions matched published references",
        },
        {
            "name": "Bench signature gate",
            "pass": bench_pass,
            "details": "deterministic node signature matched the accepted baseline",
        },
        {
            "name": "Cute Chess failure gate",
            "pass": match_pass,
            "details": "match JSON had no illegal moves, crashes, disconnects, timeouts or missing output",
        },
        {
            "name": "Strength promotion evidence",
            "pass": promotes_strength,
            "details": "requires meaningful SPRT/fixed-game evidence before enabling search/eval defaults",
        },
        {
            "name": "Regression evidence",
            "pass": blocks_regression,
            "details": "requires meaningful non-regression SPRT/fixed-game evidence for regression gates",
        },
    ]


def collect_failures(result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for command in result["commands"]:
        if command["required"] and not command["pass"]:
            failures.append(f"command failed: {command['name']}")
    if not result["bench"]["pass"]:
        failures.append("bench signature mismatch")
    for record in result["perft"]:
        if not record["pass"]:
            failures.append(f"perft mismatch: {record['fen_id']} depth {record['depth']}")
    match = result.get("match")
    if match is None:
        failures.append("missing match evidence")
    elif not match.get("pass", False):
        failures.append(f"match failed: {match.get('failure_reasons')}")
    return failures


def run_gate(args: argparse.Namespace) -> int:
    defaults = profile_defaults(args.profile)
    if args.perft_depth is not None:
        defaults["perft_depth"] = args.perft_depth

    out_dir = args.out_dir if args.out_dir is not None else DEFAULT_OUTPUT_ROOT / f"{timestamp()}-{args.profile}"
    out_dir = resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args.base_engine = resolve_path(args.base_engine)
    args.test_engine = resolve_path(args.test_engine)
    args.perft_bin = resolve_path(args.perft_bin)
    args.cutechess = resolve_path(args.cutechess)

    preflight_failures: list[str] = []
    if not args.skip_match:
        require_file(args.cutechess, "cutechess-cli", preflight_failures)

    commands: list[dict[str, Any]] = []
    if not preflight_failures:
        commands.extend(run_quality(args.profile, defaults))
        commands.extend(run_release_build())
        require_file(args.base_engine, "base engine", preflight_failures)
        require_file(args.test_engine, "test engine", preflight_failures)
        require_file(args.perft_bin, "perft binary", preflight_failures)
    if not preflight_failures:
        bench_command, bench = run_bench(args.test_engine, args.bench_nodes)
        commands.append(bench_command)
        perft = run_perft(args.perft_bin, defaults["perft_depth"])
        match, classification, match_command = run_match(args, defaults, out_dir)
        if match_command is not None:
            commands.append(match_command)
    else:
        bench = {
            "event": "engine.bench.result",
            "engine": str(args.test_engine),
            "expected_nodes": args.bench_nodes,
            "actual_nodes": None,
            "nps": None,
            "pass": False,
        }
        perft = []
        match = None
        classification = None

    result: dict[str, Any] = {
        "event": "validation.gate.result",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "pass": False,
        "commands": commands,
        "bench": bench,
        "perft": perft,
        "match": match,
        "sprt_classification": classification,
        "settings": {
            "perft_depth": defaults["perft_depth"],
            "bench_nodes": args.bench_nodes,
            "base_engine": str(args.base_engine),
            "test_engine": str(args.test_engine),
            "skip_match": args.skip_match,
        },
        "artifacts": {
            "directory": str(out_dir),
            "json": str(out_dir / "result.json"),
            "report": str(out_dir / "validation-report.md"),
        },
        "failure_reasons": preflight_failures,
    }
    result["promotion_checklist"] = build_promotion_checklist(result)
    result["failure_reasons"].extend(collect_failures(result))
    required_checks_pass = not any(
        not item["pass"]
        for item in result["promotion_checklist"]
        if item["name"] not in STATISTICAL_CHECK_NAMES
    )
    result["pass"] = len(result["failure_reasons"]) == 0 and required_checks_pass

    write_json(out_dir / "result.json", result)
    write_markdown_report(out_dir / "validation-report.md", result)

    print(f"wrote {out_dir / 'result.json'}")
    print(f"wrote {out_dir / 'validation-report.md'}")
    if result["failure_reasons"]:
        for failure in result["failure_reasons"]:
            print(f"failure: {failure}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    return run_gate(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
