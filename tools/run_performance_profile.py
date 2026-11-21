#!/usr/bin/env python3
"""Run LocalChessEngine release, PGO and profiling evidence for TASK-045."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "perf" / "task-045"
DEFAULT_RELEASE_ENGINE = REPO_ROOT / "build" / "release" / "engine" / "localchessengine"
DEFAULT_PGO_GENERATE_ENGINE = REPO_ROOT / "build" / "pgo-generate" / "engine" / "localchessengine"
DEFAULT_PGO_USE_ENGINE = REPO_ROOT / "build" / "pgo-use" / "engine" / "localchessengine"
DEFAULT_EXPECTED_NODES = 18417
OUTPUT_TAIL_LIMIT = 12_000


@dataclass(frozen=True)
class BenchRun:
    label: str
    command: list[str]
    elapsed_ms: int
    returncode: int
    nodes: int | None
    nps: int | None
    output_tail: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and self.nodes is not None and self.nps is not None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def output_tail(text: str) -> str:
    if len(text) <= OUTPUT_TAIL_LIMIT:
        return text
    return text[-OUTPUT_TAIL_LIMIT:]


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(
    name: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    required: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    elapsed_ms = round((time.monotonic() - started) * 1000)
    return {
        "name": name,
        "event": "validation.command",
        "command": command,
        "command_text": command_text(command),
        "required": required,
        "pass": completed.returncode == 0,
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


def run_bench(label: str, engine: Path, *, env: dict[str, str] | None = None) -> BenchRun:
    command = [str(engine), "bench"]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    elapsed_ms = round((time.monotonic() - started) * 1000)
    parsed = parse_bench_output(completed.stdout)
    return BenchRun(
        label=label,
        command=command,
        elapsed_ms=elapsed_ms,
        returncode=completed.returncode,
        nodes=None if parsed is None else parsed["nodes"],
        nps=None if parsed is None else parsed["nps"],
        output_tail=output_tail(completed.stdout),
    )


def bench_record(run: BenchRun, expected_nodes: int) -> dict[str, Any]:
    return {
        "event": "engine.bench.result",
        "label": run.label,
        "command": run.command,
        "command_text": command_text(run.command),
        "elapsed_ms": run.elapsed_ms,
        "returncode": run.returncode,
        "expected_nodes": expected_nodes,
        "actual_nodes": run.nodes,
        "nps": run.nps,
        "pass": run.passed and run.nodes == expected_nodes,
        "output_tail": run.output_tail,
    }


def run_python_timing(
    release_engine: Path,
    pgo_engine: Path,
    *,
    runs: int,
    expected_nodes: int,
) -> dict[str, Any]:
    timings: dict[str, list[dict[str, Any]]] = {"release": [], "pgo-use": []}
    for label, engine in (("release", release_engine), ("pgo-use", pgo_engine)):
        for index in range(runs):
            run = run_bench(label, engine)
            timings[label].append(
                {
                    "run": index + 1,
                    "elapsed_ms": run.elapsed_ms,
                    "nodes": run.nodes,
                    "nps": run.nps,
                    "pass": run.passed and run.nodes == expected_nodes,
                }
            )

    summary = {}
    for label, entries in timings.items():
        passed_entries = [entry for entry in entries if entry["pass"]]
        nps_values = [entry["nps"] for entry in entries if entry["nps"] is not None]
        summary[label] = {
            "runs": len(entries),
            "pass": len(passed_entries) == len(entries),
            "mean_elapsed_ms": mean(entry["elapsed_ms"] for entry in entries),
            "mean_nps": mean(nps_values) if nps_values else None,
        }

    release_mean = summary["release"]["mean_elapsed_ms"]
    pgo_mean = summary["pgo-use"]["mean_elapsed_ms"]
    return {
        "event": "performance.python_timing",
        "hyperfine_available": False,
        "runs": runs,
        "summary": summary,
        "speedup_ratio": None if pgo_mean == 0 else release_mean / pgo_mean,
        "timings": timings,
    }


def run_hyperfine(
    hyperfine: str,
    release_engine: Path,
    pgo_engine: Path,
    out_dir: Path,
    *,
    runs: int,
) -> dict[str, Any]:
    json_path = out_dir / "hyperfine.json"
    markdown_path = out_dir / "hyperfine.md"
    release_command = f"{shlex.quote(str(release_engine))} bench"
    pgo_command = f"{shlex.quote(str(pgo_engine))} bench"
    command = [
        hyperfine,
        "--warmup",
        "1",
        "--runs",
        str(runs),
        "--export-json",
        str(json_path),
        "--export-markdown",
        str(markdown_path),
        "--command-name",
        "release",
        release_command,
        "--command-name",
        "pgo-use",
        pgo_command,
    ]
    record = run_command("hyperfine release vs pgo-use", command, required=False)
    return {
        "event": "performance.hyperfine",
        "hyperfine_available": True,
        "command": record,
        "json": relpath(json_path),
        "markdown": relpath(markdown_path),
        "pass": record["pass"] and json_path.is_file() and markdown_path.is_file(),
    }


def xcrun_find(tool: str) -> str | None:
    completed = subprocess.run(
        ["xcrun", "--find", tool],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode != 0:
        return None
    path = completed.stdout.strip()
    return path or None


def run_xctrace(engine: Path, out_dir: Path, template: str) -> dict[str, Any]:
    trace_path = out_dir / "pgo-use-bench.trace"
    xctrace = xcrun_find("xctrace")
    if xctrace is None:
        return {
            "event": "performance.xctrace",
            "available": False,
            "pass": True,
            "trace": None,
            "reason": "xcrun could not find xctrace; install/full-select Xcode to collect traces",
        }

    if trace_path.exists():
        shutil.rmtree(trace_path)

    command = [
        "xcrun",
        "xctrace",
        "record",
        "--template",
        template,
        "--no-prompt",
        "--output",
        str(trace_path),
        "--target-stdout",
        "-",
        "--target-stderr",
        "-",
        "--launch",
        "--",
        str(engine),
        "bench",
    ]
    record = run_command("xctrace pgo-use bench", command, required=False)
    return {
        "event": "performance.xctrace",
        "available": True,
        "pass": record["pass"] and trace_path.exists(),
        "trace": relpath(trace_path) if trace_path.exists() else None,
        "command": record,
    }


def merge_profiles(raw_profiles: list[Path], merged_profile: Path) -> dict[str, Any]:
    merged_profile.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "xcrun",
        "llvm-profdata",
        "merge",
        "--output",
        str(merged_profile),
        *[str(profile) for profile in raw_profiles],
    ]
    return run_command("merge LLVM PGO profiles", command)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# TASK-045 Performance Profile Report",
        "",
        f"- Date UTC: {result['date_utc']}",
        f"- Overall result: {'PASS' if result['pass'] else 'FAIL'}",
        f"- Expected bench nodes: `{result['expected_nodes']}`",
        f"- Merged profile: `{result['pgo']['merged_profile']}`",
        "",
        "## Commands",
        "",
    ]
    for command in result["commands"]:
        status = "PASS" if command["pass"] else "FAIL"
        required = "required" if command.get("required", True) else "optional"
        lines.append(
            f"- {status} ({required}) `{command['name']}` in `{command['elapsed_ms']}` ms: "
            f"`{command.get('command_text', '')}`"
        )

    lines.extend(["", "## Bench Signatures", ""])
    for bench in result["benches"]:
        status = "PASS" if bench["pass"] else "FAIL"
        lines.append(
            f"- {status} `{bench['label']}` nodes `{bench['actual_nodes']}`, "
            f"nps `{bench['nps']}`, elapsed `{bench['elapsed_ms']}` ms."
        )

    comparison = result["comparison"]
    lines.extend(["", "## Comparison", ""])
    if comparison["hyperfine_available"]:
        lines.append(f"- hyperfine: available; JSON `{comparison['json']}`; Markdown `{comparison['markdown']}`.")
    else:
        lines.append(
            "- hyperfine: unavailable; used Python timing fallback. "
            f"Speedup ratio release/pgo-use: `{comparison['speedup_ratio']}`."
        )

    xctrace = result["xctrace"]
    lines.extend(["", "## Instruments / xctrace", ""])
    if xctrace["available"]:
        status = "PASS" if xctrace["pass"] else "FAIL"
        lines.append(f"- {status} xctrace trace: `{xctrace.get('trace')}`.")
    else:
        lines.append(f"- SKIP xctrace: {xctrace['reason']}.")

    if result["failure_reasons"]:
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend(f"- {reason}" for reason in result["failure_reasons"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--expected-nodes", type=int, default=DEFAULT_EXPECTED_NODES)
    parser.add_argument("--release-engine", type=Path, default=DEFAULT_RELEASE_ENGINE)
    parser.add_argument("--pgo-generate-engine", type=Path, default=DEFAULT_PGO_GENERATE_ENGINE)
    parser.add_argument("--pgo-use-engine", type=Path, default=DEFAULT_PGO_USE_ENGINE)
    parser.add_argument("--pgo-profile", type=Path)
    parser.add_argument("--pgo-training-runs", type=int, default=3)
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--hyperfine-runs", type=int, default=5)
    parser.add_argument("--xctrace-template", default="Time Profiler")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw-profraw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    merged_profile = resolve_path(args.pgo_profile) if args.pgo_profile else (out_dir / "merged.profdata")

    release_engine = resolve_path(args.release_engine)
    pgo_generate_engine = resolve_path(args.pgo_generate_engine)
    pgo_use_engine = resolve_path(args.pgo_use_engine)

    commands: list[dict[str, Any]] = []
    benches: list[dict[str, Any]] = []
    failure_reasons: list[str] = []

    commands.append(run_command("release configure", ["cmake", "--preset", "release"]))
    commands.append(run_command("release build engine", ["cmake", "--build", "--preset", "release", "--target", "localchessengine"]))
    release_bench = run_bench("release", release_engine)
    benches.append(bench_record(release_bench, args.expected_nodes))

    commands.append(run_command("pgo-generate configure", ["cmake", "--preset", "pgo-generate"]))
    commands.append(
        run_command(
            "pgo-generate build engine",
            ["cmake", "--build", "--preset", "pgo-generate", "--target", "localchessengine"],
        )
    )

    pgo_env = os.environ.copy()
    pgo_env["LLVM_PROFILE_FILE"] = str(raw_dir / "bench-%p-%m.profraw")
    for index in range(args.pgo_training_runs):
        training_bench = run_bench(f"pgo-generate-training-{index + 1}", pgo_generate_engine, env=pgo_env)
        benches.append(bench_record(training_bench, args.expected_nodes))

    raw_profiles = sorted(raw_dir.glob("*.profraw"))
    merge_record = merge_profiles(raw_profiles, merged_profile) if raw_profiles else {
        "name": "merge LLVM PGO profiles",
        "event": "validation.command",
        "command": [],
        "command_text": "",
        "required": True,
        "pass": False,
        "returncode": 1,
        "elapsed_ms": 0,
        "output_tail": "no .profraw files were generated",
    }
    commands.append(merge_record)

    commands.append(
        run_command(
            "pgo-use configure",
            [
                "cmake",
                "--preset",
                "pgo-use",
                f"-DLOCALCHESSENGINE_PGO_PROFILE={merged_profile}",
            ],
        )
    )
    commands.append(
        run_command(
            "pgo-use build engine",
            ["cmake", "--build", "--preset", "pgo-use", "--target", "localchessengine"],
        )
    )
    pgo_use_bench = run_bench("pgo-use", pgo_use_engine)
    benches.append(bench_record(pgo_use_bench, args.expected_nodes))

    hyperfine = shutil.which("hyperfine")
    if hyperfine is not None:
        comparison = run_hyperfine(
            hyperfine,
            release_engine,
            pgo_use_engine,
            out_dir,
            runs=args.hyperfine_runs,
        )
        commands.append(comparison["command"])
    else:
        comparison = run_python_timing(
            release_engine,
            pgo_use_engine,
            runs=args.timing_runs,
            expected_nodes=args.expected_nodes,
        )

    xctrace_result = run_xctrace(pgo_use_engine, out_dir, args.xctrace_template)
    if xctrace_result.get("command") is not None:
        commands.append(xctrace_result["command"])

    for command in commands:
        if command.get("required", True) and not command.get("pass", False):
            failure_reasons.append(f"command failed: {command['name']}")
    for bench in benches:
        if not bench["pass"]:
            failure_reasons.append(
                f"bench signature mismatch for {bench['label']}: "
                f"expected {bench['expected_nodes']} actual {bench['actual_nodes']}"
            )

    result = {
        "event": "performance.profile.result",
        "date_utc": utc_now(),
        "pass": not failure_reasons,
        "expected_nodes": args.expected_nodes,
        "commands": commands,
        "benches": benches,
        "pgo": {
            "mode_generate_preset": "pgo-generate",
            "mode_use_preset": "pgo-use",
            "raw_profiles": [relpath(path) for path in raw_profiles],
            "merged_profile": relpath(merged_profile),
            "profile_exists": merged_profile.is_file(),
        },
        "comparison": comparison,
        "xctrace": xctrace_result,
        "failure_reasons": failure_reasons,
        "artifacts": {
            "json": relpath(out_dir / "result.json"),
            "markdown": relpath(out_dir / "performance-report.md"),
        },
    }
    write_json(out_dir / "result.json", result)
    write_markdown(out_dir / "performance-report.md", result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
