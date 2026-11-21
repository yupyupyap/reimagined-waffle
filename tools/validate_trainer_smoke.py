#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER_ROOT = REPO_ROOT / "trainer"
sys.path.insert(0, str(TRAINER_ROOT / "src"))

from localchessengine_trainer.smoke_train import SmokeTrainConfig, run_smoke_training  # noqa: E402


def _load_pyproject() -> dict:
    with (TRAINER_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _validate_pyproject(pyproject: dict) -> None:
    project = pyproject.get("project", {})
    dependencies = project.get("dependencies", [])
    if project.get("name") != "localchessengine-trainer":
        raise ValueError("trainer pyproject has unexpected project.name")
    if not any(str(dependency).startswith("torch") for dependency in dependencies):
        raise ValueError("trainer pyproject must declare a torch dependency")
    if not any(str(dependency).startswith("numpy") for dependency in dependencies):
        raise ValueError("trainer pyproject must declare a numpy dependency")
    scripts = project.get("scripts", {})
    if "localchessengine-trainer-smoke" not in scripts:
        raise ValueError("trainer pyproject must expose localchessengine-trainer-smoke")
    if "localchessengine-train" not in scripts:
        raise ValueError("trainer pyproject must expose localchessengine-train")


def _validate_log(path: Path) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events = {record.get("event") for record in records}
    required = {
        "trainer.dependency.check",
        "trainer.epoch.metrics",
        "trainer.smoke_train.result",
    }
    missing = required - events
    if missing:
        raise ValueError(f"trainer smoke log is missing events: {sorted(missing)}")
    dependency = next(
        record for record in records if record.get("event") == "trainer.dependency.check"
    )
    if dependency.get("dependency") != "torch":
        raise ValueError("trainer dependency log does not identify torch")
    result = next(
        record for record in records if record.get("event") == "trainer.smoke_train.result"
    )
    if result.get("status") != "pass":
        raise ValueError("trainer smoke result did not pass")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the TASK-028 trainer smoke.")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="cpu")
    parser.add_argument("--log", type=Path, default=REPO_ROOT / "build/trainer/task-028-smoke.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--features-per-position", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_pyproject(_load_pyproject())
        result = run_smoke_training(
            SmokeTrainConfig(
                device=args.device,
                log_path=args.log,
                dataset_id="synthetic-task-028-validation",
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                features_per_position=args.features_per_position,
            )
        )
        _validate_log(args.log)
    except Exception as exc:
        print(f"FAIL trainer smoke: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"status": "pass", "log": str(args.log), **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
