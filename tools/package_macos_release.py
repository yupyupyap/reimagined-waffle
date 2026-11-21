#!/usr/bin/env python3
"""Build, sign, notarize, and verify the LocalChessEngine macOS release DMG."""

from __future__ import annotations

import argparse
import json
import plistlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def project_version() -> str:
    match = re.search(
        r"project\s*\(\s*LocalChessEngine\s+VERSION\s+([0-9][^\s)]*)",
        (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8"),
    )
    return match.group(1) if match else "0.2.0"


VERSION = project_version()
DEFAULT_APP = REPO_ROOT / "build" / "release" / "gui" / "localchessengine_gui.app"
DEFAULT_ENGINE = REPO_ROOT / "build" / "release" / "engine" / "localchessengine"
DEFAULT_COMPLIANCE_DIR = REPO_ROOT / "build" / "compliance"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "release-package"
DEFAULT_NOTARY_PROFILE = "localchessengine-notary"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def run_command(
    name: str,
    command: list[str],
    *,
    required: bool = True,
    cwd: Path = REPO_ROOT,
) -> dict[str, Any]:
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "event": "release.command",
        "name": name,
        "command": command,
        "command_text": command_text(command),
        "required": required,
        "returncode": completed.returncode,
        "pass": completed.returncode == 0,
        "elapsed_ms": round((time.monotonic() - start) * 1000),
        "output_tail": completed.stdout[-20000:],
    }


def discover_developer_id() -> str | None:
    result = run_command("codesigning identity discovery", ["security", "find-identity", "-v", "-p", "codesigning"])
    matches = re.findall(r'"(Developer ID Application: [^"]+)"', result["output_tail"])
    return matches[0] if matches else None


def copy_release_inputs(app: Path, engine: Path, compliance_dir: Path, out_dir: Path) -> dict[str, Any]:
    staging = out_dir / f"LocalChessEngine-{VERSION}"
    package_app = staging / "LocalChessEngine.app"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(app, package_app, symlinks=True)

    engine_destination = package_app / "Contents" / "MacOS" / "localchessengine"
    shutil.copy2(engine, engine_destination)
    engine_destination.chmod(engine_destination.stat().st_mode | 0o755)

    legal_source = compliance_dir / "Legal"
    if not legal_source.exists():
        raise FileNotFoundError(f"Missing compliance Legal directory: {legal_source}")
    legal_bundle_destination = package_app / "Contents" / "Resources" / "Legal"
    legal_bundle_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(legal_source, legal_bundle_destination)
    shutil.copytree(legal_source, staging / "Legal")

    archive_candidates = sorted(compliance_dir.glob("LocalChessEngine-*-compliance-sources.tar.zst"))
    if not archive_candidates:
        raise FileNotFoundError(f"Missing compliance source archive in {compliance_dir}")
    source_archive = archive_candidates[-1]
    shutil.copy2(source_archive, staging / source_archive.name)

    return {
        "event": "release.staging",
        "staging": relpath(staging),
        "app": relpath(package_app),
        "embedded_engine": relpath(engine_destination),
        "legal_bundle": relpath(legal_bundle_destination),
        "source_archive": relpath(staging / source_archive.name),
        "pass": True,
    }


def nested_code_paths(app: Path) -> list[Path]:
    paths: list[Path] = []
    for root in (app / "Contents" / "Frameworks", app / "Contents" / "PlugIns"):
        if root.exists():
            paths.extend(path for path in root.rglob("*.dylib") if path.is_file())
    paths.append(app / "Contents" / "MacOS" / "localchessengine")
    paths.append(app / "Contents" / "MacOS" / "localchessengine_gui")
    return sorted(paths, key=lambda path: len(path.parts), reverse=True)


def sign_path(path: Path, identity: str) -> dict[str, Any]:
    return run_command(
        f"codesign {relpath(path)}",
        [
            "codesign",
            "--force",
            "--timestamp",
            "--options",
            "runtime",
            "--sign",
            identity,
            str(path),
        ],
    )


def sign_release(app: Path, dmg: Path | None, identity: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in nested_code_paths(app):
        records.append(sign_path(path, identity))
    records.append(sign_path(app, identity))
    records.append(
        run_command(
            "codesign verify app",
            ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)],
        )
    )
    if dmg is not None:
        records.append(sign_path(dmg, identity))
    return records


def create_dmg(staging: Path, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    dmg = out_dir / f"LocalChessEngine-{VERSION}.dmg"
    if dmg.exists():
        dmg.unlink()
    # Avoid a macOS hdiutil failure observed when the temporary mounted volume
    # name contains spaces and matches the product/app display name.
    volume_name = f"LocalChessEngine-{VERSION}"
    record = run_command(
        "create dmg",
        [
            "hdiutil",
            "create",
            "-volname",
            volume_name,
            "-srcfolder",
            str(staging),
            "-ov",
            "-format",
            "UDZO",
            str(dmg),
        ],
    )
    return dmg, record


def parse_notary_submission_id(output: str) -> str | None:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        for key in ("id", "submissionId"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    match = re.search(r"\bid:\s*([0-9a-fA-F-]{20,})", output)
    return match.group(1) if match else None


def notarize(dmg: Path, profile: str, out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    preflight = run_command(
        "notary profile preflight",
        ["xcrun", "notarytool", "history", "--keychain-profile", profile, "--output-format", "json"],
    )
    records.append(preflight)
    if not preflight["pass"]:
        return records

    submit = run_command(
        "notary submit",
        [
            "xcrun",
            "notarytool",
            "submit",
            str(dmg),
            "--keychain-profile",
            profile,
            "--wait",
            "--output-format",
            "json",
        ],
    )
    records.append(submit)
    submission_id = parse_notary_submission_id(submit["output_tail"])
    (out_dir / "notary-submit.json").write_text(submit["output_tail"], encoding="utf-8")
    if submission_id:
        log = run_command(
            "notary log",
            [
                "xcrun",
                "notarytool",
                "log",
                submission_id,
                "--keychain-profile",
                profile,
                "--output-format",
                "json",
            ],
            required=not submit["pass"],
        )
        records.append(log)
        (out_dir / "notary-log.json").write_text(log["output_tail"], encoding="utf-8")
    if not submit["pass"]:
        return records
    records.append(run_command("stapler staple dmg", ["xcrun", "stapler", "staple", str(dmg)]))
    records.append(run_command("stapler validate dmg", ["xcrun", "stapler", "validate", str(dmg)]))
    return records


def mount_dmg(dmg: Path) -> tuple[Path | None, dict[str, Any]]:
    plist_path = Path("/tmp") / f"localchessengine-{int(time.time())}-attach.plist"
    record = run_command(
        "mount dmg",
        ["hdiutil", "attach", str(dmg), "-nobrowse", "-readonly", "-plist"],
    )
    if not record["pass"]:
        return None, record
    plist_path.write_text(record["output_tail"], encoding="utf-8")
    try:
        plist = plistlib.loads(record["output_tail"].encode("utf-8"))
    except Exception:
        return None, record
    mount_point = None
    for entity in plist.get("system-entities", []):
        value = entity.get("mount-point")
        if value:
            mount_point = Path(value)
            break
    return mount_point, record


def verify_release(app: Path, dmg: Path, *, run_gui_smoke: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.append(
        run_command(
            "codesign verify app",
            ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)],
        )
    )
    records.append(
        run_command(
            "spctl assess app",
            ["spctl", "--assess", "--type", "execute", "--verbose", str(app)],
        )
    )
    records.append(
        run_command(
            "embedded engine bench",
            [str(app / "Contents" / "MacOS" / "localchessengine"), "bench"],
        )
    )
    if run_gui_smoke:
        records.append(
            run_command(
                "gui smoke",
                [str(app / "Contents" / "MacOS" / "localchessengine_gui"), "--smoke-exit"],
            )
        )
    mount_point, mount_record = mount_dmg(dmg)
    records.append(mount_record)
    if mount_point is not None:
        mounted_app = mount_point / "LocalChessEngine.app"
        records.append(
            run_command(
                "mounted engine bench",
                [str(mounted_app / "Contents" / "MacOS" / "localchessengine"), "bench"],
            )
        )
        if run_gui_smoke:
            records.append(
                run_command(
                    "mounted gui smoke",
                    [str(mounted_app / "Contents" / "MacOS" / "localchessengine_gui"), "--smoke-exit"],
                )
            )
        records.append(run_command("detach dmg", ["hdiutil", "detach", str(mount_point)]))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", default=None)
    parser.add_argument("--notary-profile", default=DEFAULT_NOTARY_PROFILE)
    parser.add_argument("--skip-notarize", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-gui-smoke", action="store_true")
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--compliance-dir", type=Path, default=DEFAULT_COMPLIANCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    if not args.skip_build:
        records.append(
            run_command(
                "build release targets",
                ["cmake", "--build", "--preset", "release", "--target", "localchessengine", "localchessengine_gui"],
            )
        )
    records.append(
        run_command("build compliance package", ["python3", "tools/build_compliance_package.py", "--check"])
    )

    identity = args.identity or discover_developer_id()
    if not identity:
        records.append(
            {
                "event": "release.identity",
                "pass": False,
                "message": "No Developer ID Application identity found. Pass --identity explicitly.",
            }
        )
    else:
        records.append({"event": "release.identity", "identity": identity, "pass": True})

    try:
        staging_record = copy_release_inputs(args.app.resolve(), args.engine.resolve(), args.compliance_dir.resolve(), out_dir)
        records.append(staging_record)
        staging = Path(staging_record["staging"]).resolve()
        package_app = Path(staging_record["app"]).resolve()
    except Exception as exc:
        records.append({"event": "release.staging", "pass": False, "message": str(exc)})
        staging = out_dir / f"LocalChessEngine-{VERSION}"
        package_app = staging / "LocalChessEngine.app"

    if identity and package_app.exists():
        records.extend(sign_release(package_app, None, identity))
    dmg, dmg_record = create_dmg(staging, out_dir)
    records.append(dmg_record)
    if identity and dmg.exists():
        records.append(sign_path(dmg, identity))

    if args.skip_notarize:
        records.append({"event": "release.notarization", "pass": True, "skipped": True})
    elif dmg.exists():
        records.extend(notarize(dmg, args.notary_profile, out_dir))

    if dmg.exists() and package_app.exists():
        records.extend(verify_release(package_app, dmg, run_gui_smoke=not args.skip_gui_smoke))

    report = {
        "event": "release.package",
        "generated_at": utc_now(),
        "out_dir": relpath(out_dir),
        "dmg": relpath(dmg),
        "notary_profile": args.notary_profile,
        "skip_notarize": args.skip_notarize,
        "records": records,
    }
    report["pass"] = all(bool(record.get("pass", True)) for record in records)
    report_path = out_dir / "task-047-package-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": relpath(report_path), "dmg": relpath(dmg)}, indent=2))
    if not report["pass"]:
        failed = [record for record in records if not bool(record.get("pass", True))]
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
