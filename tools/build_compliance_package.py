#!/usr/bin/env python3
"""Build and validate LocalChessEngine legal/supply-chain release materials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP = REPO_ROOT / "build" / "release" / "gui" / "localchessengine_gui.app"
DEFAULT_ENGINE = REPO_ROOT / "build" / "release" / "engine" / "localchessengine"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "compliance"
LEGAL_DOCS = (
    "LICENSE",
    "COPYING.LGPL-3.0.txt",
    "COPYING.GPL-3.0.txt",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "docs/compliance/privacy.md",
    "docs/compliance/qt-relink.md",
    "docs/compliance/release-checklist.md",
)
EXPECTED_NOTICE_TOKENS = (
    "Qt",
    "LGPLv3",
    "double-conversion",
    "pcre2",
    "libb2",
    "md4c",
    "zlib",
    "doctest",
    "pybind11",
    "PyTorch",
    "NumPy",
    "Hugging Face Datasets",
    "MIT",
)
SOURCE_ARCHIVE_PATTERNS: dict[str, tuple[str, ...]] = {
    "qtbase": ("qtbase-everywhere-src-*.tar.*",),
    "qtsvg": ("qtsvg-everywhere-src-*.tar.*",),
    "double-conversion": ("google-double-conversion-v*.tar.*",),
    "pcre2": ("PCRE2Project-pcre2-pcre2-*.tar.*",),
    "libb2": ("BLAKE2-libb2-*.tar.*",),
    "md4c": ("mity-md4c-release-*.tar.*",),
    "zlib": ("madler-zlib-v*.tar.*",),
    "doctest": ("doctest-doctest-v*.tar.*",),
}
VCPKG_LICENSE_PORTS = (
    "qtbase",
    "qtsvg",
    "double-conversion",
    "pcre2",
    "libb2",
    "md4c",
    "zlib",
    "doctest",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], *, required: bool = True) -> dict[str, Any]:
    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "event": "compliance.command",
        "command": command,
        "required": required,
        "returncode": completed.returncode,
        "pass": completed.returncode == 0,
        "elapsed_ms": round((time.monotonic() - start) * 1000),
        "output_tail": completed.stdout[-12000:],
    }


def project_version() -> str:
    match = re.search(
        r"project\s*\(\s*LocalChessEngine\s+VERSION\s+([0-9][^\s)]*)",
        (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8"),
    )
    return match.group(1) if match else "0.2.0"


def resolve_vcpkg_installed(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    preset = json.loads((REPO_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    for configure in preset.get("configurePresets", []):
        cache = configure.get("cacheVariables", {})
        value = cache.get("VCPKG_INSTALLED_DIR")
        if isinstance(value, dict):
            value = value.get("value")
        if value:
            return Path(str(value)).resolve()
    return Path("/tmp/localchessengine-vcpkg-installed")


def resolve_vcpkg_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    env_root = os.environ.get("VCPKG_ROOT")
    if env_root:
        return Path(env_root).resolve()
    preset = json.loads((REPO_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    for configure in preset.get("configurePresets", []):
        toolchain = configure.get("toolchainFile")
        if isinstance(toolchain, str) and toolchain.endswith("vcpkg.cmake"):
            return Path(toolchain).resolve().parents[2]
    return Path("/tmp/localchessengine-vcpkg-root")


def resolve_triplet(explicit: str | None) -> str:
    if explicit:
        return explicit
    preset = json.loads((REPO_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    for configure in preset.get("configurePresets", []):
        cache = configure.get("cacheVariables", {})
        value = cache.get("VCPKG_TARGET_TRIPLET")
        if isinstance(value, str):
            return value
    return "arm64-osx-lce-dynamic"


def bundle_dylibs(app: Path) -> list[Path]:
    if not app.exists():
        return []
    roots = [app / "Contents" / "Frameworks", app / "Contents" / "PlugIns"]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*.dylib") if path.is_file())
    return sorted(files)


def classify_bundle_library(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith("libqt6svg"):
        return "qtsvg"
    if name.startswith("libqt6") or name.startswith("libqcocoa") or name.startswith("libqmacstyle"):
        return "qtbase"
    if "double-conversion" in name:
        return "double-conversion"
    if "pcre2" in name:
        return "pcre2"
    if "libb2" in name:
        return "libb2"
    if "md4c" in name:
        return "md4c"
    if re.match(r"libz(\.|-)", name):
        return "zlib"
    return None


def validate_legal_files() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for relative in LEGAL_DOCS:
        path = REPO_ROOT / relative
        results.append(
            {
                "event": "compliance.legal_file",
                "path": relative,
                "pass": path.is_file() and path.stat().st_size > 0,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8", errors="replace")
    for token in EXPECTED_NOTICE_TOKENS:
        results.append(
            {
                "event": "compliance.notice_token",
                "token": token,
                "pass": token.lower() in notices.lower(),
            }
        )
    return results


def validate_manifests() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    vcpkg = json.loads((REPO_ROOT / "vcpkg.json").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((REPO_ROOT / "trainer" / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = json.dumps(vcpkg.get("dependencies", []))
    build_requires = pyproject.get("build-system", {}).get("requires", [])
    project = pyproject.get("project", {})
    runtime_dependencies = project.get("dependencies", [])
    optional = project.get("optional-dependencies", {})
    results.extend(
        [
            {
                "event": "compliance.manifest_dependency",
                "manifest": "vcpkg.json",
                "dependency": "qtbase",
                "pass": "qtbase" in dependencies,
            },
            {
                "event": "compliance.manifest_dependency",
                "manifest": "vcpkg.json",
                "dependency": "qtsvg",
                "pass": "qtsvg" in dependencies,
            },
            {
                "event": "compliance.manifest_dependency",
                "manifest": "vcpkg.json",
                "dependency": "doctest",
                "pass": "doctest" in dependencies,
            },
            {
                "event": "compliance.manifest_dependency",
                "manifest": "trainer/pyproject.toml",
                "dependency": "pybind11",
                "pass": any(str(dep).startswith("pybind11") for dep in build_requires),
            },
            {
                "event": "compliance.manifest_dependency",
                "manifest": "trainer/pyproject.toml",
                "dependency": "torch",
                "pass": any(str(dep).startswith("torch") for dep in runtime_dependencies),
            },
            {
                "event": "compliance.manifest_dependency",
                "manifest": "trainer/pyproject.toml",
                "dependency": "numpy",
                "pass": any(str(dep).startswith("numpy") for dep in runtime_dependencies),
            },
            {
                "event": "compliance.manifest_dependency",
                "manifest": "trainer/pyproject.toml",
                "dependency": "datasets",
                "pass": any(str(dep).startswith("datasets") for dep in optional.get("fen-pool", [])),
            },
        ]
    )
    return results


def validate_linkage(app: Path, engine: Path, bundle_files: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    gui_exe = app / "Contents" / "MacOS" / "localchessengine_gui"
    if engine.exists():
        engine_otool = run_command(["otool", "-L", str(engine)])
        engine_output = engine_otool["output_tail"]
        results.append(engine_otool)
        results.append(
            {
                "event": "compliance.engine_qt_free",
                "path": relpath(engine),
                "pass": "Qt6" not in engine_output,
            }
        )
    else:
        results.append({"event": "compliance.engine_exists", "path": relpath(engine), "pass": False})
    if gui_exe.exists():
        gui_otool = run_command(["otool", "-L", str(gui_exe)])
        gui_output = gui_otool["output_tail"]
        results.append(gui_otool)
        results.append(
            {
                "event": "compliance.gui_dynamic_qt",
                "path": relpath(gui_exe),
                "pass": "Qt6Core" in gui_output
                and "Qt6Gui" in gui_output
                and "Qt6Widgets" in gui_output
                and "Qt6Svg" in gui_output
                and "@executable_path/../Frameworks" in gui_output,
            }
        )
    else:
        results.append({"event": "compliance.gui_exists", "path": relpath(gui_exe), "pass": False})
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8", errors="replace").lower()
    for dylib in bundle_files:
        port = classify_bundle_library(dylib)
        results.append(
            {
                "event": "compliance.bundle_dylib_notice",
                "path": relpath(dylib),
                "port": port,
                "pass": port is not None and port.lower() in notices,
            }
        )
    return results


def copy_legal_materials(out_dir: Path, vcpkg_installed: Path, triplet: str) -> tuple[Path, list[dict[str, Any]]]:
    legal_dir = out_dir / "Legal"
    if legal_dir.exists():
        shutil.rmtree(legal_dir)
    legal_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for relative in LEGAL_DOCS:
        source = REPO_ROOT / relative
        destination = legal_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "event": "compliance.legal_copy",
                "source": relative,
                "destination": relpath(destination),
                "sha256": sha256_file(destination),
            }
        )

    share_root = vcpkg_installed / triplet / "share"
    for port in VCPKG_LICENSE_PORTS:
        port_dir = share_root / port
        if not port_dir.exists():
            records.append({"event": "compliance.vcpkg_license_copy", "port": port, "pass": False})
            continue
        destination = legal_dir / "licenses" / "vcpkg" / port
        destination.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in ("copyright", "vcpkg.spdx.json"):
            source = port_dir / name
            if source.exists():
                shutil.copy2(source, destination / name)
                copied.append(name)
        records.append(
            {
                "event": "compliance.vcpkg_license_copy",
                "port": port,
                "destination": relpath(destination),
                "files": copied,
                "pass": "copyright" in copied or "vcpkg.spdx.json" in copied,
            }
        )
    return legal_dir, records


def copy_source_archives(out_dir: Path, vcpkg_root: Path, version: str) -> tuple[Path, list[dict[str, Any]]]:
    source_stage = out_dir / "source-inputs"
    if source_stage.exists():
        shutil.rmtree(source_stage)
    archives_dir = source_stage / "vcpkg-source-archives"
    archives_dir.mkdir(parents=True)
    downloads = vcpkg_root / "downloads"
    records: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    for port, patterns in SOURCE_ARCHIVE_PATTERNS.items():
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(sorted(downloads.glob(pattern)))
        matches = [path for path in matches if path.is_file() and not path.name.endswith(".part")]
        if not matches:
            records.append({"event": "compliance.source_archive", "port": port, "pass": False})
            continue
        source = matches[0]
        destination = archives_dir / source.name
        shutil.copy2(source, destination)
        digest = sha256_file(destination)
        source_hashes.append(f"{digest}  vcpkg-source-archives/{destination.name}")
        records.append(
            {
                "event": "compliance.source_archive",
                "port": port,
                "source": str(source),
                "destination": relpath(destination),
                "sha256": digest,
                "pass": True,
            }
        )
    (source_stage / "README.md").write_text(
        "\n".join(
            [
                f"# LocalChessEngine v{version} Compliance Source Inputs",
                "",
                "This directory contains source archives collected from the local vcpkg downloads",
                f"cache for Qt and redistributed dependency sources used by the macOS v{version}",
                "distribution package. It is generated by `tools/build_compliance_package.py`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (source_stage / "SHA256SUMS").write_text("\n".join(source_hashes) + "\n", encoding="utf-8")
    return source_stage, records


def create_source_tarball(source_stage: Path, out_dir: Path, version: str) -> tuple[Path, dict[str, Any]]:
    archive = out_dir / f"LocalChessEngine-{version}-compliance-sources.tar.zst"
    tar_path = out_dir / f"LocalChessEngine-{version}-compliance-sources.tar"
    if archive.exists():
        archive.unlink()
    if tar_path.exists():
        tar_path.unlink()
    tar_record = run_command(["tar", "-cf", str(tar_path), "-C", str(source_stage.parent), source_stage.name])
    zstd_record = run_command(["zstd", "-T0", "-19", "-f", "-q", str(tar_path), "-o", str(archive)])
    tar_path.unlink(missing_ok=True)
    return archive, {
        "event": "compliance.source_tarball",
        "path": relpath(archive),
        "sha256": sha256_file(archive) if archive.exists() else None,
        "bytes": archive.stat().st_size if archive.exists() else 0,
        "pass": tar_record["pass"] and zstd_record["pass"] and archive.exists(),
        "commands": [tar_record, zstd_record],
    }


def write_hashes(out_dir: Path, archive: Path) -> dict[str, Any]:
    entries: list[str] = []
    for path in sorted((out_dir / "Legal").rglob("*")):
        if path.is_file():
            entries.append(f"{sha256_file(path)}  {relpath(path)}")
    if archive.exists():
        entries.append(f"{sha256_file(archive)}  {relpath(archive)}")
    hashes = out_dir / "hashes.sha256"
    hashes.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return {
        "event": "compliance.hashes",
        "path": relpath(hashes),
        "entries": len(entries),
        "pass": hashes.exists() and len(entries) > 0,
    }


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    version = project_version()
    app = args.app.resolve()
    engine = args.engine.resolve()
    out_dir = args.out_dir.resolve()
    vcpkg_installed = resolve_vcpkg_installed(args.vcpkg_installed)
    vcpkg_root = resolve_vcpkg_root(args.vcpkg_root)
    triplet = resolve_triplet(args.triplet)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_files = bundle_dylibs(app)
    results: list[dict[str, Any]] = []
    results.extend(validate_legal_files())
    results.extend(validate_manifests())
    results.extend(validate_linkage(app, engine, bundle_files))
    legal_dir, legal_records = copy_legal_materials(out_dir, vcpkg_installed, triplet)
    results.extend(legal_records)
    source_stage, source_records = copy_source_archives(out_dir, vcpkg_root, version)
    results.extend(source_records)
    archive, archive_record = create_source_tarball(source_stage, out_dir, version)
    results.append(archive_record)
    results.append(write_hashes(out_dir, archive))

    python_project = tomllib.loads((REPO_ROOT / "trainer" / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = {
        "event": "compliance.package",
        "generated_at": utc_now(),
        "project_version": version,
        "app": relpath(app),
        "engine": relpath(engine),
        "out_dir": relpath(out_dir),
        "legal_dir": relpath(legal_dir),
        "source_archive": relpath(archive),
        "vcpkg_root": str(vcpkg_root),
        "vcpkg_installed": str(vcpkg_installed),
        "triplet": triplet,
        "bundle_dylibs": [
            {"path": relpath(path), "port": classify_bundle_library(path), "sha256": sha256_file(path)}
            for path in bundle_files
        ],
        "vcpkg_ports": list(VCPKG_LICENSE_PORTS),
        "python_build_requires": python_project.get("build-system", {}).get("requires", []),
        "python_dependencies": python_project.get("project", {}).get("dependencies", []),
        "python_optional_dependencies": python_project.get("project", {}).get("optional-dependencies", {}),
        "results": results,
    }
    manifest["pass"] = all(bool(record.get("pass", True)) for record in results)
    report_path = out_dir / "compliance-report.json"
    report_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit non-zero if any compliance check fails.")
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vcpkg-installed", type=Path)
    parser.add_argument("--vcpkg-root", type=Path)
    parser.add_argument("--triplet")
    args = parser.parse_args()

    manifest = build_package(args)
    print(json.dumps({"pass": manifest["pass"], "out_dir": manifest["out_dir"]}, indent=2))
    if args.check and not manifest["pass"]:
        failed = [record for record in manifest["results"] if not bool(record.get("pass", True))]
        print(json.dumps({"failed": failed}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
