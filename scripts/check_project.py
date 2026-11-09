#!/usr/bin/env python3
"""Read-only, offline readiness report for the maintained temporal workflow."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "numpy": "numpy", "ee": "earthengine-api", "rasterio": "rasterio",
    "matplotlib": "matplotlib", "pyproj": "pyproj", "requests": "requests",
    "pytest": "pytest", "PIL": "pillow",
}


def inspect_project(root=ROOT, cache=None):
    root = Path(root)
    cache = Path(cache) if cache is not None else root / "data/temporal/clayton_20220722.npz"
    dependencies = {}
    for module, distribution in PACKAGES.items():
        available = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(distribution) if available else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        dependencies[distribution] = {"available": available, "version": version}

    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                    text=True, timeout=10, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    cache_info = {"path": str(cache.resolve()), "exists": cache.is_file(), "sha256": None}
    if cache.is_file():
        digest = hashlib.sha256()
        with cache.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        cache_info["sha256"] = digest.hexdigest()
    missing = [name for name, info in dependencies.items() if not info["available"]]
    return {
        "python": {"executable": sys.executable, "version": sys.version.split()[0],
                   "supported": sys.version_info >= (3, 10)},
        "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "status": git("status", "--short")},
        "dependencies": dependencies, "cache": cache_info,
        "offline_prerequisites_present": sys.version_info >= (3, 10) and not missing and cache_info["exists"],
        "missing_dependencies": missing,
        "scope": "Presence and provenance only; imports, cache validity, tests and EE access are not verified.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Observation cache to fingerprint")
    parser.add_argument("--json", action="store_true", help="Print machine-readable report")
    args = parser.parse_args()
    report = inspect_project(cache=args.input)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python: {report['python']['executable']} ({report['python']['version']})")
        print(f"Git commit: {report['git']['commit'] or 'unavailable'}")
        print("Missing packages: " + (", ".join(report["missing_dependencies"]) or "none"))
        print(f"Observation cache: {report['cache']['path']}")
        print("Cache SHA-256: " + (report["cache"]["sha256"] or "missing"))
        print(report["scope"])
    return 0 if report["offline_prerequisites_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
