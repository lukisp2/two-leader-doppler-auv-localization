#!/usr/bin/env python3
"""Fail fast unless the frozen reference interpreter and packages are active."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


EXPECTED = {
    "python": "3.11.9",
    "numpy": "2.3.3",
    "gymnasium": "1.2.3",
}


def main() -> int:
    actual = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "gymnasium": importlib.metadata.version("gymnasium"),
    }
    problems = [
        f"{name}: expected {EXPECTED[name]}, found {version}"
        for name, version in actual.items()
        if version != EXPECTED[name]
    ]
    try:
        importlib.metadata.version("numba")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        problems.append("numba is installed; remove it for the reference NumPy path")

    for name in ("python", "numpy", "gymnasium"):
        print(f"{name}={actual[name]}")

    if problems:
        print("Reference-environment check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Reference-environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

