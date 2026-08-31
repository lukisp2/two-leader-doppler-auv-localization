#!/usr/bin/env python3
"""Verify hashes and, by default, the exact logical release-file tree."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess


HASH_RE = re.compile(r"[0-9a-f]{64}\Z")

# These directories are local checkout/runtime metadata rather than release
# payload. They are ignored when checking an otherwise exact tree so the same
# command works in a clone and in an extracted archive. Apple archive debris
# is deliberately not allowed: __MACOSX and ._* fail exact verification.
LOCAL_METADATA_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "venv",
    }
)
FORBIDDEN_MANIFEST_DIRS = LOCAL_METADATA_DIRS | {"__MACOSX"}
LOCAL_METADATA_FILES = frozenset({".git"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_relative(rendered: str) -> PurePosixPath:
    relative = PurePosixPath(rendered)
    if (
        not rendered
        or rendered == "."
        or relative.is_absolute()
        or relative.as_posix() != rendered
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("path is not a canonical relative POSIX path")
    if "\n" in rendered or "\r" in rendered or "\\" in rendered:
        raise ValueError("path cannot be represented portably")
    return relative


def _forbidden_manifest_path(relative: PurePosixPath) -> bool:
    return (
        any(part in FORBIDDEN_MANIFEST_DIRS for part in relative.parts)
        or relative.name == ".DS_Store"
        or relative.name.startswith("._")
        or relative.name.startswith("SHA256SUMS")
        or relative.name.startswith(".SHA256SUMS")
    )


def _git_tree_files(root: Path) -> tuple[set[str] | None, list[str]]:
    """Return the logical tracked tree only when ``root`` is the Git top level."""

    failures: list[str] = []
    try:
        top_level = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        if (root / ".git").exists() or (root / ".git").is_symlink():
            return set(), ["cannot inspect tracked tree: Git executable not found"]
        return None, []
    if top_level.returncode != 0:
        if (root / ".git").exists() or (root / ".git").is_symlink():
            message = top_level.stderr.strip() or "git rev-parse failed"
            return set(), [f"cannot inspect tracked tree: {message}"]
        return None, []
    if Path(top_level.stdout.strip()).resolve() != root:
        return None, []

    listed = subprocess.run(
        ["git", "-C", os.fspath(root), "ls-files", "-z", "--cached", "--", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if listed.returncode != 0:
        message = listed.stderr.decode("utf-8", errors="replace").strip()
        return set(), [f"cannot list tracked files: {message}"]

    files: set[str] = set()
    for raw in listed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            rendered = raw.decode("utf-8", errors="strict")
            relative = _parse_relative(rendered)
        except (UnicodeDecodeError, ValueError) as exc:
            failures.append(f"invalid tracked path: {exc}")
            continue
        if not _forbidden_manifest_path(relative):
            files.add(relative.as_posix())
    return files, failures


def _tree_files(root: Path) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    failures: list[str] = []

    def walk_error(error: OSError) -> None:
        failures.append(f"cannot walk release tree: {error}")

    for directory, dirnames, filenames in os.walk(
        root, topdown=True, onerror=walk_error, followlinks=False
    ):
        directory_path = Path(directory)
        kept_dirs = []
        for name in sorted(dirnames):
            relative = (directory_path / name).relative_to(root).as_posix()
            if name in LOCAL_METADATA_DIRS:
                continue
            if name == "__MACOSX":
                failures.append(f"unexpected directory: {relative}")
            if (directory_path / name).is_symlink():
                files.add(relative)
                failures.append(f"unsupported symbolic link: {relative}")
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs

        for name in sorted(filenames):
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if name in LOCAL_METADATA_FILES:
                continue
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                failures.append(f"cannot inspect: {relative}: {exc}")
                continue
            files.add(relative)
            if stat.S_ISLNK(mode):
                failures.append(f"unsupported symbolic link: {relative}")
            elif not stat.S_ISREG(mode):
                failures.append(f"unsupported file type: {relative}")
    return files, failures


def _allowed_extra(value: str) -> str:
    try:
        relative = _parse_relative(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if _forbidden_manifest_path(relative):
        raise argparse.ArgumentTypeError(
            "excluded release metadata cannot be allowed as an extra path"
        )
    return relative.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify every manifest hash and reject unlisted payload files by default."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--no-exact-tree",
        action="store_true",
        help="verify listed hashes only; do not reject unlisted files",
    )
    parser.add_argument(
        "--allow-extra",
        action="append",
        default=[],
        type=_allowed_extra,
        metavar="RELATIVE_PATH",
        help="allow one exact unlisted path (repeat for multiple paths)",
    )
    args = parser.parse_args()
    manifest_argument = args.manifest.absolute()
    if manifest_argument.is_symlink() or not manifest_argument.is_file():
        print(f"FAIL manifest is not a regular file: {manifest_argument}")
        return 1
    manifest = manifest_argument.resolve()
    root = manifest.parent
    failures: list[str] = []
    expected_paths: dict[str, tuple[str, int]] = {}
    checked = 0

    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        print(f"FAIL cannot read manifest: {exc}")
        return 1

    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            expected, rendered = line.split("  ", 1)
        except ValueError:
            failures.append(f"line {line_number}: malformed entry")
            continue
        if not HASH_RE.fullmatch(expected):
            failures.append(f"line {line_number}: invalid SHA-256 digest")
            continue
        try:
            relative = _parse_relative(rendered)
        except ValueError as exc:
            failures.append(f"line {line_number}: invalid path: {exc}")
            continue
        relative_text = relative.as_posix()
        if _forbidden_manifest_path(relative):
            failures.append(
                f"line {line_number}: excluded path must not be manifested: {relative_text}"
            )
            continue
        if relative_text in expected_paths:
            first_line = expected_paths[relative_text][1]
            failures.append(
                f"line {line_number}: duplicate path (first listed on line {first_line}): "
                f"{relative_text}"
            )
            continue
        expected_paths[relative_text] = (expected, line_number)

        path = root.joinpath(*relative.parts)
        if path.is_symlink():
            failures.append(f"unsupported symbolic link: {relative_text}")
        elif not path.is_file():
            failures.append(f"missing: {relative_text}")
        else:
            try:
                actual = sha256(path)
            except OSError as exc:
                failures.append(f"cannot read: {relative_text}: {exc}")
            else:
                if actual != expected:
                    failures.append(f"hash mismatch: {relative_text}")
        checked += 1

    if not args.no_exact_tree:
        actual_paths, tree_failures = _git_tree_files(root)
        tree_kind = "tracked-Git"
        if actual_paths is None:
            actual_paths, tree_failures = _tree_files(root)
            tree_kind = "filesystem"
        failures.extend(tree_failures)
        manifest_relative = manifest.relative_to(root).as_posix()
        allowed = {manifest_relative, *args.allow_extra}
        extras = sorted(actual_paths - set(expected_paths) - allowed)
        for relative in extras:
            failures.append(f"unexpected: {relative}")
        nonlogical = sorted(set(expected_paths) - actual_paths)
        for relative in nonlogical:
            failures.append(f"manifested path is outside logical tree: {relative}")
        missing_allowed = sorted(set(args.allow_extra) - actual_paths)
        for relative in missing_allowed:
            failures.append(f"allowed extra does not exist: {relative}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    suffix = (
        f" with {tree_kind} exact-tree validation"
        if not args.no_exact_tree
        else ""
    )
    print(f"Verified {checked} files{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
