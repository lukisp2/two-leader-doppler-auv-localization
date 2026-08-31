#!/usr/bin/env python3
"""Generate a deterministic SHA-256 manifest for a frozen release tree.

For a Git repository, the default inventory is the set of tracked paths from a
clean index and working tree. This prevents ignored or untracked campaign
output from entering a source release. A non-Git archive is walked directly;
``--inventory filesystem`` can also be used explicitly for such a tree.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Iterable, Iterator


EXCLUDED_DIRS = frozenset(
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
        "__MACOSX",
        "__pycache__",
        "venv",
    }
)
EXCLUDED_FILES = frozenset({".DS_Store"})


class ManifestError(RuntimeError):
    """Raised when a release tree cannot be represented safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_excluded(relative: PurePosixPath, output_relative: PurePosixPath) -> bool:
    name = relative.name
    return (
        relative == output_relative
        or any(part in EXCLUDED_DIRS for part in relative.parts)
        or name in EXCLUDED_FILES
        or name.startswith("._")
        or name.startswith("SHA256SUMS")
        or name.startswith(".SHA256SUMS")
        or (
            name.startswith(f".{output_relative.name}.")
            and name.endswith(".tmp")
        )
    )


def _validate_relative(relative: PurePosixPath) -> None:
    rendered = relative.as_posix()
    if not rendered or rendered == "." or relative.is_absolute():
        raise ManifestError(f"invalid release path: {rendered!r}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ManifestError(f"non-canonical release path: {rendered!r}")
    if "\n" in rendered or "\r" in rendered or "\\" in rendered:
        raise ManifestError(
            f"release path cannot be represented portably: {rendered!r}"
        )
    try:
        rendered.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ManifestError(f"release path is not valid UTF-8: {rendered!r}") from exc


def _checked_file(root: Path, relative: PurePosixPath) -> Path:
    _validate_relative(relative)
    path = root.joinpath(*relative.parts)
    if path.is_symlink():
        raise ManifestError(f"symbolic links are not supported: {relative.as_posix()}")
    if not path.is_file():
        raise ManifestError(f"tracked path is not a regular file: {relative.as_posix()}")
    return path


def _git_root(root: Path) -> Path | None:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return Path(completed.stdout.strip()).resolve()


def _require_clean_tracked_tree(root: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=no",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ManifestError(f"cannot inspect Git work tree: {message}")
    if completed.stdout:
        raise ManifestError(
            "tracked Git content is not clean; commit or restore tracked changes "
            "before generating the release manifest"
        )


def _git_inventory(
    root: Path, output_relative: PurePosixPath
) -> Iterator[tuple[Path, PurePosixPath]]:
    top = _git_root(root)
    if top is None:
        raise ManifestError("--inventory git requires a Git work tree")
    if top != root:
        raise ManifestError(
            f"--root must be the Git top level ({top}), not a subdirectory"
        )
    _require_clean_tracked_tree(root)
    completed = subprocess.run(
        ["git", "-C", os.fspath(root), "ls-files", "-z", "--cached", "--", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ManifestError(f"cannot list tracked files: {message}")

    relatives = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            rendered = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ManifestError("tracked release paths must be valid UTF-8") from exc
        relative = PurePosixPath(rendered)
        if not _is_excluded(relative, output_relative):
            relatives.append(relative)

    for relative in sorted(relatives, key=PurePosixPath.as_posix):
        yield _checked_file(root, relative), relative


def _filesystem_inventory(
    root: Path, output_relative: PurePosixPath
) -> Iterator[tuple[Path, PurePosixPath]]:
    def walk_error(error: OSError) -> None:
        raise ManifestError(f"cannot walk release tree: {error}") from error

    candidates: list[PurePosixPath] = []
    for directory, dirnames, filenames in os.walk(
        root, topdown=True, onerror=walk_error, followlinks=False
    ):
        directory_path = Path(directory)
        kept_dirs = []
        for name in sorted(dirnames):
            relative = PurePosixPath(
                (directory_path / name).relative_to(root).as_posix()
            )
            if _is_excluded(relative, output_relative):
                continue
            if (directory_path / name).is_symlink():
                raise ManifestError(
                    f"symbolic links are not supported: {relative.as_posix()}"
                )
            kept_dirs.append(name)
        dirnames[:] = kept_dirs

        for name in sorted(filenames):
            relative = PurePosixPath(
                (directory_path / name).relative_to(root).as_posix()
            )
            if not _is_excluded(relative, output_relative):
                candidates.append(relative)

    for relative in sorted(candidates, key=PurePosixPath.as_posix):
        yield _checked_file(root, relative), relative


def _select_inventory(root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    top = _git_root(root)
    if top == root:
        return "git"
    if top is not None:
        raise ManifestError(
            "--root is inside a Git work tree but is not its top level; use the "
            "Git top level or select --inventory filesystem explicitly"
        )
    if (root / ".git").exists() or (root / ".git").is_symlink():
        raise ManifestError(
            "Git metadata is present but the repository could not be inspected; "
            "refusing a filesystem fallback"
        )
    return "filesystem"


def files_under(
    root: Path, output: Path, inventory: str = "auto"
) -> Iterable[tuple[Path, PurePosixPath]]:
    try:
        output_relative = PurePosixPath(output.relative_to(root).as_posix())
    except ValueError as exc:
        raise ManifestError("--output must be inside --root") from exc
    if output.parent != root:
        raise ManifestError("--output must be a top-level file inside --root")

    selected = _select_inventory(root, inventory)
    if selected == "git":
        return _git_inventory(root, output_relative)
    if selected == "filesystem":
        return _filesystem_inventory(root, output_relative)
    raise ManifestError(f"unknown inventory mode: {inventory}")


def _write_atomic(output: Path, lines: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write("\n".join(lines))
        stream.write("\n")
    try:
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic manifest from tracked Git files or an archive tree."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, default=Path("SHA256SUMS"))
    parser.add_argument(
        "--inventory",
        choices=("auto", "git", "filesystem"),
        default="auto",
        help=(
            "file inventory source (default: tracked files for a Git top level; "
            "otherwise a filesystem walk)"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"release root is not a directory: {root}")
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    try:
        selected = _select_inventory(root, args.inventory)
        entries = list(files_under(root, output, args.inventory))
        lines = [
            f"{sha256(path)}  {relative.as_posix()}" for path, relative in entries
        ]
        if selected == "git":
            _require_clean_tracked_tree(root)
        _write_atomic(output, lines)
    except (ManifestError, OSError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(lines)} entries to {output} (inventory={selected}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
