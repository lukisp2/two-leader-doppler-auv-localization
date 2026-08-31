from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATE = ROOT / "scripts" / "generate_sha256_manifest.py"
VERIFY = ROOT / "scripts" / "verify_sha256_manifest.py"


def _run(script: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(value) for value in arguments)],
        check=False,
        capture_output=True,
        text=True,
    )


def _write(path: Path, value: str = "payload\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _manifest_paths(path: Path) -> list[str]:
    return [line.split("  ", 1)[1] for line in path.read_text().splitlines()]


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def test_filesystem_manifest_is_sorted_and_excludes_release_debris(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "z.txt", "z\n")
    _write(tmp_path / "a" / "b.txt", "b\n")
    _write(tmp_path / ".git" / "config")
    _write(tmp_path / ".pytest_cache" / "state")
    _write(tmp_path / "__MACOSX" / "resource")
    _write(tmp_path / ".DS_Store")
    _write(tmp_path / "._z.txt")
    _write(tmp_path / "SHA256SUMS.preliminary")
    _write(tmp_path / "SHA256SUMS.previous")
    _write(tmp_path / ".SHA256SUMS.interrupted.tmp")

    result = _run(
        GENERATE,
        "--root",
        tmp_path,
        "--inventory",
        "filesystem",
        "--output",
        "SHA256SUMS",
    )

    assert result.returncode == 0, result.stderr
    paths = _manifest_paths(tmp_path / "SHA256SUMS")
    assert paths == ["a/b.txt", "z.txt"]


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")
def test_git_inventory_uses_only_clean_tracked_files(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Manifest Test")
    _git(tmp_path, "config", "user.email", "manifest@example.invalid")
    _write(
        tmp_path / ".gitignore",
        "results/*\n"
        "!results/.gitkeep\n"
        "code/experiments_v34_mhe60_baseline_smoke/\n"
        "SHA256SUMS.preliminary\n",
    )
    _write(tmp_path / "tracked.txt")
    _write(tmp_path / "results" / ".gitkeep", "")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "fixture")
    _write(tmp_path / "results" / "ignored-run.csv")
    _write(tmp_path / "results" / "portable_inputs" / "history.json")
    _write(
        tmp_path
        / "code"
        / "experiments_v34_mhe60_baseline_smoke"
        / "campaign_summary.json"
    )
    _write(tmp_path / "SHA256SUMS.preliminary")
    _write(tmp_path / "untracked-note.txt")

    generated = _run(GENERATE, "--root", tmp_path)

    assert generated.returncode == 0, generated.stderr
    paths = _manifest_paths(tmp_path / "SHA256SUMS")
    assert paths == [".gitignore", "results/.gitkeep", "tracked.txt"]
    assert "ignored-run.csv" not in (tmp_path / "SHA256SUMS").read_text()
    assert "untracked-note.txt" not in (tmp_path / "SHA256SUMS").read_text()

    verified = _run(VERIFY, tmp_path / "SHA256SUMS")
    assert verified.returncode == 0, verified.stdout
    assert "tracked-Git exact-tree validation" in verified.stdout

    _write(tmp_path / "newly-tracked.txt")
    _git(tmp_path, "add", "newly-tracked.txt")
    tracked_extra = _run(VERIFY, tmp_path / "SHA256SUMS")
    assert tracked_extra.returncode == 1
    assert "FAIL unexpected: newly-tracked.txt" in tracked_extra.stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")
def test_git_inventory_rejects_dirty_tracked_content(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Manifest Test")
    _git(tmp_path, "config", "user.email", "manifest@example.invalid")
    _write(tmp_path / "tracked.txt", "before\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "fixture")
    _write(tmp_path / "tracked.txt", "after\n")

    result = _run(GENERATE, "--root", tmp_path)

    assert result.returncode == 2
    assert "tracked Git content is not clean" in result.stderr
    assert not (tmp_path / "SHA256SUMS").exists()


def test_verifier_checks_exact_tree_by_default(tmp_path: Path) -> None:
    _write(tmp_path / "payload.txt")
    generated = _run(
        GENERATE, "--root", tmp_path, "--inventory", "filesystem"
    )
    assert generated.returncode == 0, generated.stderr
    manifest = tmp_path / "SHA256SUMS"
    assert _run(VERIFY, manifest).returncode == 0

    _write(tmp_path / "unexpected.txt")
    exact = _run(VERIFY, manifest)
    hashes_only = _run(VERIFY, manifest, "--no-exact-tree")
    explicitly_allowed = _run(
        VERIFY, manifest, "--allow-extra", "unexpected.txt"
    )

    assert exact.returncode == 1
    assert "FAIL unexpected: unexpected.txt" in exact.stdout
    assert hashes_only.returncode == 0, hashes_only.stdout
    assert explicitly_allowed.returncode == 0, explicitly_allowed.stdout


@pytest.mark.parametrize(
    "relative",
    (
        ".DS_Store",
        "._payload.txt",
        ".SHA256SUMS.interrupted.tmp",
        "__MACOSX/resource",
        "SHA256SUMS.preliminary",
    ),
)
def test_exact_tree_rejects_excluded_packaging_debris(
    tmp_path: Path, relative: str
) -> None:
    _write(tmp_path / "payload.txt")
    generated = _run(
        GENERATE, "--root", tmp_path, "--inventory", "filesystem"
    )
    assert generated.returncode == 0, generated.stderr
    _write(tmp_path / relative)

    result = _run(VERIFY, tmp_path / "SHA256SUMS")

    assert result.returncode == 1
    assert f"FAIL unexpected: {relative}" in result.stdout


def test_exact_tree_rejects_empty_macos_metadata_directory(tmp_path: Path) -> None:
    _write(tmp_path / "payload.txt")
    generated = _run(
        GENERATE, "--root", tmp_path, "--inventory", "filesystem"
    )
    assert generated.returncode == 0, generated.stderr
    (tmp_path / "__MACOSX").mkdir()

    result = _run(VERIFY, tmp_path / "SHA256SUMS")

    assert result.returncode == 1
    assert "FAIL unexpected directory: __MACOSX" in result.stdout


def test_filesystem_exact_tree_ignores_only_fixed_runtime_cache_directories(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "payload.txt")
    generated = _run(
        GENERATE, "--root", tmp_path, "--inventory", "filesystem"
    )
    assert generated.returncode == 0, generated.stderr
    _write(tmp_path / ".pytest_cache" / "state")
    _write(tmp_path / "__pycache__" / "module.pyc")

    result = _run(VERIFY, tmp_path / "SHA256SUMS")

    assert result.returncode == 0, result.stdout


def test_verifier_rejects_noncanonical_and_duplicate_entries(tmp_path: Path) -> None:
    _write(tmp_path / "payload.txt")
    digest = hashlib.sha256((tmp_path / "payload.txt").read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  payload.txt\n"
        f"{digest}  payload.txt\n"
        f"{digest}  ../outside.txt\n",
        encoding="utf-8",
    )

    result = _run(VERIFY, tmp_path / "SHA256SUMS")

    assert result.returncode == 1
    assert "duplicate path" in result.stdout
    assert "path is not a canonical relative POSIX path" in result.stdout


def test_generator_requires_top_level_output(tmp_path: Path) -> None:
    _write(tmp_path / "payload.txt")

    result = _run(
        GENERATE,
        "--root",
        tmp_path,
        "--inventory",
        "filesystem",
        "--output",
        "nested/SHA256SUMS",
    )

    assert result.returncode == 2
    assert "--output must be a top-level file" in result.stderr
    assert not (tmp_path / "nested" / "SHA256SUMS").exists()
