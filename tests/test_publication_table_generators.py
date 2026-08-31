from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"


def _environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(CODE), environment.get("PYTHONPATH", "")))
    )
    return environment


def _run(
    script: str,
    *arguments: str,
    tmp_path: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CODE / script), *arguments],
        cwd=ROOT,
        env=_environment(tmp_path),
        check=check,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_tables_5_and_6_generate_from_public_defaults_and_explicit_paths(
    tmp_path: Path,
) -> None:
    table5 = tmp_path / "table5.tex"
    _run(
        "make_publication_table_source_policy_ablation.py",
        "--output",
        str(table5),
        tmp_path=tmp_path,
    )
    text5 = table5.read_text(encoding="utf-8")
    assert r"\label{tab:source_policy_ablation}" in text5
    assert "Both links & Information-guided" in text5

    table6 = tmp_path / "table6.tex"
    _run(
        "make_publication_table_source_policy_contrasts.py",
        "--summary",
        str(ROOT / "data/tables/leader_source_policy_summary.json"),
        "--rows",
        str(ROOT / "data/tables/leader_source_policy_episode_rows.csv"),
        "--output",
        str(table6),
        tmp_path=tmp_path,
    )
    text6 = table6.read_text(encoding="utf-8")
    assert r"\label{tab:source_policy_contrasts}" in text6
    assert "paired Newcombe hybrid-score" in text6

    campaign = tmp_path / "doi_campaign"
    campaign.mkdir()
    shutil.copyfile(
        ROOT / "data/tables/leader_source_policy_summary.json",
        campaign / "campaign_summary.json",
    )
    shutil.copyfile(
        ROOT / "data/tables/leader_source_policy_episode_rows.csv",
        campaign / "episode_arm_summary.csv",
    )
    compatible = tmp_path / "table5_campaign_dir.tex"
    _run(
        "make_publication_table_source_policy_ablation.py",
        "--campaign-dir",
        str(campaign),
        "--output",
        str(compatible),
        tmp_path=tmp_path,
    )
    assert compatible.read_text(encoding="utf-8") == text5


def test_table_8_generates_from_public_defaults(tmp_path: Path) -> None:
    output = tmp_path / "table8.tex"
    _run(
        "make_publication_table_planner_component_ablation.py",
        "--output",
        str(output),
        tmp_path=tmp_path,
    )
    text = output.read_text(encoding="utf-8")
    assert r"\label{tab:planner_component_ablation}" in text
    assert "Uniform random feasible" in text


def test_table_9_fails_closed_then_generates_labelled_descriptive_output(
    tmp_path: Path,
) -> None:
    refused = tmp_path / "table9_refused.tex"
    failure = _run(
        "make_publication_table_closed_loop_stress.py",
        "--output",
        str(refused),
        tmp_path=tmp_path,
        check=False,
    )
    assert failure.returncode != 0
    assert "--allow-invalid-descriptive" in failure.stderr
    assert not refused.exists()

    output = tmp_path / "table9.tex"
    completed = _run(
        "make_publication_table_closed_loop_stress.py",
        "--allow-invalid-descriptive",
        "--output",
        str(output),
        tmp_path=tmp_path,
    )
    assert "WARNING" in completed.stderr
    text = output.read_text(encoding="utf-8")
    assert text.startswith("% WARNING: descriptive output")
    assert r"\label{tab:integrated_stress}" in text
    assert "Nominal & 97 & 96 & 97 & 100" in text
    assert "Dead-reckoning scale" in text
    assert "4 / 24" in text
