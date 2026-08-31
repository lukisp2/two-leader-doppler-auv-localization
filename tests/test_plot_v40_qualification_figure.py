from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_v40_qualification_figure.py"
SPEC = importlib.util.spec_from_file_location("plot_v40_qualification_figure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plotter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plotter
SPEC.loader.exec_module(plotter)


CELL_COUNTS = {
    ("kinematic", "no_current"): (70, 85, 60, 78),
    ("kinematic", "bottom_track_visible"): (75, 90, 65, 85),
    ("low_order_dynamic", "no_current"): (65, 78, 55, 70),
    ("low_order_dynamic", "bottom_track_visible"): (60, 80, 50, 72),
}
REFERENCE_ONLY = {
    ("kinematic", "no_current"): (5, 4),
    ("kinematic", "bottom_track_visible"): (4, 3),
    ("low_order_dynamic", "no_current"): (6, 5),
    ("low_order_dynamic", "bottom_track_visible"): (3, 2),
}


def paired_endpoint(reference: int, treatment: int, reference_only: int) -> dict:
    both = reference - reference_only
    treatment_only = treatment - both
    neither = 100 - both - treatment_only - reference_only
    difference, lower, upper = plotter.paired_newcombe_method10_interval(
        both,
        treatment_only,
        reference_only,
        neither,
    )
    return {
        "concordant_success": both,
        "treatment_only": treatment_only,
        "reference_only": reference_only,
        "concordant_failure": neither,
        "risk_difference": difference,
        "paired_newcombe_method10_95": [lower, upper],
        "mcnemar_exact_two_sided_p": 0.5,
    }


def synthetic_document() -> dict:
    arms = []
    contrasts = {}
    for execution, current in plotter.CELL_ORDER:
        fixed_terminal, active_terminal, fixed_tail, active_tail = CELL_COUNTS[
            (execution, current)
        ]
        for policy, terminal, tail in (
            (plotter.POLICY_FIXED, fixed_terminal, fixed_tail),
            (plotter.POLICY_ACTIVE, active_terminal, active_tail),
        ):
            arms.append(
                {
                    "arm": f"{policy}__{execution}__{current}",
                    "policy": policy,
                    "execution": execution,
                    "current": current,
                    "N": 100,
                    "terminal_success_n": terminal,
                    "terminal_success_rate": terminal / 100.0,
                    "Tail80_n": tail,
                    "Tail80_rate": tail / 100.0,
                }
            )
        terminal_reference_only, tail_reference_only = REFERENCE_ONLY[
            (execution, current)
        ]
        contrasts[f"{execution}__{current}"] = {
            "pairs": 100,
            "direction": "treatment_minus_reference",
            "terminal_joint_success": paired_endpoint(
                fixed_terminal,
                active_terminal,
                terminal_reference_only,
            ),
            "tail80_joint_success": paired_endpoint(
                fixed_tail,
                active_tail,
                tail_reference_only,
            ),
        }
    arms.reverse()
    return {
        "schema_version": 1,
        "analysis_version": "synthetic_test_only",
        "integrity_valid": True,
        "arms": arms,
        "paired_main_effects": {
            "treatment_minus_reference_convention": True,
            "active_minus_fixed_within_execution_current": contrasts,
        },
    }


def write_document(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def png_metadata(path: Path) -> tuple[int, int, float | None]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("not a PNG")
    offset = 8
    width = height = None
    dpi = None
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        if kind == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        if kind == b"pHYs":
            x_pixels_per_metre, _, unit = struct.unpack(">IIB", payload)
            if unit == 1:
                dpi = x_pixels_per_metre * 0.0254
        offset += 12 + length
        if kind == b"IEND":
            break
    if width is None or height is None:
        raise AssertionError("PNG lacks IHDR")
    return width, height, dpi


class QualificationFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        plotter.configure_style()

    def test_newcombe_reference_values(self) -> None:
        observed = plotter.paired_newcombe_method10_interval(20, 12, 2, 16)
        for value, expected in zip(observed, (0.2, 0.0562, 0.3292)):
            self.assertAlmostEqual(value, expected, delta=5.0e-5)

    def test_synthetic_input_builds_two_panels_with_external_legend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "publication.json"
            write_document(path, synthetic_document())
            data = plotter.load_publication_results(path)
            self.assertEqual(len(data.arms), 8)
            self.assertEqual(len(data.effects), 4)
            figure = plotter.build_figure(data)
            try:
                self.assertEqual(len(figure.axes), 2)
                self.assertEqual(len(figure.legends), 1)
                self.assertTrue(all(axis.get_legend() is None for axis in figure.axes))
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                legend_bounds = figure.legends[0].get_window_extent(renderer)
                self.assertTrue(
                    all(
                        not legend_bounds.overlaps(axis.get_window_extent(renderer))
                        for axis in figure.axes
                    )
                )
                visible_text = " ".join(
                    [axis.get_title(loc="left") for axis in figure.axes]
                    + [axis.get_xlabel() for axis in figure.axes]
                    + [axis.get_ylabel() for axis in figure.axes]
                    + [tick.get_text() for axis in figure.axes for tick in axis.get_xticklabels()]
                    + [tick.get_text() for axis in figure.axes for tick in axis.get_yticklabels()]
                    + [text.get_text() for text in figure.legends[0].get_texts()]
                )
                self.assertIsNone(re.search(r"\bV\d+\b", visible_text, flags=re.IGNORECASE))
                self.assertIn("eight arms", visible_text)
                self.assertIn("Information-guided acquisition", visible_text)
            finally:
                plotter.plt.close(figure)

    def test_writes_vector_pdf_and_600_dpi_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "publication.json"
            write_document(input_path, synthetic_document())
            data = plotter.load_publication_results(input_path)
            outputs = plotter.write_figure(data, root / "figures", dpi=600)
            self.assertEqual(set(outputs), {"pdf", "png"})
            self.assertTrue(outputs["pdf"].read_bytes().startswith(b"%PDF"))
            self.assertGreater(outputs["pdf"].stat().st_size, 10_000)
            width, height, dpi = png_metadata(outputs["png"])
            self.assertEqual(width, round(plotter.FIGURE_WIDTH_IN * 600))
            self.assertEqual(height, round(plotter.FIGURE_HEIGHT_IN * 600))
            self.assertIsNotNone(dpi)
            assert dpi is not None
            self.assertGreaterEqual(dpi, 599.0)

    def test_validation_fails_closed_for_integrity_counts_and_ci(self) -> None:
        cases = []
        invalid_integrity = synthetic_document()
        invalid_integrity["integrity_valid"] = False
        cases.append((invalid_integrity, "integrity"))

        duplicate_arm = synthetic_document()
        duplicate_arm["arms"][0] = dict(duplicate_arm["arms"][1])
        cases.append((duplicate_arm, "exact 2 x 2 x 2"))

        invalid_ci = synthetic_document()
        endpoint = invalid_ci["paired_main_effects"][
            "active_minus_fixed_within_execution_current"
        ]["kinematic__no_current"]["terminal_joint_success"]
        endpoint["paired_newcombe_method10_95"][0] -= 0.01
        cases.append((invalid_ci, "stored CI lower disagrees"))

        nonfinite = synthetic_document()
        nonfinite["arms"][0]["Tail80_rate"] = "nan"
        cases.append((nonfinite, "not finite"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (document, message) in enumerate(cases):
                with self.subTest(message=message):
                    path = root / f"invalid_{index}.json"
                    write_document(path, document)
                    with self.assertRaisesRegex(plotter.FigureInputError, message):
                        plotter.load_publication_results(path)

    def test_cli_is_data_independent_until_given_synthetic_json(self) -> None:
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "mpl"
            config.mkdir()
            environment["MPLCONFIGDIR"] = str(config)
            help_run = subprocess.run(
                [sys.executable, str(SCRIPT), "--help"],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertIn("usage:", help_run.stdout.lower())

            input_path = root / "publication.json"
            write_document(input_path, synthetic_document())
            output = root / "figures"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(input_path),
                    "--output-dir",
                    str(output),
                    "--dpi",
                    "300",
                ],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertIn("PDF:", completed.stdout)
            self.assertTrue((output / f"{plotter.DEFAULT_BASENAME}.pdf").is_file())
            self.assertTrue((output / f"{plotter.DEFAULT_BASENAME}_300dpi.png").is_file())

    def test_dpi_and_basename_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "publication.json"
            write_document(path, synthetic_document())
            data = plotter.load_publication_results(path)
            with mock.patch.object(plotter, "build_figure") as build:
                with self.assertRaisesRegex(plotter.FigureInputError, "at least 300"):
                    plotter.write_figure(data, Path(temporary), dpi=299)
                with self.assertRaisesRegex(plotter.FigureInputError, "plain filename"):
                    plotter.write_figure(data, Path(temporary), basename="../escape")
                build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
