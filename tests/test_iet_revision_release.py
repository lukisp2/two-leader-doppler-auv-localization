"""Checks use released rows and metadata; no simulated episode is executed."""
from pathlib import Path
import csv
import json
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import project_iet_revision_campaign as projector

class RevisionReleaseTests(unittest.TestCase):
    def test_private_root_projection_fails_closed(self):
        self.assertEqual(projector.public_root_label('/arbitrary/release_repo_staging/code/a'),'repository/code/a')
        self.assertEqual(projector.public_root_label('/arbitrary/UUV_ART2/a'),'historical_source/a')
        with self.assertRaises(ValueError):
            projector.public_root_label('/unexpected/unknown/a')

    def test_complete_cohort_and_outcome_independent_selection(self):
        d=json.loads((ROOT/'data/provenance/iet_difficult_initial_conditions/selected_scenarios.json').read_text())
        self.assertFalse(d['selection_uses_outcomes'])
        self.assertEqual([r['seed'] for r in d['screened_initial_states']],list(range(52000,53000)))
        eligible=[r['seed'] for r in d['screened_initial_states']
                  if r['initial_formation_error_m']>=350 and r['initial_heading_error_deg']>=120]
        self.assertEqual(len(eligible),57)
        self.assertEqual(d['selected_seeds'],eligible[:50])
        self.assertFalse(any(50000<=s<=50999 for s in d['selected_seeds']))

    def test_rows_reproduce_counts_and_error_summaries(self):
        with (ROOT/'data/tables/iet_difficult_initial_episode_rows.csv').open() as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),100)
        summary=json.loads((ROOT/'data/tables/iet_difficult_initial_summary.json').read_text())
        for controller, expected in [('baseline_pid',(22,2,1)),('delay_aware',(48,46,48))]:
            rr=[r for r in rows if r['controller']==controller]
            self.assertEqual(len(rr),50)
            self.assertEqual(len({r['episode_seed'] for r in rr}),50)
            counts=tuple(sum(r[k].lower() in {'true','1'} for r in rr) for k in
                         ('terminal_joint_success','tail80_joint_success','dwell15_joint_success'))
            self.assertEqual(counts,expected)
            for key in ('terminal_localization_error_m','terminal_formation_error_m'):
                x=np.array([float(r[key]) for r in rr])
                self.assertTrue(np.isfinite(x).all())
                self.assertAlmostEqual(float(np.median(x)),summary['metrics_by_controller'][controller][key]['median'])

if __name__=='__main__':
    unittest.main()
