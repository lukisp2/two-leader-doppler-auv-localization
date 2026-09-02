# Controller-repair qualification audit

## Evidentiary status

The controller-repair campaign is a complete, integrity-valid, paired
qualification: 100 seeds, four arms per seed, and 400 episode-arm outcomes.
The estimator, active-acquisition planner, evidence-qualified gate, dynamic
plant, horizon, and success definitions were held fixed. Within each current
condition, the two controller arms used matching sensor-noise and current
tapes and were bitwise identical before the first TRACK action.

The delay-aware tracker increased terminal joint success from 38% to 95%
without current and from 26% to 95% with ground-velocity-visible horizontal
current. Tail80 joint success increased from 2% to 91% and from 6% to 92%,
respectively. The reference-only discordant count was zero for all four primary
paired comparisons. Mean post-TRACK saturation fell by 0.913 and 0.906, and
mean normalized action-curvature RMS fell by 0.527 and 0.524.

The campaign nevertheless did **not** pass its complete prespecified
acceptance gate. In the no-current condition, terminal-localization p95 was
0.9932 m for the reference and 1.5600 m for the delay-aware tracker. The
treatment-minus-reference difference, 0.5669 m, exceeded the frozen 0.50-m
non-inferiority margin by 0.0669 m. All other prespecified screens passed.
This outcome must be reported as a strong controller improvement accompanied
by one failed composite acceptance condition, not as an unqualified pass.

## Decision-label correction

The frozen runner emitted `CONTROLLER_REPAIR_REJECT_OR_INVALID` whenever either
integrity or acceptance failed. That inherited label combines two scientifically
different states. Here, `integrity_valid` is true and `acceptance_pass` is
false. The unambiguous public interpretation is therefore:

```text
VALID_COMPLETE__COMPOSITE_ACCEPTANCE_NOT_MET
```

The frozen summary and decision are preserved byte-for-byte in the raw
campaign and with path-only sanitization in `data/provenance/controller_repair/`.
The public semantic audit is additive; it does not rewrite the historical
decision artifact or alter any numerical value.

## Reserved-seed marker correction

The inherited fields `final_holdout_sealed` and
`sealed_seed_range_untouched` store `[50000, 50999]`. Their defensible meaning
is limited to a reserved-range marker and a runner exclusion rule. A stored
range is not evidence that no external process ever inspected those seeds.

This release verifies only that the controller-repair qualification selected
seeds 51000--51099 and that this selected cohort is disjoint from the reserved
range 50000--50999. It makes no global "never opened" claim. Manuscript and
review-response language must follow this narrower interpretation.

## Public artifacts

- `data/tables/controller_repair_episode_rows.csv`: all 400 outcome rows;
- `data/tables/controller_repair_publication_arms.csv`: four publication cells;
- `data/tables/controller_repair_results.json`: row-derived distributions,
  paired effects, frozen screens, hashes, and semantic audit;
- `data/provenance/controller_repair/`: path-sanitized frozen control records,
  additive semantic audit, and export manifest.

The complete 922-file local campaign remains outside Git. Use
`scripts/project_v41_doi_campaign.py` to produce the path-sanitized 924-file
DOI view: 400 episode JSON files, 400 trajectory NPZ files, 100 noise tapes,
15 frozen source files, the remaining campaign records, approved public
environment metadata, and the projection manifest.
