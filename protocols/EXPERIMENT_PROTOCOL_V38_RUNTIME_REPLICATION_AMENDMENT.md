# V38 runtime-only replication amendment

## Status and scope

This amendment authorizes one exact replication of the frozen V38
leader-source/acquisition-policy campaign on fresh seeds 48600--48699.

The amendment was opened only because the completed 48400--48499 campaign
failed its global runtime-integrity gate after one decision update took
2.9143316249828786 s in seed 48448, arm
`both_leaders__fixed_s_turn`. Five sequential diagnostic repeats of that exact
seed, arm, exogenous tape, and publication configuration produced maxima from
1.2127373749390244 to 1.276237832964398 s. All five passed the frozen 2.0 s
threshold and produced identical scientific outcomes. This supports a
nondeterministic scheduling or I/O delay diagnosis, but it does not
retroactively validate the original campaign.

## Frozen scientific specification

The replication changes none of the following:

- physical environment or vehicle model;
- six source-policy arms and their execution order;
- exogenous tape generation;
- estimator model and publication settings;
- active planner, fixed S-turn policy, and action bank;
- acquire-to-track gate and all of its thresholds;
- fixed 220-action horizon;
- endpoint definitions and statistical scoring;
- maximum combined decision-runtime definition;
- strict runtime-integrity requirement: every recorded maximum must be below
  2.0 s.

No efficacy result from the original campaign was used to tune the
replication.

## Runner-only change

The frozen runner remains unchanged. The versioned replication runner:

1. requires `--seed-start 48600`;
2. constructs exactly 100 consecutive campaign seeds, 48600--48699;
3. rejects any overlap with the closed 49900--50999 range;
4. audits the selected seed block for prior use before creating the campaign;
5. records the selected block, this justification, the reference campaign,
   and source hashes in the immutable contract;
6. snapshots its own runner, this amendment, its tests, and the complete
   loaded source closure;
7. permits a one-seed preflight only when that seed lies outside the campaign
   block and the closed final range.

## Execution rule

Run the six arms sequentially. Do not run another heavy numerical campaign in
parallel. A preflight pass does not contribute to the campaign analysis.
The full replication is valid only if every frozen integrity check passes,
including the unchanged strict 2.0 s runtime gate.
