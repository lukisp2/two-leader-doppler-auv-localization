# Publication figure from the qualification postprocessor

`scripts/plot_v40_qualification_figure.py` creates one compact, monochrome
two-panel figure from the validated compact JSON written by the read-only
qualification postprocessor. The figure generator does **not** accept or open a
campaign directory. It reads no episode records, traces, campaign summary, or
decision file.

## Figure content

- Panel (a) reports terminal-success and Tail80-success proportions for all
  eight policy x execution x current arms. Marker shape identifies the endpoint;
  open and filled markers distinguish the fixed S-turn and information-guided
  acquisition.
- Panel (b) reports paired information-guided-minus-fixed-S-turn risk
  differences in the four
  execution x current cells. Whiskers are paired Newcombe method-10 95%
  confidence intervals. The dashed vertical line marks no policy effect.

The visible notation uses descriptive method names rather than internal
development-version identifiers. The four cells are labeled as kinematic or
delayed first-order execution, each without current or with visible current. The
legend is placed below both axes, outside the plotted data.

Suggested manuscript caption:

> Task sensitivity to command response and visible current. (a) Absolute
> terminal and Tail80 rates in all eight arms; open markers denote fixed S-turn
> and filled markers information-guided acquisition. (b) Paired
> information-guided-minus-fixed risk differences with paired Newcombe method-10
> 95% intervals. Positive values favor information-guided acquisition. Both
> Doppler links are used throughout, and visible current is measured through the
> idealized ground-velocity-aided channel.

## Input validation

The generator fails closed unless the JSON contains:

- `integrity_valid: true` and the supported compact-output schema;
- exactly eight unique policy x execution x current arms with 100 episodes
  each;
- integer success counts and rates that agree exactly with those counts;
- exactly four paired active-minus-fixed contrasts with 100 pairs each;
- complete paired 2 x 2 binary tables for terminal and Tail80 success;
- stored risk differences and Newcombe intervals that can be reproduced from
  the paired counts; and
- arm-level marginal differences that agree with the paired risk differences.

This validation is an additional presentation guard. It does not replace the
contract, hash, trace, and outcome audit performed by
`scripts/postprocess_v40_qualification.py`.

## Usage after postprocessing

From the repository root:

```bash
python3 scripts/plot_v40_qualification_figure.py \
  data/tables/v40_publication_results.json \
  --output-dir results/figures
```

The default outputs are:

- `results/figures/fig_policy_execution_current.pdf` -- vector PDF with
  embedded TrueType fonts; and
- `results/figures/fig_policy_execution_current_600dpi.png` -- 600 dpi PNG.

The PNG resolution can be changed with `--dpi`, but values below 300 dpi are
rejected. `--basename` accepts a plain filename only and cannot redirect output
outside `--output-dir`.

## Synthetic tests

The tests construct an eight-arm factorial document in a temporary directory.
They never inspect qualification results:

```bash
python3 -m unittest tests/test_plot_v40_qualification_figure.py -v
```

They verify schema guards, paired-count and Newcombe consistency, absence of
internal version labels in visible figure text, legend placement outside both
axes, vector PDF creation, and PNG dimensions/resolution at 600 dpi.
