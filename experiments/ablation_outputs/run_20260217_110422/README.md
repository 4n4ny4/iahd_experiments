# Ablation Analysis - ROUGE-Gated Head Detection

Run directory:

- `experiments/ablation_outputs/run_20260217_110422`

## Experiment setup

- Model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Dataset: `data/haystack_plan_100_per_task.csv`
- Attempts: `842` for each condition
- Top shared heads ablated:
  - `L15H30`
  - `L16H1`
  - `L19H3`
  - `L20H14`
- Random controls: 3 random 4-head sets

## Core results

Baseline:

- ROUGE success: `139/842` (`16.51%`)
- Value match: `474/842` (`56.29%`)

Top shared ablation:

- ROUGE success: `108/842` (`12.83%`)
- Value match: `486/842` (`57.72%`)
- Delta vs baseline:
  - ROUGE: `-3.68` percentage points
  - Value: `+1.43` percentage points

Random controls (delta vs baseline):

- `random_ablation_1`: ROUGE `-3.56` pp, Value `+1.66` pp
- `random_ablation_2`: ROUGE `-0.12` pp, Value `-0.48` pp
- `random_ablation_3`: ROUGE `-0.95` pp, Value `+0.12` pp
- Random mean:
  - ROUGE: `-1.54` pp
  - Value: `+0.44` pp

## Visualizations generated

Original run plots:

- `ablation_condition_overall.png`
- `ablation_per_task_delta_rouge.png`

Additional comparison plots:

- `viz_overall_delta_vs_baseline.png`
  - ROUGE and Value deltas for each condition in one figure.
- `viz_top_vs_random_controls.png`
  - Top-shared ROUGE drop vs random-control distribution.
- `viz_per_task_top_vs_random.png`
  - Per-task ROUGE deltas: top-shared ablation vs random-control mean (with min/max error bars).

## Interpretation

1. Ablating the top shared heads causes a clear aggregate ROUGE drop (`-3.68` pp), indicating these heads likely contribute to retrieval behavior under the ROUGE-gated pipeline.

2. Evidence is suggestive but not fully decisive:
   - one random set (`random_ablation_1`) is nearly as harmful as top-shared on ROUGE.
   - with only 3 random sets, variance is not tightly estimated.

3. Value-match does not decrease under top-shared ablation, reinforcing that ROUGE and value metrics are capturing different behavior.

4. Per-task effects are concentrated on stronger structured tasks (`employees_count_*`, `holder_record_amount`, `headquarters_state`, `incorporation_year`).

## Recommended next steps

- Increase random controls to at least 10 sets for the same `k=4`.
- Run a `k` sweep (`k=1,2,4,8,12`) and compare top-shared vs random at each `k`.
- Add significance-style summaries (bootstrap or randomization test) on top-vs-random ROUGE deltas.
- Repeat with non-ROUGE-aware detection criteria for entity-heavy tasks.
