# Single-Depth Retrieval Head Experiment Findings

This README summarizes run:

- `experiments/single_depth_outputs/run_20260217_095958`
- Model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Head space: `32 layers x 32 heads = 1024 total heads`
- Retrieval-head threshold: `0.1`
- Decode budget: `20` tokens
- Rows scanned: `849`

## What each metric means

- `Attempts`: rows that passed preprocessing and were actually evaluated.
- `ROUGE OK`: rows with ROUGE-1 recall against needle sentence `> 50`.
- `Value Match`: decoded output contains `needle_value`.
- `#Heads`: selected retrieval heads for a task (aggregated from ROUGE-OK rows).

## Task-level summary

| Task | Attempts | ROUGE OK | ROUGE Rate | Value Match | Value Rate | #Heads |
|---|---:|---:|---:|---:|---:|---:|
| `employees_count_full_time` | 49 | 22 | 44.9% | 8 | 16.3% | 11 |
| `employees_count_total` | 100 | 42 | 42.0% | 8 | 8.0% | 19 |
| `holder_record_amount` | 98 | 39 | 39.8% | 20 | 20.4% | 22 |
| `incorporation_year` | 100 | 16 | 16.0% | 71 | 71.0% | 25 |
| `headquarters_state` | 100 | 9 | 9.0% | 57 | 57.0% | 24 |
| `incorporation_state` | 98 | 6 | 6.1% | 93 | 94.9% | 21 |
| `registrant_name` | 99 | 4 | 4.0% | 96 | 97.0% | 13 |
| `ceo_lastname` | 98 | 0 | 0.0% | 55 | 56.1% | 0 |
| `headquarters_city` | 100 | 0 | 0.0% | 66 | 66.0% | 0 |

Overall:

- Attempts: `842`
- ROUGE OK: `138` (`16.4%`)
- Value Match: `474` (`56.3%`)

## Plot-by-plot interpretation

### 1) `retrieval_success_rate.png`

- Compares ROUGE-gated success vs value-match success per task.
- Key read: several tasks have much higher value-match than ROUGE success, indicating sentence overlap is a stricter gate than answer correctness.

### 2) `rouge_vs_value.png`

- Scatter with `x = ROUGE success rate`, `y = value-match rate`.
- Points far above diagonal indicate value extraction without strong sentence-level overlap.
- Notable in this run: `registrant_name`, `incorporation_state`, `headquarters_city`, `ceo_lastname`.

### 3) `head_sparsity.png`

- Shows fraction of all 1024 heads selected per task.
- Head usage is sparse: about `0.0%` to `2.4%` depending on task.

### 4) `overlap_heatmap_jaccard.png`

- Symmetric Jaccard overlap across task head sets.
- Reveals a strong overlap cluster among structured fields (counts/records/incorporation/headquarters_state).
- `ceo_lastname` and `headquarters_city` appear as zero-overlap rows due to no selected heads.

### 5) `overlap_heatmap_pct.png`

- Directional overlap: `% of row task's heads contained in column task`.
- Useful to identify subset relations (asymmetric sharing), not just raw symmetric overlap.

### 6) `layer_distribution.png`

- Per-task distribution of selected heads by layer.
- Lets you compare where each task's retrieval heads concentrate.

### 7) `layer_distribution_aggregate.png`

- Aggregate retrieval-head counts by layer across all tasks.
- Highlights globally active retrieval layers.

### 8) `top_shared_heads.png`

- Heads used by the most tasks (top-k shared heads).
- Useful shortlist for causal follow-up (ablation/intervention).

### 9) `overlap_by_data_type.png`

- Left panel: average overlap by type pair (`numerical-numerical`, `categorical-categorical`, `categorical-numerical`), with pair count and max overlap.
- Right panel: top cross-type task pairs.
- Shows that meaningful overlap exists even across data-type boundaries, not only within type.

### 10) `pairwise_overlap_bar.png`

- Pairwise Jaccard overlaps ranked high to low.
- Top pair in this run: `employees_count_total` x `holder_record_amount` (`0.864`).

### 11) `grid_task_count.png`

- Layer x head map of how many tasks use each head.
- Quickly shows broadly reused heads vs task-specific heads.

### 12) `grid_union_intersection.png`

- Union map (all heads used by any task) and intersection map (used by all tasks).
- In this run: union is small, intersection is empty.

### 13) `grid_<task>.png` (per-task binary maps)

- Binary head maps per task.
- `grid_ceo_lastname.png` and `grid_headquarters_city.png` are empty, matching zero ROUGE successes.

### 14) `ablation_candidate_ranking.png`

- Ranked list of heads to ablate first.
- Ranking combines:
  - how many tasks a head appears in (`support_count`)
  - success-weighted support (`weighted_support`, based on task ROUGE success rates)
  - whether the head spans multiple data types (`type_count`)
- Interpretation: higher-ranked heads are better first candidates for cross-task ablation tests.

### 15) `ablation_candidate_taskmap.png`

- Head-by-task heatmap for top ablation candidates.
- Cell value is the task ROUGE success rate for tasks where that head is present.
- Interpretation: identifies whether a candidate head is broadly useful or concentrated in a subset of tasks.

### 16) `ablation_candidates.json`

- Machine-readable top candidate list (default top-20).
- Each entry includes:
  - `head` and `label`
  - `support_count`
  - `weighted_support`
  - `type_count`
  - `priority_score`
  - `tasks`

## Main takeaways from this experiment

1. Retrieval circuitry is sparse.
   - Union of selected heads: `34 / 1024` (`3.3%`).
   - Per-task selection: `0` to `25` heads.

2. No universal retrieval core.
   - Intersection across all tasks: `0` heads.

3. Strong partial sharing exists.
   - Several task pairs show high overlap (Jaccard up to `0.864`).

4. ROUGE gate and value-match capture different behaviors.
   - High value-match can coexist with low/zero ROUGE success.
   - This affects which tasks can accumulate retrieval-head evidence.

5. Two tasks had zero selected heads but non-trivial value success.
   - `ceo_lastname`: 55/98 value matches.
   - `headquarters_city`: 66/100 value matches.
   - Interpretation: current ROUGE gate is likely too strict for these answer styles.

6. Top ablation candidates are strongly shared across tasks and data types.
   - Highest-priority heads from this run include:
     - `L15H30`
     - `L16H1`
     - `L19H3`
     - `L20H14`
   - These heads each appear across 7 tasks and both data types, making them strong first-pass ablation targets.

## Artifacts

- Report: `analysis_report.md`
- LaTeX table: `overlap_table.tex`
- Plots: `plots/`
- Evaluated rows: `final_experiment_rows.csv`
- Ablation candidate list: `plots/ablation_candidates.json`

## Recommended next experiments

- Try a less strict ROUGE criterion or value-based gating for head accumulation.
- Increase `max_decode` for low-ROUGE tasks to test generation truncation effects.
- Run targeted head ablations on top shared heads to test causal importance.
