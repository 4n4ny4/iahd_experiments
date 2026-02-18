# Experiments

## Scripts

| Script | Purpose |
|--------|--------|
| `run_single_depth_haystack_plan.py` | Single-depth head discovery; **success = value match** (decoded contains `needle_value`, with number normalization). Writes `task_heads.json`, `task_head_rankings.json` to `outputs/single_depth/`. |
| `run_ablation_shared_heads.py` | **Per-task** ablation: for each task, ablates top k=1, 10, 100 heads from that task’s ranking. Primary metric = value-match rate. Input: `task_head_rankings.json` (or legacy `ablation_candidates.json`). Writes to `outputs/ablation/`. |
| `analyze_results.py` | Plots and summary for a single-depth run (head sparsity, overlap, etc.); can produce `ablation_candidates.json`. |
| `analyze_head_grid.py` | Head-layer heatmap for a run. |
| `analyze_overlap_table.py` | Overlap table (e.g. LaTeX) for a run. |
| `analyze_ablation_run.py` | Summary and metrics for an ablation run. |

## Outputs

All run directories live under **`outputs/`**:

- **`outputs/single_depth/run_<timestamp>/`** — `task_heads.json`, `task_head_rankings.json`, `task_success.json`, `task_value_match.json`, `run_meta.json`, `final_experiment_rows.csv`. Run `analyze_results.py` to add `plots/`.
- **`outputs/ablation/run_<timestamp>/`** — `condition_results.json`, `summary.json`, `per_task_ablation_summary.json`, `run_meta.json`, and value-match delta plots.

Pass `--run_dir <path>` to analysis scripts to point at a specific run.
