# Experiments

## Scripts

| Script | Purpose |
|--------|--------|
| `run_single_depth_haystack_plan.py` | Single-depth head discovery; success = ROUGE-1 recall > 50. Writes to `outputs/single_depth/`. |
| `run_single_depth_haystack_output_gate.py` | Same pipeline; success = needle_value in decoded output. Balanced rows per task. Writes to `outputs/single_depth_output_gate/`. |
| `run_ablation_shared_heads.py` | Ablate shared heads from a candidate set. Writes to `outputs/ablation/`. |
| `analyze_results.py` | Plots and summary for a single-depth run; can produce `ablation_candidates.json`. |
| `analyze_head_grid.py` | Head-layer heatmap for a run. |
| `analyze_overlap_table.py` | Overlap table (e.g. LaTeX) for a run. |
| `analyze_ablation_run.py` | Summary and metrics for an ablation run. |

## Outputs

All run directories live under **`outputs/`**:

- `outputs/single_depth/run_<timestamp>/`
- `outputs/single_depth_output_gate/run_<timestamp>/`
- `outputs/ablation/run_<timestamp>/`

Pass `--run_dir <path>` to analysis scripts to point at a specific run.
