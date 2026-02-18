# IAHD Experiments

Experiments for identifying and ablating retrieval heads in a haystack (needle-in-document) setup with a causal LM.

## Directory layout

```
iahd_experiments/
├── data/                    # Input CSVs (haystack rows per task)
│   ├── haystack_plan.csv
│   └── haystack_plan_100_per_task.csv
├── experiments/
│   ├── outputs/             # All run outputs (one subdir per pipeline)
│   │   ├── single_depth/           # Value-match-gated head discovery + rankings
│   │   └── ablation/               # Per-task ablation (k=1,10,100 heads)
│   ├── run_*.py             # Run pipelines (head discovery, ablation)
│   └── analyze_*.py         # Analysis and plotting
├── requirements.txt
└── README.md
```

## Pipeline overview

1. **Head discovery (single depth)**  
   `run_single_depth_haystack_plan.py` runs a haystack CSV through the model. **Success = value match**: decoded output contains `needle_value` (with number normalization: commas stripped, number words → digits). For each successful row, attention-to-needle scores are accumulated; retrieval heads are selected by threshold and written as:
   - `task_heads.json` — list of `[layer, head]` per task (thresholded).
   - `task_head_rankings.json` — per-task ranked heads with `avg_score` (used by ablation).

2. **Analysis**  
   Point analysis scripts at a run directory (e.g. `experiments/outputs/single_depth/run_YYYYMMDD_HHMMSS`):
   - `analyze_results.py` — plots, head sparsity, overlap heatmaps, ablation candidate list.
   - `analyze_head_grid.py` — head heatmaps.
   - `analyze_overlap_table.py` — overlap table (e.g. LaTeX).

3. **Ablation**  
   `run_ablation_shared_heads.py` does **per-task** ablation: for each task it ablates the top 1, 10, and 100 ranked heads (or fewer if the task has fewer heads). It expects `task_head_rankings.json` from a single-depth run (or the legacy `ablation_candidates.json` list). **Primary metric = value-match rate.** Outputs: `per_task_ablation_summary.json`, delta plots, and overall-by-k summary.  
   `analyze_ablation_run.py` summarizes an ablation run.

## Quick start

From the repo root:

```bash
pip install -r requirements.txt
```

**Single-depth (value-match gate):**
```bash
python experiments/run_single_depth_haystack_plan.py --haystack_csv data/haystack_plan_100_per_task.csv
```

**Per-task ablation (k=1, 10, 100):**
```bash
python experiments/run_ablation_shared_heads.py \
  --haystack_csv data/haystack_plan_100_per_task.csv \
  --candidate_heads_json experiments/outputs/single_depth/run_<TIMESTAMP>/task_head_rankings.json \
  --ablation_k 1,10,100
```

**Analyze a single-depth run:**
```bash
python experiments/analyze_results.py --run_dir experiments/outputs/single_depth/run_<TIMESTAMP>
```

Outputs are written under `experiments/outputs/<pipeline>/run_<timestamp>/`.
