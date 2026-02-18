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
│   │   ├── single_depth/           # ROUGE-gated single-depth head discovery
│   │   ├── single_depth_output_gate/  # Value-in-output-gated single-depth
│   │   └── ablation/               # Ablation runs (shared-head removal)
│   ├── run_*.py             # Run pipelines (head discovery, ablation)
│   └── analyze_*.py         # Analysis and plotting
├── requirements.txt
└── README.md
```

## Pipeline overview

1. **Head discovery (single depth)**  
   Run a haystack CSV through the model; for each “success” (by ROUGE or value match), accumulate attention-to-needle scores and select retrieval heads by threshold.
   - **ROUGE gate**: `run_single_depth_haystack_plan.py` — success = ROUGE-1 recall > 50 vs needle sentence.
   - **Output gate**: `run_single_depth_haystack_output_gate.py` — success = decoded output contains `needle_value`; uses `data/haystack_plan_100_per_task.csv` with equal rows per task.

2. **Analysis**  
   Point analysis scripts at a run directory (e.g. `experiments/outputs/single_depth/run_YYYYMMDD_HHMMSS`):
   - `analyze_results.py` — plots, ROUGE vs value summary.
   - `analyze_head_grid.py` — head heatmaps.
   - `analyze_overlap_table.py` — overlap table (e.g. LaTeX).

3. **Ablation**  
   `run_ablation_shared_heads.py` takes candidate heads (e.g. from `analyze_results.py` → `ablation_candidates.json`) and runs shared-head ablation; results go under `experiments/outputs/ablation/`.  
   `analyze_ablation_run.py` summarizes an ablation run.

## Quick start

From the repo root:

```bash
pip install -r requirements.txt
```

**Single-depth (ROUGE gate):**
```bash
python experiments/run_single_depth_haystack_plan.py --haystack_csv data/haystack_plan_100_per_task.csv
```

**Single-depth (output gate, balanced tasks):**
```bash
python experiments/run_single_depth_haystack_output_gate.py
```

**Analyze a run:**
```bash
python experiments/analyze_results.py --run_dir experiments/outputs/single_depth/run_20260217_095958
```

Outputs are written under `experiments/outputs/<pipeline>/run_<timestamp>/`.
