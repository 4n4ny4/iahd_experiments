# Running the ROUGE-Gated Ablation Pipeline

## Summary of Changes

`run_single_depth_haystack_plan.py` now supports a `--gate` option:

- **`rouge`** (default): ROUGE-1 recall vs needle sentence > 50% — surfaces retrieval heads for more tasks including entity extraction
- **`value`**: Value match only (previous behavior)
- **`hybrid`**: Value match AND ROUGE > threshold — correct answers with clear retrieval signal

## Step 1: Head Identification (ROUGE-gated)

```bash
cd /home/ubuntu/iahd_experiments

python experiments/run_single_depth_haystack_plan.py \
  --haystack_csv data/haystack_plan_100_per_task.csv \
  --gate rouge \
  --output_dir experiments/outputs/single_depth
```

This creates `experiments/outputs/single_depth/run_<TIMESTAMP>/` with:
- `task_head_rankings.json` — per-task ranked heads (use for ablation)
- `task_heads.json`
- `task_value_match.json`, `task_rouge_ok.json`
- `run_meta.json` — includes `gate` and `rouge_threshold`

**Optional flags:**
- `--rouge_threshold 0.3` — lower threshold to include more rows (default 0.5)
- `--threshold 0.05` — lower head score threshold to surface more heads
- `--max_rows 50` — limit rows for quick tests

## Step 2: Per-Task Ablations

```bash
# Replace <TIMESTAMP> with the run from Step 1
RUN_DIR=experiments/outputs/single_depth/run_<TIMESTAMP>

python experiments/run_ablation_shared_heads.py \
  --haystack_csv data/haystack_plan_100_per_task.csv \
  --candidate_heads_json ${RUN_DIR}/task_head_rankings.json \
  --ablation_k "1,10,100" \
  --output_dir experiments/outputs/ablation
```

Ablation evaluates **value match** as the success metric.

## One-Liner (uses latest run)

```bash
cd /home/ubuntu/iahd_experiments

# Step 1
python experiments/run_single_depth_haystack_plan.py \
  --haystack_csv data/haystack_plan_100_per_task.csv \
  --gate rouge

# Step 2
LATEST=$(ls -td experiments/outputs/single_depth/run_* 2>/dev/null | head -1)
python experiments/run_ablation_shared_heads.py \
  --haystack_csv data/haystack_plan_100_per_task.csv \
  --candidate_heads_json "${LATEST}/task_head_rankings.json" \
  --ablation_k "1,10,100"
```

## Using Existing ROUGE Run (no re-run)

If you have the ROUGE run from `single_depth_rouge`, you can run ablation with its `task_heads.json`:

```bash
python experiments/run_ablation_shared_heads.py \
  --haystack_csv data/haystack_plan_100_per_task.csv \
  --candidate_heads_json experiments/outputs/single_depth_rouge/run_20260217_095958/task_heads.json \
  --ablation_k "1,10,100"
```

Note: `task_heads` has no scores, so head order is by (layer, head). For ranked heads by retrieval score, run Step 1 with `--gate rouge`.

## Step 3: Visualize Ablation Results

To regenerate or view plots from an ablation run:

```bash
python experiments/visualize_ablation_run.py experiments/outputs/ablation/run_<TIMESTAMP>
```

This creates PNG plots in the run directory:
- `ablation_within_task_delta_heatmap.png` — Delta accuracy (pp) vs baseline per task
- `ablation_within_task_accuracy.png` — Grouped bars (Baseline, k=1,5,10) per task
- `ablation_within_task_aggregate.png` — Aggregate within-task accuracy
- `ablation_across_task_accuracy.png` — Aggregate when ablating global top heads
