# Recommended Next Steps

## Why this roadmap

Current results show promising causal signal from top-shared head ablation, but control variance remains high and ROUGE gating appears to miss entity-style retrieval behavior. The plan below maximizes evidence quality and paper readiness.

## Phase 1: Strengthen causal evidence (highest priority)

1. Increase random controls for `k=4`:
   - Target `10-20` random 4-head sets.
   - Keep all other settings fixed for fair comparison.

2. Run ablation `k`-sweep:
   - `k = 1, 2, 4, 8, 12`.
   - For each `k`, compare top-shared vs matched random controls.

3. Report key outcomes:
   - Overall ROUGE/value deltas vs baseline.
   - Per-task deltas.
   - Top-shared minus random-mean gap at each `k`.

### Success criterion

Top-shared ablations consistently outperform random controls (larger retrieval drop) across multiple `k` values.

---

## Phase 2: Fix the gating limitation

1. Evaluate detection gates:
   - ROUGE-only (current baseline),
   - value-only,
   - hybrid gate (`value_match OR semantic_match`) for entity tasks.

2. Re-run head discovery under each gate.
3. Compare:
   - discovered head sets,
   - overlap patterns,
   - downstream ablation impact.

### Why this matters

Two tasks had zero ROUGE success but strong value match, suggesting ROUGE-only gating under-detects entity retrieval heads.

---

## Phase 3: Wu vs shuffle-robust method (main paper comparison)

1. Use sentence-shuffled contexts (same tasks, same row budget).
2. Compare:
   - Wu-style ROUGE-gated detector,
   - shuffle-robust counterfactual/value-sensitive detector.

3. Evaluate by causal ablation effectiveness, not just overlap plots.

### Success criterion

The shuffle-robust method discovers heads whose ablation causes larger and more consistent retrieval drops than Wu baseline under shuffled text.

---

## Deliverables for a paper-ready package

### Core figures

- Top-shared vs random-control gap by `k`.
- Per-task delta heatmaps.
- Method comparison under shuffled text.
- Gate-sensitivity comparison (ROUGE vs value/hybrid).

### Core tables

- Effect sizes with confidence intervals.
- Head-set overlaps across methods/gates.
- Robustness across seeds.

### Appendix

- Random head sets used.
- Seeds, run configs, and exact command lines.
- Additional per-task breakdowns.

---

## Suggested two-week execution plan

### Week 1

- Complete `k=4` with more random controls.
- Run `k=8` and `k=12`.
- Build summary plots/tables with control-adjusted effect sizes.

### Week 2

- Implement and benchmark hybrid/entity-aware gate.
- Run sentence-shuffle comparison: Wu vs shuffle-robust method.
- Finalize methods and results narrative for drafting.
