"""
Generates a LaTeX-ready overlap table and a concise Markdown summary
of key research findings from the single-depth experiment.
"""
import argparse
import json
import os

import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    args = parser.parse_args()

    task_heads = load_json(os.path.join(args.run_dir, "task_heads.json"))
    task_success = load_json(os.path.join(args.run_dir, "task_success.json"))
    task_attempts = load_json(os.path.join(args.run_dir, "task_attempts.json"))
    task_value_match = load_json(os.path.join(args.run_dir, "task_value_match.json"))

    total = args.num_layers * args.num_heads
    tasks = sorted(task_heads.keys())
    head_sets = {t: set(tuple(h) for h in task_heads[t]) for t in tasks}

    # ---- Jaccard matrix ----
    n = len(tasks)
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A, B = head_sets[tasks[i]], head_sets[tasks[j]]
            jaccard[i, j] = len(A & B) / len(A | B) if A | B else 0

    # ---- Intersection / Union counts ----
    all_heads = set()
    for s in head_sets.values():
        all_heads |= s
    inter_all = head_sets[tasks[0]] if tasks else set()
    for s in head_sets.values():
        inter_all &= s

    # ---- Markdown report ----
    lines = []
    lines.append("# Retrieval Head Overlap Analysis\n")
    lines.append(f"Model: Llama-3-8B-Instruct ({args.num_layers}L x {args.num_heads}H = {total} heads)\n")

    lines.append("## Per-Task Summary\n")
    lines.append(f"| {'Task':<28} | {'Attempts':>8} | {'ROUGE OK':>8} | {'Value Match':>11} | {'#Heads':>6} | {'%Heads':>6} |")
    lines.append(f"|{'-'*30}|{'-'*10}|{'-'*10}|{'-'*13}|{'-'*8}|{'-'*8}|")
    for t in tasks:
        att = task_attempts.get(t, 0)
        suc = task_success.get(t, 0)
        val = task_value_match.get(t, 0)
        nh = len(head_sets[t])
        pct = nh / total * 100
        lines.append(f"| {t:<28} | {att:>8} | {suc:>8} | {val:>11} | {nh:>6} | {pct:>5.1f}% |")

    lines.append(f"\n## Overlap Statistics\n")
    lines.append(f"- Union of all retrieval heads: **{len(all_heads)}** ({len(all_heads)/total*100:.1f}%)")
    lines.append(f"- Intersection of all retrieval heads: **{len(inter_all)}** ({len(inter_all)/total*100:.1f}%)")

    lines.append(f"\n## Jaccard Overlap Matrix\n")
    header = "| " + " | ".join([""] + [t[:12] for t in tasks]) + " |"
    sep = "| " + " | ".join(["---"] * (n + 1)) + " |"
    lines.append(header)
    lines.append(sep)
    for i, t in enumerate(tasks):
        row = f"| {t[:12]} | " + " | ".join(f"{jaccard[i, j]:.2f}" for j in range(n)) + " |"
        lines.append(row)

    lines.append(f"\n## Top Pairwise Overlaps\n")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((tasks[i], tasks[j], jaccard[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for t1, t2, jac in pairs[:10]:
        lines.append(f"- {t1} x {t2}: Jaccard = {jac:.3f}")

    lines.append(f"\n## Key Research Findings\n")
    mean_jac = np.mean([jaccard[i, j] for i in range(n) for j in range(i + 1, n)]) if n > 1 else 0
    lines.append(f"1. **Average pairwise Jaccard overlap**: {mean_jac:.3f}")
    lines.append(f"2. **Retrieval heads are {'highly' if mean_jac > 0.5 else 'moderately' if mean_jac > 0.2 else 'minimally'} shared** across the {n} tested tasks.")
    lines.append(f"3. **Head sparsity**: Tasks use between {min(len(s) for s in head_sets.values())} and {max(len(s) for s in head_sets.values())} retrieval heads ({min(len(s)/total*100 for s in head_sets.values()):.1f}%-{max(len(s)/total*100 for s in head_sets.values()):.1f}% of all heads).")
    lines.append(f"4. **Core retrieval heads**: {len(inter_all)} heads are retrieval heads for *every* task tested.")

    report = "\n".join(lines)
    out_path = os.path.join(args.run_dir, "analysis_report.md")
    with open(out_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {out_path}")

    # ---- LaTeX table ----
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    cols = "l" + "c" * n
    latex_lines.append(f"\\begin{{tabular}}{{{cols}}}")
    latex_lines.append("\\toprule")
    latex_lines.append(" & ".join([""] + [t.replace("_", "\\_")[:14] for t in tasks]) + " \\\\")
    latex_lines.append("\\midrule")
    for i, t in enumerate(tasks):
        cells = []
        for j in range(n):
            if i == j:
                cells.append("---")
            else:
                cells.append(f"{jaccard[i, j]:.2f}")
        latex_lines.append(t.replace("_", "\\_")[:14] + " & " + " & ".join(cells) + " \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Jaccard overlap of retrieval heads across EDGAR tasks.}")
    latex_lines.append("\\label{tab:overlap}")
    latex_lines.append("\\end{table}")

    latex_path = os.path.join(args.run_dir, "overlap_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
