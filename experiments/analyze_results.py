import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

TASK_DATA_TYPE = {
    "registrant_name": "categorical",
    "headquarters_city": "categorical",
    "headquarters_state": "categorical",
    "incorporation_state": "categorical",
    "ceo_lastname": "categorical",
    "incorporation_year": "numerical",
    "employees_count_total": "numerical",
    "employees_count_full_time": "numerical",
    "holder_record_amount": "numerical",
}

TASK_SHORT_NAME = {
    "registrant_name": "registrant",
    "headquarters_city": "hq_city",
    "headquarters_state": "hq_state",
    "incorporation_state": "inc_state",
    "ceo_lastname": "ceo_last",
    "incorporation_year": "inc_year",
    "employees_count_total": "emp_total",
    "employees_count_full_time": "emp_full_time",
    "holder_record_amount": "holder_amt",
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_retrieval_success_rate(task_success, task_attempts, task_value_match, out_dir):
    tasks = sorted(task_attempts.keys())
    rouge_rates = [
        task_success.get(t, 0) / max(1, task_attempts.get(t, 1)) * 100 for t in tasks
    ]
    value_rates = [
        task_value_match.get(t, 0) / max(1, task_attempts.get(t, 1)) * 100 for t in tasks
    ]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, rouge_rates, width, label="ROUGE gate (sentence)", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, value_rates, width, label="Value match (answer)", color="#55A868")

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Retrieval Success Rate by Task")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "retrieval_success_rate.png"), dpi=150)
    plt.close(fig)


def plot_head_sparsity(task_heads, num_layers, num_heads, out_dir):
    tasks = sorted(task_heads.keys())
    total_heads = num_layers * num_heads
    counts = [len(task_heads[t]) for t in tasks]
    percents = [c / total_heads * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(tasks, percents, color="#DD8452")
    ax.set_ylabel("% of All Heads")
    ax.set_title(f"Retrieval Head Sparsity by Task (total heads = {total_heads})")
    ax.set_xticklabels(tasks, rotation=30, ha="right")

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{c}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "head_sparsity.png"), dpi=150)
    plt.close(fig)


def plot_overlap_heatmap(task_heads, out_dir):
    tasks = sorted(task_heads.keys())
    n = len(tasks)
    head_sets = {t: set(tuple(h) for h in task_heads[t]) for t in tasks}

    jaccard = np.zeros((n, n))
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            A, B = head_sets[t1], head_sets[t2]
            if not A and not B:
                jaccard[i, j] = 0.0
            else:
                jaccard[i, j] = len(A & B) / len(A | B)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(jaccard, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tasks)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{jaccard[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Jaccard Overlap")
    ax.set_title("Retrieval Head Overlap Across Tasks (Jaccard)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlap_heatmap_jaccard.png"), dpi=150)
    plt.close(fig)

    pct_overlap = np.zeros((n, n))
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            A, B = head_sets[t1], head_sets[t2]
            if not A:
                pct_overlap[i, j] = 0.0
            else:
                pct_overlap[i, j] = len(A & B) / len(A)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pct_overlap, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tasks)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{pct_overlap[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="% of Row's Heads in Column")
    ax.set_title("Retrieval Head Overlap (% of Row in Column)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlap_heatmap_pct.png"), dpi=150)
    plt.close(fig)


def plot_layer_distribution(task_heads, num_layers, out_dir):
    tasks = sorted(task_heads.keys())

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(num_layers)
    width = 0.8 / max(1, len(tasks))

    for idx, task in enumerate(tasks):
        layer_counts = np.zeros(num_layers)
        for h in task_heads[task]:
            layer_counts[h[0]] += 1
        offset = (idx - len(tasks) / 2 + 0.5) * width
        ax.bar(x + offset, layer_counts, width, label=task, alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of Retrieval Heads")
    ax.set_title("Retrieval Head Distribution by Layer")
    ax.set_xticks(x)
    ax.legend(fontsize=6, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_distribution.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    all_layers = np.zeros(num_layers)
    for task in tasks:
        for h in task_heads[task]:
            all_layers[h[0]] += 1
    ax.bar(x, all_layers, color="#C44E52")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total Retrieval Heads (all tasks)")
    ax.set_title("Aggregate Retrieval Head Distribution by Layer")
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_distribution_aggregate.png"), dpi=150)
    plt.close(fig)


def plot_top_heads(task_heads, out_dir, top_k=20):
    head_task_count = {}
    tasks = sorted(task_heads.keys())
    for task in tasks:
        for h in task_heads[task]:
            key = tuple(h)
            head_task_count.setdefault(key, set())
            head_task_count[key].add(task)

    ranked = sorted(head_task_count.items(), key=lambda x: len(x[1]), reverse=True)[:top_k]

    labels = [f"L{h[0]}H{h[1]}" for h, _ in ranked]
    counts = [len(t) for _, t in ranked]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(labels)), counts, color="#8172B2")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of Tasks Using This Head")
    ax.set_title(f"Top {top_k} Most Shared Retrieval Heads")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "top_shared_heads.png"), dpi=150)
    plt.close(fig)


def plot_rouge_vs_value(task_success, task_attempts, task_value_match, out_dir):
    tasks = sorted(task_attempts.keys())
    rouge_rates = [task_success.get(t, 0) / max(1, task_attempts.get(t, 1)) * 100 for t in tasks]
    value_rates = [task_value_match.get(t, 0) / max(1, task_attempts.get(t, 1)) * 100 for t in tasks]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(rouge_rates, value_rates, s=80, color="#4C72B0", zorder=5)
    for i, t in enumerate(tasks):
        ax.annotate(t, (rouge_rates[i], value_rates[i]), fontsize=7, ha="left", va="bottom")

    lims = [0, 105]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("ROUGE Gate Success Rate (%)")
    ax.set_ylabel("Value Match Rate (%)")
    ax.set_title("ROUGE Gate vs Value Match by Task")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rouge_vs_value.png"), dpi=150)
    plt.close(fig)


def plot_overlap_by_data_type(task_heads, out_dir):
    tasks = sorted(task_heads.keys())
    head_sets = {t: set(tuple(h) for h in task_heads[t]) for t in tasks}
    type_labels = sorted(set(TASK_DATA_TYPE.get(t, "other") for t in tasks))

    # Aggregate pairwise Jaccard by type pair.
    grouped = {}
    cross_pairs = []
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            if i >= j:
                continue
            A, B = head_sets[t1], head_sets[t2]
            jac = len(A & B) / len(A | B) if (A or B) else 0.0

            ty1 = TASK_DATA_TYPE.get(t1, "other")
            ty2 = TASK_DATA_TYPE.get(t2, "other")
            pair_key = tuple(sorted((ty1, ty2)))
            grouped.setdefault(pair_key, []).append(jac)

            if ty1 != ty2:
                s1 = TASK_SHORT_NAME.get(t1, t1)
                s2 = TASK_SHORT_NAME.get(t2, t2)
                cross_pairs.append((f"{s1} vs {s2}", jac))

    mean_grid = np.zeros((len(type_labels), len(type_labels)))
    count_grid = np.zeros((len(type_labels), len(type_labels)))
    max_grid = np.zeros((len(type_labels), len(type_labels)))

    for i, ty1 in enumerate(type_labels):
        for j, ty2 in enumerate(type_labels):
            key = tuple(sorted((ty1, ty2)))
            vals = grouped.get(key, [])
            if vals:
                mean_grid[i, j] = float(np.mean(vals))
                count_grid[i, j] = len(vals)
                max_grid[i, j] = float(np.max(vals))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    im = axes[0].imshow(mean_grid, cmap="Purples", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(type_labels)))
    axes[0].set_xticklabels(type_labels, rotation=20, ha="right")
    axes[0].set_yticks(range(len(type_labels)))
    axes[0].set_yticklabels(type_labels)
    axes[0].set_title("Average Head Overlap by Data Type")
    for i in range(len(type_labels)):
        for j in range(len(type_labels)):
            axes[0].text(
                j,
                i,
                f"{mean_grid[i, j]:.2f}\n(n={int(count_grid[i, j])})\nmax={max_grid[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(im, ax=axes[0], label="Mean Jaccard")

    cross_pairs.sort(key=lambda x: x[1], reverse=True)
    top_cross = cross_pairs[:10]
    if top_cross:
        labels = [x[0] for x in top_cross]
        vals = [x[1] for x in top_cross]
        y = np.arange(len(labels))
        axes[1].barh(y, vals, color="#55A868")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1.05)
        axes[1].set_xlabel("Jaccard Overlap")
        axes[1].set_title("Top Cross-Type Task Overlaps")
    else:
        axes[1].text(0.5, 0.5, "No cross-type task pairs found", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlap_by_data_type.png"), dpi=150)
    plt.close(fig)


def build_ablation_candidates(task_heads, task_success, task_attempts):
    tasks = sorted(task_heads.keys())
    rouge_rate = {
        t: (task_success.get(t, 0) / max(1, task_attempts.get(t, 1)))
        for t in tasks
    }

    head_to_tasks = {}
    for t in tasks:
        for h in task_heads.get(t, []):
            key = tuple(h)
            head_to_tasks.setdefault(key, set()).add(t)

    candidates = []
    for head, supporting_tasks in head_to_tasks.items():
        support_count = len(supporting_tasks)
        weighted_support = float(sum(rouge_rate[t] for t in supporting_tasks))
        type_count = len(set(TASK_DATA_TYPE.get(t, "other") for t in supporting_tasks))
        # Prioritize heads that are both successful and cross-task/cross-type.
        priority_score = weighted_support + 0.1 * support_count + 0.15 * (type_count - 1)

        candidates.append(
            {
                "head": [head[0], head[1]],
                "label": f"L{head[0]}H{head[1]}",
                "support_count": support_count,
                "weighted_support": weighted_support,
                "type_count": type_count,
                "priority_score": priority_score,
                "tasks": sorted(supporting_tasks),
            }
        )

    candidates.sort(
        key=lambda x: (
            x["priority_score"],
            x["support_count"],
            x["weighted_support"],
        ),
        reverse=True,
    )
    return candidates


def plot_ablation_candidates(task_heads, task_success, task_attempts, out_dir, top_k=20):
    tasks = sorted(task_heads.keys())
    rouge_rate = {
        t: (task_success.get(t, 0) / max(1, task_attempts.get(t, 1)))
        for t in tasks
    }
    candidates = build_ablation_candidates(task_heads, task_success, task_attempts)
    top = candidates[:top_k]

    if not top:
        return

    # Plot 1: ranked ablation candidates.
    labels = [c["label"] for c in top]
    scores = [c["priority_score"] for c in top]
    support = [c["support_count"] for c in top]
    weighted = [c["weighted_support"] for c in top]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(labels))))
    bars = ax.barh(y, scores, color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Ablation Priority Score")
    ax.set_title("Top Heads to Ablate First (Cross-Task Retrieval Candidates)")

    for i, (bar, sc, ws) in enumerate(zip(bars, support, weighted)):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"tasks={sc}, ws={ws:.2f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_candidate_ranking.png"), dpi=150)
    plt.close(fig)

    # Plot 2: head-by-task support map, weighted by task ROUGE success.
    matrix = np.zeros((len(top), len(tasks)))
    for i, cand in enumerate(top):
        task_set = set(cand["tasks"])
        for j, t in enumerate(tasks):
            if t in task_set:
                matrix[i, j] = rouge_rate[t]

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(tasks)), max(6, 0.35 * len(top))))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(rouge_rate.values() or [1.0]))
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_SHORT_NAME.get(t, t) for t in tasks], rotation=30, ha="right")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)
    ax.set_title("Ablation Candidate Head Support by Task (cell = ROUGE success rate)")
    ax.set_xlabel("Task")
    ax.set_ylabel("Head")
    fig.colorbar(im, ax=ax, label="Task ROUGE Success Rate")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_candidate_taskmap.png"), dpi=150)
    plt.close(fig)

    # Save candidate ranking for direct use in ablation scripts.
    out_path = os.path.join(out_dir, "ablation_candidates.json")
    with open(out_path, "w") as f:
        json.dump(top, f, indent=2)


def print_summary(task_heads, task_success, task_attempts, task_value_match, num_layers, num_heads):
    total_heads = num_layers * num_heads
    tasks = sorted(task_attempts.keys())

    print("=" * 80)
    print("RETRIEVAL HEAD ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nModel: {num_layers} layers x {num_heads} heads = {total_heads} total heads\n")

    print(f"{'Task':<30} {'Attempts':>8} {'ROUGE OK':>8} {'ValMatch':>8} {'Heads':>6} {'%Heads':>7}")
    print("-" * 80)
    for t in tasks:
        att = task_attempts.get(t, 0)
        suc = task_success.get(t, 0)
        val = task_value_match.get(t, 0)
        nh = len(task_heads.get(t, []))
        pct = nh / total_heads * 100
        print(f"{t:<30} {att:>8} {suc:>8} {val:>8} {nh:>6} {pct:>6.1f}%")

    all_heads = set()
    for t in tasks:
        for h in task_heads.get(t, []):
            all_heads.add(tuple(h))
    print(f"\nUnique retrieval heads across all tasks: {len(all_heads)} ({len(all_heads)/total_heads*100:.1f}%)")

    if len(tasks) >= 2:
        overlaps = []
        for i, t1 in enumerate(tasks):
            for j, t2 in enumerate(tasks):
                if i >= j:
                    continue
                A = set(tuple(h) for h in task_heads.get(t1, []))
                B = set(tuple(h) for h in task_heads.get(t2, []))
                if A or B:
                    jac = len(A & B) / len(A | B)
                    overlaps.append((t1, t2, jac))
        overlaps.sort(key=lambda x: x[2], reverse=True)
        print(f"\nTop pairwise overlaps (Jaccard):")
        for t1, t2, jac in overlaps[:10]:
            print(f"  {t1} x {t2}: {jac:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to run output directory")
    parser.add_argument("--num_layers", type=int, default=32, help="Number of model layers")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--top_k", type=int, default=20, help="Top K shared heads to show")
    parser.add_argument(
        "--ablation_top_k",
        type=int,
        default=20,
        help="Number of top ablation candidates to plot/save",
    )
    args = parser.parse_args()

    task_heads = load_json(os.path.join(args.run_dir, "task_heads.json"))
    task_success = load_json(os.path.join(args.run_dir, "task_success.json"))
    task_attempts = load_json(os.path.join(args.run_dir, "task_attempts.json"))
    task_value_match = load_json(os.path.join(args.run_dir, "task_value_match.json"))

    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print_summary(task_heads, task_success, task_attempts, task_value_match, args.num_layers, args.num_heads)

    plot_retrieval_success_rate(task_success, task_attempts, task_value_match, plots_dir)
    plot_head_sparsity(task_heads, args.num_layers, args.num_heads, plots_dir)
    plot_overlap_heatmap(task_heads, plots_dir)
    plot_layer_distribution(task_heads, args.num_layers, plots_dir)
    plot_top_heads(task_heads, plots_dir, top_k=args.top_k)
    plot_rouge_vs_value(task_success, task_attempts, task_value_match, plots_dir)
    plot_overlap_by_data_type(task_heads, plots_dir)
    plot_ablation_candidates(
        task_heads,
        task_success,
        task_attempts,
        plots_dir,
        top_k=args.ablation_top_k,
    )

    print(f"\nPlots saved to: {plots_dir}/")


if __name__ == "__main__":
    main()
