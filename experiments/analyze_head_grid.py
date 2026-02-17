"""
Generates a per-task retrieval head grid (layer x head heatmap) and a
union/intersection grid across all tasks, similar to Figure 3 in the Wu paper.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def make_binary_grid(heads_list, num_layers, num_heads):
    grid = np.zeros((num_layers, num_heads))
    for h in heads_list:
        grid[h[0]][h[1]] = 1.0
    return grid


def plot_single_task_grid(task, heads_list, num_layers, num_heads, out_dir):
    grid = make_binary_grid(heads_list, num_layers, num_heads)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = ListedColormap(["white", "#4C72B0"])
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(f"Retrieval Heads for: {task}  (n={len(heads_list)})")
    ax.set_xticks(range(0, num_heads, max(1, num_heads // 8)))
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 8)))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"grid_{task}.png"), dpi=150)
    plt.close(fig)


def plot_task_count_grid(task_heads, num_layers, num_heads, out_dir):
    tasks = sorted(task_heads.keys())
    count_grid = np.zeros((num_layers, num_heads))
    for task in tasks:
        for h in task_heads[task]:
            count_grid[h[0]][h[1]] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(count_grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=len(tasks))
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(f"Number of Tasks Using Each Head  (max = {len(tasks)})")
    ax.set_xticks(range(0, num_heads, max(1, num_heads // 8)))
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 8)))
    fig.colorbar(im, ax=ax, label="Task Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "grid_task_count.png"), dpi=150)
    plt.close(fig)


def plot_union_intersection_grid(task_heads, num_layers, num_heads, out_dir):
    tasks = sorted(task_heads.keys())
    if len(tasks) < 2:
        return

    head_sets = [set(tuple(h) for h in task_heads[t]) for t in tasks]

    union = head_sets[0]
    intersection = head_sets[0]
    for s in head_sets[1:]:
        union = union | s
        intersection = intersection & s

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    union_grid = make_binary_grid(list(union), num_layers, num_heads)
    cmap_blue = ListedColormap(["white", "#4C72B0"])
    axes[0].imshow(union_grid, cmap=cmap_blue, aspect="auto", vmin=0, vmax=1)
    axes[0].set_xlabel("Head Index")
    axes[0].set_ylabel("Layer Index")
    axes[0].set_title(f"Union of All Tasks  (n={len(union)})")

    inter_grid = make_binary_grid(list(intersection), num_layers, num_heads)
    cmap_red = ListedColormap(["white", "#C44E52"])
    axes[1].imshow(inter_grid, cmap=cmap_red, aspect="auto", vmin=0, vmax=1)
    axes[1].set_xlabel("Head Index")
    axes[1].set_ylabel("Layer Index")
    axes[1].set_title(f"Intersection of All Tasks  (n={len(intersection)})")

    for ax in axes:
        ax.set_xticks(range(0, num_heads, max(1, num_heads // 8)))
        ax.set_yticks(range(0, num_layers, max(1, num_layers // 8)))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "grid_union_intersection.png"), dpi=150)
    plt.close(fig)


def plot_pairwise_overlap_bar(task_heads, out_dir):
    tasks = sorted(task_heads.keys())
    head_sets = {t: set(tuple(h) for h in task_heads[t]) for t in tasks}

    pairs = []
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            if i >= j:
                continue
            A, B = head_sets[t1], head_sets[t2]
            inter = len(A & B)
            union = len(A | B)
            jaccard = inter / union if union else 0
            pairs.append((f"{t1}\nvs\n{t2}", jaccard, inter))

    pairs.sort(key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]
    intersections = [p[2] for p in pairs]

    fig, ax = plt.subplots(figsize=(max(12, len(pairs) * 0.8), 6))
    bars = ax.bar(range(len(labels)), scores, color="#55A868")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    ax.set_ylabel("Jaccard Overlap")
    ax.set_title("Pairwise Retrieval Head Overlap")
    ax.set_ylim(0, 1.05)

    for bar, inter in zip(bars, intersections):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"|∩|={inter}", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pairwise_overlap_bar.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=32)
    args = parser.parse_args()

    task_heads = load_json(os.path.join(args.run_dir, "task_heads.json"))

    plots_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    tasks = sorted(task_heads.keys())
    for task in tasks:
        plot_single_task_grid(task, task_heads[task], args.num_layers, args.num_heads, plots_dir)

    plot_task_count_grid(task_heads, args.num_layers, args.num_heads, plots_dir)
    plot_union_intersection_grid(task_heads, args.num_layers, args.num_heads, plots_dir)
    plot_pairwise_overlap_bar(task_heads, plots_dir)

    print(f"Head grid plots saved to: {plots_dir}/")


if __name__ == "__main__":
    main()
