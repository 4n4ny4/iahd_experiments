import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def overall_rates(res):
    attempts = sum(res["task_attempts"].values())
    rouge = sum(res["task_success"].values()) / max(1, attempts)
    value = sum(res["task_value_match"].values()) / max(1, attempts)
    return rouge, value


def plot_overall_delta(summary, out_dir):
    baseline = summary["baseline"]
    conds = summary["conditions"]
    names = list(conds.keys())
    rouge_delta = [100 * conds[n]["delta_rouge_rate_vs_baseline"] for n in names]
    value_delta = [100 * conds[n]["delta_value_rate_vs_baseline"] for n in names]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.8), 5))
    ax.bar(x - width / 2, rouge_delta, width, label="ROUGE delta", color="#4C72B0")
    ax.bar(x + width / 2, value_delta, width, label="Value delta", color="#55A868")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Delta vs baseline (percentage points)")
    ax.set_title(
        "Overall Ablation Effect vs Baseline\n"
        f"Baseline: ROUGE={100*baseline['rouge_rate']:.1f}%, Value={100*baseline['value_rate']:.1f}%"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "viz_overall_delta_vs_baseline.png"), dpi=150)
    plt.close(fig)


def plot_random_control_gap(summary, out_dir):
    conds = summary["conditions"]
    top = conds["top_shared_ablation"]["delta_rouge_rate_vs_baseline"]
    random_keys = sorted([k for k in conds if k.startswith("random_ablation_")])
    random_vals = [conds[k]["delta_rouge_rate_vs_baseline"] for k in random_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(np.arange(len(random_vals)), [100 * v for v in random_vals], color="#DD8452", label="Random controls")
    ax.axhline(100 * top, color="#C44E52", linestyle="--", label="Top-shared ablation")
    ax.axhline(100 * np.mean(random_vals), color="#8172B2", linestyle=":", label="Random mean")
    ax.set_xticks(np.arange(len(random_vals)))
    ax.set_xticklabels(random_keys, rotation=20, ha="right")
    ax.set_ylabel("ROUGE delta vs baseline (percentage points)")
    ax.set_title("Top-shared Ablation vs Random Controls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "viz_top_vs_random_controls.png"), dpi=150)
    plt.close(fig)


def plot_per_task_delta(condition_results, out_dir):
    baseline = condition_results["baseline"]
    top = condition_results["top_shared_ablation"]
    random_keys = sorted([k for k in condition_results if k.startswith("random_ablation_")])
    tasks = sorted(baseline["task_attempts"].keys())

    top_delta = []
    rand_mean = []
    rand_min = []
    rand_max = []

    for task in tasks:
        b_rate = baseline["task_success"].get(task, 0) / max(1, baseline["task_attempts"].get(task, 0))
        t_rate = top["task_success"].get(task, 0) / max(1, top["task_attempts"].get(task, 0))
        top_delta.append(100 * (t_rate - b_rate))

        rv = []
        for rk in random_keys:
            rr = condition_results[rk]["task_success"].get(task, 0) / max(1, condition_results[rk]["task_attempts"].get(task, 0))
            rv.append(100 * (rr - b_rate))
        rand_mean.append(float(np.mean(rv)))
        rand_min.append(float(np.min(rv)))
        rand_max.append(float(np.max(rv)))

    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, top_delta, width=0.38, label="Top-shared", color="#4C72B0")
    ax.bar(x + 0.2, rand_mean, width=0.38, label="Random mean", color="#55A868")
    ax.errorbar(x + 0.2, rand_mean, yerr=[np.array(rand_mean) - np.array(rand_min), np.array(rand_max) - np.array(rand_mean)],
                fmt="none", ecolor="#2F6B4F", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_ylabel("ROUGE delta vs baseline (percentage points)")
    ax.set_title("Per-task ROUGE Impact: Top-shared vs Random Controls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "viz_per_task_top_vs_random.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    summary = load_json(os.path.join(args.run_dir, "summary.json"))
    condition_results = load_json(os.path.join(args.run_dir, "condition_results.json"))

    plot_overall_delta(summary, args.run_dir)
    plot_random_control_gap(summary, args.run_dir)
    plot_per_task_delta(condition_results, args.run_dir)

    print(f"Saved visualizations to: {args.run_dir}")


if __name__ == "__main__":
    main()
