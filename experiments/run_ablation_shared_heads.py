import argparse
import csv
import json
import os
import random
import re
from contextlib import nullcontext
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


TASK_QUESTIONS = {
    "registrant_name": "What is the registrant name?",
    "headquarters_city": "What is the headquarters city?",
    "headquarters_state": "What is the headquarters state?",
    "incorporation_state": "What is the incorporation state?",
    "incorporation_year": "What is the incorporation year?",
    "employees_count_total": "What is the total employee count?",
    "employees_count_full_time": "What is the full-time employee count?",
    "ceo_lastname": "What is the CEO's last name?",
    "holder_record_amount": "What is the holder record amount?",
}


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip().lower()


def insert_needle_at_depth(tokenizer, context, needle, question, depth_percent, buffer_tokens=None):
    tokens_context = tokenizer.encode(context)
    tokens_needle = tokenizer.encode(needle)

    if buffer_tokens is None:
        messages = [
            {
                "role": "user",
                "content": f"<document></document>\nBased on the content of the document, Question: {question}\nAnswer:",
            }
        ]
        base_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        base_len = base_ids.shape[-1] if hasattr(base_ids, "shape") else len(base_ids)
        buffer_tokens = base_len + len(tokens_needle)

    target_len = max(0, len(tokens_context) - buffer_tokens)
    if len(tokens_context) + len(tokens_needle) > target_len:
        tokens_context = tokens_context[: max(0, target_len - len(tokens_needle))]

    if depth_percent >= 100:
        tokens_new_context = tokens_context + tokens_needle
    else:
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        tokens_new_context = tokens_context[:insertion_point]
        period_tokens = tokenizer.encode(".")
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    return tokenizer.decode(tokens_new_context)


def greedy_decode(model, tokenizer, prompt_ids, max_decode):
    with torch.no_grad():
        outputs = model(
            input_ids=prompt_ids[:, :-1],
            use_cache=True,
            return_dict=True,
            output_attentions=False,
        )
        past_kv = outputs.past_key_values
        inp = prompt_ids[:, -1:]
        current_position = prompt_ids.size(1) - 1
        device = prompt_ids.device

        generated = []
        for _ in range(max_decode):
            position_ids = torch.tensor([[current_position]], dtype=torch.long, device=device)
            out = model(
                input_ids=inp,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_id = out.logits[:, -1].argmax(dim=-1)
            generated.append(next_id.item())
            if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
                break
            current_position += 1
            inp = next_id.unsqueeze(1)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


class HeadAblationHooks:
    """Zero selected attention head channels before o_proj."""

    def __init__(self, model, heads):
        self.model = model
        self.heads = heads
        self.handles = []
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.by_layer = {}
        for l, h in heads:
            self.by_layer.setdefault(l, set()).add(h)

    def __enter__(self):
        for layer_idx, layer_heads in self.by_layer.items():
            if layer_idx >= len(self.model.model.layers):
                continue
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj

            def pre_hook(_module, inputs, heads=sorted(layer_heads)):
                x = inputs[0]
                if x.ndim != 3:
                    return inputs
                x = x.clone()
                for head_idx in heads:
                    start = head_idx * self.head_dim
                    end = start + self.head_dim
                    x[..., start:end] = 0
                return (x,)

            self.handles.append(o_proj.register_forward_pre_hook(pre_hook))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def load_top_heads(candidate_heads_json, top_k):
    with open(candidate_heads_json, "r") as f:
        candidates = json.load(f)
    top = []
    for c in candidates[:top_k]:
        head = c.get("head")
        if isinstance(head, list) and len(head) == 2:
            top.append((int(head[0]), int(head[1])))
    return top


def sample_random_head_sets(num_layers, num_heads, k, n_sets, seed, exclude):
    rng = random.Random(seed)
    universe = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    allowed = [x for x in universe if x not in set(exclude)]
    if len(allowed) < k:
        raise ValueError("Not enough heads left to sample random control sets.")
    out = []
    for _ in range(n_sets):
        out.append(sorted(rng.sample(allowed, k)))
    return out


def evaluate_condition(
    model,
    tokenizer,
    scorer,
    args,
    condition_name,
    ablation_heads=None,
):
    task_attempts = {}
    task_success = {}
    task_value_match = {}
    rows_seen_per_task = {}
    rows_scanned = 0

    hook_ctx = HeadAblationHooks(model, ablation_heads) if ablation_heads else nullcontext()
    with hook_ctx:
        with open(args.haystack_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=f"{condition_name} rows"):
                rows_scanned += 1
                if args.max_rows is not None and rows_scanned > args.max_rows:
                    break

                task = row.get("task")
                needle = row.get("needle_sentence", "")
                needle_value = row.get("needle_value", "")
                haystack = row.get("haystack_text", "")
                if not task or not needle or not haystack:
                    continue

                rows_seen_per_task.setdefault(task, 0)
                if rows_seen_per_task[task] >= args.max_rows_per_task:
                    continue
                rows_seen_per_task[task] += 1

                task_attempts.setdefault(task, 0)
                task_success.setdefault(task, 0)
                task_value_match.setdefault(task, 0)

                if normalize_text(needle) in normalize_text(haystack):
                    continue

                question = TASK_QUESTIONS.get(task, f"What is the {task}?")
                context = insert_needle_at_depth(tokenizer, haystack, needle, question, 50)
                context_tokens = tokenizer.encode(context)
                if len(context_tokens) > args.target_tokens:
                    context = tokenizer.decode(context_tokens[: args.target_tokens])
                elif len(context_tokens) < args.target_tokens:
                    padding_tokens = tokenizer.encode(" " * (args.target_tokens - len(context_tokens)))
                    context = tokenizer.decode(context_tokens + padding_tokens)

                if normalize_text(needle) not in normalize_text(context):
                    continue

                task_attempts[task] += 1
                messages = [
                    {
                        "role": "user",
                        "content": f"<document>{context}</document>\nBased on the content of the document, Question: {question}\nAnswer:",
                    }
                ]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )
                if hasattr(inputs, "input_ids"):
                    inputs = inputs["input_ids"].to(model.device)
                else:
                    inputs = inputs.to(model.device)

                decoded = greedy_decode(model, tokenizer, inputs, args.max_decode)
                if needle_value and normalize_text(needle_value) in normalize_text(decoded):
                    task_value_match[task] += 1

                rouge = scorer.score(needle, decoded)["rouge1"].recall * 100
                if rouge > 50:
                    task_success[task] += 1

    return {
        "rows_scanned": rows_scanned,
        "task_attempts": task_attempts,
        "task_success": task_success,
        "task_value_match": task_value_match,
        "ablation_heads": [list(x) for x in (ablation_heads or [])],
    }


def summarize_conditions(results):
    summary = {}
    baseline = results["baseline"]
    base_attempts = sum(baseline["task_attempts"].values())
    base_success = sum(baseline["task_success"].values())
    base_value = sum(baseline["task_value_match"].values())
    base_rouge_rate = base_success / max(1, base_attempts)
    base_value_rate = base_value / max(1, base_attempts)

    summary["baseline"] = {
        "attempts": base_attempts,
        "rouge_success": base_success,
        "value_match": base_value,
        "rouge_rate": base_rouge_rate,
        "value_rate": base_value_rate,
    }
    summary["conditions"] = {}

    for name, res in results.items():
        if name == "baseline":
            continue
        attempts = sum(res["task_attempts"].values())
        success = sum(res["task_success"].values())
        value = sum(res["task_value_match"].values())
        rouge_rate = success / max(1, attempts)
        value_rate = value / max(1, attempts)
        summary["conditions"][name] = {
            "attempts": attempts,
            "rouge_success": success,
            "value_match": value,
            "rouge_rate": rouge_rate,
            "value_rate": value_rate,
            "delta_rouge_rate_vs_baseline": rouge_rate - base_rouge_rate,
            "delta_value_rate_vs_baseline": value_rate - base_value_rate,
        }

    return summary


def plot_ablation_summary(results, out_dir):
    baseline = results["baseline"]
    all_conditions = [k for k in results.keys()]

    def overall_rates(res):
        att = sum(res["task_attempts"].values())
        suc = sum(res["task_success"].values())
        val = sum(res["task_value_match"].values())
        return 100 * suc / max(1, att), 100 * val / max(1, att)

    rouge_rates = []
    value_rates = []
    labels = []
    for name in all_conditions:
        rr, vr = overall_rates(results[name])
        labels.append(name)
        rouge_rates.append(rr)
        value_rates.append(vr)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.7), 5))
    ax.bar(x - width / 2, rouge_rates, width, label="ROUGE success", color="#4C72B0")
    ax.bar(x + width / 2, value_rates, width, label="Value match", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Baseline vs Ablation Conditions")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_condition_overall.png"), dpi=150)
    plt.close(fig)

    tasks = sorted(baseline["task_attempts"].keys())
    non_base = [c for c in all_conditions if c != "baseline"]
    if not non_base:
        return

    delta = np.zeros((len(non_base), len(tasks)))
    for i, cond in enumerate(non_base):
        for j, task in enumerate(tasks):
            b_att = baseline["task_attempts"].get(task, 0)
            c_att = results[cond]["task_attempts"].get(task, 0)
            b_rate = baseline["task_success"].get(task, 0) / max(1, b_att)
            c_rate = results[cond]["task_success"].get(task, 0) / max(1, c_att)
            delta[i, j] = c_rate - b_rate

    fig, ax = plt.subplots(figsize=(max(11, len(tasks) * 1.1), max(4.5, len(non_base) * 1.2)))
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(len(non_base)))
    ax.set_yticklabels(non_base)
    ax.set_title("Delta ROUGE Success Rate vs Baseline (Ablation - Baseline)")
    for i in range(len(non_base)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Delta success rate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_per_task_delta_rouge.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haystack_csv", required=True)
    parser.add_argument("--candidate_heads_json", required=True, help="Path to ablation_candidates.json from analysis step")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_decode", type=int, default=20)
    parser.add_argument("--target_tokens", type=int, default=7000)
    parser.add_argument("--max_rows_per_task", type=int, default=100)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--top_k_heads", type=int, default=4, help="Top shared heads to ablate")
    parser.add_argument("--n_random_sets", type=int, default=3, help="Number of random control conditions")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--output_dir", default="experiments/ablation_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    top_heads = load_top_heads(args.candidate_heads_json, args.top_k_heads)
    random_sets = sample_random_head_sets(
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        len(top_heads),
        args.n_random_sets,
        args.random_seed,
        exclude=top_heads,
    )

    conditions = [("baseline", None), ("top_shared_ablation", top_heads)]
    for i, rand_heads in enumerate(random_sets, start=1):
        conditions.append((f"random_ablation_{i}", rand_heads))

    results = {}
    for name, ablation_heads in conditions:
        results[name] = evaluate_condition(
            model=model,
            tokenizer=tokenizer,
            scorer=scorer,
            args=args,
            condition_name=name,
            ablation_heads=ablation_heads,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_conditions(results)

    with open(os.path.join(run_dir, "condition_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "max_decode": args.max_decode,
                "target_tokens": args.target_tokens,
                "max_rows_per_task": args.max_rows_per_task,
                "top_k_heads": args.top_k_heads,
                "n_random_sets": args.n_random_sets,
                "random_seed": args.random_seed,
                "top_heads": [list(x) for x in top_heads],
            },
            f,
            indent=2,
        )

    plot_ablation_summary(results, run_dir)
    print(f"Ablation run saved to: {run_dir}")


if __name__ == "__main__":
    main()
