import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

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


def find_needle_span(prompt_ids, needle_ids):
    span_len = len(needle_ids)
    needle_set = set(needle_ids)
    if span_len == 0:
        return -1, -1
    for i in range(len(prompt_ids)):
        span = prompt_ids[i : i + span_len]
        overlap = len(set(span) & needle_set) / max(1, len(needle_set))
        if overlap > 0.9:
            return i, i + span_len
    return -1, -1


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


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip().lower()


def retrieval_calculate(attentions, retrieval_score, inp_id, prompt_ids, needle_start, needle_end, topk=1):
    for layer_idx in range(len(attentions)):
        head_dim = attentions[layer_idx].shape[1]
        for head_idx in range(head_dim):
            values, idx = attentions[layer_idx][0][head_idx][-1].topk(topk)
            for i in idx:
                if needle_start <= i < needle_end and inp_id == prompt_ids[i].item():
                    retrieval_score[layer_idx][head_idx] += 1 / (needle_end - needle_start)
                    break


def greedy_decode_with_retrieval(model, tokenizer, prompt_ids, needle_ids, max_decode):
    needle_start, needle_end = find_needle_span(prompt_ids[0].tolist(), needle_ids)
    if needle_start < 0:
        return "", None

    retrieval_score = np.zeros(
        (model.config.num_hidden_layers, model.config.num_attention_heads),
        dtype=float,
    )

    with torch.no_grad():
        # Do not request attentions on prefill: with long prompts this is O(seq^2)
        # and can OOM. We only need attentions during incremental decode below.
        outputs = model(input_ids=prompt_ids[:, :-1], use_cache=True, return_dict=True, output_attentions=False)
        past_kv = outputs.past_key_values
        inp = prompt_ids[:, -1:]
        # Explicit position_ids for incremental decode (avoids RoPE shape mismatch with cache)
        current_position = prompt_ids.size(1) - 1
        device = prompt_ids.device

        generated = []
        for _ in range(max_decode):
            position_ids = torch.tensor(
                [[current_position]],
                dtype=torch.long,
                device=device,
            )
            out = model(
                input_ids=inp,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_id = out.logits[:, -1].argmax(dim=-1)
            generated.append(next_id.item())
            retrieval_calculate(out.attentions, retrieval_score, next_id.item(), prompt_ids[0], needle_start, needle_end)

            if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
                break
            current_position += 1
            inp = next_id.unsqueeze(1)

    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return decoded, retrieval_score


def head_set_from_scores(avg_scores, threshold):
    heads = set()
    for l in range(avg_scores.shape[0]):
        for h in range(avg_scores.shape[1]):
            if avg_scores[l, h] >= threshold:
                heads.add((l, h))
    return heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haystack_csv", required=True)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_decode", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--target_tokens", type=int, default=7000)
    parser.add_argument("--output_dir", default="experiments/outputs/single_depth")
    parser.add_argument("--max_rows", type=int, default=None, help="Max CSV rows to process (default: all). Use for quick tests.")
    parser.add_argument(
        "--max_rows_per_task",
        type=int,
        default=100,
        help="Max CSV rows to process per task (default: 100).",
    )
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

    task_scores = {}
    task_success = {}
    task_attempts = {}
    task_value_match = {}
    task_rows_seen = {}
    target_tasks = set(TASK_QUESTIONS.keys())
    capped_target_tasks = set()
    total_rows = 0
    final_rows = []

    with open(args.haystack_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="CSV rows", file=sys.stderr):
            if len(capped_target_tasks) == len(target_tasks):
                break
            total_rows += 1
            if args.max_rows is not None and total_rows > args.max_rows:
                break
            task = row.get("task")
            needle = row.get("needle_sentence", "")
            needle_value = row.get("needle_value", "")
            haystack = row.get("haystack_text", "")
            if not task or not needle or not haystack:
                continue

            task_scores.setdefault(task, [])
            task_success.setdefault(task, 0)
            task_attempts.setdefault(task, 0)
            task_value_match.setdefault(task, 0)
            task_rows_seen.setdefault(task, 0)
            if task_rows_seen[task] >= args.max_rows_per_task:
                continue
            task_rows_seen[task] += 1
            if task in target_tasks and task_rows_seen[task] >= args.max_rows_per_task:
                capped_target_tasks.add(task)

            if normalize_text(needle) in normalize_text(haystack):
                continue

            question = TASK_QUESTIONS.get(task, f"What is the {task}?")
            context = insert_needle_at_depth(tokenizer, haystack, needle, question, 50)
            context_tokens = tokenizer.encode(context)
            target_len = args.target_tokens
            if len(context_tokens) > target_len:
                context = tokenizer.decode(context_tokens[:target_len])
            elif len(context_tokens) < target_len:
                padding_tokens = tokenizer.encode(" " * (target_len - len(context_tokens)))
                context = tokenizer.decode(context_tokens + padding_tokens)

            if normalize_text(needle) not in normalize_text(context):
                continue
            task_attempts[task] += 1
            final_rows.append(
                {
                    "filename": row.get("filename", ""),
                    "task": task,
                    "needle_sentence": needle,
                    "needle_value": needle_value,
                    "haystack_text": haystack,
                    "context_with_needle": context,
                    "needle_in_haystack": normalize_text(needle) in normalize_text(haystack),
                }
            )

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

            needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]
            decoded, score = greedy_decode_with_retrieval(model, tokenizer, inputs, needle_ids, args.max_decode)
            if score is None:
                continue

            if needle_value and normalize_text(needle_value) in normalize_text(decoded):
                task_value_match[task] += 1

            rouge = scorer.score(needle, decoded)["rouge1"].recall * 100
            if rouge > 50:
                task_scores[task].append(score)
                task_success[task] += 1

    task_heads = {}
    task_avg_scores = {}
    for task, scores_list in task_scores.items():
        if not scores_list:
            task_heads[task] = []
            task_avg_scores[task] = None
            continue
        avg_scores = np.mean(np.stack(scores_list, axis=0), axis=0)
        task_avg_scores[task] = avg_scores
        task_heads[task] = sorted([list(x) for x in head_set_from_scores(avg_scores, args.threshold)])

    with open(os.path.join(run_dir, "task_heads.json"), "w") as f:
        json.dump(task_heads, f, indent=2)

    with open(os.path.join(run_dir, "task_success.json"), "w") as f:
        json.dump(task_success, f, indent=2)

    with open(os.path.join(run_dir, "task_attempts.json"), "w") as f:
        json.dump(task_attempts, f, indent=2)

    with open(os.path.join(run_dir, "task_value_match.json"), "w") as f:
        json.dump(task_value_match, f, indent=2)

    final_csv = os.path.join(run_dir, "final_experiment_rows.csv")
    with open(final_csv, "w", newline="") as f:
        fieldnames = [
            "filename",
            "task",
            "needle_sentence",
            "needle_value",
            "haystack_text",
            "context_with_needle",
            "needle_in_haystack",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "threshold": args.threshold,
                "max_decode": args.max_decode,
                "total_rows": total_rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
