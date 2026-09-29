"""
Compare two chat checkpoints on the same evaluation tasks.

Example:
torchrun --standalone --nproc_per_node=2 -m scripts.chat_compare -- \
    --model-a-source=distill --model-b-source=rl \
    --tasks=MMLU|GSM8K
"""

import argparse
import csv
import gc
import json
import os

import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval


ALL_TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"]
BASELINES = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "MMLU": 0.25,
    "GSM8K": 0.0,
    "HumanEval": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two nanochat checkpoints")
    parser.add_argument("--model-a-source", choices=["sft", "rl", "distill"], default="distill")
    parser.add_argument("--model-a-tag", default=None, help="model A checkpoint tag")
    parser.add_argument("--model-a-step", type=int, default=None, help="model A checkpoint step")
    parser.add_argument("--model-a-name", default=None, help="display name for model A")
    parser.add_argument("--model-b-source", choices=["sft", "rl", "distill"], default="rl")
    parser.add_argument("--model-b-tag", default=None, help="model B checkpoint tag")
    parser.add_argument("--model-b-step", type=int, default=None, help="model B checkpoint step")
    parser.add_argument("--model-b-name", default=None, help="display name for model B")
    parser.add_argument(
        "--tasks",
        default="|".join(ALL_TASKS),
        help="task names separated by | (default: all standard chat tasks)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="categorical batch size")
    parser.add_argument("--num-samples", type=int, default=1, help="samples per generative problem")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
    parser.add_argument("--max-problems", type=int, default=None, help="limit examples per task")
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps", ""], default="")
    parser.add_argument(
        "--output",
        default="chat_compare_results",
        help="output basename under the nanochat cache directory",
    )
    return parser.parse_args()


def compute_chatcore(results: dict[str, float]) -> float | None:
    if not all(task in results for task in ALL_TASKS):
        return None
    centered = [
        (results[task] - BASELINES[task]) / (1.0 - BASELINES[task])
        for task in ALL_TASKS
    ]
    return sum(centered) / len(centered)


def evaluate_checkpoint(args, source, model_tag, step, tasks):
    model, tokenizer, meta = load_model(
        source,
        args.device,
        phase="eval",
        model_tag=model_tag,
        step=step,
    )
    engine = Engine(model, tokenizer)
    results = {}
    for task_name in tasks:
        print0(f"Evaluating {source}:{model_tag or 'auto'}:{step or 'last'} on {task_name}")
        results[task_name] = run_chat_eval(
            task_name,
            model,
            tokenizer,
            engine,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
        )
        print0(f"{task_name}: {100 * results[task_name]:.2f}%")

    chatcore = compute_chatcore(results)
    if chatcore is not None:
        print0(f"ChatCORE: {chatcore:.4f}")

    del engine, model
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    if args.ddp:
        dist.barrier()

    return {
        "source": source,
        "model_tag": model_tag,
        "step": step,
        "checkpoint_meta": meta,
        "tasks": results,
        "chatcore": chatcore,
    }


def write_results(output_base, records):
    base_dir = get_base_dir()
    os.makedirs(base_dir, exist_ok=True)
    json_path = os.path.join(base_dir, output_base + ".json")
    csv_path = os.path.join(base_dir, output_base + ".csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    task_names = sorted({task for record in records for task in record["tasks"]})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "source", "model_tag", "step", "chatcore", *task_names],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "name": record["name"],
                "source": record["source"],
                "model_tag": record["model_tag"],
                "step": record["step"],
                "chatcore": record["chatcore"],
                **record["tasks"],
            })
    return json_path, csv_path


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_samples <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch-size, num-samples, and max-new-tokens must be positive")
    tasks = args.tasks.split("|")
    unknown_tasks = [task for task in tasks if task not in ALL_TASKS]
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}; choose from {ALL_TASKS}")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    args.ddp, ddp_rank, _, _, args.device = compute_init(device_type)
    master_process = ddp_rank == 0
    if args.ddp:
        dist.barrier()

    model_a_name = args.model_a_name or f"{args.model_a_source}:{args.model_a_tag or 'auto'}"
    model_b_name = args.model_b_name or f"{args.model_b_source}:{args.model_b_tag or 'auto'}"
    records = []
    for name, source, model_tag, step in [
        (model_a_name, args.model_a_source, args.model_a_tag, args.model_a_step),
        (model_b_name, args.model_b_source, args.model_b_tag, args.model_b_step),
    ]:
        result = evaluate_checkpoint(args, source, model_tag, step, tasks)
        result["name"] = name
        records.append(result)

    if master_process:
        json_path, csv_path = write_results(args.output, records)
        print0("\nComparison:")
        for task_name in tasks:
            a = records[0]["tasks"][task_name]
            b = records[1]["tasks"][task_name]
            print0(f"{task_name:16s} {model_a_name}: {100*a:6.2f}% | "
                   f"{model_b_name}: {100*b:6.2f}% | delta: {100*(a-b):+6.2f}%")
        if records[0]["chatcore"] is not None:
            print0(
                f"ChatCORE          {model_a_name}: {records[0]['chatcore']:.4f} | "
                f"{model_b_name}: {records[1]['chatcore']:.4f} | "
                f"delta: {records[0]['chatcore'] - records[1]['chatcore']:+.4f}"
            )
        print0(f"Saved JSON: {json_path}")
        print0(f"Saved CSV:  {csv_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
