"""
Usage:
  python run_eval.py                          # runs mmlu, hellaswag, custom
  python run_eval.py --tasks mmlu hellaswag
  python run_eval.py --tasks custom_qa
"""
import argparse
import json
from pathlib import Path

import lm_eval
from lm_eval.models.utils import handle_stop_sequences

from vllm_model import VLLMEndpointModel

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run(tasks, base_url, model, limit, num_fewshot):
    lm = VLLMEndpointModel(base_url=base_url, model=model)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=True,
    )

    out_path = RESULTS_DIR / "results.json"
    out_path.write_text(json.dumps(results["results"], indent=2))
    print(f"\nSaved → {out_path}\n")

    print(f"{'Task':<30} {'Metric':<25} {'Value':>8}")
    print("-" * 65)
    for task, metrics in results["results"].items():
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{task:<30} {metric:<25} {value:>8.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "hellaswag", "custom_qa"])
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--limit", type=int, default=None, help="Cap samples per task (useful for quick runs)")
    parser.add_argument("--num-fewshot", type=int, default=0)
    args = parser.parse_args()

    run(args.tasks, args.base_url, args.model, args.limit, args.num_fewshot)


if __name__ == "__main__":
    main()
