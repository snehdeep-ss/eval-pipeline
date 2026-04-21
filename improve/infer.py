"""
Runs ARC-Challenge inference with optimized prompts + self-consistency.
Usage:
  python infer.py --mode baseline          # 0-shot, no CoT
  python infer.py --mode optimized         # 4-shot CoT + self-consistency k=5
  python infer.py --mode optimized --limit 100
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import httpx
from scipy import stats

from optimize_prompt import build_prompt, build_ensemble_prompts, extract_answer

BASE_URL = "http://localhost:8000/v1"
MODEL = "default"
DATA_PATH = Path(__file__).parent / "data" / "arc_test.jsonl"
RESULTS_DIR = Path(__file__).parent / ".." / "eval_runner" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)


def generate(client: httpx.Client, prompt: str, temperature: float, max_tokens: int, base_url: str) -> str:
    resp = client.post(
        f"{base_url}/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_p": 1.0, "seed": SEED},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def majority_vote(answers: list) -> str | None:
    valid = [a for a in answers if a]
    return Counter(valid).most_common(1)[0][0] if valid else None


def run(examples, mode: str, client: httpx.Client, base_url: str) -> list[dict]:
    results = []
    for ex in examples:
        if mode == "baseline":
            prompt = f"Question: {ex['question']}\n{ex['options']}\nAnswer:"
            raw = generate(client, prompt, temperature=0.0, max_tokens=8, base_url=base_url)
            pred = extract_answer(raw, ex["labels"])
        else:
            # self-consistency (k=5) across ensemble of 3 prompt phrasings = 15 total votes
            prompts = build_ensemble_prompts(ex, k=4)
            raws = []
            for p in prompts:
                raws += [generate(client, p, temperature=0.5, max_tokens=128, base_url=base_url) for _ in range(5)]
            answers = [extract_answer(r, ex["labels"]) for r in raws]
            pred = majority_vote(answers)

        correct = pred == ex["answer"]
        results.append({"id": ex["id"], "question": ex["question"], "answer": ex["answer"], "pred": pred, "correct": correct})
        print(f"[{'OK' if correct else 'X'}] pred={pred} gold={ex['answer']}")

    return results


def accuracy_ci(results: list[dict]):
    correct = [int(r["correct"]) for r in results]
    acc = sum(correct) / len(correct)
    n = len(correct)
    se = (acc * (1 - acc) / n) ** 0.5
    ci = 1.96 * se
    return acc, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    examples = [json.loads(l) for l in DATA_PATH.read_text().splitlines() if l.strip()]
    if args.limit:
        examples = random.sample(examples, args.limit)

    with httpx.Client() as client:
        results = run(examples, args.mode, client, args.base_url)

    acc, ci = accuracy_ci(results)
    print(f"\nMode: {args.mode} | Accuracy: {acc:.4f} ± {ci:.4f} (95% CI) | n={len(results)}")

    out = RESULTS_DIR / f"arc_{args.mode}.json"
    out.write_text(json.dumps({"mode": args.mode, "accuracy": acc, "ci": ci, "n": len(results), "results": results}, indent=2))
    print(f"Saved → {out}")

    # print 10 before/after examples
    print("\n--- Sample predictions ---")
    for r in results[:10]:
        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"[{status}] Q: {r['question'][:80]}... | gold={r['answer']} pred={r['pred']}")


if __name__ == "__main__":
    main()
