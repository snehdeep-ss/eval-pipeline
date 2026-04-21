"""
Usage:
  python validate.py                      # verify determinism + validate custom task outputs
  python validate.py --prompt "Hello"     # test determinism on a specific prompt
"""
import argparse
import json
import re
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000/v1"
MODEL = "default"


def generate(prompt: str, client: httpx.Client, seed: int = 42, base_url: str = BASE_URL) -> str:
    resp = client.post(
        f"{base_url}/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()


def check_determinism(prompt: str, runs: int = 3, base_url: str = BASE_URL) -> bool:
    with httpx.Client() as client:
        outputs = [generate(prompt, client, base_url=base_url) for _ in range(runs)]

    all_same = len(set(outputs)) == 1
    print(f"Prompt: {prompt!r}")
    for i, o in enumerate(outputs):
        print(f"  Run {i+1}: {o!r}")
    print(f"  Deterministic: {'YES' if all_same else 'NO — nondeterminism detected'}\n")
    return all_same


CUSTOM_TASK_SCHEMA = {
    "What is the time complexity of binary search?": r"O\(log n\)|O\(logn\)",
    "Which data structure uses LIFO order?": r"stack",
    "What does HTTP stand for?": r"hypertext transfer protocol",
    "What is the default port for HTTPS?": r"443",
    "What Python keyword is used to define a function?": r"def",
}


def validate_custom_outputs(results_path: Path) -> None:
    if not results_path.exists():
        print(f"No results found at {results_path} — run eval first.")
        return

    data = json.loads(results_path.read_text())
    samples = data.get("custom_qa", {}).get("samples", data.get("samples", []))
    if not samples:
        score = data.get("custom_qa", {}).get("contains_match,none")
        if score is not None:
            print(f"custom_qa contains_match score: {score:.4f} (from last eval run)")
        else:
            print("No custom_qa samples found — run: python eval_runner/run_eval.py --tasks custom_qa first")
        return

    passed = failed = 0
    for s in samples:
        question = s.get("doc", {}).get("question", "")
        prediction = s.get("resps", [[""]])[0][0].strip().lower()
        pattern = CUSTOM_TASK_SCHEMA.get(question)
        if pattern is None:
            continue
        ok = bool(re.search(pattern, prediction, re.IGNORECASE))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {question!r}\n       got: {prediction!r}\n")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"Custom task validation: {passed} passed, {failed} failed out of {passed+failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Explain what a transformer model is in one sentence.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--skip-determinism", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    if not args.skip_determinism:
        print("=== Determinism Check ===")
        prompts = [
            args.prompt,
            "What is 2 + 2?",
            "Name a planet in the solar system.",
        ]
        results = [check_determinism(p, args.runs, base_url=args.base_url) for p in prompts]
        if not all(results):
            print("WARNING: nondeterminism detected. vLLM may not support seed on all backends.\n")

    if not args.skip_validation:
        print("=== Custom Task Output Validation ===")
        results_path = Path(__file__).parent.parent / "eval_runner" / "results" / "results.json"
        validate_custom_outputs(results_path)


if __name__ == "__main__":
    main()
