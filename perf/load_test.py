"""
Usage:
  python load_test.py                        # default: 20 requests, concurrency 5
  python load_test.py --concurrency 10 --n 50
  python load_test.py --long                 # use long prompts
"""
import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000/v1"
MODEL = "default"

SHORT_PROMPTS = [
    "What is 2 + 2?",
    "Name the capital of Japan.",
    "What color is the sky?",
    "Define recursion.",
    "What is a neural network?",
]

LONG_PROMPTS = [
    "Explain the transformer architecture in detail, covering self-attention, positional encoding, and why it replaced RNNs for sequence modeling tasks.",
    "Describe the full lifecycle of an HTTP request from a browser to a server and back, including DNS resolution, TCP handshake, TLS, and response parsing.",
    "Walk through how gradient descent with momentum works, why momentum helps escape local minima, and how it differs from Adam optimizer.",
    "Explain how a database index works internally using a B-tree, when you should and should not use indexes, and what a covering index is.",
    "Describe the CAP theorem and explain the trade-offs made by Cassandra, DynamoDB, and PostgreSQL in terms of consistency, availability, and partition tolerance.",
]


async def single_request(client: httpx.AsyncClient, prompt: str, max_tokens: int, idx: int):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    t_start = time.perf_counter()
    ttft = None
    token_times = []

    async with client.stream("POST", f"{BASE_URL}/chat/completions", json=payload, timeout=120) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t_start
                token_times.append(now)

    t_end = time.perf_counter()
    total_time = t_end - t_start
    n_tokens = len(token_times)
    tpot = (t_end - (t_start + ttft)) / n_tokens if n_tokens > 1 and ttft else None

    return {
        "idx": idx,
        "prompt_len": len(prompt.split()),
        "total_time": round(total_time, 4),
        "ttft": round(ttft, 4) if ttft else None,
        "tpot": round(tpot, 4) if tpot else None,
        "tokens_generated": n_tokens,
    }


async def run_load_test(prompts: list[str], concurrency: int, max_tokens: int):
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(client, prompt, idx):
        async with semaphore:
            return await single_request(client, prompt, max_tokens, idx)

    async with httpx.AsyncClient() as client:
        tasks = [bounded(client, p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

    return results


def percentile(data, p):
    sorted_data = sorted(x for x in data if x is not None)
    if not sorted_data:
        return None
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return round(sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo), 4)


def print_summary(results, label):
    latencies = [r["total_time"] for r in results]
    ttfts = [r["ttft"] for r in results if r["ttft"]]
    tpots = [r["tpot"] for r in results if r["tpot"]]

    print(f"\n=== {label} ===")
    print(f"Requests       : {len(results)}")
    print(f"Latency P50    : {percentile(latencies, 50)}s")
    print(f"Latency P95    : {percentile(latencies, 95)}s")
    print(f"Latency P99    : {percentile(latencies, 99)}s")
    print(f"TTFT avg       : {sum(ttfts)/len(ttfts):.4f}s" if ttfts else "TTFT: N/A")
    print(f"TPOT avg       : {sum(tpots)/len(tpots):.4f}s/tok" if tpots else "TPOT: N/A")


def save_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved → {path}")


def try_gpu_util():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        util, mem_used, mem_total = out.strip().split(", ")
        print(f"\nGPU util: {util}% | VRAM: {mem_used}/{mem_total} MiB")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    pool = LONG_PROMPTS if args.long else SHORT_PROMPTS
    prompts = (pool * (args.n // len(pool) + 1))[: args.n]

    label = f"{'long' if args.long else 'short'} prompts | concurrency={args.concurrency}"
    print(f"Running {args.n} requests ({label})")

    results = asyncio.run(run_load_test(prompts, args.concurrency, args.max_tokens))

    print_summary(results, label)
    try_gpu_util()

    out = Path(__file__).parent / "metrics.csv"
    save_csv(results, out)


if __name__ == "__main__":
    main()
