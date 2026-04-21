"""
Usage:
  python client.py                          # single prompt
  python client.py --concurrent 5           # 5 concurrent requests
  python client.py --prompt "Hello" --stream
"""
import argparse
import asyncio
import time

import httpx

BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "default"


def build_payload(prompt: str, stream: bool, max_tokens: int, temperature: float, top_p: float, stop: list[str] | None):
    return {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        **({"stop": stop} if stop else {}),
    }


def stream_generate(prompt: str, max_tokens=256, temperature=0.7, top_p=0.9, stop=None, base_url=BASE_URL):
    payload = build_payload(prompt, stream=True, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
    collected = []
    with httpx.Client(timeout=120) as client:
        with client.stream("POST", f"{base_url}/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                import json
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    collected.append(delta)
    print()
    return "".join(collected)


def generate(prompt: str, max_tokens=256, temperature=0.7, top_p=0.9, stop=None, base_url=BASE_URL):
    payload = build_payload(prompt, stream=False, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def async_generate(client: httpx.AsyncClient, prompt: str, idx: int, max_tokens=256, temperature=0.7, top_p=0.9, base_url=BASE_URL):
    payload = build_payload(prompt, stream=False, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=None)
    t0 = time.perf_counter()
    resp = await client.post(f"{base_url}/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0
    content = resp.json()["choices"][0]["message"]["content"]
    print(f"[{idx}] ({elapsed:.2f}s) {content[:80]}...")
    return elapsed, content


async def run_concurrent(prompts: list[str], base_url=BASE_URL):
    async with httpx.AsyncClient() as client:
        tasks = [async_generate(client, p, i, base_url=base_url) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
    times = [r[0] for r in results]
    print(f"\nConcurrent {len(prompts)} requests — avg {sum(times)/len(times):.2f}s, max {max(times):.2f}s")
    return results


SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Explain gradient descent in one sentence.",
    "Write a haiku about machine learning.",
    "What is 17 * 13?",
    "Name three uses of transformers in NLP.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=SAMPLE_PROMPTS[0])
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--concurrent", type=int, default=0, help="Run N concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stop", nargs="*", default=None)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    if args.concurrent > 0:
        prompts = (SAMPLE_PROMPTS * 10)[: args.concurrent]
        asyncio.run(run_concurrent(prompts, base_url=args.base_url))
    elif args.stream:
        print(f"Prompt: {args.prompt}\n--- Streaming ---")
        stream_generate(args.prompt, args.max_tokens, args.temperature, args.top_p, args.stop, args.base_url)
    else:
        print(f"Prompt: {args.prompt}\n--- Response ---")
        t0 = time.perf_counter()
        out = generate(args.prompt, args.max_tokens, args.temperature, args.top_p, args.stop, args.base_url)
        print(f"{out}\n({time.perf_counter()-t0:.2f}s)")


if __name__ == "__main__":
    main()
