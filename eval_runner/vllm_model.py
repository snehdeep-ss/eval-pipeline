import json
import hashlib
from pathlib import Path

import httpx
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 32


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _load_cache(key: str):
    p = CACHE_DIR / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save_cache(key: str, value):
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value))


@register_model("vllm_endpoint")
class VLLMEndpointModel(LM):
    def __init__(self, base_url="http://localhost:8000/v1", model="default", max_tokens=256, temperature=0.0, batch_size=BATCH_SIZE, **kwargs):
        super().__init__()
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = int(batch_size)
        self._client = httpx.Client(timeout=300)

    def _batch_complete(self, prompts: list[str], max_tokens: int, temperature: float, logprobs: int = None, echo: bool = False) -> list[dict]:
        payload = {
            "model": self.model,
            "prompt": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
        }
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if echo:
            payload["echo"] = True

        resp = self._client.post(f"{self.base_url}/completions", json=payload)
        resp.raise_for_status()
        choices = resp.json()["choices"]
        # vLLM returns choices sorted by index
        return sorted(choices, key=lambda c: c["index"])

    def loglikelihood(self, requests):
        args = [r.args for r in requests]
        results = [None] * len(args)
        uncached_indices, uncached_prompts = [], []

        for i, (ctx, cont) in enumerate(args):
            prompt = ctx + cont
            key = _cache_key(f"ll:{self.model}:{prompt}")
            cached = _load_cache(key)
            if cached is not None:
                results[i] = tuple(cached)
            else:
                uncached_indices.append(i)
                uncached_prompts.append(ctx + cont)

        for batch_start in range(0, len(uncached_prompts), self.batch_size):
            batch = uncached_prompts[batch_start: batch_start + self.batch_size]
            choices = self._batch_complete(batch, max_tokens=1, temperature=0.0, logprobs=1, echo=True)
            for j, choice in enumerate(choices):
                token_logprobs = choice["logprobs"]["token_logprobs"]
                ll = sum(x for x in token_logprobs if x is not None)
                result = (ll, True)
                global_i = uncached_indices[batch_start + j]
                results[global_i] = result
                _save_cache(_cache_key(f"ll:{self.model}:{uncached_prompts[batch_start + j]}"), list(result))

        return results

    def loglikelihood_rolling(self, requests):
        return [self.loglikelihood([r])[0] for r in requests]

    def generate_until(self, requests):
        args = [r.args for r in requests]
        results = [None] * len(args)
        uncached_indices, uncached_prompts, uncached_kwargs = [], [], []

        for i, (ctx, gen_kwargs) in enumerate(args):
            stop = gen_kwargs.get("until", [])
            max_tokens = gen_kwargs.get("max_gen_toks", self.max_tokens)
            key = _cache_key(f"gen:{self.model}:{ctx}:{stop}:{max_tokens}")
            cached = _load_cache(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_prompts.append(ctx)
                uncached_kwargs.append(gen_kwargs)

        for batch_start in range(0, len(uncached_prompts), self.batch_size):
            batch_prompts = uncached_prompts[batch_start: batch_start + self.batch_size]
            batch_kwargs = uncached_kwargs[batch_start: batch_start + self.batch_size]
            max_tokens = max(kw.get("max_gen_toks", self.max_tokens) for kw in batch_kwargs)
            choices = self._batch_complete(batch_prompts, max_tokens=max_tokens, temperature=self.temperature)

            for j, choice in enumerate(choices):
                text = choice["text"]
                stop = batch_kwargs[j].get("until", [])
                for s in stop:
                    if s in text:
                        text = text[: text.index(s)]
                global_i = uncached_indices[batch_start + j]
                ctx = uncached_prompts[batch_start + j]
                results[global_i] = text
                _save_cache(_cache_key(f"gen:{self.model}:{ctx}:{stop}:{max_tokens}"), text)

        return results
