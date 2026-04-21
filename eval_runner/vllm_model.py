import json
import hashlib
import os
from pathlib import Path

import httpx
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _load_cache(key: str):
    p = CACHE_DIR / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save_cache(key: str, value):
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value))


@register_model("vllm_endpoint")
class VLLMEndpointModel(LM):
    def __init__(self, base_url="http://localhost:8000/v1", model="default", max_tokens=256, temperature=0.0, **kwargs):
        super().__init__()
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = httpx.Client(timeout=120)

    def _complete(self, prompt: str) -> str:
        key = _cache_key(f"{self.model}:{self.temperature}:{prompt}")
        cached = _load_cache(key)
        if cached is not None:
            return cached

        resp = self._client.post(
            f"{self.base_url}/completions",
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": 1.0,
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["text"]
        _save_cache(key, text)
        return text

    def loglikelihood(self, requests):
        results = []
        for ctx, cont in [r.args for r in requests]:
            prompt = ctx + cont
            key = _cache_key(f"ll:{self.model}:{prompt}")
            cached = _load_cache(key)
            if cached is not None:
                results.append(tuple(cached))
                continue

            resp = self._client.post(
                f"{self.base_url}/completions",
                json={"model": self.model, "prompt": prompt, "max_tokens": 1, "logprobs": 1, "echo": True, "temperature": 0.0},
            )
            resp.raise_for_status()
            data = resp.json()
            token_logprobs = data["choices"][0]["logprobs"]["token_logprobs"]
            ll = sum(x for x in token_logprobs if x is not None)
            result = (ll, True)
            _save_cache(key, list(result))
            results.append(result)
        return results

    def loglikelihood_rolling(self, requests):
        return [self.loglikelihood([r])[0] for r in requests]

    def generate_until(self, requests):
        results = []
        for ctx, gen_kwargs in [r.args for r in requests]:
            stop = gen_kwargs.get("until", [])
            max_tokens = gen_kwargs.get("max_gen_toks", self.max_tokens)
            key = _cache_key(f"gen:{self.model}:{ctx}:{stop}:{max_tokens}")
            cached = _load_cache(key)
            if cached is not None:
                results.append(cached)
                continue
            text = self._complete(ctx)
            for s in stop:
                if s in text:
                    text = text[: text.index(s)]
            _save_cache(key, text)
            results.append(text)
        return results
