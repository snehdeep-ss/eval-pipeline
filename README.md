# LLM Eval Pipeline

vLLM inference server + lm-eval benchmarks + perf testing + inference-time improvements.

## What I Learned

- **Prompt ensembling was the key lift** — running 3 differently-phrased instructions and majority voting across 15 total samples pushed ARC-Challenge from +2% to +3%. Single-phrasing self-consistency plateaued earlier than expected.
- **Batching the lm-eval wrapper mattered more than expected** — sending 32 prompts per request instead of 1 cut eval time by ~4x. The model was mostly idle between sequential requests without it.
- **TTFT on H100 is remarkably stable** — 31ms regardless of prompt length. vLLM's paged attention absorbs context length differences almost entirely at the prefill stage.
- **Exact match is a trap for open-ended tasks** — the model was often correct but scored 0 because of punctuation or extra words. Substring matching got custom_qa from 0% to 80% without changing the model or prompts at all.

## Setup

```bash
pip install -r requirements.txt
```

Tested on RunPod H100 SXM (80GB). Defaults to `mistralai/Mistral-7B-Instruct-v0.2`.

---

## Parts

### A — Serve
```bash
make serve                        # start vLLM server on :8000
make generate                     # start /generate endpoint on :8001
make client                       # single request
make concurrent                   # 5 concurrent requests
```

Query the `/generate` endpoint:
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_tokens": 128, "temperature": 0.7}'
```

Custom model:
```bash
python serve/serve.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### B — Eval
```bash
make eval
# or with specific tasks:
cd eval_runner && python run_eval.py --tasks mmlu hellaswag custom_qa --limit 100
```

Results saved to `eval_runner/results/results.json`. Custom task in `eval_runner/tasks/custom_qa/`.

**Baseline results (Mistral-7B, H100, n=100):**

| Task | Score |
|------|-------|
| MMLU | 59.67% |
| HellaSwag | 57.00% |
| custom_qa | 80.00% |

### C — Perf
```bash
make perf          # short prompts, concurrency=5
make perf-long     # long prompts, concurrency=5
```

Raw results saved to `perf/metrics.csv`. Plots generated via:
```bash
cd perf && jupyter nbconvert --to script analysis.ipynb --stdout | python
```
Plots saved as `perf/latency_distribution.png` and `perf/latency_vs_prompt_len.png`.

**Results (H100, Mistral-7B, 20 requests, concurrency=5):**

| Metric | Short prompts | Long prompts |
|--------|--------------|--------------|
| P50 latency | 0.63s | 0.80s |
| P95 latency | 0.82s | 0.81s |
| TTFT avg | 31ms | 32ms |
| TPOT avg | 6ms/tok | 6ms/tok |
| GPU util | 100% | 100% |

### D — Guardrails
```bash
make validate
```

Checks determinism (3 runs, temperature=0, seed=42) and validates custom task outputs.

### E — Improve
```bash
make improve
```

ARC-Challenge inference-time improvement. Strategy: 4-shot TF-IDF CoT + self-consistency k=5 + prompt ensembling (3 phrasings × 5 samples = 15 votes per question).

| Mode | Accuracy |
|------|----------|
| Baseline (0-shot) | 65.00% |
| Optimized | 68.00% |
| **Lift** | **+3.00%** |

See `improve/report.md` for full ablation and analysis.

---

## Layout

```
serve/               vLLM server + /generate endpoint + client
eval_runner/         lm-eval wrapper + custom task + results/
perf/                load test + metrics.csv + analysis notebook + plots
guardrails/          determinism check + output validation
improve/             ARC-Challenge inference-time improvements + report
```
