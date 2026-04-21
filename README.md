# LLM Eval Pipeline

vLLM inference server + lm-eval benchmarks + perf testing + inference-time improvements.

## Setup

```bash
pip install -r requirements.txt
```

Tested on RunPod with an A100. Defaults to `mistralai/Mistral-7B-Instruct-v0.2`.

---

## Parts

### A — Serve
```bash
make serve                        # start vLLM server on :8000
make client                       # single request
make concurrent                   # 5 concurrent requests
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

Custom task lives in `eval_runner/tasks/custom_qa/`. Results saved to `eval_runner/results/results.json`.

### C — Perf
```bash
make perf          # short prompts
make perf-long     # long prompts
```

Outputs `perf/metrics.csv`. Open `perf/analysis.ipynb` to plot.

### D — Guardrails
```bash
make validate
```

Checks determinism (3 runs, temperature=0) and regex-validates custom task outputs.

### E — Improve
```bash
make improve
```

Runs baseline and optimized inference on ARC-Challenge. Strategy: 4-shot TF-IDF CoT + self-consistency (k=5). See `improve/report.md`.

---

## Layout

```
serve/          vLLM server + client
eval_runner/    lm-eval wrapper + custom task
perf/           load test + notebook
guardrails/     determinism check + output validation
improve/        ARC-Challenge inference-time improvements
```
