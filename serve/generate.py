"""
Simple /generate endpoint wrapping the vLLM server.

Start:
  python generate.py

Query:
  curl -X POST http://localhost:8001/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is machine learning?"}'

Optional params:
  {
    "prompt": "your question here",
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["\n"]
  }
"""
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

VLLM_URL = "http://localhost:8000/v1/completions"
MODEL = "default"

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list[str] | None = None


@app.post("/generate")
def generate(req: GenerateRequest):
    payload = {
        "model": MODEL,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }
    if req.stop:
        payload["stop"] = req.stop

    resp = httpx.post(VLLM_URL, json=payload, timeout=120)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["text"].strip()
    return {"prompt": req.prompt, "response": text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
