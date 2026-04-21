"""
Start vLLM server:
  python serve.py
  python serve.py --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000
"""
import argparse
import subprocess
import sys

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--served-model-name", "default",
        "--enable-chunked-prefill",
    ]

    print(f"Starting vLLM server: {args.model} on {args.host}:{args.port}")
    print(f"OpenAI-compatible endpoint: http://{args.host}:{args.port}/v1")
    print("Ctrl+C to stop\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
