"""
Builds prompts for ARC-Challenge using:
  - Chain-of-thought instruction
  - Top-k few-shot examples selected by TF-IDF similarity
Usage: imported by infer.py
"""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POOL_PATH = Path(__file__).parent / "data" / "arc_fewshot_pool.jsonl"

COT_INSTRUCTION = (
    "You are an expert at multiple-choice science questions. "
    "For each question, reason step by step, then output ONLY the answer letter on the last line as 'Answer: X'."
)

ENSEMBLE_INSTRUCTIONS = [
    COT_INSTRUCTION,
    (
        "You are a science teacher. Read the question carefully, eliminate wrong answers, "
        "then output ONLY the correct letter on the last line as 'Answer: X'."
    ),
    (
        "Solve this science multiple-choice question. Think through each option, "
        "then output ONLY the best answer letter on the last line as 'Answer: X'."
    ),
]

_pool = None
_vectorizer = None
_pool_vecs = None


def _load_pool():
    global _pool, _vectorizer, _pool_vecs
    if _pool is not None:
        return
    _pool = [json.loads(l) for l in POOL_PATH.read_text().splitlines() if l.strip()]
    corpus = [r["question"] for r in _pool]
    _vectorizer = TfidfVectorizer().fit(corpus)
    _pool_vecs = _vectorizer.transform(corpus)


def get_few_shots(question: str, k: int = 4) -> list[dict]:
    _load_pool()
    q_vec = _vectorizer.transform([question])
    sims = cosine_similarity(q_vec, _pool_vecs)[0]
    top_k = sims.argsort()[-k:][::-1]
    return [_pool[i] for i in top_k]


def format_example_block(ex: dict, include_cot: bool = True) -> str:
    block = f"Question: {ex['question']}\n{ex['options']}\n"
    if include_cot:
        block += f"Let's think step by step. The answer is {ex['answer']}.\nAnswer: {ex['answer']}"
    else:
        block += f"Answer: {ex['answer']}"
    return block


def build_prompt(example: dict, k: int = 4, cot: bool = True, instruction: str = None) -> str:
    shots = get_few_shots(example["question"], k=k)
    shot_text = "\n\n".join(format_example_block(s, include_cot=cot) for s in shots)
    query = f"Question: {example['question']}\n{example['options']}\n"
    query += "Let's think step by step." if cot else "Answer:"
    inst = instruction or COT_INSTRUCTION
    return f"{inst}\n\n{shot_text}\n\n{query}"


def build_ensemble_prompts(example: dict, k: int = 4) -> list[str]:
    return [build_prompt(example, k=k, cot=True, instruction=inst) for inst in ENSEMBLE_INSTRUCTIONS]


def extract_answer(text: str, valid_labels: list[str]) -> str | None:
    # prefer explicit "Answer: X" line
    m = re.search(r"Answer:\s*([A-E])", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid_labels:
        return m.group(1).upper()
    # fallback: last standalone letter
    for token in reversed(text.split()):
        t = token.strip(".,\n").upper()
        if t in valid_labels:
            return t
    return None
