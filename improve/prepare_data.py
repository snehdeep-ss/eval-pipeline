"""
Downloads ARC-Challenge and builds a few-shot pool + test split.
Usage: python prepare_data.py
Output: data/arc_test.jsonl, data/arc_fewshot_pool.jsonl
"""
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def format_example(row) -> dict:
    choices = row["choices"]
    options = "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))
    return {
        "id": row["id"],
        "question": row["question"],
        "options": options,
        "labels": choices["label"],
        "texts": choices["text"],
        "answer": row["answerKey"],
    }


def main():
    ds = load_dataset("ai2_arc", "ARC-Challenge")

    test = [format_example(r) for r in ds["test"]]
    pool = [format_example(r) for r in ds["train"]]

    (OUT / "arc_test.jsonl").write_text("\n".join(json.dumps(r) for r in test))
    (OUT / "arc_fewshot_pool.jsonl").write_text("\n".join(json.dumps(r) for r in pool))

    print(f"Test: {len(test)} | Few-shot pool: {len(pool)}")


if __name__ == "__main__":
    main()
