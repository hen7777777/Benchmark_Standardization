"""External harness task functions: frozen prompts and official RULER scores."""
import os
import re
from functools import cache
from pathlib import Path


@cache
def tokenizer():
    from transformers import AutoTokenizer
    # Prefer RULER_SOURCES (consolidated frozen sources) when set; otherwise fall
    # back to the legacy layout where this file was copied into RUN/tasks/.
    sources = os.environ.get("RULER_SOURCES")
    if sources:
        return AutoTokenizer.from_pretrained(Path(sources) / "tokenizer", local_files_only=True)
    return AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "sources/tokenizer", local_files_only=True)


def doc_to_text(doc):
    return tokenizer().apply_chat_template(
        [{"role": "user", "content": doc["input"]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    ) + doc["answer_prefix"]


def process_results(doc, results):
    pred = re.sub(r"[\x00-\x1f]", "\n", results[0].strip()).strip().lower()
    matches = [float(ref.lower() in pred) for ref in doc["outputs"]]
    # QA references are alternative correct answers, not multiple required answers.
    value = max(matches) if doc["task"] in ("qa_1", "qa_2") else sum(matches) / len(matches)
    return {"ruler_score": value * 100.0}
