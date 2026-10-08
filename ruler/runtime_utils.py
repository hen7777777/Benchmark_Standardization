"""Frozen RULER inputs; checkpoint-specific chat rendering; official score semantics."""
import os
import re
from functools import cache


@cache
def tokenizer():
    from transformers import AutoTokenizer
    checkpoint = os.environ.get('RULER_EVAL_CHECKPOINT')
    if not checkpoint:
        raise ValueError('RULER_EVAL_CHECKPOINT must identify the evaluated checkpoint')
    value = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if not value.chat_template:
        raise ValueError('RULER evaluation requires a checkpoint chat_template')
    return value


def doc_to_text(doc):
    return tokenizer().apply_chat_template(
        [{'role': 'user', 'content': doc['input']}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    ) + doc['answer_prefix']


def process_results(doc, results):
    pred = re.sub(r'[\x00-\x1f]', '\n', results[0].strip()).strip().lower()
    matches = [float(ref.lower() in pred) for ref in doc['outputs']]
    value = max(matches) if doc['task'] in ('qa_1', 'qa_2') else sum(matches) / len(matches)
    return {'ruler_score': value * 100.0}
