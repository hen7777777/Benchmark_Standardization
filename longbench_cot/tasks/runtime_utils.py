"""LongBench CoT runtime utilities (single-pass, official prompts + scoring).

LongBench v1 has no official chain-of-thought protocol in THUDM/LongBench; this
module implements a reason-then-answer variant that appends a "Let's think step
by step" instruction to the official direct prompts and reuses the official
per-dataset metrics (ported from THUDM/LongBench, MIT). LongBench v2 CoT follows
the official 0-shot two-pass CoT (prompts/0shot_cot.txt + 0shot_cot_ans.txt),
merged into one generation, with the official answer-extraction regex and
accuracy scoring.

The final answer is extracted from the text following the LAST "Final answer:"
marker (case-insensitive). If the marker is absent the full generation is scored
verbatim, so the metric degrades gracefully rather than throwing.
"""
from __future__ import annotations

import re
import string
from collections import Counter

try:
    import jieba
    from rouge import Rouge
except ImportError:  # pragma: no cover - installed with lm_eval[longbench]
    jieba = None
    Rouge = None


_FINAL_MARKER = "final answer:"


def extract_final_answer(text: str) -> str:
    """Return the part of the generation after the last 'Final answer:' marker."""
    idx = text.lower().rfind(_FINAL_MARKER)
    if idx >= 0:
        return text[idx + len(_FINAL_MARKER):].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# v1 metrics (ported from THUDM/LongBench via lm_eval/tasks/longbench/metrics.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    return f1_score(normalized_prediction.split(), normalized_ground_truth.split())


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [t for t in (normalize_zh_answer(x) for x in prediction_tokens) if len(t) > 0]
    ground_truth_tokens = [t for t in (normalize_zh_answer(x) for x in ground_truth_tokens) if len(t) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


_rouge = None


def rouge_score(prediction: str, ground_truth: str, **kwargs):
    global _rouge
    if Rouge is None:
        return 0.0
    if _rouge is None:
        _rouge = Rouge()
    try:
        scores = _rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def classification_score(prediction: str, ground_truth: str, **kwargs):
    all_classes = kwargs["all_classes"]
    em_match_list = [c for c in all_classes if c in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def count_score(prediction: str, ground_truth: str, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_score(prediction: str, ground_truth: str, **kwargs):
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs):
    matches = re.findall(r"段落(\d+)", ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def _best_over_answers(doc, generation, score_fn, **extra):
    pred = extract_final_answer(generation)
    best = 0.0
    for ground_truth in doc["answers"]:
        best = max(best, score_fn(pred, ground_truth, **extra))
    return best


# --- v1 process_results (one per metric type; the leaf YAMLs select these) ---

def v1_qa_f1(doc, results):
    val = _best_over_answers(doc, results[0], qa_f1_score)
    return {"score": val, "qa_f1_score": val}


def v1_qa_f1_zh(doc, results):
    val = _best_over_answers(doc, results[0], qa_f1_zh_score)
    return {"score": val, "qa_f1_zh_score": val}


def v1_rouge(doc, results):
    val = _best_over_answers(doc, results[0], rouge_score)
    return {"score": val, "rouge_score": val}


def v1_rouge_zh(doc, results):
    val = _best_over_answers(doc, results[0], rouge_zh_score)
    return {"score": val, "rouge_zh_score": val}


def v1_classification(doc, results):
    val = _best_over_answers(doc, results[0], classification_score, all_classes=doc["all_classes"])
    return {"score": val, "classification_score": val}


def v1_count(doc, results):
    val = _best_over_answers(doc, results[0], count_score)
    return {"score": val, "count_score": val}


def v1_retrieval(doc, results):
    val = _best_over_answers(doc, results[0], retrieval_score)
    return {"score": val, "retrieval_score": val}


def v1_retrieval_zh(doc, results):
    val = _best_over_answers(doc, results[0], retrieval_zh_score)
    return {"score": val, "retrieval_zh_score": val}


# ---------------------------------------------------------------------------
# v2 CoT (official 0-shot two-pass CoT merged into one generation)
# ---------------------------------------------------------------------------

_V2_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def v2_doc_to_text(doc) -> str:
    choices = doc["choices"]
    return (
        "Please read the following text and answer the questions below.\n\n"
        "<text>\n" + doc["context"].strip() + "\n</text>\n\n"
        "What is the correct answer to this question: " + doc["question"].strip() + "\n"
        "Choices:\n"
        "(A) " + str(choices[0]) + "\n"
        "(B) " + str(choices[1]) + "\n"
        "(C) " + str(choices[2]) + "\n"
        "(D) " + str(choices[3]) + "\n\n"
        "Let's think step by step, then give the single, most likely answer choice. "
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )


def _extract_v2(response: str):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return None


def v2_process(doc, results):
    pred = _extract_v2(results[0])
    if pred is None or pred not in _V2_INDEX:
        return {"acc": 0.0}
    return {"acc": 1.0 if _V2_INDEX[pred] == int(doc["answer"]) else 0.0}
