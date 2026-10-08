#!/usr/bin/env python3
"""Generate the LongBench v1/v2 CoT task YAMLs from the pinned harness' direct
task definitions. This is a one-time reproducible generator: it reads the
official direct prompts and metric assignments out of the harness' longbench /
longbench2 YAMLs and writes self-contained CoT leaf + group YAMLs into
longbench_cot/tasks. The runtime helpers live in tasks/runtime_utils.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

# Allow reading the harness' custom `!function <module>.<name>` tags.
def _function_ctor(loader, node):
    return loader.construct_scalar(node)


yaml.SafeLoader.add_constructor("!function", _function_ctor)

SUITE = Path(__file__).resolve().parents[1]
HARNESS = Path("/data/lpg_projects/mj/test/vendor/lm-evaluation-harness/lm_eval/tasks")
OUT = SUITE / "longbench_cot" / "tasks"

V1_METRIC = {
    "get_qa_f1_with_score": ("v1_qa_f1", "qa_f1_score"),
    "get_qa_f1_zh_with_score": ("v1_qa_f1_zh", "qa_f1_zh_score"),
    "get_rouge_with_score": ("v1_rouge", "rouge_score"),
    "get_rouge_zh_with_score": ("v1_rouge_zh", "rouge_zh_score"),
    "get_classification_with_score": ("v1_classification", "classification_score"),
    "get_count_with_score": ("v1_count", "count_score"),
    "get_retrieval_with_score": ("v1_retrieval", "retrieval_score"),
    "get_retrieval_zh_with_score": ("v1_retrieval_zh", "retrieval_zh_score"),
    "get_code_sim_with_score": ("v1_code_sim", "code_sim_score"),
}

COT_SUFFIX = (
    "\n\nLet's think step by step, then write your final answer on a new line "
    'starting with "Final answer:".'
)

V1_EXCLUDE = {"lcc", "repobench-p"}  # code completion: CoT is not applicable


def read_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def v1_datasets():
    rows = []
    for path in sorted((HARNESS / "longbench").glob("*.yaml")):
        name = path.stem
        if name.startswith("_") or name.endswith("_e"):
            continue
        if name in V1_EXCLUDE:
            continue
        cfg = read_yaml(path)
        fn = cfg.get("process_results")
        fn = fn.split(".")[-1] if isinstance(fn, str) else fn
        if fn not in V1_METRIC:
            print(f"  !! skip {name}: unexpected process_results {fn!r}", file=sys.stderr)
            continue
        proc_fn, metric = V1_METRIC[fn]
        rows.append(dict(name=name, prompt=cfg["doc_to_text"], proc_fn=proc_fn, metric=metric))
    return rows


def v2_datasets():
    rows = []
    for path in sorted((HARNESS / "longbench2").glob("*.yaml")):
        if path.stem.startswith("_") or path.name == "_longbench_common_yaml":
            continue
        cfg = read_yaml(path)
        rows.append(dict(name=cfg["dataset_name"]))
    return rows


def render_v1_leaf(row):
    prompt = row["prompt"] + COT_SUFFIX
    task = f"longbench_v1_cot_{row['name']}"
    return (
        f"task: {task}\n"
        "tag:\n"
        "  - longbench_v1_cot_tasks\n"
        "dataset_path: Xnhyacinth/LongBench\n"
        "test_split: test\n"
        f"dataset_name: {row['name']}\n"
        f"doc_to_text: {json.dumps(prompt, ensure_ascii=False)}\n"
        "doc_to_target: '{{answers}}'\n"
        f"process_results: !function runtime_utils.{row['proc_fn']}\n"
        "generation_kwargs:\n"
        "  max_gen_toks: 1024\n"
        "  do_sample: false\n"
        "  temperature: 0.0\n"
        "  until: []\n"
        "metric_list:\n"
        "  - metric: score\n"
        "    aggregation: mean\n"
        "    higher_is_better: true\n"
        f"  - metric: {row['metric']}\n"
        "    aggregation: mean\n"
        "    higher_is_better: true\n"
        "metadata:\n"
        "  version: 1\n"
    )


def render_v1_group(names):
    lines = [
        "group: longbench_v1_cot",
        "group_alias: LongBench v1 chain-of-thought",
        "task:",
    ]
    lines += [f"  - longbench_v1_cot_{n}" for n in names]
    lines += [
        "aggregate_metric_list:",
        "  - metric: score",
        "    weight_by_size: false",
        "metadata:",
        "  version: 1",
    ]
    return "\n".join(lines) + "\n"


def render_v2_leaf(name):
    task = f"longbench_v2_cot_{name}"
    return (
        f"task: {task}\n"
        "tag:\n"
        "  - longbench_v2_cot_tasks\n"
        "dataset_path: recursal/longbench-v2\n"
        "test_split: train\n"
        f"dataset_name: {name}\n"
        "output_type: generate_until\n"
        "doc_to_text: !function runtime_utils.v2_doc_to_text\n"
        "doc_to_target: answer\n"
        "process_results: !function runtime_utils.v2_process\n"
        "num_fewshot: 0\n"
        "generation_kwargs:\n"
        "  max_gen_toks: 1024\n"
        "  do_sample: false\n"
        "  temperature: 0.0\n"
        "  until: []\n"
        "metric_list:\n"
        "  - metric: acc\n"
        "    aggregation: mean\n"
        "    higher_is_better: true\n"
        "metadata:\n"
        "  version: 1\n"
    )


def render_v2_group(names):
    lines = [
        "group: longbench_v2_cot",
        "group_alias: LongBench v2 chain-of-thought",
        "task:",
    ]
    lines += [f"  - longbench_v2_cot_{n}" for n in names]
    lines += [
        "aggregate_metric_list:",
        "  - metric: acc",
        "    weight_by_size: true",
        "metadata:",
        "  version: 1",
    ]
    return "\n".join(lines) + "\n"


def main():
    v1 = v1_datasets()
    v2 = v2_datasets()
    OUT.mkdir(parents=True, exist_ok=True)
    for row in v1:
        (OUT / f"longbench_v1_cot_{row['name']}.yaml").write_text(render_v1_leaf(row), encoding="utf-8")
    (OUT / "longbench_v1_cot.yaml").write_text(
        render_v1_group([r["name"] for r in v1]), encoding="utf-8")
    for row in v2:
        (OUT / f"longbench_v2_cot_{row['name']}.yaml").write_text(render_v2_leaf(row["name"]), encoding="utf-8")
    (OUT / "longbench_v2_cot.yaml").write_text(
        render_v2_group([r["name"] for r in v2]), encoding="utf-8")
    print(f"v1 tasks: {len(v1)} -> {[r['name'] for r in v1]}")
    print(f"v2 tasks: {len(v2)} -> {[r['name'] for r in v2]}")


if __name__ == "__main__":
    main()
