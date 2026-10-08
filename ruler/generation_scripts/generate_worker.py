"""Run one pinned official RULER task and annotate the frozen 100-row pool."""
import argparse
import json
import os
import random
import runpy
import sys
from pathlib import Path

import yaml

def render_prompt(tokenizer, doc):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": doc["input"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) + doc["answer_prefix"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sources-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--qa-seed", type=int, required=True)
    opts = parser.parse_args()
    run = opts.run_dir.resolve()
    sources = opts.sources_dir.resolve()
    generator = sources / "generator"
    sys.path[:0] = [str(generator / "synthetic"), str(generator)]
    config = yaml.safe_load((sources / "task_configs.yaml").read_text())[opts.task]
    constants = runpy.run_path(str(generator / "synthetic/constants.py"))["TASKS"][config["task"]]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sources / "tokenizer", local_files_only=True)
    # Raw task+answer-prefix token counts are bounded by L minus this overhead.
    overhead = len(tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False), add_special_tokens=False))
    args = ["ruler", "--save_dir", str(run / "tmp/raw" / str(opts.length)),
            "--save_name", opts.task, "--subset", "test",
            "--tokenizer_path", str(sources / "tokenizer"), "--tokenizer_type", "hf",
            "--max_seq_length", str(opts.length), "--tokens_to_generate", str(constants["tokens_to_generate"]),
            "--model_template_token", str(overhead), "--num_samples", "100",
            "--random_seed", str(opts.seed), "--template", constants["template"] + constants["answer_prefix"]]
    for key, value in config["args"].items():
        args.extend([f"--{key}", str(value)])
    sys.argv = args
    loaded = runpy.run_path(str(generator / "synthetic" / f"{config['task']}.py"), run_name="ruler_source")
    qa_provenance = []
    if config["task"] == "qa":
        qas = loaded["QAS"]
        source_indices = random.Random(opts.qa_seed).sample(range(len(qas)), 100)
        source_data = json.loads((generator / "synthetic/json" / f"{config['args']['dataset']}.json").read_text())
        if config["args"]["dataset"] == "squad":
            original_qas = [q for d in source_data["data"] for p in d["paragraphs"] for q in p["qas"] if not q["is_impossible"]]
            source_ids = [q["id"] for q in original_qas]
        else:
            source_ids = [q["_id"] for q in source_data]
        qa_provenance = [{"qa_source_index": i, "qa_source_id": source_ids[i]} for i in source_indices]
        selected = [qas[i] for i in source_indices]
        # Mutate in place: runpy functions share this list through their globals.
        qas[:] = selected
    loaded["main"]()
    raw = run / "tmp/raw" / str(opts.length) / opts.task / "test.jsonl"
    rows = [json.loads(line) for line in raw.open()]
    if len(rows) != 100:
        raise ValueError(f"Expected 100 rows: {opts.task}, {opts.length}")
    target = run / "data/100" / str(opts.length) / f"{opts.task}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        row["upstream_index"] = row.pop("index")
        row.update(index=i, sample_id=f"{opts.task}/{opts.length}/{i:03d}", task=opts.task,
                   max_seq_length=opts.length, generation_seed=opts.seed,
                   max_gen_toks=constants["tokens_to_generate"], chat_overhead_reserved=overhead)
        prompt = render_prompt(tok, row)
        row["prompt_tokens"] = len(tok.encode(prompt, add_special_tokens=False))
        row["total_tokens_reserved"] = row["prompt_tokens"] + row["max_gen_toks"]
        if row["total_tokens_reserved"] > opts.length:
            raise ValueError(f"Prompt budget exceeded: {row['sample_id']}")
        if not row["outputs"] or not all(isinstance(s, str) and s for s in row["outputs"]):
            raise ValueError(f"Missing labels: {row['sample_id']}")
        if qa_provenance:
            row.update(qa_provenance[i], qa_selection_seed=opts.qa_seed)
        if "token_position_answer" in row:
            raw_tokens = len(tok.encode(row["input"], add_special_tokens=False))
            row["first_answer_depth_fraction"] = row["token_position_answer"] / raw_tokens
    with target.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Completed {opts.task} {opts.length}: 100 samples", flush=True)


if __name__ == "__main__":
    main()
