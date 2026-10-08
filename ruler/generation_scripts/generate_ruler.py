"""Generate all seven lengths, then freeze nested 10/50/100 subsets and task YAMLs."""
import argparse
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
SUITE_ROOT = SCRIPTS_DIR.parents[1]  # harness_builtin_suite
DEFAULT_SOURCES = SUITE_ROOT / "ruler" / "sources"
LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
MASTER_SEED = 20261004


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=SUITE_ROOT / "runs" / "ruler_generation")
    parser.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--tasks", nargs="+")
    opts = parser.parse_args()
    run = opts.run_dir.resolve()
    sources = opts.sources_dir.resolve()
    configs = yaml.safe_load((sources / "task_configs.yaml").read_text())
    task_names = list(configs)
    selected_tasks = opts.tasks or task_names
    jobs = [(length, name) for length in opts.lengths for name in selected_tasks]
    env = dict(os.environ, PYTHONHASHSEED="0", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NLTK_DATA=str(sources / "nltk_data"), HF_HUB_OFFLINE="1")
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    def generate(job):
        length, name = job
        ti, li = task_names.index(name), LENGTHS.index(length)
        seed = MASTER_SEED + li * 1000 + ti
        qa_seed = MASTER_SEED + 200000 + ti
        target = run / "data/100" / str(length) / f"{name}.jsonl"
        cmd = [sys.executable, str(SCRIPTS_DIR / "generate_worker.py"), "--run-dir", str(run),
               "--sources-dir", str(sources),
               "--task", name, "--length", str(length), "--seed", str(seed), "--qa-seed", str(qa_seed)]
        if not target.exists():
            with (run / "logs" / f"generate_{length}_{name}.log").open("w") as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Ready {name} {length}", flush=True)
        return {"length": length, "task": name, "generation_seed": seed,
                "qa_selection_seed": qa_seed if configs[name]["task"] == "qa" else None,
                "command": cmd, "log": f"logs/generate_{length}_{name}.log"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=opts.workers) as pool:
        records = list(pool.map(generate, jobs))
    (run / "metadata/generation_commands.json").write_text(json.dumps(records, indent=2) + "\n")
    manifests = []
    summaries = []
    tasks_dir = run / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    shutil.copyfile(SCRIPTS_DIR / "harness_utils.py", tasks_dir / "ruler_local_utils.py")
    for length, name in jobs:
        source = run / "data/100" / str(length) / f"{name}.jsonl"
        rows = [json.loads(line) for line in source.open()]
        if len(rows) != 100:
            raise ValueError(f"Incomplete pool: {source}")
        subset_seed = MASTER_SEED + 100000 + task_names.index(name)
        order = list(range(100))
        random.Random(subset_seed).shuffle(order)
        for size in [10, 50, 100]:
            ids = order[:size] if size < 100 else list(range(100))
            data_file = run / "data" / str(size) / str(length) / f"{name}.jsonl"
            if size < 100:
                data_file.parent.mkdir(parents=True, exist_ok=True)
                with data_file.open("w") as f:
                    for i in ids:
                        f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
            picked = [rows[i] for i in ids]
            manifests.append({"length": length, "task": name, "size": size,
                              "subset_seed": subset_seed, "pool_indices": ids,
                              "sample_ids": [r["sample_id"] for r in picked]})
            tokens = [r["total_tokens_reserved"] for r in picked]
            depths = [r["first_answer_depth_fraction"] for r in picked if "first_answer_depth_fraction" in r]
            depth_bins = [sum(min(9, int(d * 10)) == b for d in depths) for b in range(10)] if depths else None
            summaries.append({"length": length, "task": name, "size": size,
                              "min_total_tokens": min(tokens), "mean_total_tokens": sum(tokens) / size,
                              "max_total_tokens": max(tokens), "mean_fill_fraction": sum(tokens) / size / length,
                              "first_answer_depth_decile_counts": depth_bins})
            task_id = f"ruler_local_{size}_{length}_{name}"
            cfg = {"task": task_id, "dataset_path": "json",
                   "dataset_kwargs": {"data_files": {"test": str(data_file)}},
                   "output_type": "generate_until", "test_split": "test",
                   "doc_to_text": "__DOC_FUNCTION__", "doc_to_target": "{{outputs}}",
                   "process_results": "__METRIC_FUNCTION__", "num_fewshot": 0,
                   "metric_list": [{"metric": "ruler_score", "aggregation": "mean", "higher_is_better": True}],
                   "generation_kwargs": {"do_sample": False, "temperature": 0.0,
                                          "max_gen_toks": picked[0]["max_gen_toks"], "until": []},
                   "metadata": {"version": 1, "sample_count": size, "length": length}}
            text = yaml.safe_dump(cfg, sort_keys=False)
            text = text.replace("__DOC_FUNCTION__", "!function ruler_local_utils.doc_to_text")
            text = text.replace("__METRIC_FUNCTION__", "!function ruler_local_utils.process_results")
            (tasks_dir / f"{task_id}.yaml").write_text(text)
    for length in opts.lengths:
        for size in [10, 50, 100]:
            group = {"group": f"ruler_local_{size}_{length}",
                     "task": [f"ruler_local_{size}_{length}_{name}" for name in selected_tasks],
                     "aggregate_metric_list": [{"metric": "ruler_score", "weight_by_size": False}]}
            (tasks_dir / f"ruler_local_{size}_{length}.yaml").write_text(yaml.safe_dump(group, sort_keys=False))
    (run / "metadata/subsets.json").write_text(json.dumps(manifests, indent=2) + "\n")
    (run / "metadata/length_statistics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Done: {len(jobs)} pools, {len(manifests)} version/task/length cells", flush=True)


if __name__ == "__main__":
    main()
