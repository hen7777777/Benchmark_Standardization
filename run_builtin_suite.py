#!/usr/bin/env python3
"""Evaluate one checkpoint with lm-evaluation-harness.

Two arguments are always required: a local HuggingFace checkpoint directory and
--gpu. The suite never picks a GPU for you: check nvidia-smi, then pass the
physical index (or a comma-separated list matching --tensor-parallel-size).
math, code, complex_reasoning, general_reasoning and longbench come from the
harness itself (no --include_path), so their scores are the harness' own
built-in protocols. RULER instead runs this
project's frozen dataset through harness task templates, because the built-in
ruler group regenerates 500 samples per task and length on every load and needs
network access for its SQuAD/HotpotQA sources.

The checkpoint is loaded only once. By default (--model-backend server) the
suite starts a single vLLM OpenAI-compatible server and every stage talks to
it over HTTP, so a multi-stage run pays the model load a single time and the
GPU is released in a finally block after the last stage. --model-backend
inprocess restores the old behaviour, where every lm_eval process built its
own engine.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from vllm_server import VllmServer, pick_free_port, server_argv, write_server_record

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "builtin_tasks.json").read_text())
DEFAULTS = CONFIG["defaults"]
STAGE_DEFS = {stage["name"]: stage for stage in CONFIG["stages"]}
ALL_STAGES = [stage["name"] for stage in CONFIG["stages"]]
HARNESS_COMMIT = CONFIG["harness_commit"]
CACHE = ROOT / "cache"
RULER_SPEC = json.loads((ROOT / "ruler" / "tasks.json").read_text())
RULER_VERSIONS = {int(key): value for key, value in RULER_SPEC["versions"].items()}
RULER_LENGTHS = {int(key): value for key, value in RULER_SPEC["lengths"].items()}
RULER_TASK_ORDER = list(RULER_SPEC["task_order"])
RULER_SAMPLE_VERSIONS = tuple(sorted(RULER_VERSIONS))
RULER_TASK_COUNT = len(RULER_TASK_ORDER)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_checkpoint(path: Path) -> dict:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("checkpoint must be a local, complete HF model directory with config.json")
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        raise ValueError("checkpoint has no complete HF weights; a LoRA adapter alone is not supported")
    if any(ch in str(path) for ch in ",\n"):
        raise ValueError("checkpoint path cannot contain a comma or newline in native --model_args")
    return json.loads((path / "config.json").read_text())


def parse_gpus(value: str, count: int) -> str:
    """Validate the physical GPU indices the user chose; never auto-select one.

    Auto-selection is deliberately unsupported: on a shared machine the card that
    looks idle can be claimed a second later, and silently moving a run to another
    card makes the result hard to attribute. Pass the indices explicitly.
    """
    ids = [item.strip() for item in str(value).split(",")]
    if len(ids) != count or not all(item.isdigit() for item in ids):
        raise ValueError(
            "--gpu must list exactly --tensor-parallel-size physical GPU indices "
            "(for example --gpu 1, or --gpu 1,3 when --tensor-parallel-size 2). "
            "Run nvidia-smi and choose the card(s) yourself; auto-selection is disabled."
        )
    if len(set(ids)) != len(ids):
        raise ValueError("--gpu has duplicate physical indices: " + str(value))
    return ",".join(ids)


def cache_env(offline: bool) -> dict:
    env = os.environ.copy()
    mapping = {
        "TMPDIR": ROOT / "cache/tmp",
        "HF_HOME": CACHE / "hf",
        "HF_DATASETS_CACHE": CACHE / "datasets",
        "HF_MODULES_CACHE": CACHE / "modules",
        "HF_HUB_CACHE": CACHE / "hub",
        "XDG_CACHE_HOME": CACHE / "xdg",
        "VLLM_CACHE_ROOT": CACHE / "vllm",
        "NLTK_DATA": CACHE / "nltk",
        "XDG_DATA_HOME": CACHE / "xdg-data",
    }
    for key, value in mapping.items():
        value.mkdir(parents=True, exist_ok=True)
        env[key] = str(value)
    env.update(HF_ALLOW_CODE_EVAL="1", TOKENIZERS_PARALLELISM="false", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    if offline:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_EVALUATE_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    else:
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE", "TRANSFORMERS_OFFLINE"):
            env.pop(key, None)
    return env


def newest_mtime(root: Path) -> float:
    """Newest mtime (epoch seconds) anywhere under ``root``; 0.0 if it is absent or empty."""
    if not root.is_dir():
        return 0.0
    newest = root.stat().st_mtime
    for current, _dirs, files in os.walk(root):
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(current, name)).st_mtime)
            except FileNotFoundError:
                continue
    return newest


def _terminate_group(process: subprocess.Popen, grace: float = 15.0) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_logged(argv, path: Path, env: dict, results_dir: Path | None = None, shutdown_grace: float = 60.0):
    """Run one subprocess with the log at ``path`` and return (code, seconds, note).

    Two shutdown hangs are handled, both of which leave a finished run alive:

    * the CLI already crashed (its log ends in a traceback) but the vLLM engine
      keeps the process group alive;
    * the CLI already wrote ``results_*.json`` into ``results_dir`` and then
      stopped producing any output without exiting.

    In the second case the results are valid, so the run counts as a success and
    ``note`` records that the suite had to terminate the stalled process group.
    """
    started = time.monotonic()
    note = None
    suffix = None
    with path.open("w") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        preexisting = set(results_dir.rglob("results_*.json")) if results_dir is not None and results_dir.is_dir() else set()
        try:
            while process.poll() is None:
                now = time.time()
                silent = now - path.stat().st_mtime
                if results_dir is not None and silent > shutdown_grace:
                    # lm-eval writes <output_path>/<model_name>/results_<stamp>.json, then the
                    # sample jsonl files, then the final table. Treat any file under the results
                    # tree as activity so a long sample dump is not mistaken for a hang.
                    idle = min(silent, now - newest_mtime(results_dir))
                    landed = [item for item in results_dir.rglob("results_*.json") if item not in preexisting]
                    if landed and idle > shutdown_grace:
                        _terminate_group(process)
                        names = ", ".join(sorted(str(item.relative_to(results_dir)) for item in landed))
                        suffix = (
                            f"[suite] results already written ({names}) and no output for {int(idle)}s; "
                            "the CLI would not exit, so the suite terminated its process group.\n"
                        )
                        note = (
                            f"results_*.json present ({names}) but the CLI did not exit after "
                            f"{int(shutdown_grace)}s of total silence; suite terminated the stalled process group"
                        )
                        break
                # Engines can keep an already-crashed CLI alive during shutdown.
                if silent > 90:
                    with path.open("rb") as tail:
                        tail.seek(max(0, path.stat().st_size - 16000))
                        text = tail.read().decode(errors="replace").strip()
                    last = text.splitlines()[-1] if text else ""
                    if "Traceback (most recent call last):" in text and re.match(r"^[\w.]*(Error|Exception):", last):
                        _terminate_group(process)
                        raise RuntimeError("native subprocess crashed and stalled at shutdown: " + str(path))
                time.sleep(1)
        except BaseException:
            if process.poll() is None:
                _terminate_group(process)
            raise
    # Write the notice only after the log handle is closed so it is not overwritten.
    if suffix:
        with path.open("a") as append:
            append.write("\n" + suffix)
    elapsed = round(time.monotonic() - started, 3)
    if note is not None:
        return 0, elapsed, note
    return process.returncode, elapsed, None



def run_parallel(jobs, env: dict, limit: int = 0):
    """Run every job's ``argv`` at the same time and return one result per job.

    ``jobs`` is a list of dicts; each needs ``argv`` and ``log`` and may carry any
    extra keys, which are handed back untouched in the result. ``limit > 0`` caps
    how many subprocesses run concurrently. No stage cancels another one, so all
    logs are on disk when the caller decides what failed.
    """
    results = []
    pending = list(jobs)
    active = []

    def launch(job):
        stream = job["log"].open("w")
        process = subprocess.Popen(job["argv"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        active.append(dict(job=job, process=process, stream=stream, started=time.monotonic()))

    try:
        while pending or active:
            while pending and (limit <= 0 or len(active) < limit):
                launch(pending.pop(0))
            for item in [entry for entry in active if entry["process"].poll() is not None]:
                active.remove(item)
                item["stream"].close()
                results.append(
                    dict(job=item["job"], exit_code=item["process"].returncode, elapsed_seconds=round(time.monotonic() - item["started"], 3))
                )
            if pending or active:
                time.sleep(0.5)
    except BaseException:
        for item in active:
            if item["process"].poll() is None:
                _terminate_group(item["process"])
            item["stream"].close()
        raise
    return results


def stage_concurrency(definition: dict, override: int = 0) -> int:
    """In-flight HTTP requests one stage may keep against the shared server.

    The default mirrors the engine limit the stage used before the shared
    server existed (``max_num_seqs``), so server mode batches the same way an
    in-process engine did. ``--num-concurrent`` overrides it for every stage
    when more throughput is wanted than the old engine allowed.
    """
    if override:
        return override
    return int(definition.get("num_concurrent", definition.get("max_num_seqs", 1)))


def inprocess_model_args(checkpoint: Path, args, context: int, definition: dict) -> str:
    values = dict(
        pretrained=str(checkpoint),
        dtype="bfloat16",
        max_model_len=context,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_num_seqs=definition.get("max_num_seqs", 1),
        max_num_batched_tokens=2048,
        generation_config="vllm",
        add_bos_token=False,
        seed=args.seed,
    )
    if definition.get("truncation_side"):
        values["truncation_side"] = definition["truncation_side"]
    return ",".join(f"{key}={value}" for key, value in values.items())


def server_model_args(checkpoint: Path, definition: dict, plan: dict) -> str:
    """Model args for the HTTP client that talks to the shared vLLM server.

    ``local-completions`` posts straight to ``base_url``, so it has to be the
    full /v1/completions endpoint. ``tokenizer_backend=huggingface`` loads only
    the tokenizer on the client side; the weights stay in the server, so the
    engine is never rebuilt between stages. ``max_length`` must cover the same
    context the in-process engine was given, because this client left-truncates
    the prompt to ``max_length - max_gen_toks`` before sending it.
    ``num_concurrent > 1`` is the shape TemplateAPI expects: concurrency
    lives in the async request pool, not in a batch dimension, and vLLM does
    the continuous batching on its side. ``batch_size`` must not appear here
    because the evaluator already injects the CLI ``--batch_size``.
    """
    return ",".join(
        [
            f"base_url={plan['base_url']}",
            f"model={plan['served_model_name']}",
            f"tokenizer={checkpoint}",
            "tokenizer_backend=huggingface",
            f"max_length={plan['context']}",
            f"num_concurrent={stage_concurrency(definition, plan['num_concurrent'])}",
            # batch_size is intentionally absent: lm_eval's evaluator injects the CLI
            # --batch_size into the model constructor, so repeating it here raises
            # "got multiple values for keyword argument 'batch_size'".
            f"timeout={plan['request_timeout']}",
            "add_bos_token=False",
        ]
    )


def harness_root_and_commit() -> tuple[Path, str]:
    import lm_eval

    root = Path(lm_eval.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return root, commit


def resolve_ruler_data_root(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_value = os.environ.get("RULER_DATA_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(ROOT / (DEFAULTS.get("ruler_data_root") or "datasets/long_context/ruler"))
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (ROOT / candidate)
        resolved = resolved.resolve()
        if (resolved / RULER_VERSIONS[RULER_SAMPLE_VERSIONS[-1]]).is_dir():
            return resolved
    raise ValueError(
        "RULER_DATA_ROOT_MISSING: no frozen RULER data found. Looked at: "
        + ", ".join(str(item) for item in candidates)
        + ". Copy datasets/long_context/ruler into this folder or point RULER_DATA_ROOT at it."
    )


def ruler_task_prefix(samples: int, length: int) -> str:
    version = RULER_VERSIONS.get(samples)
    label = RULER_LENGTHS.get(length)
    if version is None or label is None:
        raise ValueError(
            f"RULER_UNKNOWN_SETTING: samples={samples} length={length}; "
            f"versions={RULER_SAMPLE_VERSIONS} lengths={sorted(RULER_LENGTHS)}"
        )
    return f"ruler_local_{version}_{label}"


def render_ruler_leaf(prefix: str, task: str, data_file: Path, samples: int, length: int) -> str:
    module = RULER_SPEC["functions_module"]
    return (
        f"task: {prefix}_{task}\n"
        "dataset_path: json\n"
        "dataset_kwargs:\n"
        "  data_files:\n"
        f"    test: {data_file}\n"
        "output_type: generate_until\n"
        "test_split: test\n"
        f"doc_to_text: !function {module}.doc_to_text\n"
        "doc_to_target: '{{outputs}}'\n"
        f"process_results: !function {module}.process_results\n"
        "num_fewshot: 0\n"
        "metric_list:\n"
        f'- metric: {RULER_SPEC["metric"]}\n'
        "  aggregation: mean\n"
        "  higher_is_better: true\n"
        "generation_kwargs:\n"
        "  do_sample: false\n"
        "  temperature: 0.0\n"
        f'  max_gen_toks: {RULER_SPEC["max_gen_toks"][task]}\n'
        "  until: []\n"
        "metadata:\n"
        "  version: 1\n"
        f"  sample_count: {samples}\n"
        f"  length: {length}\n"
        f"  length_label: {RULER_LENGTHS[length]}\n"
    )


def render_ruler_group(prefix: str, tasks) -> str:
    lines = [f"group: {prefix}", "task:"]
    lines += [f"- {prefix}_{task}" for task in tasks]
    lines += ["aggregate_metric_list:", f'- metric: {RULER_SPEC["metric"]}', "  weight_by_size: false"]
    return "\n".join(lines) + "\n"


def prepare_ruler_include(run_dir: Path, samples: int, lengths, data_root: Path) -> tuple[Path, str]:
    """Generate the harness task definitions for the requested frozen RULER slices."""
    include = run_dir / "ruler" / "tasks"
    include.mkdir(parents=True, exist_ok=True)
    version = RULER_VERSIONS.get(samples)
    if version is None:
        raise ValueError(f"RULER_UNKNOWN_SETTING: samples must be one of {RULER_SAMPLE_VERSIONS}")
    wanted = []
    for length in lengths:
        prefix = ruler_task_prefix(samples, length)
        folder = data_root / version / RULER_LENGTHS[length]
        files = {path.stem: path for path in sorted(folder.glob("*.jsonl"))}
        missing = [task for task in RULER_TASK_ORDER if task not in files]
        if missing:
            raise ValueError(f"RULER_DATA_INCOMPLETE: {folder} is missing " + ",".join(missing))
        for task in RULER_TASK_ORDER:
            (include / f"{prefix}_{task}.yaml").write_text(render_ruler_leaf(prefix, task, files[task], samples, length))
        (include / f"{prefix}.yaml").write_text(render_ruler_group(prefix, RULER_TASK_ORDER))
        wanted.append(prefix)
    (include / f'{RULER_SPEC["functions_module"]}.py').write_text((ROOT / "ruler" / "runtime_utils.py").read_text())
    return include, ",".join(wanted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Complete local HF checkpoint directory (first of two required arguments)")
    parser.add_argument("--stages", default=DEFAULTS["stages"], help='"all" or a comma-separated subset of: ' + ",".join(ALL_STAGES))
    parser.add_argument("--smoke", action="store_true", help="One or two examples per task")
    parser.add_argument("--limit", type=int, default=2, help="Examples per task in --smoke mode")
    parser.add_argument(
        "--gpu",
        required=True,
        help="Physical GPU index, or a comma-separated list matching --tensor-parallel-size "
        "(e.g. --gpu 1, or --gpu 1,3 with --tensor-parallel-size 2). Required: the suite never "
        "auto-selects a card, so check nvidia-smi first.",
    )
    parser.add_argument(
        "--shutdown-grace",
        type=float,
        default=60.0,
        help="Seconds of silence after results_*.json is written before the suite kills a CLI that "
        "will not exit (default: 60; the results are kept and the stage counts as successful)",
    )
    parser.add_argument("--context-length", type=int, default=DEFAULTS["context_length"])
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULTS["tensor_parallel_size"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULTS["gpu_memory_utilization"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--name", help="Output directory name; generated automatically by default")
    parser.add_argument(
        "--apply-chat-template", action="store_true", help="Pass --apply_chat_template to every task (off by default: harness defaults)"
    )
    parser.add_argument(
        "--cache-requests",
        action="store_true",
        help="Reuse cached prompts (harness --cache_requests true); prompts are tokenizer dependent",
    )
    parser.add_argument(
        "--allow-network", action="store_true", help="Do not force offline mode; only needed for the first prefetch on a new machine"
    )
    parser.add_argument(
        "--ruler-samples",
        type=int,
        choices=RULER_SAMPLE_VERSIONS,
        default=DEFAULTS["ruler_samples"],
        help="Frozen RULER subset: " + ", ".join(str(v) for v in RULER_SAMPLE_VERSIONS) + " samples per task and length",
    )
    parser.add_argument("--ruler-lengths", type=int, nargs="+", default=DEFAULTS["ruler_lengths"])
    parser.add_argument(
        "--ruler-data-root", default=None, help="Directory holding <version>/<length>/<task>.jsonl (default: datasets/long_context/ruler)"
    )
    parser.add_argument(
        "--preflight-jobs",
        type=int,
        default=0,
        help="How many stage preflights run at the same time (0 = all stages at once). Preflights are CPU-only and "
        "never touch the GPU, so overlapping them just removes the serial wait; each stage still writes its own log.",
    )
    parser.add_argument(
        "--preflight-only", action="store_true", help="Resolve task names and audit inputs, then stop before model inference"
    )
    parser.add_argument(
        "--model-backend",
        choices=("server", "inprocess"),
        default=DEFAULTS.get("model_backend", "server"),
        help="server = load the checkpoint once into a shared vLLM OpenAI server and run every stage "
        "against it (default); inprocess = the old behaviour, one --model vllm engine per stage",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=0,
        help="Port for the shared vLLM server (0 = pick a free port on 127.0.0.1)",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for the shared server to answer /v1/models (default: 900)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="Per-request HTTP timeout of the lm_eval API client (default: 1800s; long-context "
        "generations need a large value)",
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=0,
        help="In-flight requests per stage against the shared server (0 = per-stage default: that "
        "stage's max_num_seqs, i.e. the batching the in-process engine used)",
    )
    parser.add_argument(
        "--server-keep-alive",
        action="store_true",
        help="Leave the shared server running when the suite exits (debugging only; the GPU stays occupied)",
    )
    args = parser.parse_args()

    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")
    if not stages or len(set(stages)) != len(stages) or any(x not in STAGE_DEFS for x in stages):
        parser.error('--stages must be "all" or unique names from ' + ",".join(ALL_STAGES))
    if args.limit < 1 or (args.limit != 2 and not args.smoke):
        parser.error("--limit is only available together with --smoke")
    if args.context_length < 2048 or args.tensor_parallel_size < 1:
        parser.error("context length must be >=2048 and tensor parallel size >=1")
    if not 0 < args.gpu_memory_utilization <= 0.95:
        parser.error("gpu memory utilization must be in (0,0.95]")
    if args.preflight_jobs < 0:
        parser.error("--preflight-jobs must be >= 0 (0 = no limit)")
    if args.server_port and not 1 <= args.server_port <= 65535:
        parser.error("--server-port must be in 1..65535 (0 picks a free port)")
    if args.server_timeout <= 0 or args.request_timeout <= 0:
        parser.error("--server-timeout and --request-timeout must be > 0")
    if args.num_concurrent < 0:
        parser.error("--num-concurrent must be >= 0 (0 = per-stage default)")
    if "ruler" in stages:
        if len(set(args.ruler_lengths)) != len(args.ruler_lengths):
            parser.error("--ruler-lengths must be unique frozen lengths")
        unknown = [item for item in args.ruler_lengths if item not in RULER_LENGTHS]
        if unknown:
            parser.error("--ruler-lengths must be frozen budgets " + str(sorted(RULER_LENGTHS)) + "; got " + str(unknown))
    if "ruler" in stages and max(args.ruler_lengths) > args.context_length:
        raise ValueError("RULER_LENGTH_EXCEEDS_CONTEXT: pass a larger --context-length or fewer --ruler-lengths")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model_config = check_checkpoint(checkpoint)
    context = min(args.context_length, model_config.get("max_position_embeddings", args.context_length))
    if context < 2048:
        raise ValueError("checkpoint context window is below 2048 tokens")

    harness_path, commit = harness_root_and_commit()
    if commit != HARNESS_COMMIT:
        raise ValueError(f"Harness commit changed: {commit} (expected {HARNESS_COMMIT})")

    ruler_root = resolve_ruler_data_root(args.ruler_data_root) if "ruler" in stages else None

    name = args.name or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + re.sub(r"[^A-Za-z0-9_.-]", "_", checkpoint.name)
        + "_"
        + uuid.uuid4().hex[:6]
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,150}", name):
        raise ValueError("--name must be a plain directory name without path separators")
    run_dir = ROOT / "runs" / name
    if run_dir.exists():
        raise ValueError("OUTPUT_ALREADY_EXISTS: " + str(run_dir))

    offline = not args.allow_network
    env = cache_env(offline)
    if "ruler" in stages:
        env["RULER_EVAL_CHECKPOINT"] = str(checkpoint)
        if ruler_root is not None:
            env["RULER_DATA_ROOT"] = str(ruler_root)
    gpu = parse_gpus(args.gpu, args.tensor_parallel_size)
    env["CUDA_VISIBLE_DEVICES"] = gpu

    server_plan = None
    if args.model_backend == "server":
        # local-completions is the harness' HTTP client; it needs these packages.
        # Fail before touching the GPU instead of after the model is loaded.
        missing = [name for name in ("aiohttp", "tenacity", "requests") if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError(
                "--model-backend server needs the harness API client packages (missing: "
                + ", ".join(missing)
                + "); install them (see requirements.txt) or pass --model-backend inprocess"
            )
        # Resolve the port and the served-model alias before any stage argv is
        # built, so command.json records exactly what will run. The server itself
        # starts after preflight: a broken task set should fail without occupying
        # a GPU, and preflight never needs the weights.
        served_model_name = re.sub(r"[^A-Za-z0-9_.-]", "_", checkpoint.name) or "checkpoint"
        server_plan = dict(
            host="127.0.0.1",
            port=args.server_port or pick_free_port(),
            served_model_name=served_model_name,
            api_key=uuid.uuid4().hex,
            context=context,
            request_timeout=args.request_timeout,
            num_concurrent=args.num_concurrent,
            max_num_seqs=max(stage_concurrency(STAGE_DEFS[stage], args.num_concurrent) for stage in stages),
            max_num_batched_tokens=2048,
        )
        server_plan["base_url"] = f"http://{server_plan['host']}:{server_plan['port']}/v1/completions"
        env["OPENAI_API_KEY"] = server_plan["api_key"]

    versions = {key: importlib.metadata.version(key) for key in ("lm_eval", "vllm", "torch", "transformers", "datasets", "evaluate")}
    if "ruler" in stages:
        cells = RULER_TASK_COUNT * args.ruler_samples * len(args.ruler_lengths)
        print(
            f"NOTE: RULER runs the frozen local protocol: 13 tasks x {args.ruler_samples} samples x "
            f"{len(args.ruler_lengths)} length(s) = {cells} samples from {ruler_root}; "
            "offline, nothing is regenerated. Lengths: " + ", ".join(str(x) for x in args.ruler_lengths),
            flush=True,
        )
    plan = []
    for stage in stages:
        plan.append(dict(stage=stage, definition=STAGE_DEFS[stage]))

    run_dir.mkdir(parents=True)
    manifest = dict(
        suite="harness_builtin_suite",
        checkpoint=str(checkpoint),
        requested_context=args.context_length,
        effective_context=context,
        checkpoint_config_sha256=sha256(checkpoint / "config.json"),
        smoke=args.smoke,
        limit=args.limit if args.smoke else None,
        gpu=gpu,
        model_backend=args.model_backend,
        num_concurrent=args.num_concurrent,
        request_timeout=args.request_timeout,
        server_port=args.server_port or (server_plan["port"] if server_plan else None),
        stages=stages,
        seed=args.seed,
        shutdown_grace=args.shutdown_grace,
        preflight_jobs=args.preflight_jobs,
        apply_chat_template=args.apply_chat_template,
        cache_requests=args.cache_requests,
        offline=offline,
        ruler_samples=args.ruler_samples if "ruler" in stages else None,
        ruler_lengths=args.ruler_lengths if "ruler" in stages else None,
        ruler_data_root=str(ruler_root) if ruler_root is not None else None,
        harness_root=str(harness_path),
        harness_commit=commit,
        versions=versions,
        task_config=CONFIG["harness_commit"],
        cache=str(CACHE),
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="preflight",
        stage_status={},
        environment={
            key: env.get(key)
            for key in ("TMPDIR", "HF_HOME", "HF_DATASETS_CACHE", "HF_MODULES_CACHE", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE")
        },
    )
    write_json(run_dir / "manifest.json", manifest)
    print("RUN:", run_dir, "MODE:", "smoke" if args.smoke else "full", "GPU:", gpu, "OFFLINE:", offline, flush=True)

    server = None
    server_record = None
    try:
        # Build one job per stage first: task resolution, include generation and the
        # native argv are cheap and sequential, while the preflight subprocesses are
        # independent CPU-only checks that can all run at once.
        preflight_jobs = []
        for item in plan:
            stage, definition = item["stage"], item["definition"]
            directory = run_dir / stage
            directory.mkdir()
            tasks = ",".join(definition["tasks"])
            include = None
            if definition.get("ruler_local"):
                include, tasks = prepare_ruler_include(run_dir, args.ruler_samples, args.ruler_lengths, ruler_root)
            elif definition.get("include"):
                include = str((ROOT / definition["include"]).resolve())
            if definition.get("ruler_local"):
                preflight = [
                    sys.executable,
                    str(ROOT / "preflight_ruler.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--include-path",
                    str(include),
                    "--tasks",
                    tasks,
                    "--context-length",
                    str(context),
                    "--output",
                    str(directory / "input_audit.json"),
                ]
                if args.smoke:
                    preflight += ["--limit", str(args.limit)]
            else:
                preflight = [
                    sys.executable,
                    str(ROOT / "preflight.py"),
                    "--tasks",
                    tasks,
                    "--checkpoint",
                    str(checkpoint),
                    "--cache-dir",
                    str(CACHE),
                    "--output",
                    str(directory / "task_report.json"),
                    "--check-cache",
                ]
                if include is not None:
                    preflight += ["--include-path", include]

            if args.model_backend == "server":
                model_flags = [
                    "--model",
                    "local-completions",
                    "--model_args",
                    server_model_args(checkpoint, definition, server_plan),
                ]
            else:
                model_flags = ["--model", "vllm", "--model_args", inprocess_model_args(checkpoint, args, context, definition)]
            # API models do their concurrency in the async request pool, so the
            # evaluator batch dimension stays at 1; pass the real engine batch
            # only for the in-process backend.
            batch_size = "1" if args.model_backend == "server" else str(definition["batch_size"])
            native = [
                sys.executable,
                "-m",
                "lm_eval",
                "run",
                *model_flags,
                "--tasks",
                tasks,
                "--batch_size",
                batch_size,
                "--seed",
                str(args.seed),
                "--log_samples",
                "--show_config",
                "--output_path",
                str(directory / "results"),
            ]
            if include is not None:
                native += ["--include_path", str(include)]
            if definition.get("unsafe_code"):
                native += ["--confirm_run_unsafe_code"]
            if args.apply_chat_template:
                native += ["--apply_chat_template"]
            if args.cache_requests:
                native += ["--cache_requests", "true"]
            if args.smoke:
                native += ["--limit", str(args.limit)]

            manifest["stage_status"][stage] = "preflight_pending"
            preflight_jobs.append(
                dict(
                    stage=stage,
                    directory=directory,
                    argv=preflight,
                    native=native,
                    declared_tasks=definition["tasks"],
                    resolved_tasks=tasks,
                    include=include,
                    log=directory / "preflight.log",
                )
            )
        write_json(run_dir / "manifest.json", manifest)
        parallel = "all at once" if args.preflight_jobs <= 0 else f"{args.preflight_jobs} at a time"
        print(f"PREFLIGHT: {len(preflight_jobs)} stage(s) starting in parallel ({parallel})", flush=True)

        started = time.monotonic()
        results = run_parallel(preflight_jobs, env, args.preflight_jobs)
        wall = round(time.monotonic() - started, 3)
        manifest["preflight_wall_seconds"] = wall
        failed = []
        for result in sorted(results, key=lambda item: stages.index(item["job"]["stage"])):
            job = result["job"]
            stage, directory = job["stage"], job["directory"]
            write_json(
                directory / "preflight_exit.json",
                dict(exit_code=result["exit_code"], elapsed_seconds=result["elapsed_seconds"], watchdog=None),
            )
            write_json(
                directory / "command.json",
                dict(
                    argv=job["native"],
                    model_backend=args.model_backend,
                    tasks=job["declared_tasks"],
                    resolved_task_string=job["resolved_tasks"],
                    include_path=str(job["include"]) if job["include"] else None,
                    limit=args.limit if args.smoke else None,
                    preflight_argv=job["argv"],
                    preflight_seconds=result["elapsed_seconds"],
                ),
            )
            if result["exit_code"]:
                failed.append(stage)
                manifest["stage_status"][stage] = "preflight_failed"
                print(f"PREFLIGHT FAILED: {stage} (exit {result['exit_code']}, {result['elapsed_seconds']}s)", flush=True)
                tail = (directory / "preflight.log").read_text(errors="replace").strip().splitlines()
                if tail:
                    print("  " + tail[-1], flush=True)
            else:
                manifest["stage_status"][stage] = "preflight_passed"
                print(f"PREFLIGHT PASSED: {stage} ({result['elapsed_seconds']}s)", flush=True)
        write_json(run_dir / "manifest.json", manifest)
        print(
            f"PREFLIGHT DONE: {len(results) - len(failed)}/{len(results)} passed in {wall}s wall "
            f"({sum(item['elapsed_seconds'] for item in results)}s of stage time)",
            flush=True,
        )
        if failed:
            raise RuntimeError("preflight failed for: " + ", ".join(failed) + "; see " + str(run_dir) + "/<stage>/preflight.log")
        if args.preflight_only:
            manifest["status"] = "preflight_only_complete"
            write_json(run_dir / "manifest.json", manifest)
            return 0

        manifest["status"] = "running"
        write_json(run_dir / "manifest.json", manifest)
        if server_plan is not None:
            print(
                f"SERVER: loading {checkpoint.name} once for {len(stages)} stage(s) on GPU {gpu} "
                f"(port {server_plan['port']}); every stage will talk to it over HTTP",
                flush=True,
            )
            server = VllmServer(
                server_argv(
                    python=sys.executable,
                    checkpoint=checkpoint,
                    served_model_name=server_plan["served_model_name"],
                    host=server_plan["host"],
                    port=server_plan["port"],
                    dtype="bfloat16",
                    context=server_plan["context"],
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_num_seqs=server_plan["max_num_seqs"],
                    max_num_batched_tokens=server_plan["max_num_batched_tokens"],
                    seed=args.seed,
                    api_key=server_plan["api_key"],
                ),
                log_path=run_dir / "server.log",
                host=server_plan["host"],
                port=server_plan["port"],
                served_model_name=server_plan["served_model_name"],
                api_key=server_plan["api_key"],
                cwd=ROOT,
            )
            ready = server.start(env, timeout=args.server_timeout)
            server_record = server.describe()
            server_record["concurrency"] = {
                stage: stage_concurrency(STAGE_DEFS[stage], args.num_concurrent) for stage in stages
            }
            server_record["max_num_seqs"] = server_plan["max_num_seqs"]
            manifest["server"] = server_record
            write_server_record(run_dir / "server.json", server_record)
            write_json(run_dir / "manifest.json", manifest)
            print(f"SERVER READY in {ready}s: {server.base_url} (model={server_plan['served_model_name']})", flush=True)
        for item in plan:
            stage = item["stage"]
            directory = run_dir / stage
            native = json.loads((directory / "command.json").read_text())["argv"]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            print("NATIVE RUN:", stage, flush=True)
            code, elapsed, watchdog = run_logged(
                native,
                directory / "native.log",
                env,
                results_dir=directory / "results",
                shutdown_grace=args.shutdown_grace,
            )
            write_json(
                directory / "exit.json",
                dict(exit_code=code, elapsed_seconds=elapsed, watchdog=watchdog, shutdown_grace=args.shutdown_grace),
            )
            if watchdog:
                print("WATCHDOG:", stage, watchdog, flush=True)
            manifest["stage_status"][stage] = "completed" if code == 0 else "failed"
            write_json(run_dir / "manifest.json", manifest)
            if code:
                raise RuntimeError(f'{stage} native CLI exited {code}; see {directory / "native.log"}')
            print("NATIVE COMPLETED:", stage, elapsed, "seconds", flush=True)
        subprocess.run([sys.executable, str(ROOT / "summarize.py"), str(run_dir)], check=True, env=env)
        manifest["status"] = "complete"
        print("SUMMARY:", run_dir / "summary.md", flush=True)
        return 0
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_json(run_dir / "manifest.json", manifest)
        subprocess.run([sys.executable, str(ROOT / "summarize.py"), str(run_dir), "--partial"], check=False, env=env)
        raise
    finally:
        if server is not None:
            if args.server_keep_alive:
                print(f"SERVER LEFT RUNNING (--server-keep-alive): {server.base_url}", flush=True)
            else:
                print("SERVER: stopping the shared vLLM process group", flush=True)
                stopped = server.stop()
                server_record = dict(server_record or {}, **stopped)
                manifest["server"] = server_record
                write_server_record(run_dir / "server.json", server_record)
                print(f"SERVER STOPPED: pid={stopped['pid']} exit={stopped['exit_code']}", flush=True)
        manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(run_dir / "manifest.json", manifest)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
