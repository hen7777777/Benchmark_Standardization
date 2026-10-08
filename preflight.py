#!/usr/bin/env python3
"""Resolve harness built-in task names and check that their data is available.

No model inference. Fails fast if a task name is unknown, the checkpoint
tokenizer cannot be loaded locally, or (with --check-cache) the dataset a task
needs is not present in the suite cache while running offline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def leaves(node):
    for key, value in node.items():
        if isinstance(value, dict):
            yield from leaves(value)
        elif isinstance(value, tuple):
            yield from (leaves(value[1]) if isinstance(value[1], dict) else [(key, value[1])])
        else:
            yield key, value


parser = argparse.ArgumentParser()
parser.add_argument('--tasks', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--cache-dir', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--metadata')
parser.add_argument('--check-cache', action='store_true')
parser.add_argument('--include-path', default=None,
                    help='Directory with additional task YAMLs and runtime utils (e.g. longbench_cot/tasks)')
parser.add_argument('--skip-external', action='store_true',
                    help='Skip the dataset cache check (tasks whose data is generated at run time, e.g. RULER)')
a = parser.parse_args()

metadata = json.loads(a.metadata) if a.metadata else {}
metadata.setdefault('tokenizer', a.checkpoint)
metadata.setdefault('pretrained', a.checkpoint)

failures = []
try:
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
    tokenizer = 'ok'
except Exception as exc:                                    # noqa: BLE001
    tokenizer = f'FAIL {type(exc).__name__}: {exc}'

from lm_eval.tasks import TaskManager, get_task_dict               # noqa: E402

manager = TaskManager(metadata=metadata, include_path=a.include_path) if a.include_path else TaskManager(metadata=metadata)
report = {'tasks': a.tasks, 'checkpoint': a.checkpoint, 'tokenizer': tokenizer,
          'metadata': {k: v for k, v in metadata.items() if k != 'tokenizer'},
          'offline': bool(os.environ.get('HF_HUB_OFFLINE')), 'leaf_tasks': [], 'cache_checks': {}}
if tokenizer != 'ok':
    failures.append('tokenizer: ' + tokenizer)

tree = get_task_dict(a.tasks.split(','), task_manager=manager)
leaf = list(leaves(tree))
report['resolved_task_count'] = len(leaf)

cache_dir = Path(a.cache_dir)
seen = {}
for name, task in leaf:
    config = task.config
    entry = dict(name=name, num_fewshot=config.num_fewshot, output_type=config.output_type,
                 dataset_path=config.dataset_path or '', dataset_name=config.dataset_name,
                 testsplit=config.test_split, custom_dataset=callable(config.custom_dataset),
                 metrics=[str(m.get('metric')) if isinstance(m, dict) else str(m) for m in (config.metric_list or [])],
                 filters=[f.get('name') for f in (config.filter_list or []) if isinstance(f, dict)])
    report['leaf_tasks'].append(entry)
    if config.dataset_path:
        seen.setdefault((config.dataset_path, config.dataset_name, json.dumps(config.dataset_kwargs, sort_keys=True)), []).append(name)

if a.check_cache and not a.skip_external:
    import datasets
    for (path, config_name, kwargs_json), names in seen.items():
        kwargs = json.loads(kwargs_json) if kwargs_json not in (None, 'null') else {}
        key = f'{path}|{config_name}'
        try:
            datasets.load_dataset(path, name=config_name, **kwargs)
            report['cache_checks'][key] = 'ok'
        except Exception as exc:                        # noqa: BLE001
            report['cache_checks'][key] = f'FAIL {type(exc).__name__}: {str(exc)[:160]}'
            failures.append(f'cache {key} for {",".join(names[:3])}')
elif a.check_cache:
    report['cache_checks'] = 'skipped (--skip-external)'

report['failures'] = failures
report['status'] = 'failed' if failures else 'passed'
# harness task configs can carry callables (e.g. !function metrics), so stringify unknown types
Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
print(f'PREFLIGHT {report["status"]}: {len(leaf)} leaf tasks resolved, '
      f'{len(report["cache_checks"]) if isinstance(report["cache_checks"], dict) else 0} dataset checks')
for failure in failures:
    print('  FAIL', failure)
raise SystemExit(1 if failures else 0)
