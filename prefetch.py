#!/usr/bin/env python3
"""Download everything the built-in tasks need into this folder's cache/.

Run once before the first offline evaluation (needs network or the local proxy).
It loads each built-in task's dataset through the harness task object itself and
warms the `evaluate` metric module cache used by the code tasks.

The RULER stage is skipped on purpose: it reads this project's frozen dataset from
datasets/long_context/ruler, so it needs neither HuggingFace nor a proxy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'builtin_tasks.json').read_text())
CACHE = ROOT / 'cache'
PROXY_DEFAULT = 17891


def leaves(node):
    for key, value in node.items():
        if isinstance(value, dict):
            yield from leaves(value)
        elif isinstance(value, tuple):
            yield from (leaves(value[1]) if isinstance(value[1], dict) else [(key, value[1])])
        else:
            yield key, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy-port', type=int, default=PROXY_DEFAULT,
                        help='local proxy port used for HuggingFace access (0 = do not set a proxy)')
    parser.add_argument('--skip-modules', action='store_true', help='skip warming the evaluate code_eval module')
    args = parser.parse_args()

    if args.proxy_port:
        proxy = f'http://127.0.0.1:{args.proxy_port}'
        os.environ.update(http_proxy=proxy, https_proxy=proxy, HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_EVALUATE_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ.pop(key, None)
    os.environ.update(
        HF_HOME=str(CACHE / 'hf'), HF_DATASETS_CACHE=str(CACHE / 'datasets'),
        HF_MODULES_CACHE=str(CACHE / 'modules'), HF_HUB_CACHE=str(CACHE / 'hub'),
        NLTK_DATA=str(CACHE / 'nltk'), TOKENIZERS_PARALLELISM='false',
        # the harness code tasks run a sanity compute while importing their utils
        HF_ALLOW_CODE_EVAL='1')
    for name in ('tmp', 'hf', 'datasets', 'modules', 'hub', 'xdg', 'vllm', 'nltk'):
        (CACHE / name).mkdir(parents=True, exist_ok=True)

    report = {'proxy_port': args.proxy_port, 'datasets': {}, 'modules': {}, 'nltk': {},
              'ruler': 'local frozen data under datasets/long_context/ruler; nothing to download'}

    if not args.skip_modules:
        try:
            import evaluate
            module = evaluate.load('code_eval')
            report['modules']['code_eval'] = f'ok ({type(module).__name__})'
        except Exception as exc:                          # noqa: BLE001
            report['modules']['code_eval'] = f'FAIL {type(exc).__name__}: {str(exc)[:160]}'
    print('modules:', report['modules'], flush=True)

    try:
        import nltk
        for resource in ('punkt_tab', 'punkt'):
            try:
                report['nltk'][resource] = 'ok' if nltk.download(resource, download_dir=str(CACHE / 'nltk')) else 'failed'
            except Exception as exc:                      # noqa: BLE001
                report['nltk'][resource] = f'FAIL {type(exc).__name__}: {str(exc)[:120]}'
    except Exception as exc:                              # noqa: BLE001
        report['nltk']['import'] = f'FAIL {type(exc).__name__}: {str(exc)[:120]}'
    print('nltk:', report['nltk'], flush=True)

    from lm_eval.tasks import TaskManager, get_task_dict

    stages = [stage for stage in CONFIG['stages'] if stage.get('kind') != 'local_frozen']
    for stage in stages:
        try:
            tree = get_task_dict(stage['tasks'], task_manager=TaskManager())
        except Exception as exc:                          # noqa: BLE001
            report['datasets'][stage['name'] + '/<resolve>'] = f'FAIL {type(exc).__name__}: {str(exc)[:160]}'
            print(stage['name'], 'resolve FAILED:', type(exc).__name__, str(exc)[:160], flush=True)
            continue
        for name, task in leaves(tree):
            key = f"{stage['name']}/{name}"
            try:
                task.download()
                report['datasets'][key] = f"ok dataset={task.config.dataset_path or 'custom'}"
            except Exception as exc:                      # noqa: BLE001
                report['datasets'][key] = f'FAIL {type(exc).__name__}: {str(exc)[:160]}'
            print(key, '->', report['datasets'][key], flush=True)

    report['cache_dir'] = str(CACHE)
    report['du'] = subprocess.run(['du', '-sh', str(CACHE)], capture_output=True, text=True).stdout.strip()
    (ROOT / 'prefetch_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    failed = [key for key, value in report['datasets'].items() if str(value).startswith('FAIL')]
    failed += [f'modules/{key}' for key, value in report['modules'].items() if str(value).startswith('FAIL')]
    print('PREFETCH', 'FAILED' if failed else 'OK', 'cache:', report['du'])
    for key in failed:
        print('  FAIL', key)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
