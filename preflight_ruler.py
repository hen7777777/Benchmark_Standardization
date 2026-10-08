#!/usr/bin/env python3
"""Frozen-RULER preflight: recount every prompt with the evaluated checkpoint.

RULER sample budgets were fixed with a pinned Qwen3-0.6B tokenizer. Another
checkpoint tokenizes the same text differently, so this audit recomputes the
real token count and refuses to run any sample whose prompt plus generation
budget exceeds the effective context. Nothing is truncated or dropped silently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer          # noqa: E402
from lm_eval.tasks import TaskManager, get_task_dict      # noqa: E402


def leaves(tree):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from leaves(value)
        elif isinstance(value, tuple):
            if isinstance(value[1], dict):
                yield from leaves(value[1])
            else:
                yield key, value[1]
        else:
            yield key, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--include-path', required=True)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--context-length', type=int, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError('RULER protocol requires a checkpoint tokenizer with a chat_template')

    tasks = get_task_dict(args.tasks.split(','), task_manager=TaskManager(include_path=args.include_path))
    entries, failures = {}, []
    requests, maximum = 0, 0
    audit_path = Path(args.output).with_name('input_lengths.jsonl')
    with audit_path.open('w') as audit:
        for name, task in leaves(tasks):
            task.build_all_requests(limit=args.limit, apply_chat_template=False, cache_requests=False)
            seen = set()
            for request in task.instances:
                if request.request_type != 'generate_until':
                    raise ValueError(name + ': RULER expects generate_until, got ' + request.request_type)
                prompt, kwargs = request.args[0], request.args[1]
                budget = kwargs['max_gen_toks']
                length = len(tokenizer.encode(prompt, add_special_tokens=False))
                allowed = args.context_length - budget
                if allowed < 1:
                    raise ValueError(f'OUTPUT_BUDGET_EXCEEDS_CONTEXT {name}: {budget} > {args.context_length}')
                if length > allowed:
                    raise ValueError(
                        f'INPUT_TOO_LONG {name}/{request.doc_id}: {length}+{budget}>{args.context_length}; '
                        'raise --context-length or drop that --ruler-lengths entry')
                maximum = max(maximum, length + budget)
                requests += 1
                if request.doc_id in seen:
                    continue
                seen.add(request.doc_id)
                doc = request.doc
                audit.write(json.dumps(dict(
                    task=name, doc_id=request.doc_id, sample_id=doc['sample_id'],
                    nominal_length=doc['max_seq_length'], input_tokens=length, output_budget=budget,
                    actual_total_tokens_reserved=length + budget,
                    reference_prompt_tokens=doc['prompt_tokens'],
                    reference_total_tokens_reserved=doc['total_tokens_reserved'],
                    reference_length_delta=length + budget - doc['total_tokens_reserved'],
                )) + '\n')
            entries[name] = dict(
                unique_samples=len(seen),
                nominal_length=task.config.metadata.get('length'),
                version_samples=task.config.metadata.get('sample_count'),
            )

    report = dict(
        tasks=args.tasks, checkpoint=args.checkpoint, context_length=args.context_length, limit=args.limit,
        leaf_tasks=entries, unique_samples=sum(item['unique_samples'] for item in entries.values()),
        requests=requests, truncated_samples=0, max_input_plus_budget=maximum,
        length_policy='reject overlength; never truncate', status='failed' if failures else 'passed',
        failures=failures, audit=str(audit_path),
    )
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
    print(f'PREFLIGHT {report["status"]}: {len(entries)} leaf tasks, '
          f'{report["unique_samples"]} unique samples, max input+budget {maximum}', flush=True)
    for failure in failures:
        print('  FAIL', failure)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
