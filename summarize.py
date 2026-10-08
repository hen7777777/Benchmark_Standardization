#!/usr/bin/env python3
"""Collect the native harness result files of one run into a compact summary."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'builtin_tasks.json').read_text())
STAGE_DEFS = {stage['name']: stage for stage in CONFIG['stages']}


def numeric(metrics: dict) -> dict:
    return {key: value for key, value in (metrics or {}).items()
            if isinstance(value, (int, float)) and not key.endswith('_stderr')}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    run = args.run_directory
    manifest = json.loads((run / 'manifest.json').read_text())

    ruler_samples = manifest.get('ruler_samples')
    ruler_lengths = manifest.get('ruler_lengths') or []

    rows, errors = [], []
    if manifest['status'] not in ('running', 'complete') and not args.partial:
        errors.append('Run status: ' + manifest['status'])
    for stage in manifest['stages']:
        directory = run / stage
        exit_file = directory / 'exit.json'
        exit_code = json.loads(exit_file.read_text())['exit_code'] if exit_file.exists() else None
        if not args.partial and (manifest.get('stage_status', {}).get(stage) != 'completed' or exit_code != 0):
            errors.append(f'{stage}: status={manifest.get("stage_status", {}).get(stage)}, exit_code={exit_code}')
        found = sorted((directory / 'results').rglob('results_*.json'))
        if not found:
            if not args.partial:
                errors.append(f'{stage}: no results_*.json')
            continue
        results = json.loads(found[0].read_text())
        leaves, groups = results.get('results', {}), results.get('groups', {})
        subtasks = results.get('group_subtasks', {})
        definition = STAGE_DEFS.get(stage, {})
        primary = {(task, metric) for task, metric in definition.get('primary', [])}
        if definition.get('ruler_local'):
            primary |= {(f'ruler_local_{ruler_samples}_{length}', 'ruler_score,none')
                        for length in ruler_lengths}
        for name, metrics in list(groups.items()) + list(leaves.items()):
            is_group = name in groups
            for metric, value in numeric(metrics).items():
                rows.append(dict(stage=stage, task=name, metric=metric, score=value,
                                 samples=metrics.get('sample_count', {}).get(metric)
                                 if isinstance(metrics.get('sample_count'), dict) else None,
                                 level='group' if is_group else 'task',
                                 primary=bool((name, metric) in primary)))
        if definition.get('ruler_local'):
            for length in ruler_lengths:
                group_name = f'ruler_local_{ruler_samples}_{length}'
                members = subtasks.get(group_name) or []
                values = [numeric(leaves.get(member, {})).get('ruler_score,none') for member in members]
                values = [value for value in values if value is not None]
                if values:
                    rows.append(dict(stage=stage, task='ruler', metric=f'macro,{length}',
                                     score=sum(values) / len(values), samples=len(values),
                                     level='macro', primary=True))
        # keep the leaf rows of grouped benchmarks so per-dataset numbers stay visible
        for group, members in subtasks.items():
            if group not in groups:
                continue
            group_metrics = numeric(groups[group])
            for member in members:
                for metric, value in numeric(leaves.get(member, {})).items():
                    if metric in group_metrics or (group, metric) in primary:
                        rows.append(dict(stage=stage, task=member, metric=metric, score=value,
                                         samples=None, level='task', primary=False))

    deduped = {}
    for row in rows:
        key = (row['stage'], row['task'], row['metric'], row['level'])
        if key in deduped:
            deduped[key]['primary'] = deduped[key]['primary'] or row['primary']
        else:
            deduped[key] = row
    rows = list(deduped.values())
    report = dict(checkpoint=manifest['checkpoint'], smoke=manifest['smoke'],
                  effective_context=manifest['effective_context'], harness_commit=manifest['harness_commit'],
                  offline=manifest['offline'], stages=manifest['stages'],
                  ruler_samples=manifest.get('ruler_samples'), ruler_lengths=manifest.get('ruler_lengths'),
                  ruler_data_root=manifest.get('ruler_data_root'),
                  partial=args.partial, errors=errors, rows=rows)
    (run / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (run / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['stage', 'task', 'metric', 'score', 'samples', 'level', 'primary'])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 统一评测入口汇总', '',
             f"- checkpoint: `{manifest['checkpoint']}`",
             f"- harness commit: `{manifest['harness_commit']}`",
             f"- mode: {'smoke' if manifest['smoke'] else 'full'}; offline: {manifest['offline']}",
             f"- stages: {', '.join(manifest['stages'])}",
             f"- 有效上下文: {manifest['effective_context']}", '']
    if manifest.get('ruler_samples'):
        lines += [f"- RULER 冻结数据: {manifest['ruler_samples']} 样本/任务/长度, "
                  f"长度 {', '.join(str(x) for x in manifest.get('ruler_lengths') or [])}, "
                  f"根目录 `{manifest.get('ruler_data_root')}`", '']
    if errors:
        lines += ['## 未通过', ''] + [f'- {error}' for error in errors] + ['']
    lines += ['## 主要指标', '', '| stage | task | metric | score |', '|---|---|---|---|']
    for row in rows:
        if row['primary']:
            lines.append(f"| {row['stage']} | {row['task']} | {row['metric']} | {row['score']:.4f} |")
    if manifest.get('ruler_samples') and 'ruler' in manifest['stages']:
        order = ['niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2',
                 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_1', 'qa_2']
        detail = []
        for row in rows:
            if (row['stage'], row['level'], row['metric']) != ('ruler', 'task', 'ruler_score,none'):
                continue
            parts = row['task'].split('_')
            if len(parts) < 5 or not parts[3].isdigit():
                continue
            name = '_'.join(parts[4:])
            detail.append((int(parts[3]), order.index(name) if name in order else len(order), name, row['score']))
        if detail:
            lines += ['', '## RULER 分项（每长度 13 任务，0-100）', '',
                      '| 长度 | 任务 | ruler_score |', '|---:|---|---:|']
            for length, _, name, score in sorted(detail):
                lines.append(f'| {length} | {name} | {score:.2f} |')
            lines += ['']
    lines += ['', f"全部 {len(rows)} 行在 `summary.csv` / `summary.json`（含各个子数据集与组聚合）。", '']
    (run / 'summary.md').write_text('\n'.join(lines))
    print('SUMMARY', 'FAILED' if errors else 'OK', f'{len(rows)} rows')
    for error in errors:
        print('  ERROR', error)
    return 0 if (not errors or args.partial) else 1


if __name__ == '__main__':
    raise SystemExit(main())
