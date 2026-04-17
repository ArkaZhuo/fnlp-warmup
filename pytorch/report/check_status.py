from __future__ import annotations

import json
from pathlib import Path

import nbformat

root = Path('/home/df_05/A_fnlp/pytorch')

checks = {
    'task1_outputs_exists': (root / 'outputs/task1').exists(),
    'task2_outputs_exists': (root / 'outputs/task2').exists(),
    'task3_outputs_exists': (root / 'outputs/task3').exists(),
    'task2_req_exists': (root / 'outputs/task2/req').exists(),
    'glove_50d_exists': (root / 'data/glove/glove.6B.50d.txt').exists(),
    'task3_lm_metrics_exists': (root / 'outputs/task3/lm_baseline/metrics.json').exists(),
}
print('artifact_checks =', checks)

for nb_name in ['task1_report.ipynb', 'task2_report.ipynb', 'task3_report.ipynb']:
    nb_path = root / 'report' / nb_name
    ok = nb_path.exists()
    executed = False
    output_cells = 0
    if ok:
        nb = nbformat.read(nb_path, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        output_cells = sum(1 for c in code_cells if c.get('outputs'))
        executed = all(c.get('execution_count') is not None for c in code_cells)
    print(f'{nb_name}: exists={ok} executed={executed} output_cells={output_cells}')


def best_run(task_dir: Path, key: str, higher_is_better: bool = True) -> tuple[str, dict]:
    best_name = ''
    best_metrics: dict = {}
    best_val = float('-inf') if higher_is_better else float('inf')
    for p in task_dir.glob('**/metrics.json'):
        m = json.loads(p.read_text(encoding='utf-8'))
        if key not in m:
            continue
        val = float(m[key])
        if (higher_is_better and val > best_val) or ((not higher_is_better) and val < best_val):
            best_val = val
            best_name = p.parent.name
            best_metrics = m
    return best_name, best_metrics


t1_name, t1_metrics = best_run(root / 'outputs/task1', key='test_acc')
t2_name, t2_metrics = best_run(root / 'outputs/task2', key='test_acc')
t3_name, t3_metrics = best_run(root / 'outputs/task3', key='best_test_exact')
t3_lm_name, t3_lm_metrics = best_run(root / 'outputs/task3', key='best_val_ppl', higher_is_better=False)

print('task1_best =', {'run': t1_name, **t1_metrics})
print('task2_best =', {'run': t2_name, **t2_metrics})
print('task3_best_add =', {'run': t3_name, **t3_metrics})
print('task3_best_lm =', {'run': t3_lm_name, **t3_lm_metrics})

required_task2_req_runs = {
    'req_cnn_ce_adam_lr1e3_base',
    'req_cnn_ce_adam_lr5e4_base',
    'req_cnn_mse_adam_lr1e3_base',
    'req_cnn_mse_adam_lr5e4_base',
    'req_cnn_ce_sgd_lr1e3_base',
    'req_cnn_ce_adam_lr1e3_k128',
    'req_cnn_ce_adam_lr1e3_k2345',
    'req_cnn_ce_adam_lr1e3_base50d',
    'req_cnn_ce_adam_lr1e3_glove50d',
}
existing_req_runs = {
    p.parent.name
    for p in (root / 'outputs/task2/req').glob('*/metrics.json')
}
missing_req = sorted(required_task2_req_runs - existing_req_runs)
print('task2_req_missing_runs =', missing_req)

glove_metrics_path = root / 'outputs/task2/req/req_cnn_ce_adam_lr1e3_glove50d/metrics.json'
if glove_metrics_path.exists():
    glove_metrics = json.loads(glove_metrics_path.read_text(encoding='utf-8'))
    print('task2_glove_loaded_vectors =', glove_metrics.get('loaded_glove_vectors'))
else:
    print('task2_glove_loaded_vectors = missing_metrics')
