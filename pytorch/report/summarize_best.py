from __future__ import annotations

import json
from pathlib import Path

root = Path('/home/df_05/A_fnlp/pytorch')


def best_run(task_dir: Path, key: str, higher_is_better: bool = True) -> tuple[str, dict, Path]:
    best_name = ''
    best_metrics: dict = {}
    best_path: Path | None = None
    best_val = float('-inf') if higher_is_better else float('inf')
    for p in task_dir.glob('**/metrics.json'):
        m = json.loads(p.read_text(encoding='utf-8'))
        if key not in m:
            continue
        v = float(m[key])
        if (higher_is_better and v > best_val) or ((not higher_is_better) and v < best_val):
            best_val = v
            best_name = p.parent.name
            best_metrics = m
            best_path = p.parent
    if best_path is None:
        raise RuntimeError(f'No metrics with key={key} under {task_dir}')
    return best_name, best_metrics, best_path


t1_name, t1_m, t1_p = best_run(root / 'outputs/task1', 'test_acc')
t2_name, t2_m, t2_p = best_run(root / 'outputs/task2', 'test_acc')
t3_add_name, t3_add_m, t3_add_p = best_run(root / 'outputs/task3', 'best_test_exact')
t3_lm_name, t3_lm_m, t3_lm_p = best_run(root / 'outputs/task3', 'best_val_ppl', higher_is_better=False)

print('TASK1_BEST', {'run': t1_name, 'test_acc': t1_m.get('test_acc'), 'path': str(t1_p)})
print('TASK2_BEST', {'run': t2_name, 'test_acc': t2_m.get('test_acc'), 'path': str(t2_p)})
print('TASK3_ADD_BEST', {'run': t3_add_name, 'best_test_exact': t3_add_m.get('best_test_exact'), 'path': str(t3_add_p)})
print('TASK3_LM_BEST', {'run': t3_lm_name, 'best_val_ppl': t3_lm_m.get('best_val_ppl'), 'path': str(t3_lm_p)})

req_glove = root / 'outputs/task2/req/req_cnn_ce_adam_lr1e3_glove50d/metrics.json'
req_base50 = root / 'outputs/task2/req/req_cnn_ce_adam_lr1e3_base50d/metrics.json'
if req_glove.exists() and req_base50.exists():
    m_glove = json.loads(req_glove.read_text(encoding='utf-8'))
    m_base50 = json.loads(req_base50.read_text(encoding='utf-8'))
    print(
        'TASK2_GLOVE_COMPARE',
        {
            'base50d_test_acc': m_base50.get('test_acc'),
            'glove50d_test_acc': m_glove.get('test_acc'),
            'loaded_glove_vectors': m_glove.get('loaded_glove_vectors'),
        },
    )
