import hashlib
import json
import os
import pickle
import platform
import subprocess
import time

import numpy as np
import pandas as pd

RUN_DIR = 'results/saved_runs'

def _git_rev():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None

def _cohort_fingerprint(lung):
    if lung is None:
        return None
    parts = ['|'.join(map(str, lung.tcga_x.index)), '|'.join(map(str, lung.trcr_x.index))]
    y = pd.to_numeric(pd.Series(np.asarray(lung.tcga_y).ravel()), errors='coerce')
    return {'tcga_n': int(len(lung.tcga_x)), 'trcr_n': int(len(lung.trcr_x)), 'n_labelled': int(y.notna().sum()), 'n_censored': int(y.isna().sum()),
            'n_features': {k: len(v) for k, v in lung.cols.items()}, 'hash': hashlib.sha1('||'.join(parts).encode()).hexdigest()[:12]}

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return {'__ndarray__': True, 'shape': list(x.shape), 'head': _jsonable(x.ravel()[:8].tolist())}
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return repr(x)

_UNPICKLABLE = ('transform',)
_HEAVY = ('models', 'x_train', 'y_train', 'x_test', 'y_test', 'x_external', 'y_external', 'surv_train', 'params')

def _strip_unpicklable(result, keep_heavy=False):
    drop = set(_UNPICKLABLE) if keep_heavy else set(_UNPICKLABLE) | set(_HEAVY)
    if not isinstance(result, dict):
        return result, []
    out, removed = {}, set()
    for k, v in result.items():
        if isinstance(v, dict):
            kept = {kk: vv for kk, vv in v.items() if kk not in drop}
            removed |= (set(v) & drop)
            out[k] = kept
        else:
            out[k] = v
    return out, sorted(removed)

def save_run(result, name, kind, base=None, fold_kwargs=None, split=None, folds=None, lung=None, known_genes=None, grid=None, notes='', extra=None, run_dir=RUN_DIR, overwrite=False, keep_heavy=False):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f'{name}.pkl')
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f'{path} exists; pass overwrite=True or pick another name')
    ids = None
    if split is not None and lung is not None:
        tr, te = split
        ids = {'train': list(map(str, lung.tcga_x.index[tr])), 'test': list(map(str, lung.tcga_x.index[te]))}
    result, dropped = _strip_unpicklable(result, keep_heavy=keep_heavy)
    meta = {
        'name': name, 'kind': kind, 'notes': notes, 'dropped_keys': dropped,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_rev': _git_rev(), 'host': platform.node(),
        'base': _jsonable(base), 'fold_kwargs': _jsonable(fold_kwargs), 'grid': _jsonable(grid),
        'n_known_genes': None if known_genes is None else len(set(known_genes)),
        'cohort': _cohort_fingerprint(lung),
        'summary': _summarize(result, kind),
    }
    payload = {'result': result, 'meta': meta, 'split': split, 'folds': folds, 'split_ids': ids, 'known_genes': None if known_genes is None else sorted(map(str, known_genes)), 'extra': extra}
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    with open(os.path.join(run_dir, f'{name}.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"saved {path}")
    if dropped:
        print(f"  not stored (regenerable from the saved config): {', '.join(dropped)}")
    if meta['summary']:
        print('  ' + '; '.join(f"{k} {v}" for k, v in meta['summary'].items()))
    return path

def load_run(name, run_dir=RUN_DIR, lung=None):
    with open(os.path.join(run_dir, f'{name}.pkl'), 'rb') as f:
        payload = pickle.load(f)
    if lung is not None and payload['meta'].get('cohort'):
        now, then = _cohort_fingerprint(lung), payload['meta']['cohort']
        if now['hash'] != then['hash']:
            print(f"[warn] cohort differs from the one this run used (then {then['tcga_n']} TCGA / {then['n_censored']} censored, now {now['tcga_n']} / {now['n_censored']}). Stored positional indices no longer refer to the same patients; use split_ids instead.")
    return payload

def _summarize(result, kind):
    try:
        return {v: f"int {r['metrics']['internal']['auc']:.3f} / ext {r['metrics']['external']['auc']:.3f}" for v, r in result.items()}
    except Exception:
        pass
    return {}

CURVE_KEYS = ('prc_prec', 'prc_rec', 'prc_thresholds', 'fpr', 'tpr', 'roc_thresholds', 'thresholds')

def _scalars(m, keep_curves=False):
    if keep_curves:
        return dict(m)
    return {k: v for k, v in m.items() if k not in CURVE_KEYS and not isinstance(v, (list, tuple, np.ndarray))}

def curves(result, version, split='internal'):
    m = result[version]['metrics'][split]
    return {k: np.asarray(m[k]) for k in CURVE_KEYS if k in m}

def metrics_table(result, kind='fixed_split', keep_curves=False):
    rows = []
    for v, r in result.items():
        for split_name, m in r['metrics'].items():
            rows.append({'version': v, 'split': split_name, **_scalars(m, keep_curves)})
    return pd.DataFrame(rows)

def tuning_table(result):
    rows = []
    for v, r in result.items():
        if isinstance(r, dict) and 'best_params' in r:
            rows.append({'version': v, 'inner_best_auc': r.get('inner_best_auc'), 'internal_auc': r['metrics']['internal']['auc'], 'gap': (r.get('inner_best_auc') or np.nan) - r['metrics']['internal']['auc'], **(r.get('best_params') or {})})
    return pd.DataFrame(rows)

def predictions(result, version, split='internal'):
    e = result[version]['ens'][split]
    return pd.DataFrame({'id': e['ids'], 'logit': e['logits'], 'prob': e['probs_cal'], 'target': e['targets']})

def merge_results(*results):
    out = {}
    for r in results:
        dup = set(r) & set(out)
        if dup:
            raise ValueError(f'duplicate version name(s): {sorted(dup)}; rename before merging')
        out.update(r)
    return out

def trials_table(result, version=None):
    rows = []
    for v, r in result.items():
        if version is not None and v != version:
            continue
        trials = (r or {}).get('trials') or []
        for t in trials:
            rows.append({'version': v, **t})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(['version', 'value'], ascending=[True, False])
        df['rank'] = df.groupby('version')['value'].rank(ascending=False, method='min').astype(int)
        df['gap_from_best'] = df['value'] - df.groupby('version')['value'].transform('max')
    return df