import numpy as np
import pandas as pd
from scipy import stats
# from sklearn.metrics import (auc, balanced_accuracy_score, f1_score, precision_recall_curve, roc_auc_score)
from train_util import compute_metrics

Z95 = 1.959964 # two-sided 95% normal quantile

def mdd(se, alpha=0.05, power=0.80, m=1):
    """ Minimum detectable difference: the smallest TRUE difference a design with this standard error would detect with probability 'power'. Power is computed from observed effect size.
    MDD = (z_{1 - alpha/(2m)} + z_power) * SE """
    za = stats.norm.ppf(1 - alpha / (2 * m))
    zb = stats.norm.ppf(power)
    return (za + zb) * np.asarray(se, dtype=float)

def _aligned(res, versions, split):
    """ Predictions for every version on the same patients, in the same order """
    ids = None
    for v in versions:
        e = res[v]['ens'][split]
        order = np.argsort(np.asarray(e['ids']).astype(str))
        vid = np.asarray(e['ids']).astype(str)[order]
        if ids is None:
            ids, y = vid, np.asarray(e['targets'])[order].astype(int)
        elif not np.array_equal(ids, vid):
            raise ValueError(f"version {v} was scored on different patients for split '{split}'; paired comparison requires an identical test set")
        res[v].setdefault('_aligned', {})[split] = np.asarray(e['probs_cal'])[order]
    probs = {v: res[v]['_aligned'][split] for v in versions}
    return ids, y, probs

def _midrank(x):
    order = np.argsort(x)
    ranked, i, n = np.zeros(len(x), dtype=float), 0, len(x)
    xs = x[order]
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        ranked[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranked
    return out

def delong_test(y_true, prob_a, prob_b):
    """ DeLong's test for two AUCs on the same patients """
    y = np.asarray(y_true).astype(int)
    pos, neg = np.asarray(prob_a)[y == 1], np.asarray(prob_a)[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return (np.nan,) * 5
    preds = np.vstack([np.asarray(prob_a), np.asarray(prob_b)])
    k = preds.shape[0]
    pos_all, neg_all = preds[:, y == 1], preds[:, y == 0]
    tx = np.array([_midrank(pos_all[r]) for r in range(k)])
    ty = np.array([_midrank(neg_all[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos_all[r], neg_all[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    s = np.atleast_2d(s)
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    se = float(np.sqrt(max(var_diff, 0)))
    diff = float(aucs[0] - aucs[1])
    p = float(2 * stats.norm.sf(abs(diff) / se)) if se > 0 else (1.0 if diff == 0 else 0.0)
    return float(aucs[0]), float(aucs[1]), diff, se, p

def _fast_auc(y, p):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(p, kind='mergesort')
    sp = p[order]
    _, inv, counts = np.unique(sp, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    avg_rank = (end - counts + end + 1) / 2.0 # average 1-based rank within each tie group
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = avg_rank[inv]
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def _fast_auprc(y, p):
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return np.nan
    desc = np.argsort(-p, kind='mergesort')
    ps, ys = p[desc], y[desc]
    distinct = np.flatnonzero(np.diff(ps))
    idx = np.r_[distinct, len(ys) - 1]
    tps = np.cumsum(ys)[idx].astype(float)
    fps = 1.0 + idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    last = int(np.searchsorted(tps, tps[-1])) # drop the flat tail sklearn also drops
    sl = slice(last, None, -1)
    precision = np.r_[precision[sl], 1.0]
    recall = np.r_[recall[sl], 0.0]
    trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(-trapz(precision, recall))

CAL_EPS = 1e-6

def _metric(y, p, name, threshold=0.5):
    y = np.asarray(y).astype(np.int8)
    p = np.asarray(p, dtype=float)
    if name == 'auc':
        return _fast_auc(y, p)
    if name in ('bal_acc', 'f1'):
        yh = p >= (0.5 if threshold is None else threshold)
        pos, neg = y == 1, ~(y == 1)
        tp = float(np.count_nonzero(yh & pos))
        fp = float(np.count_nonzero(yh & neg))
        fn = float(np.count_nonzero(~yh & pos))
        tn = float(np.count_nonzero(~yh & neg))
        if name == 'f1':
            denom = 2 * tp + fp + fn
            return 2 * tp / denom if denom else 0.0 # matches f1_score(zero_division=0)
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        return float(np.nanmean([rec, spec]))
    if name == 'auprc':
        return _fast_auprc(y, p)
    if name in ('brier', 'bss'):
        b = float(np.mean((np.clip(p, CAL_EPS, 1 - CAL_EPS) - y) ** 2))
        if name == 'brier':
            return b
        pbar = float(y.mean())
        ref = pbar * (1.0 - pbar)
        return 1.0 - b / ref if ref > 0 else np.nan
    return compute_metrics(y, p, threshold, plot_metrics=False).get(name, np.nan)

def paired_bootstrap(y, prob_a, prob_b, metric='auc', thr_a=None, thr_b=None, n_boot=2000, seed=0):
    """ Bootstraps the difference between two models on the same patients. Each resample draws patients once and scores both models on them, so the correlation between the two models is
    preserved and only the difference is bootstrapped. Returns the observed difference, 95% percentile interval, SD of the replicate distribution, and fraction of resamples favoring model A 
    (a bootstrap analogue of a one-sided p-value) """
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    obs = _metric(y, prob_a, metric, thr_a) - _metric(y, prob_b, metric, thr_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(_metric(y[idx], prob_a[idx], metric, thr_a) - _metric(y[idx], prob_b[idx], metric, thr_b))
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {'diff': float(obs) if np.isfinite(obs) else np.nan, 'lo95': np.nan, 'hi95': np.nan, 'se': np.nan, 'frac_favouring_a': np.nan, 'n_boot_used': 0}
    return {'diff': float(obs), 'lo95': float(np.percentile(d, 2.5)), 'hi95': float(np.percentile(d, 97.5)), 'se': float(d.std(ddof=1)) if len(d) > 1 else np.nan,
            'frac_favouring_a': float((d > 0).mean()), 'n_boot_used': int(len(d))}

def paired_permutation(y, prob_a, prob_b, metric='auc', thr_a=None, thr_b=None, n_perm=5000, seed=0):
    """ Two-sided paired permutation test on the difference between two models scored on the same patients """
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int)
    a, b = np.asarray(prob_a, dtype=float), np.asarray(prob_b, dtype=float)
    obs = _metric(y, a, metric, thr_a) - _metric(y, b, metric, thr_b)
    if not np.isfinite(obs):
        return {'perm_diff': np.nan, 'perm_p': np.nan, 'n_perm': 0}
    count = 0
    for _ in range(n_perm):
        swap = rng.rand(len(y)) < 0.5
        pa, pb = np.where(swap, b, a), np.where(swap, a, b)
        # Thresholds stay with their model: swapping a patient moves its prediction, not the operating point
        d = _metric(y, pa, metric, thr_a) - _metric(y, pb, metric, thr_b)
        if np.isfinite(d) and abs(d) >= abs(obs) - 1e-12:
            count += 1
    return {'perm_diff': float(obs), 'perm_p': float((count + 1) / (n_perm + 1)), 'n_perm': int(n_perm)}

def paired_compare(res, ref='CEM', versions=None, splits=('internal', 'external'), metrics=('auc', 'auprc', 'bal_acc'), n_boot=2000, n_perm=5000, seed=0, holm=True):
    versions = [v for v in (versions or res.keys()) if v != ref]
    rows = []
    for split in splits:
        ids, y, probs = _aligned(res, [ref] + versions, split)
        for v in versions:
            thr_ref, thr_v = res[ref].get('threshold'), res[v].get('threshold')
            for metric in metrics:
                bs = paired_bootstrap(y, probs[ref], probs[v], metric=metric, thr_a=thr_ref, thr_b=thr_v, n_boot=n_boot, seed=seed)
                row = {'split': split, 'metric': metric, 'ref': ref, 'variant': v, 'n': int(len(y)), 'n_pos': int(y.sum()), f'{ref}': _metric(y, probs[ref], metric, thr_ref),
                       'variant_val': _metric(y, probs[v], metric, thr_v), **{k: bs[k] for k in ('diff', 'lo95', 'hi95', 'frac_favouring_a')}, 'boot_se': bs['se'],
                       'mdd80': float(mdd(bs['se'])) if np.isfinite(bs['se']) else np.nan}
                if metric == 'auc':
                    *_, se, p = delong_test(y, probs[ref], probs[v])
                    row.update({'delong_se': se, 'delong_p': p})
                if n_perm:
                    row.update(paired_permutation(y, probs[ref], probs[v], metric=metric, thr_a=thr_ref, thr_b=thr_v, n_perm=n_perm, seed=seed))
                rows.append(row)
    out = pd.DataFrame(rows)
    if holm:
        for raw, adj_col in (('delong_p', 'delong_p_holm'), ('perm_p', 'perm_p_holm')):
            if raw not in out.columns:
                continue
            out[adj_col] = np.nan
            for _, grp in out.groupby(['split', 'metric']):
                p = grp[raw].to_numpy(dtype=float)
                if np.all(np.isnan(p)):
                    continue
                out.loc[grp.index, adj_col] = _holm(p)
    return out

def _holm(p):
    """ Holm step-down adjusted p-values """
    p = np.asarray(p, dtype=float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj, running = np.full(len(p), np.nan), 0.0
    n_valid = int(np.isfinite(p).sum())
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        running = max(running, (n_valid - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj

def combine_splits(cmp, metric='auc', splits=('oof', 'internal', 'external'), se_from='boot'):
    """ Inverse-variance combination of the paired differences across the scored sets """
    if se_from not in ('boot', 'ci'):
        raise ValueError(f"se_from must be 'boot' or 'ci', got {se_from!r}")
    sub = cmp[(cmp['metric'] == metric) & (cmp['split'].isin(splits))]
    rows = []
    for variant, g in sub.groupby('variant'):
        if se_from == 'boot' and 'boot_se' in g.columns and np.isfinite(g['boot_se']).any():
            se = g['boot_se']
        else:
            se = (g['hi95'] - g['lo95']) / (2 * Z95)
        ok = np.isfinite(se) & (se > 0)
        if not ok.any():
            continue
        d_i, w = g['diff'].to_numpy()[ok.to_numpy()], (1 / se.pow(2)).to_numpy()[ok.to_numpy()]
        d = float((d_i * w).sum() / w.sum())
        se_c = float(np.sqrt(1 / w.sum()))
        q = float((w * (d_i - d) ** 2).sum())
        rows.append({'variant': variant, 'metric': metric, 'n_splits': int(ok.sum()), 'splits': ','.join(g['split'].to_numpy()[ok.to_numpy()]), 'diff': d, 'lo95': d - Z95 * se_c, 
                     'hi95': d + Z95 * se_c, 'boot_se': se_c, 'mdd80': float(mdd(se_c)), 'se_from': se_from, 'p': float(2 * stats.norm.sf(abs(d) / se_c)) if se_c > 0 else np.nan,
                     'Q': q, 'Q_p': float(stats.chi2.sf(q, max(ok.sum() - 1, 1)))})
    return pd.DataFrame(rows)