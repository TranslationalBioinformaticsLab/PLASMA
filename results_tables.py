import os
import numpy as np
import pandas as pd
from IPython.display import display
from compare_variants import Z95, mdd

DASH = '-'
METRIC_LABEL = dict(auc='AUROC', auprc='AUPRC', avg_prec='Avg prec', bal_acc='Bal acc', acc='Acc', prec='Prec', rec='Recall', spec='Spec', f1='F1',
                    brier='Brier', bss='BSS', cal_slope='Cal slope', cal_citl='Cal CIL')
SPLIT_LABEL = dict(internal='TCGA test', external='TRACERx', oof='TCGA OOF')
""" Decimals for one metric. The right number is not the same for all of them: AUROC on ~23 events and ~27 non-events moves in steps of 1/621, so a third digit 
is real, whereas sensitivity moves in steps of 1/23 and a third digit asserts precision the denominator can't provide """
DECIMALS = dict(auc=3, auprc=3, avg_prec=3, brier=3, cal_slope=2, cal_citl=2, default=2)

def _fmt(point, lo, hi, d=2):
    if point is None or not np.isfinite(point):
        return '—'
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return f'{point:.{d}f}'
    sep = ' to ' if (lo < 0 or hi < 0) else DASH
    return f'{point:.{d}f} ({lo:.{d}f}{sep}{hi:.{d}f})'

def _p(p, d=3):
    if p is None or not np.isfinite(p):
        return '—'
    return f'< 0.001' if p < 0.001 else f'{p:.{d}f}'

def _dec(decimals, key):
    if isinstance(decimals, dict):
        return int(decimals.get(key, decimals.get('default', 2)))
    return int(decimals)

def performance_table(res, split, metrics=('auc', 'auprc', 'bal_acc', 'rec', 'spec', 'prec', 'f1'), names=None, order=None, decimals=DECIMALS, out=None, long=False, counts=True):
    multiple_splits = isinstance(split, list)
    def get_table(split_):
        models = [m for m in (order or res) if split_ in (res[m].get('metrics') or {})]
        if order is None:
            models.sort(key=lambda m: res[m]['metrics'][split_].get('auc', -np.inf), reverse=True)
        rows, tidy = [], []
        for m in models:
            pt = res[m]['metrics'][split_]
            ci = (res[m].get('ci') or {}).get(split_) or {}
            row = {'Model': (names or {}).get(m, m)}
            for k in metrics:
                lo, _, hi = ci.get(k, (np.nan, np.nan, np.nan))
                row[METRIC_LABEL.get(k, k)] = _fmt(pt.get(k), lo, hi, _dec(decimals, k))
                tidy.append({'model': m, 'split': split_, 'metric': k, 'estimate': pt.get(k, np.nan), 'lo95': lo, 'hi95': hi})
            if counts and all(k in pt for k in ('tp', 'fp', 'fn', 'tn')):
                row['TN'], row['FP'], row['FN'], row['TP'] = int(pt['tn']), int(pt['fp']), int(pt['fn']), int(pt['tp'])
                tidy += [{'model': m, 'split': split_, 'metric': k, 'estimate': int(pt[k]), 'lo95': np.nan, 'hi95': np.nan} for k in ('tn', 'fp', 'fn', 'tp')]
            rows.append(row)
        disp = pd.DataFrame(rows)
        out_str = f'{SPLIT_LABEL.get(split_, split_)} (n = {_n_scored(res, models[0], split_)})'
        if 'brier' in metrics:
            nib = _no_info_brier(res, models[0], split_)
            if nib is not None:
                out_str += f' The no-information Brier score for this set, obtained by predicting its event rate for every patient, is {nib:.3f}.'
        print(out_str)
        tidy = pd.DataFrame(tidy)
        disp.insert(0, 'Cohort', SPLIT_LABEL.get(split_, split_)); tidy.insert(0, 'Cohort', SPLIT_LABEL.get(split_, split_))
        return disp, tidy
    if multiple_splits:
        disps, tidys = [], []
        for s in split:
            d, t = get_table(s)
            disps.append(d); tidys.append(t)
        disp = pd.concat(disps, ignore_index=True)
        tidy = pd.concat(tidys, ignore_index=True)    
    else: disp, tidy = get_table(split)
    if out: save_table(disp, out)
    display(disp)
    return (disp, pd.DataFrame(tidy)) if long else disp

def _n_scored(res, model, split):
    e = (res[model].get('ens') or {}).get(split) or {}
    t = e.get('targets')
    return len(t) if t is not None else '—'

def _no_info_brier(res, model, split):
    """ p(1 - p) for the scored set: the Brier score of a model that assigns every patient the cohort event rate """
    e = (res[model].get('ens') or {}).get(split) or {}
    t = e.get('targets')
    if t is None or len(t) == 0:
        return None
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return None
    p = float(t.mean())
    return p * (1.0 - p)

def _se_of_diff(g):
    """ SE of the paired difference: boot_se where paired_compare recorded it, else inverted from the interval """
    if 'boot_se' in g.columns and np.isfinite(pd.to_numeric(g['boot_se'], errors='coerce')).any():
        return pd.to_numeric(g['boot_se'], errors='coerce').to_numpy(dtype=float)
    return ((g['hi95'] - g['lo95']) / (2 * Z95)).to_numpy(dtype=float)
 
def resolution_table(cmp, pooled=None, metric='auc', splits=('internal', 'external'), alpha=0.05, power=0.80, holm=True, decimals=3, out=None):
    """ What size of difference each cohort could resolve — the companion table to comparison_table """
    sub = cmp[cmp['metric'] == metric]
    if sub.empty:
        return pd.DataFrame()
    m = int(sub['variant'].nunique())
    mlabel = METRIC_LABEL.get(metric, metric)
    se_col, mdd_col = f'SE of Δ', f'Detectable Δ ({power:.0%})'
    holm_col = f'Detectable Δ ({power:.0%}, Holm m={m})'
 
    def _row(label, g, n, n_pos):
        se = _se_of_diff(g)
        se = se[np.isfinite(se) & (se > 0)]
        if len(se) == 0:
            return None
        med = float(np.median(se))
        r = {'Cohort': label, 'Patients': n, 'Events': n_pos, 'Comparators': m,
             se_col: round(med, decimals),
             mdd_col: round(float(mdd(med, alpha, power, 1)), decimals)}
        if holm and m > 1:
            r[holm_col] = round(float(mdd(med, alpha, power, m)), decimals)
        return r
 
    rows = []
    for s in splits:
        g = sub[sub['split'] == s]
        if g.empty:
            continue
        rows.append(_row(SPLIT_LABEL.get(s, s), g, _first_int(g, 'n'), _first_int(g, 'n_pos')))
    pooled_splits = []
    if pooled is not None and len(pooled):
        pg = pooled[pooled['metric'] == metric] if 'metric' in pooled.columns else pooled
        if len(pg):
            if 'splits' in pg.columns:
                pooled_splits = [s for s in str(pg['splits'].iloc[0]).split(',') if s]
            pooled_splits = pooled_splits or list(splits)
            gc = sub[sub['split'].isin(pooled_splits)]
            # sum the DISTINCT per-split counts, not the rows: cmp holds one row per comparator x split
            n = _sum_distinct(gc, 'n')
            n_pos = _sum_distinct(gc, 'n_pos')
            rows.append(_row('Pooled (inverse variance)', pg, n, n_pos))
    disp = pd.DataFrame([r for r in rows if r]).dropna(axis=1, how='all')
    if disp.empty:
        return disp
    print(f"Resolution: at {power:.0%} power these data could detect differences in {mlabel} of " + '; '.join(f'{r["Cohort"]} {r[mdd_col]:.{decimals}f}' for _, r in disp.iterrows()) + '. Smaller true differences had a lower chance of being detected, so a non-significant result is not evidence that the models perform equally.')
    if out:
        save_table(disp, out)
    return disp
 
def _first_int(g, col):
    if col not in g.columns:
        return None
    v = pd.to_numeric(g[col], errors='coerce').dropna()
    return int(v.iloc[0]) if len(v) else None
 
def _sum_distinct(g, col, by='split'):
    if col not in g.columns or g.empty:
        return None
    v = g.groupby(by)[col].first()
    v = pd.to_numeric(v, errors='coerce').dropna()
    return int(v.sum()) if len(v) else None
 
def comparison_table(cmp, pooled=None, metric='auc', ref='CEM', names=None, order=None, splits=('internal', 'external'), decimals=3, out=None, delong=False, cohort_p=False, show_q=False, 
                     layout='wide', show_raw_p=False, power=False, note=None):
    """ The paired difference from ref, per cohort and pooled. Positive differences favour ref; per-cohort p-values are Holm-corrected within each cohort, as paired_compare returns them; the
    pooled column is corrected separately if a 'p_holm' column is present, otherwise its raw p is shown. The pooled p comes from inverse-variance combination, not from a permutation test """
    if layout not in ('wide', 'long'):
        raise ValueError(f"layout must be 'wide' or 'long'; got {layout!r}")
    sub = cmp[cmp['metric'] == metric]
    variants = [v for v in (order or sub['variant'].unique())]
    mlabel = METRIC_LABEL.get(metric, metric)
    ref_label = (names or {}).get(ref, ref)
    use_delong = bool(delong) and metric == 'auc' and 'delong_p' in cmp.columns
    build = _comparison_long if layout == 'long' else _comparison_wide
    disp = build(sub, pooled, variants, names, splits, decimals, mlabel, use_delong, show_raw_p, show_q, cohort_p)
    shows_cohort_p = cohort_p or layout == 'long'
    if power:
        rt = resolution_table(cmp, pooled, metric=metric, splits=splits, decimals=decimals)
        if not rt.empty:
            disp.attrs['resolution'] = rt
    if out:
        save_table(disp, out)
    display(disp)
    return disp

def _comparison_wide(sub, pooled, variants, names, splits, decimals, mlabel, use_delong, show_raw_p, show_q, cohort_p):
    rows = []
    for v in variants:
        row = {'Model': (names or {}).get(v, v)}
        for s in splits:
            short = SPLIT_LABEL.get(s, s)
            g = sub[(sub['variant'] == v) & (sub['split'] == s)]
            if g.empty:
                row[f'{short} Δ (95% CI)'] = '—'
                continue
            r = g.iloc[0]
            row[f'{short} Δ (95% CI)'] = _fmt(r['diff'], r['lo95'], r['hi95'], decimals)
            if cohort_p:
                corrected = 'perm_p_holm' in r.index and np.isfinite(r['perm_p_holm'])
                if show_raw_p:
                    row[f'{short} perm. P'] = _p(r.get('perm_p'))
                row[f"{short} perm. P ({'Holm' if corrected else 'uncorr.'})"] = _p(r['perm_p_holm'] if corrected else r.get('perm_p'))
            if use_delong and 'delong_p' in r.index and np.isfinite(r['delong_p']):
                d_corr = 'delong_p_holm' in r.index and np.isfinite(r['delong_p_holm'])
                if show_raw_p:
                    row[f'{short} DeLong P'] = _p(r['delong_p'])
                row[f"{short} DeLong P ({'Holm' if d_corr else 'uncorr.'})"] = _p(r['delong_p_holm'] if d_corr else r['delong_p'])
        if pooled is not None:
            g = pooled[pooled['variant'] == v]
            if not g.empty:
                r = g.iloc[0]
                pooled_corrected = 'p_holm' in r.index and np.isfinite(r['p_holm'])
                row['Pooled Δ (95% CI)'] = _fmt(r['diff'], r['lo95'], r['hi95'], decimals)
                if show_raw_p:
                    row['Pooled P'] = _p(r.get('p'))
                row[f"Pooled P ({'Holm' if pooled_corrected else 'uncorr.'})"] = _p(r['p_holm'] if pooled_corrected else r.get('p'))
                if show_q:
                    row['Het. P'] = _p(r.get('Q_p'))
        rows.append(row)
    return pd.DataFrame(rows)

def _comparison_long(sub, pooled, variants, names, splits, decimals, mlabel, use_delong, show_raw_p, show_q, cohort_p=True):
    dcol = f'Δ {mlabel} (95% CI)'
    cols = ['Model', 'Cohort', dcol]
    if show_raw_p:
        cols.append('Perm. P')
    cols.append('Perm. P (Holm)')
    if use_delong:
        if show_raw_p:
            cols.append('DeLong P')
        cols.append('DeLong P (Holm)')
    if show_q and pooled is not None:
        cols.append('Het. P')
    rows = []
    for v in variants:
        label = (names or {}).get(v, v)
        for s in splits:
            g = sub[(sub['variant'] == v) & (sub['split'] == s)]
            if g.empty:
                continue
            r = g.iloc[0]
            row = dict.fromkeys(cols, '—')
            row['Model'], row['Cohort'] = label, SPLIT_LABEL.get(s, s)
            row[dcol] = _fmt(r['diff'], r['lo95'], r['hi95'], decimals)
            if show_raw_p:
                row['Perm. P'] = _p(r.get('perm_p'))
            has_holm = 'perm_p_holm' in r.index and np.isfinite(r['perm_p_holm'])
            row['Perm. P (Holm)'] = _p(r['perm_p_holm'] if has_holm else r.get('perm_p'))
            if use_delong and 'delong_p' in r.index and np.isfinite(r['delong_p']):
                if show_raw_p:
                    row['DeLong P'] = _p(r['delong_p'])
                d_holm = 'delong_p_holm' in r.index and np.isfinite(r['delong_p_holm'])
                row['DeLong P (Holm)'] = _p(r['delong_p_holm'] if d_holm else r['delong_p'])
            rows.append(row)
        if pooled is not None:
            g = pooled[pooled['variant'] == v]
            if not g.empty:
                r = g.iloc[0]
                row = dict.fromkeys(cols, '—')
                row['Model'], row['Cohort'] = label, 'Pooled'
                row[dcol] = _fmt(r['diff'], r['lo95'], r['hi95'], decimals)
                if show_raw_p:
                    row['Perm. P'] = _p(r.get('p'))
                p_holm = 'p_holm' in r.index and np.isfinite(r['p_holm'])
                row['Perm. P (Holm)'] = _p(r['p_holm'] if p_holm else r.get('p'))
                if show_q and 'Het. P' in cols:
                    row['Het. P'] = _p(r.get('Q_p'))
                rows.append(row)
    return pd.DataFrame(rows, columns=cols)

def save_table(df, stem):
    os.makedirs(os.path.dirname(stem) or '.', exist_ok=True)
    cap = df.attrs.get('caption', '')
    written = []
    path = f'{stem}.{f}'
    if f == 'csv':
        df.to_csv(path, index=False)
    else:
        raise ValueError(f'unknown format {f}')
    written.append(path)
    return written