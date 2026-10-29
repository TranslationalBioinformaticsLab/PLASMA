import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D as _L2D
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.backend_bases import _get_renderer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc
from scipy.stats import norm

LINE_PROPS = {
    'CEM': {'color': '#FC1C7D', 'lw': 1.9, 'solid_joinstyle': 'round'},
    'C':   {'color': '#0073E6', 'lw': 1.4}, 'E':   {'color': '#059669', 'lw': 1.4},
    'M':   {'color': '#D97706', 'lw': 1.4}, 'CE':  {'color': '#7C3AED', 'lw': 1.4},
    'CM':  {'color': '#0891B2', 'lw': 1.4}, 'EM':  {'color': '#BE185D', 'lw': 1.4},
    'LR':  {'color': '#02B902', 'lw': 1.4}, 'RF':  {'color': '#961C04', 'lw': 1.4},
    'MLP': {'color': '#0073E6', 'lw': 1.4}, 'SVM': {'color': '#6600FF', 'lw': 1.4},
    'KNN': {'color': '#FFD700', 'lw': 1.4}, 'XGB': {'color': '#444444', 'lw': 1.4},
}
SPLIT_TITLE = {'oof': 'TCGA training (out-of-fold)', 'internal': 'TCGA held-out test', 'external': 'TRACERx'}
SPLIT_COLOR = {'oof': '#6B7280', 'internal': '#2563EB', 'external': '#059669'}
BAR_KEYS = ['auc', 'auprc', 'bal_acc', 'prec', 'rec', 'spec', 'f1']
LABELS = {'auc': 'AUROC', 'auprc': 'AUPRC', 'avg_prec': 'AP', 'acc': 'ACC', 'bal_acc': 'BAL ACC', 'prec': 'PREC', 'rec': 'SENS', 'spec': 'SPEC', 'f1': 'F1', 'mcc': 'MCC', 'brier': 'Brier'}

def _yp(results, version, split):
    e = results[version]['ens'][split]
    return np.asarray(e['targets']).astype(int), np.asarray(e['probs_cal'], dtype=float)

def _props(version, line_props=None):
    lp = dict((line_props or LINE_PROPS).get(version, {}))
    lp.setdefault('lw', 1.4)
    if 'color' not in lp:
        lp['color'] = plt.get_cmap('tab10')(abs(hash(version)) % 10)
    return lp

def _name(version, names=None):
    return version if names is None else names.get(version, version)

def _legend(ax, loc='lower right', ncol=1, fontsize=8):
    h, l = ax.get_legend_handles_labels()
    leg = ax.legend(h[::-1], l[::-1], loc=loc, fontsize=fontsize, ncol=ncol, frameon=True, fancybox=True, framealpha=0.85, edgecolor='0.7', handlelength=1.2, handletextpad=.6, labelspacing=.2, columnspacing=.8, borderpad=.4)
    leg.set_zorder(10)
    return leg

def _housekeeping(ax, xlabel, ylabel, title, labelsize=8, titlesize=9, legend=True, loc='lower right', legendsize=7.5):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel, fontsize=labelsize, fontweight=540)
    ax.set_ylabel(ylabel, fontsize=labelsize, fontweight=540)
    ax.set_title(title, fontsize=titlesize, fontweight='bold', pad=5)
    ax.tick_params(axis='both', labelsize=labelsize)
    if legend:
        _legend(ax, loc=loc, fontsize=legendsize)

def _metric_value(results, version, split, key):
    m = (results[version].get('metrics') or {}).get(split) or {}
    if key in m and np.isfinite(m[key]):
        return float(m[key])
    y, p = _yp(results, version, split)
    if key == 'auc':
        return float(roc_auc_score(y, p))
    if key == 'auprc':
        prec, rec, _ = precision_recall_curve(y, p)
        return float(auc(rec, prec))
    if key == 'avg_prec':
        return float(average_precision_score(y, p))
    raise KeyError(f"no stored '{key}' for {version}/{split} and no fallback defined")

def _sorted_by(results, versions, split, metric, house):
    vals = {v: _metric_value(results, v, split, metric) for v in versions}
    return dict(sorted(vals.items(), key=lambda it: (it[1], it[0] == house)))

def _curve(results, version, split, kind):
    m = (results[version].get('metrics') or {}).get(split) or {}
    keys = ('fpr', 'tpr') if kind == 'roc' else ('prc_rec', 'prc_prec')
    if all(k in m and m[k] is not None and len(m[k]) for k in keys):
        return np.asarray(m[keys[0]], dtype=float), np.asarray(m[keys[1]], dtype=float)
    y, p = _yp(results, version, split)
    if kind == 'roc':
        fpr, tpr, _ = roc_curve(y, p)
        return fpr, tpr
    prec, rec, _ = precision_recall_curve(y, p)
    return rec, prec

def _prevalence(results, versions, split):
    for v in versions:
        m = (results[v].get('metrics') or {}).get(split) or {}
        if all(k in m for k in ('tn', 'fp', 'fn', 'tp')):
            n = m['tn'] + m['fp'] + m['fn'] + m['tp']
            if n:
                return float((m['tp'] + m['fn']) / n)
    y, _ = _yp(results, versions[0], split)
    return float(y.mean())

def _panel_extent(ax, renderer, right=False):
    bb = ax.get_window_extent(renderer)
    xs = [bb.x1 if right else bb.x0]
    parts = [ax.yaxis.label] + list(ax.get_yticklabels(which='both'))
    for part in parts:
        if part.get_visible() and part.get_text():
            pb = part.get_window_extent(renderer)
            xs.append(pb.x1 if right else pb.x0)
    # Tick labels drawn as AnnotationBbox (e.g. italicize_gene_yticklabels) instead of real tick labels
    for art in ax.artists:
        if isinstance(art, AnnotationBbox) and art.get_visible():
            pb = art.get_window_extent(renderer)
            if pb.width > 0:
                xs.append(pb.x1 if right else pb.x0)
    x_px = max(xs) if right else min(xs)
    return ax.figure.transFigure.inverted().transform((x_px, 0))[0]
 
def _panel_label(ax, text, right=False, dy=0.015, dx=0.0, align='outer', fontsize=10, x=None, draw=True, **kw):
    t = ax.text(0, 0, text, fontsize=fontsize, fontweight='bold', va='bottom', ha='right' if right else 'left', clip_on=False, zorder=10, **kw)
    if align == 'box':
        t.set_transform(ax.transAxes)
        t.set_position((1.0 if right else 0.0, 1.0 + dy))
        return t
    fig = ax.figure
    t.set_transform(mtransforms.blended_transform_factory(fig.transFigure, ax.transAxes))
    def place():
        xx = x if x is not None else _panel_extent(ax, _renderer(fig), right)
        t.set_position((xx + dx, 1.0 + dy))
    _register_placer(fig, place)
    if draw:
        _draw_once(fig)
    place()
    return t

def _renderer(fig):
    get = getattr(fig.canvas, 'get_renderer', None)
    if get is not None:
        return get()
    return _get_renderer(fig)
 
def _draw_once(fig):
    try:
        fig.canvas.draw()
        return fig.canvas.get_renderer()
    except Exception:
        return _renderer(fig)
 
def _register_placer(fig, fn):
    fig._panel_label_placers = getattr(fig, '_panel_label_placers', []) + [fn]
 
def refresh_panel_labels(fig):
    placers = getattr(fig, '_panel_label_placers', [])
    if placers:
        _draw_once(fig)
        for fn in placers:
            fn()
    return len(placers)
 
def panel_label(ax, text, **kw):
    # Alias for _panel_label, for hand built figures like the shap + association panel
    return _panel_label(ax, text, **kw)
 
def panel_labels(axes, letters='ABCDEFGH', column=True, right=False, tol=0.02, **kw):
    axes = list(axes)
    fig = axes[0].figure

    def measure():
        r = _renderer(fig)
        ext = {ax: _panel_extent(ax, r, right) for ax in axes}
        if not column:
            return ext
        pos = {ax: (ax.get_position().x1 if right else ax.get_position().x0) for ax in axes}
        pick = max if right else min
        return {ax: pick(ext[a] for a in axes if abs(pos[a] - pos[ax]) <= tol) for ax in axes}
 
    _draw_once(fig)
    shared = measure()
    texts = [_panel_label(ax, lt, right=right, x=shared[ax], draw=False, **kw) for ax, lt in zip(axes, letters)]
    def replace_group():
        fresh = measure()
        for a, t in zip(axes, texts):
            t.set_position((fresh[a], t.get_position()[1]))
    _register_placer(fig, replace_group)
    return texts

def plot_roc_pr(results, splits=('internal', 'external'), versions=None, house='CEM', names=None, line_props=None, sig=None, fsize=None, dpi=300, wspace=.25, hspace=.3, save=None, show=True,
                suptitle=None, label_panels=False, use_pct=False, n_grid=101):
    versions = [v for v in (versions or results.keys()) if v in results]
    splits = [s for s in splits if all(s in results[v]['ens'] for v in versions)]
    sig = sig or {}
    fsize = fsize or (7.5, 3.6 * len(splits))
    fig, axes = plt.subplots(len(splits), 2, figsize=fsize, dpi=dpi, gridspec_kw={'wspace': wspace, 'hspace': hspace}, squeeze=False)
    labels = iter('ABCDEFG')
    grid = np.linspace(0, 1, n_grid)
    for row, split in enumerate(splits):
        ax_roc, ax_pr = axes[row]
        ax_roc.plot([0, 1], [0, 1], color='#B2B2B2', lw=1.5, ls='--', zorder=1, label='Random chance')
        for i, (v, auc_v) in enumerate(_sorted_by(results, versions, split, 'auc', house).items()):
            fpr, tpr = _curve(results, v, split, 'roc')
            star = '*' if v in sig.get(split, set()) else ''
            lbl = f"{_name(v, names)}{star} = {auc_v * 100:.1f}%" if use_pct else f"{_name(v, names)}{star} = {auc_v:.3f}"
            ax_roc.step(fpr, tpr, **_props(v, line_props), label=lbl, zorder=2 + i)
        _housekeeping(ax_roc, 'False Positive Rate', 'True Positive Rate', f'{SPLIT_TITLE.get(split, split)} ROC')
        # No-skill line = prevalence of THIS set; AUPRC is not comparable across sets with different prevalence
        prev = _prevalence(results, versions, split)
        ns_lbl = f'No skill ({prev * 100:.1f}%)' if use_pct else f'No skill ({prev:.3f})'
        ax_pr.axhline(prev, color='#B2B2B2', lw=1.5, ls='--', zorder=1, label=ns_lbl)
        for i, (v, auprc_v) in enumerate(_sorted_by(results, versions, split, 'auprc', house).items()):
            rec, prec = _curve(results, v, split, 'pr')
            star = '*' if v in sig.get(split, set()) else ''
            lbl = f"{_name(v, names)}{star} = {auprc_v * 100:.1f}%" if use_pct else f"{_name(v, names)}{star} = {auprc_v:.3f}"
            ax_pr.step(rec, prec, where='post', **_props(v, line_props), label=lbl, zorder=2 + i)
        _housekeeping(ax_pr, 'Recall', 'Precision', f'{SPLIT_TITLE.get(split, split)} PR', loc='lower right')
        if sig.get(split):
            ax_roc.text(.98, .02, '* paired difference vs %s, p < 0.05' % _name(house, names), transform=ax_roc.transAxes, ha='right', va='bottom', fontsize=7, style='italic', color='gray')
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=0.995)
        fig.subplots_adjust(top=0.93)
    if label_panels:
        letters = list('ABCDEF')
        panel_labels([axes[r][0] for r in range(len(splits))], letters[0::2])
        panel_labels([axes[r][1] for r in range(len(splits))], letters[1::2])
    return _finish(fig, save, show, dpi)

def _finish(fig, save, show, dpi):
    if save:
        os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
        fig.savefig(save, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), min(1.0, c + h)

def plot_observed_association(cohorts, order=None, labels=None, colors=None, fsize=(4.2, 3.0), dpi=600, ylabel='Two-year recurrence', title=None, annotate_n=True, tests=None,
                              bracket_gap=0.045, bracket_step=0.085, ymax=None, out=None, show=False, ax=None):
    """ Observed event rate per category, per cohort, with Wilson intervals """
    cats = list(order) if order else list(dict.fromkeys(next(iter(cohorts.values()))[0].dropna()))
    names = list(cohorts)
    colors = colors or {names[0]: '#2563EB', **({names[1]: '#059669'} if len(names) > 1 else {})}
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=fsize)
    else:
        fig = ax.figure
    width = 0.8 / max(len(names), 1)
    x = np.arange(len(cats))
    bar_pos, bar_top = {}, {}
    for i, nm in enumerate(names):
        st, y = cohorts[nm]
        st = pd.Series(np.asarray(st)).reset_index(drop=True)
        y = np.asarray(y).astype(int)
        rates, los, his, ns = [], [], [], []
        for c in cats:
            m = (st == c).to_numpy()
            n, k = int(m.sum()), int(y[m].sum()) if m.sum() else 0
            rates.append(k / n if n else np.nan)
            lo, hi = _wilson(k, n)
            los.append(rates[-1] - lo if n else np.nan)
            his.append(hi - rates[-1] if n else np.nan)
            ns.append(n)
        pos = x + i * width
        ax.bar(pos, rates, width * .9, yerr=[los, his], capsize=2.5, label=nm, color=colors.get(nm, f'C{i}'), alpha=1, error_kw=dict(lw=1, alpha=1))
        if annotate_n:
            for xx, r, n in zip(pos, rates, ns):
                if n:
                    ax.text(xx, 0.015, f'$n$={n}', ha='center', va='bottom', fontsize=7, color='white')
        for c, xx, r, h in zip(cats, pos, rates, his):
            bar_pos[(nm, c)] = xx
            bar_top[(nm, c)] = (r + h) if np.isfinite(r) else 0.0      # top of the error bar, for bracket placement
    drawn_top = None
    for i, (nm, (ca, cb, p)) in enumerate(sorted((tests or {}).items())):
        if (nm, ca) not in bar_pos or (nm, cb) not in bar_pos:
            continue
        xa, xb = bar_pos[(nm, ca)], bar_pos[(nm, cb)]
        yb = max(bar_top[(nm, ca)], bar_top[(nm, cb)]) + bracket_gap
        if drawn_top is not None:
            yb = max(yb, drawn_top + bracket_step)
        ax.plot([xa, xa, xb, xb], [yb - 0.013, yb, yb, yb - 0.013], lw=0.9, color='#111827', clip_on=False)
        ax.text((xa + xb) / 2, yb + 0.006, f"{nm}: " + ('$P$ < 0.001' if p < 0.001 else f'$P$ = {p:.3f}'),
                ha='center', va='bottom', fontsize=7.5, color='#111827')
        drawn_top = yb
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([(labels or {}).get(c, str(c).capitalize()) for c in cats], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9, fontweight=540, labelpad=5)
    # Headroom for however many brackets were stacked, plus room for their labels
    ax.set_ylim(0, ymax if ymax else ((drawn_top + 0.10) if tests and drawn_top else 0.75))
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.tick_params(axis='y', labelsize=8)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=7.5, frameon=True, framealpha=.85, edgecolor='0.7', loc='upper right', borderpad=.35)
    for sp in ax.spines.values():
        sp.set_color('black')
    if out and created:
        fig.savefig(out, bbox_inches='tight', dpi=dpi)
    if created:
        plt.show() if show else plt.close(fig)
    return ax

def pool_log_or(estimates):
    """ Inverse-variance pooled odds ratio from a set of independent estimates """
    lo_, w_ = [], []
    for o, lo, hi in estimates.values():
        l = np.log(o)
        se = (np.log(hi) - np.log(lo)) / (2 * 1.96)
        lo_.append(l)
        w_.append(1 / se ** 2)
    lo_, w_ = np.asarray(lo_), np.asarray(w_)
    b = (lo_ * w_).sum() / w_.sum()
    se = np.sqrt(1 / w_.sum())
    return float(np.exp(b)), float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se)), \
        float(2 * norm.sf(abs(b / se)))

def plot_or_forest(estimates, fsize=(4.2, 2.0), dpi=600, xlabel='Odds ratio (current vs former smoker)', title=None, legend=True, legend_loc='lower right', out=None, show=False, ax=None, note=None):
    """ Forest plot of odds ratios on a log axis, one row per cohort plus the pooled estimate """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=fsize)
    else:
        fig = ax.figure
    labs = list(estimates)
    y = np.arange(len(labs))[::-1]
    for yy, lab in zip(y, labs):
        o, lo, hi = estimates[lab]
        pooled = lab.lower().startswith('pool')
        adj = ('adj' in lab.lower()) or ('*' in lab)
        col = '#FC1C7D' if pooled else ('#6B7280' if adj else '#111827')
        face = 'none' if (pooled and adj) else col
        ax.plot([lo, hi], [yy, yy], color=col, lw=2 if pooled else 1.5, solid_capstyle='butt', zorder=3)
        ax.plot([o], [yy], marker='D' if pooled else 'o', ms=6 if pooled else (5 if adj else 5.5), color=col, markerfacecolor=face, markeredgewidth=1.5 if (pooled and adj) else 1.0, zorder=4)
    ax.axvline(1.0, color='#B2B2B2', lw=1.2, ls='--', zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks([0.2, 0.5, 1.0, 2.0])
    ax.set_xticklabels(['0.2', '0.5', '1.0', '2.0'], fontsize=8)
    ax.minorticks_off()
    ax.set_xlim(0.17, 2.2)
    ax.set_xlabel(xlabel, fontsize=9, fontweight=540)
    if note:
        ax.annotate(note, xy=(.53, -0.28), xycoords='axes fraction', fontsize=7, color='#52514e', ha='left', va='top', annotation_clip=False)
    ax.set_ylim(-0.6, len(labs) - 0.4)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')
    if legend:
        kinds = [('unadjusted', '#111827', 'o', 5, None), ('adjusted', '#6B7280', 'o', 4.5, None), ('pooled', '#FC1C7D', 'D', 5.5, None), ('pooled, adjusted', '#FC1C6D', 'D', 5.5, 'none')]
        p = [l.lower().startswith('pool') for l in labs]
        a = [('adj' in l.lower()) or ('*' in l) for l in labs]
        def _has(k):
            if k == 'unadjusted': return any(not pi and not ai for pi, ai in zip(p, a))
            if k.startswith('adjusted'): return any(ai and not pi for pi, ai in zip(p, a))
            if k == 'pooled': return any(pi and not ai for pi, ai in zip(p, a))
            if k == 'pooled, adjusted':  return any(pi and ai for pi, ai in zip(p, a))
            return False
        present = [t for t in kinds if _has(t[0])]
        handles = [_L2D([], [], color=c, marker=m, lw=1.5, label=lab, markerfacecolor=(f or c)) for lab, c, m, s, f in present] # , markeredgewidth=1.5
        if handles:
            ax.legend(handles=handles, fontsize=7, loc=legend_loc, frameon=True, framealpha=.85, edgecolor='0.7', borderpad=.35, markerscale=0.7, labelspacing=.35) # handlelength=1.4, labelspacing=.25, borderpad=.35
    for sp in ax.spines.values():
        sp.set_color('black')
    if out and created:
        fig.savefig(out, bbox_inches='tight', dpi=dpi)
    if created:
        plt.show() if show else plt.close(fig)
    return ax