import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shap
import matplotlib.transforms as mtransforms
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
import os
import json

class EnsembleWrapper(nn.Module):
    """ The cross-fitted ensemble as a single differentiable function of one concatenated input. SHAP wants f(X) with X a single matrix; PLASMA takes three blocks. This splits the 
    concatenated input back into clinical / expression / mutation by the column counts and averages the members' logits (exactly what train_crossfit does when it scores a patient, so the 
    explained function and the reported function are the same """
    def __init__(self, models, n_clin, n_expr, n_mut):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_clin, self.n_expr, self.n_mut = n_clin, n_expr, n_mut
        for m in self.models:
            m.eval()

    def forward(self, x):
        c = x[:, :self.n_clin]
        e = x[:, self.n_clin:self.n_clin + self.n_expr]
        m = x[:, self.n_clin + self.n_expr:]
        outs = []
        for mod in self.models:
            kw = {}
            if 'C' in mod.version: kw['x_c'] = c
            if 'E' in mod.version: kw['x_e'] = e
            if 'M' in mod.version: kw['x_m'] = m
            outs.append(mod(**kw)[0]) # logits; the clinical skip is already inside
        return torch.stack(outs, dim=0).mean(dim=0).unsqueeze(-1)

def _frame(entry, which):
    """ The scaled feature frame for one split, in the column order the model expects """
    p = entry['params']
    cols = p.features[0] + p.features[1] + p.features[2]
    x = {'train': entry['x_train'], 'test': entry['x_test'], 'external': entry['x_external']}[which]
    return x[cols]

def explain(entry, device, background='train', explain_on='test', n_background=100, n_samples=200, raw=None, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    p = entry['params']
    n_clin, n_expr, n_mut = (len(p.features[i]) for i in range(3))
    names = list(p.features[0]) + list(p.features[1]) + list(p.features[2])
    wrapper = EnsembleWrapper(entry['models'], n_clin, n_expr, n_mut).to(device).eval()
    bg_df = _frame(entry, background)
    if n_background and len(bg_df) > n_background:
        bg_df = bg_df.sample(n=n_background, random_state=seed)
    if explain_on == 'all':
        fg_df = pd.concat([_frame(entry, k) for k in ('train', 'test', 'external')])
    else:
        fg_df = _frame(entry, explain_on)
    bg = torch.tensor(bg_df.to_numpy(), dtype=torch.float32, device=device)
    fg = torch.tensor(fg_df.to_numpy(), dtype=torch.float32, device=device)
    if verbose:
        print(f'explaining {len(entry["models"])} models on {len(fg_df)} patients ({explain_on}), background {len(bg_df)} ({background})')
    expl = shap.GradientExplainer(wrapper, bg)
    vals = expl.shap_values(fg, nsamples=n_samples)
    if isinstance(vals, list):
        vals = vals[0]
    vals = np.asarray(vals).reshape(len(fg_df), len(names))
    with torch.no_grad():
        base = float(wrapper(bg).mean().item())
    disp = fg_df
    if raw is not None:
        disp = raw.reindex(index=fg_df.index, columns=fg_df.columns)
        missing = disp.isna().all()
        if missing.any():                       # columns the raw frame doesn't carry fall back to the scaled values
            disp.loc[:, missing] = fg_df.loc[:, missing]
    return {'values': vals, 'data': fg_df, 'display': disp, 'names': names, 'base': base, 'blocks': {'clin': (0, n_clin), 'expr': (n_clin, n_clin + n_expr), 
            'mut': (n_clin + n_expr, n_clin + n_expr + n_mut)}, 'explain_on': explain_on, 'background': background}

def save_explanation(ex, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(ex['values'], index=ex['data'].index, columns=ex['names']).to_csv(f'{out_dir}/shap_values.csv')
    ex['data'].to_csv(f'{out_dir}/data_scaled.csv')
    ex['display'].to_csv(f'{out_dir}/data_display.csv')
    meta = {k: ex[k] for k in ('names', 'base', 'blocks', 'explain_on', 'background')}
    with open(f'{out_dir}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

def load_explanation(out_dir):
    """ Inverse of save_explanation """
    def _read(name):
        # round_trip parsing restores the floats to_csv wrote exactly, rather than to the last-digit approximation
        return pd.read_csv(os.path.join(out_dir, name), index_col=0, float_precision='round_trip')
    vals, data, disp = _read('shap_values.csv'), _read('data_scaled.csv'), _read('data_display.csv')
    with open(os.path.join(out_dir, 'meta.json')) as f:
        meta = json.load(f)
    names = list(meta['names'])
    if list(vals.columns) != names or list(data.columns) != names:
        raise ValueError(f'{out_dir}: column order in the CSVs does not match meta.json names')
    if not vals.index.equals(data.index):
        raise ValueError(f'{out_dir}: shap_values.csv and data_scaled.csv cover different patients or orders')
    return {'values': vals.to_numpy(dtype=float), 'data': data, 'display': disp.reindex(index=data.index, columns=data.columns), 'names': names, 'base': float(meta['base']),
            'blocks': {k: tuple(v) for k, v in meta['blocks'].items()}, 'explain_on': meta['explain_on'], 'background': meta['background']}

def top_features(ex, n=20, block=None):
    """ Features ranked by mean |SHAP|, with the direction of their association. 'corr_with_shap' is the correlation between a feature's value and its attribution: positive means higher 
    values push the prediction toward recurrence (equivalent to how colors are interpreted in the beeswarm summary plot) """
    v, d = ex['values'], ex['data']
    lo, hi = ex['blocks'][block] if block else (0, v.shape[1])
    rows = []
    for j in range(lo, hi):
        x = d.iloc[:, j].to_numpy(dtype=float)
        s = v[:, j]
        corr = np.corrcoef(x, s)[0, 1] if np.std(x) > 0 and np.std(s) > 0 else np.nan
        rows.append({'feature': ex['names'][j], 'mean_abs_shap': float(np.abs(s).mean()), 'mean_shap': float(s.mean()), 'corr_with_shap': float(corr)})
    return pd.DataFrame(rows).sort_values('mean_abs_shap', ascending=False).head(n).reset_index(drop=True)

def modality_contribution(ex):
    """ Total attribution per modality block — how much of the prediction each data type accounts for (note though that some modalities share attribution) """
    v = ex['values']
    rows = []
    for name, (lo, hi) in ex['blocks'].items():
        blk = np.abs(v[:, lo:hi])
        rows.append({'block': name, 'n_features': hi - lo, 'total_abs_shap': float(blk.sum(axis=1).mean()), 'mean_abs_shap_per_feature': float(blk.mean())})
    df = pd.DataFrame(rows)
    df['share'] = (df['total_abs_shap'] / df['total_abs_shap'].sum()).round(3)
    return df

def _explanation(ex, use_display=True):
    return shap.Explanation(values=ex['values'], base_values=np.full(len(ex['values']), ex['base']), data=(ex['display'] if use_display else ex['data']).to_numpy(), feature_names=ex['names'])

def clean_feature(f):
    if 'ORF' in f:
        f = f.replace('ORF', 'orf')
    if f.endswith('_mut'):
        return rf"$\it{{{f[:-4]}}}$ mutations"
    if f.endswith('_expr'):
        return rf"$\it{{{f[:-5]}}}$ expression"
    if f.startswith('stage'):
        return f"Stage {f.replace('stage_', '').upper()}" if '_' in f else 'Stage'
    special = {'path_T': 'Pathologic T', 'path_N': 'Pathologic N', 'age': 'Age', 'sex': 'Sex', 'current_smoker': 'Current smoker', 'ref_smoker': 'Former smoker', 'never_smoker': 'Never smoker'}
    return special.get(f, f.replace('_', ' ').capitalize())

def _add_shap_arrows(ax, labels=('early', 'late'), y_frac=0.04, frac=0.60, base_frac=0.02, text_pad_frac=0.005, color='#1A1919', lw=2):
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    xmin, xmax = ax.get_xlim()
    x0 = 0.0
    half = min(max(1e-9, x0 - xmin), max(1e-9, xmax - x0))
    right_head, left_head = x0 + frac * half, x0 - frac * half
    right_base, left_base = x0 + base_frac * half, x0 - base_frac * half
    pad, y = text_pad_frac * (xmax - xmin), y_frac
    ax.annotate('', xy=(right_head, y), xytext=(right_base, y), xycoords=trans, textcoords=trans, arrowprops=dict(arrowstyle='-|>', color=color, lw=lw))
    ax.text(min(right_head + pad, xmax - pad), y, labels[0], transform=trans, va='center', ha='left', fontsize=8, fontweight='bold', color=color)
    ax.annotate('', xy=(left_head, y), xytext=(left_base, y), xycoords=trans, textcoords=trans, arrowprops=dict(arrowstyle='-|>', color=color, lw=lw))
    ax.text(max(left_head - pad, xmin + pad), y, labels[1], transform=trans, va='center', ha='right', fontsize=8, fontweight='bold', color=color)

def plot_beeswarm(ex, n=20, w=15, h=10, dpi=600, out=None, show=False, title='SHAP feature importance', arrow_labels=('early', 'late'), arrow_y=0.04, frac=0.60, base_frac=0.02, bottom_pad=0.2, features=None, ax=None, **kwargs):
    phi, X, names = ex['values'], ex['data'].to_numpy(), list(ex['names'])
    if features is not None:
        keep = [f for f in features if f in names]
        missing = [f for f in features if f not in names]
        if missing:
            print(f"  warning: {len(missing)} feature(s) not in the model, skipped: {missing[:5]} {'...' if len(missing) > 5 else ''}")
        idx = np.array([names.index(f) for f in keep])
        phi, X, names = phi[:, idx], X[:, idx], keep
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure
    # Make SHAP draw into the requested axes.
    before = set(map(id, fig.axes))
    plt.sca(ax)
    shap.summary_plot(phi, X, feature_names=[clean_feature(f) for f in names], show=False, max_display=n, plot_size=None, **kwargs)
    added = [a for a in fig.axes if id(a) not in before]
    ax.set_title(title, color='black', fontsize=9, fontweight='bold') # pad=8, 
    ax.set_xlabel(ax.get_xlabel(), color='black', fontsize=9) # labelpad=5, 
    ax.tick_params(axis='x', colors='black', labelsize=8)
    ax.tick_params(axis='y', reset=True, color='black', labelsize=9, right=False) # pad=5, 
    if added: # absent when shap draws no colourbar
        cb_ax = added[-1]
        cb_ax.tick_params(labelsize=9)
        cb_ax.set_ylabel('Feature value', fontsize=9, labelpad=0)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - bottom_pad, ymax)
    ax.grid(True, axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.9)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color('black')
    _add_shap_arrows(ax, labels=arrow_labels, y_frac=arrow_y, frac=frac, base_frac=base_frac, lw=1.5)
    if out and created:
        fig.savefig(out, bbox_inches='tight', dpi=dpi)
    if created:
        plt.show() if show else plt.close(fig)
    return fig

def italicize_gene_yticklabels(ax, fontsize=9, pad=6, suffixes=(' expression', ' mutations')):
    labs = [t.get_text() for t in ax.get_yticklabels()]
    parts = []
    for s in labs:
        s = s.replace(r'$\it{', '').replace(r'$\mathit{', '').replace('}$', '')
        for suf in suffixes:
            if s.endswith(suf):
                parts.append((s[:-len(suf)], suf))
                break
        else:
            parts.append(s)
    ax.set_yticklabels([''] * len(parts))
    for ytick, part in zip(ax.get_yticks(), parts):
        boxes = [TextArea(part, textprops=dict(fontsize=fontsize))] if isinstance(part, str) else [TextArea(part[0], textprops=dict(fontsize=fontsize, style='italic')), TextArea(part[1], textprops=dict(fontsize=fontsize))]
        ax.add_artist(AnnotationBbox(HPacker(children=boxes, align='baseline', pad=0, sep=0), (0, ytick), xybox=(-pad, 0), xycoords=('axes fraction', 'data'), boxcoords='offset points', box_alignment=(1, 0.5), frameon=False, annotation_clip=False))