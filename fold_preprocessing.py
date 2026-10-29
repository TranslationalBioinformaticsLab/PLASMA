import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from custom_transformers import *

""" const_t: Gets rid of constant columns
mut_min_count: Positive number of mutations for a gene in the training fold (sensitivity analysis showed feature set was unchanged across thresholds of 0, 3, 5, and 10; set to 10 for computational efficiency)
min_expr, min_frac: Expr gene is kept if its log2 val exceeds min_expr in >=N% of training samples. On log2(x+1) data, 1.0 corresponds to a raw value of 1, so this is "detectably expressed in at least 3/5 of tumors." 
                    it doesn't matter what threshold is chosen for min_frac, the genes it drops are never selected
min_var: Applied alongside the above, on the same log2 scale. Removes genes that are expressed but flat across patients
expr_var_percentile: Filters expr genes by variance within each fold, retaining the top decile (sensitivity analysis across 25, 50, 75, and 90 showed mean pairwise Jaccard similarity between folds' selected feature 
                     sets rose monotonically, while the number of genes selected in all ten folds peaked at the 90th percentile)
corr_threshold: Pairs above this correlation are considered redundant and one gets dropped; high enough to only remove near-duplicates
corr_metric_mut: Mutation columns are binary, so Jaccard is better than Pearson (it ignores the co-absent cell, which dominates when both genes are mutated in a small minority of patients)
min_mut_positives: Genes below this are excluded from the correlation analysis entirely; with 2 positives, a Jaccard of 1.0 means two patients happen to share both mutations, which is coincidence rather than redundancy
expr_k, mut_k: How many features each supervised selector keeps 
expr_score: ANOVA F-statistic for univariate ranking, also supports auc, mi, snr, mw, and an ensemble; F assumes roughly normal within-class distributions, which log2 expression approximately satisfies
expr_known_bonus, mut_boost_factor: The literature prior. Expression features get a N-place rank promotion if they're in known_genes; mutation features get their importance multiplied by n.n
expr_stability_rounds, expr_stability_frac: Not used (bootstrapped stratified selection)
to_scale: Which modalities to include for scaling
scale_clin: Z-score every clinical column instead of age only """

FOLD_DEFAULTS = dict(
    const_t=1e-6,
    mut_min_count=10,
    min_expr=1.0, min_frac=0.20, 
    min_var=0.01,
    expr_var_percentile=90,
    corr_threshold=0.95, 
    corr_metric_mut='jaccard', 
    min_mut_positives=3,
    expr_k=50, mut_k=50, 
    expr_score='f', 
    expr_known_bonus=0, mut_boost_factor=1,
    expr_stability_rounds=0, expr_stability_frac=0.7,
    to_scale=('clin', 'expr'),
    scale_clin=False,
)

def set_fold_defaults(**kwargs):
    """ Override FOLD_DEFAULTS """
    unknown = sorted(set(kwargs) - set(FOLD_DEFAULTS))
    if unknown:
        raise KeyError(f"Unknown fold option(s): {unknown}. Valid keys: {sorted(FOLD_DEFAULTS)}")
    FOLD_DEFAULTS.update(kwargs)
    return dict(FOLD_DEFAULTS)

def get_fold_defaults(**overrides):
    unknown = sorted(set(overrides) - set(FOLD_DEFAULTS))
    if unknown:
        raise KeyError(f"Unknown fold option(s): {unknown}. Valid keys: {sorted(FOLD_DEFAULTS)}")
    return {**FOLD_DEFAULTS, **overrides}

MUT_RANKER_PARAMS = dict(
    n_estimators=600, eta=0.05, max_depth=10, min_child_weight=1, subsample=0.8, colsample_bytree=0.5, reg_alpha=0.0, reg_lambda=2.0,
)
 
def fit_fold_pipeline(x_train, y_train, refcols, known_genes=(), seed=42, verbose=False, **overrides):
    """ Fit every data-dependent preprocessing step on this fold's training data; x_train, y_train = outer-fold training split (TCGA); refcols = global column dictionary 
    {'clin','expr','mut'} (LUNG.cols); known_genes = set of approved bare symbols (no suffix) """
    cfg = get_fold_defaults(**overrides) # raises on unknown keys rather than silently ignoring them
    known_bare = set(known_genes)
    known_expr = [f'{g}_expr' for g in known_bare]
    known_mut = [f'{g}_mut' for g in known_bare]
    steps = {}
    # 1. Unsupervised filters
    X = x_train
    ordered = [
        ('drop_const', RemoveNearlyConstant(refcols, t=cfg['const_t'], verbose=verbose)),
        ('rare_mut', RemoveRareMutations(refcols, min_count=cfg['mut_min_count'], verbose=verbose)),
        ('low_expr', RemoveLowExpression(refcols, min_expr=cfg['min_expr'], min_frac=cfg['min_frac'], min_var=cfg['min_var'], verbose=verbose)),
        ('expr_var', FilterVariance(refcols, coltype='expr', percentile=cfg['expr_var_percentile'], verbose=verbose)),
        ('corr', MiniCorrelationAnalysis(refcols, known_genes=sorted(known_bare), threshold=cfg['corr_threshold'], primary_method='variance', secondary_method='corr count',
                                         corr_metric_mut=cfg['corr_metric_mut'], min_mut_positives=cfg['min_mut_positives'], verbose=verbose)),
    ]
    for name, step in ordered:
        step.fit(X, y_train)
        X = step.transform(X)
        steps[name] = step
    # 2. Supervised selection
    expr_sel = WeightedUnivar(refcols, coltype='expr', score=cfg['expr_score'], known_genes=known_expr, k=cfg['expr_k'], known_rank_bonus=cfg['expr_known_bonus'],
                                  stability_rounds=cfg['expr_stability_rounds'], stability_frac=cfg['expr_stability_frac'], random_state=seed, verbose=verbose)
    expr_sel.fit(X, y_train)
    X = expr_sel.transform(X)
    steps['expr_sel'] = expr_sel
    ranker = XGBClassifier(random_state=seed, n_jobs=-1, importance_type='total_gain', **MUT_RANKER_PARAMS)
    mut_sel = SelectFeatures(refcols, coltype='mut', clf=ranker, top_k=cfg['mut_k'], boost_known=known_mut, boost_factor=cfg['mut_boost_factor'], verbose=verbose)
    mut_sel.fit(X, y_train)
    X = mut_sel.transform(X)
    steps['mut_sel'] = mut_sel
    cols = get_current_cols(X, refcols)
    ordered_cols = cols['clin'] + cols['expr'] + cols['mut']
    # 3. Scaling
    scaler = SelectScalers(cols, to_scale=list(cfg['to_scale']), force_standard=('clin',) if cfg['scale_clin'] else (), verbose=verbose)
    scaler.fit(X)
    steps['scaler'] = scaler
    def transform(df):
        return scaler.transform(df[ordered_cols].copy())
    n_features = {k: len(v) for k, v in cols.items()}
    if verbose: print(f"  features: {n_features}")
    return {'transform': transform, 'cols': cols, 'steps': steps, 'n_features': n_features}