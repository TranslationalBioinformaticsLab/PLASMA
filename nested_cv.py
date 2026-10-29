import copy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from fold_preprocessing import fit_fold_pipeline
from train_util import ModelParams, MultiModalDataset, train_seed, evaluate, generate_seeds, set_global_determinism, Calibrator, choose_threshold, compute_metrics, bootstrap_metrics
from model import PLASMA

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _auc_masked(targets, scores):
    """ AUROC over determinate rows only (censored patients have NaN labels and can't be scored) """
    t = np.asarray(targets, dtype=float)
    det = np.isfinite(t)
    if det.sum() == 0 or len(np.unique(t[det])) < 2:
        return float('nan')
    return roc_auc_score(t[det], np.asarray(scores)[det])

def _dataset(params, x, y, surv=None):
    return MultiModalDataset(x[params.features[0]], x[params.features[1]], x[params.features[2]], y, surv=surv, bins=params.surv['bins'])

def _loader(ds, bs, shuffle=False, seed=None, pin_memory=True):
    g = torch.Generator().manual_seed(seed) if (shuffle and seed is not None) else None
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g, drop_last=False, num_workers=0, pin_memory=pin_memory)

def train_crossfit(params, x_train, y_train, eval_sets, device, seeds, k_fit=5, split_seed=0, calibrator_kind='platt', pin_memory=True, fixed_epochs=True, surv=None, return_models=False, final_refit=False):
    """ Cross-fitted training: calibration and thresholding on out-of-fold predictions (replaces the inner-val slice, which was too small for those decisions) Per call: (1) split x_train 
    into k_fit stratified folds. (2) for each fold and seed, train on the other k_fit-1 folds and predict the held-out fold. (3) pool those predictions --> every training patient now has an 
    OOF prediction from a model that did not see them, so calibration and threshold are fit on ~len(x_train) patients instead of ~30, as before. (4) average logits across all (fold, seed) 
    models to score eval_sets. Fits per call: k_fit * len(seeds), plus len(seeds) more when final_refit (not used) """
    y_arr = pd.to_numeric(pd.Series(np.asarray(y_train).ravel()), errors='coerce').to_numpy(dtype=float)
    det = np.flatnonzero(np.isfinite(y_arr))      # censored rows train but are never scored, so they are not split
    cens = np.flatnonzero(~np.isfinite(y_arr))
    skf = StratifiedKFold(n_splits=k_fit, shuffle=True, random_state=split_seed)
    ds_eval = {k: _dataset(params, xv, yv) for k, (xv, yv) in eval_sets.items()}
    oof_logits = np.full(len(y_arr), np.nan, dtype=float)   # filled for determinate rows only
    kept_models = []
    eval_logits = {k: [] for k in ds_eval}
    per_model_best = {}
    for j, (fit_det, oof_det) in enumerate(skf.split(det, y_arr[det])):
        fit_all_idx = np.sort(np.concatenate([det[fit_det], cens])) if len(cens) else det[fit_det]
        oof_idx = det[oof_det]
        x_pool, y_pool = x_train.iloc[fit_all_idx], y_train.iloc[fit_all_idx]
        x_oof, y_oof = x_train.iloc[oof_idx], y_train.iloc[oof_idx]
        x_fit, y_fit = x_pool, y_pool
        ds_val = None
        fold_params = copy.deepcopy(params)
        if fixed_epochs is not True:
            fold_params.epochs = int(fixed_epochs)
        last_fold_params = fold_params 
        ds_fit = _dataset(fold_params, x_fit, y_fit, surv=surv)
        ds_oof = _dataset(fold_params, x_oof, y_oof)
        fold_oof, fold_eval = [], {k: [] for k in ds_eval}
        for s in seeds:
            set_global_determinism(s + 100 * j)
            fit_l = _loader(ds_fit, fold_params.bs, shuffle=True, seed=s + 100 * j, pin_memory=pin_memory)
            val_l = None if ds_val is None else _loader(ds_val, fold_params.bs, pin_memory=pin_memory)
            model = PLASMA(fold_params).to(device)
            model, best_score, best_epoch, _ = train_seed(model, fold_params, fit_l, val_l, device)
            per_model_best[(j, s)] = {'best_score': best_score, 'best_epoch': best_epoch}
            fold_oof.append(evaluate(model, _loader(ds_oof, fold_params.bs, pin_memory=pin_memory), None, device, fold_params)[2]['logits'])
            for k, ds in ds_eval.items():
                fold_eval[k].append(evaluate(model, _loader(ds, fold_params.bs, pin_memory=pin_memory), None, device, fold_params)[2]['logits'])
            if return_models:
                kept_models.append(model.eval()) # kept for SHAP, the ensemble IS the model that gets reported
            else:
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        oof_logits[oof_idx] = np.mean(fold_oof, axis=0) # seed-averaged, in logit space
        for k in ds_eval:
            eval_logits[k].append(np.mean(fold_eval[k], axis=0))
    assert not np.isnan(oof_logits[det]).any(), 'some determinate training patients have no OOF prediction'
    # Calibration and threshold on the pooled OOF predictions — determinate rows only, since censored patients have no label to calibrate or threshold against
    y_det = y_arr[det].astype(int)
    cal = Calibrator(kind=calibrator_kind).fit(oof_logits[det], y_det)
    oof_cal = cal.transform(oof_logits[det])
    thr, _ = choose_threshold(y_true=y_det, y_prob=oof_cal, target=params.threshold_target)
    metrics = {'oof': compute_metrics(y_det, oof_cal, threshold=thr, get_cal=False)}
    ens = {'oof': {'logits': oof_logits[det], 'probs_cal': oof_cal, 'targets': y_det, 'ids': np.asarray(x_train.index)[det]}}
    if final_refit: # not used
        p_ref = params if fixed_epochs is None else last_fold_params
        refit_models, refit_eval = [], {k: [] for k in ds_eval}
        ds_all, val_full = _dataset(p_ref, x_train, y_train, surv=surv), None
        for s in seeds:
            set_global_determinism(s)
            fit_l = _loader(ds_all, p_ref.bs, shuffle=True, seed=s, pin_memory=pin_memory)
            model, _, _, _ = train_seed(PLASMA(p_ref).to(device), p_ref, fit_l, val_full, device)
            for k, ds in ds_eval.items():
                refit_eval[k].append(evaluate(model, _loader(ds, p_ref.bs, pin_memory=pin_memory), None, device, p_ref)[2]['logits'])
            if return_models:
                refit_models.append(model.eval())
            else:
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        eval_logits = refit_eval
        if return_models:
            kept_models = refit_models
    for k, (xv, yv) in eval_sets.items():
        lg = np.mean(eval_logits[k], axis=0) # average over the scoring models
        pc = cal.transform(lg)
        tg = np.asarray(yv).ravel().astype(int)
        metrics[k] = compute_metrics(tg, pc, threshold=thr)
        ens[k] = {'logits': lg, 'probs_cal': pc, 'targets': tg, 'ids': np.asarray(xv.index)}
    out = {'metrics': metrics, 'threshold': float(thr), 'ens': ens, 'per_seed_best': per_model_best, 'calibrator': cal}
    if return_models:
        # k_fit * len(seeds) fitted modules, in eval mode. Explaining their averaged logit explains exactly the function whose metrics are reported above (which a refit model would not be)
        out['models'] = kept_models
    return out

def _fit_and_score(params, x_fit_pool, y_fit_pool, x_score, y_score, device, seed, pin_memory, fixed_epochs=True, surv=None):
    """ Train one model and return AUC on x_score. For ranking configurations only """
    p = copy.deepcopy(params)
    if fixed_epochs is not True: # see train_crossfit: True keeps params.epochs so the grid can tune it
        p.epochs = int(fixed_epochs)
    ds_fit, ds_val = _dataset(p, x_fit_pool, y_fit_pool, surv=surv), None
    set_global_determinism(seed)
    fit_l = _loader(ds_fit, p.bs, shuffle=True, seed=seed, pin_memory=pin_memory)
    val_l = None if ds_val is None else _loader(ds_val, p.bs, pin_memory=pin_memory)
    model = PLASMA(p).to(device)
    model, _, _, _ = train_seed(model, p, fit_l, val_l, device)
    score_l = _loader(_dataset(p, x_score, y_score), p.bs, pin_memory=pin_memory)
    auc_, _, _ = evaluate(model, score_l, None, device, p)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(auc_)

def inner_tune(x_train, y_train, cols, base_overrides, search_space, device, n_trials=30, k_inner=3, seed=0, pin_memory=True, verbose=False, fixed_epochs=True, surv=None, sampler=None):
    y_arr = pd.to_numeric(pd.Series(np.asarray(y_train).ravel()), errors='coerce').to_numpy(dtype=float)
    det = np.flatnonzero(np.isfinite(y_arr))      # censored rows train but are never scored
    cens = np.flatnonzero(~np.isfinite(y_arr))
    skf = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=seed)
    inner_folds = [((np.sort(np.concatenate([det[a], cens])) if len(cens) else det[a]), det[b]) for a, b in skf.split(det, y_arr[det])]

    def objective(trial):
        overrides = _deep_merge(base_overrides, search_space(trial))
        params = ModelParams(cols, **overrides)
        aucs = []
        for j, (tr, te) in enumerate(inner_folds):
            aucs.append(_fit_and_score(params, x_train.iloc[tr], y_train.iloc[tr], x_train.iloc[te], y_train.iloc[te], device=device, seed=seed + 1000 * (j + 1), pin_memory=pin_memory, fixed_epochs=fixed_epochs, surv=surv))
        return float(np.nanmean(aucs))

    study = optuna.create_study(direction='maximize', sampler=sampler if sampler is not None else optuna.samplers.RandomSampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = _deep_merge(base_overrides, _params_to_overrides(study.best_trial, search_space))
    if verbose:
        print(f"    inner best AUC {study.best_value:.4f}  params {study.best_trial.params}")
    return best, study

def _params_to_overrides(trial, search_space):
    """ Re-run search_space against a FrozenTrial to recover the override dict. optuna stores flat param names; the search_space function knows the structure """
    import optuna
    fixed = optuna.trial.FixedTrial(trial.params)
    return search_space(fixed)

def _deep_merge(base, override):
    """ Override wins. One level of nested-dict merging (sufficient for ModelParams) """
    out = dict(base)
    for k, v in override.items():
        out[k] = {**out[k], **v} if (k in out and isinstance(out[k], dict) and isinstance(v, dict)) else v
    return out

def make_fixed_split(y, test_frac=0.2, seed=42, include_censored=True):
    """ One stratified train/test partition, censored patients never enter the test set """
    y_arr = pd.to_numeric(pd.Series(np.asarray(y).ravel()), errors='coerce').to_numpy(dtype=float)
    det = np.flatnonzero(np.isfinite(y_arr))
    cens = np.flatnonzero(~np.isfinite(y_arr))
    tr_det, te_det = train_test_split(det, test_size=test_frac, stratify=y_arr[det], random_state=seed)
    train_idx = np.concatenate([tr_det, cens]) if (include_censored and len(cens)) else tr_det
    return np.sort(train_idx), np.sort(te_det)

def fixed_split_eval(lung, device, split, base, versions=('CEM',), n_seeds=5, k_fit=5, fixed_epochs=True, fold_kwargs=None, known_genes=None, search_space=None, n_trials=30, k_inner=3, 
                     tune_versions=None, sampler=None, n_boot=1000, boot_seed=0, seed=42, pin_memory=True, verbose=True, return_models=False, final_refit=False):
    """ Train and score on one train/test partition, with confidence intervals from bootstrapping the test patients. The feature pipeline is fit on the training patients only, calibration 
    and the operating point come from cross-fitted out-of-fold predictions within the training set, censored patients train but are never scored, and seed logits are averaged before
    scoring. So a fixed-split number here is directly comparable to one outer fold of run_nested_cv. Confidence intervals represent sampling uncertainty in the test patients """
    fold_kwargs = fold_kwargs or {}
    known = set(known_genes) if known_genes is not None else set()
    tr_idx, te_idx = split
    x_tr, y_tr = lung.tcga_x.iloc[tr_idx], lung.tcga_y.iloc[tr_idx]
    x_te, y_te = lung.tcga_x.iloc[te_idx], lung.tcga_y.iloc[te_idx]
    surv_all = getattr(lung, 'tcga_surv', None)
    surv_tr = None if surv_all is None else surv_all.iloc[tr_idx]
    if base.get('loss') == 'survival' and surv_tr is None:
        raise ValueError("loss='survival' needs lung.tcga_surv; rebuild TCGA with keep_censored=True")
    assert y_te.notna().all(), 'censored patients must not be in the test set'
    det_tr = y_tr.notna().to_numpy()
    fold = fit_fold_pipeline(x_tr[det_tr], y_tr[det_tr], lung.cols, known, seed=seed, verbose=verbose, **fold_kwargs)
    xt_tr, xt_te, xt_trcr = (fold['transform'](d) for d in (x_tr, x_te, lung.trcr_x))
    if verbose:
        print(f"fixed split: train {len(x_tr)} ({int(det_tr.sum())} labelled + {int((~det_tr).sum())} censored), test {len(x_te)}, TRACERx {len(lung.trcr_x)}")
    tune_versions = set(versions if tune_versions is None else tune_versions)
    out = {}
    for version in versions:
        ov = _deep_merge(base, {'version': version})
        best_params, inner_best, trials = None, float('nan'), []
        if search_space is not None and version in tune_versions:
            space = search_space[version] if isinstance(search_space, dict) else search_space
            smp = sampler() if callable(sampler) else sampler
            ov, study = inner_tune(xt_tr, y_tr, fold['cols'], ov, space, device, n_trials=n_trials, k_inner=k_inner, seed=seed, pin_memory=pin_memory, verbose=verbose, fixed_epochs=fixed_epochs, surv=surv_tr, sampler=smp)
            best_params, inner_best = study.best_trial.params, float(study.best_value)
            trials = [{'number': t.number, 'value': t.value, **t.params} for t in study.trials if t.value is not None]
            if verbose:
                print(f"  {version:4} tuned over {n_trials} trials: inner-CV AUC {inner_best:.3f}  {best_params}")
        params = ModelParams(fold['cols'], **ov)
        seeds = generate_seeds(n=n_seeds, seed=seed, print_seeds=False)
        res = train_crossfit(params, xt_tr, y_tr, eval_sets={'internal': (xt_te, y_te), 'external': (xt_trcr, lung.trcr_y)}, device=device, seeds=seeds, k_fit=k_fit, 
                             split_seed=seed, pin_memory=pin_memory, fixed_epochs=fixed_epochs, surv=surv_tr, return_models=return_models, final_refit=final_refit)
        ci = {k: bootstrap_metrics(res['ens'][k]['targets'], res['ens'][k]['probs_cal'], threshold=res['threshold'], n_boot=n_boot, seed=boot_seed) for k in ('internal', 'external')}
        out[version] = {'metrics': res['metrics'], 'ci': ci, 'threshold': res['threshold'], 'ens': res['ens'], 'n_features': fold['n_features'], 'per_seed_best': res['per_seed_best'],
                        'best_params': best_params, 'inner_best_auc': inner_best, 'overrides': ov, 'trials': trials}
        if return_models:
            assert 'models' in res, 'train_crossfit did not return models — is return_models reaching it?'
            out[version].update({'models': res['models'], 'params': params, 'transform': fold['transform'], 'x_train': xt_tr, 'y_train': y_tr, 'x_test': xt_te, 'y_test': y_te,
                                 'x_external': xt_trcr, 'y_external': lung.trcr_y, 'surv_train': surv_tr, 'fixed_epochs': fixed_epochs, 'k_fit': k_fit})
        if verbose:
            for split_name in ('internal', 'external'):
                m, c = res['metrics'][split_name], ci[split_name]
                print(f"  {version:4} {split_name:8} AUC {m['auc']:.3f} [{c['auc'][0]:.3f}-{c['auc'][2]:.3f}] | AUPRC {m['auprc']:.3f} [{c['auprc'][0]:.3f}-{c['auprc'][2]:.3f}] | bal_acc {m['bal_acc']:.3f} [{c['bal_acc'][0]:.3f}-{c['bal_acc'][2]:.3f}]")
            if np.isfinite(inner_best):
                print(f"  {version:4} inner-CV {inner_best:.3f} vs test {res['metrics']['internal']['auc']:.3f} (gap {inner_best - res['metrics']['internal']['auc']:+.3f}: how optimistic the tuning was)")
    return out

def fixed_split_table(res, metrics=('auc', 'auprc', 'bal_acc', 'prec', 'rec', 'spec', 'f1')):
    """ Point estimate with a 95% bootstrap interval per metric """
    rows = []
    for version, r in res.items():
        for split_name in ('internal', 'external'):
            for m in metrics:
                lo, med, hi = r['ci'][split_name][m]
                rows.append({'version': version, 'split': split_name, 'metric': m, 'estimate': r['metrics'][split_name][m], 'lo95': lo, 'median': med, 'hi95': hi, 'inner_best_auc': r.get('inner_best_auc', float('nan'))})
    return pd.DataFrame(rows)