import warnings
import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from fold_preprocessing import fit_fold_pipeline
from train_util import Calibrator, choose_threshold, compute_metrics, bootstrap_metrics

import optuna

_SK_18_PLUS = tuple(int(x) for x in sklearn.__version__.split('.')[:2]) >= (1, 8)

def _make_lr(seed, **hp):
    pen = hp.get('penalty', 'l2')
    kw = dict(C=hp.get('C', 1.0), solver='liblinear', max_iter=1000, class_weight='balanced', random_state=seed)
    kw['l1_ratio' if _SK_18_PLUS else 'penalty'] = (1.0 if pen == 'l1' else 0.0) if _SK_18_PLUS else pen
    return LogisticRegression(**kw)

def _make_rf(seed, **hp):
    return RandomForestClassifier(n_estimators=hp.get('n_estimators', 200), max_depth=hp.get('max_depth', 3), min_samples_leaf=hp.get('min_samples_leaf', 5),
                                  max_features=hp.get('max_features', 'sqrt'), class_weight='balanced', random_state=seed, n_jobs=-1)

def _make_mlp(seed, **hp):
    """validation_fraction matches PLASMA's inner_val_frac; sklearn carves it out internally, stratified."""
    return MLPClassifier(hidden_layer_sizes=hp.get('hidden_layer_sizes', (64, 32, 16)), alpha=hp.get('alpha', 1e-4), learning_rate_init=hp.get('learning_rate_init', 1e-3),
                         activation=hp.get('activation', 'relu'), early_stopping=True, validation_fraction=0.15, n_iter_no_change=10, batch_size=12, max_iter=500, random_state=seed)

def _make_svm(seed, **hp):
    return SVC(kernel=hp.get('kernel', 'rbf'), C=hp.get('C', 1.0), gamma=hp.get('gamma', 'scale'), shrinking=hp.get('shrinking', True), class_weight='balanced', probability=False, 
               random_state=seed)

def _make_knn(seed, **hp):
    return KNeighborsClassifier(n_neighbors=hp.get('n_neighbors', 10), weights=hp.get('weights', 'uniform'), metric=hp.get('metric', 'minkowski'), p=hp.get('p', 2), leaf_size=hp.get('leaf_size', 30))

MODEL_FACTORIES = {'LR': _make_lr, 'RF': _make_rf, 'MLP': _make_mlp, 'SVM': _make_svm, 'KNN': _make_knn}

# Only genuinely stochastic models honour n_seeds > 1: refitting a deterministic model gives identical scores
STOCHASTIC = {'LR': False, 'RF': True, 'MLP': True, 'SVM': False, 'KNN': False}

GRIDS = {
    'LR':  {'C': [1e-2, 1e-1, 1.0, 10.0], 'penalty': ['l1', 'l2']}, # 8
    'RF':  {'max_depth': [3, 5, None], 'min_samples_leaf': [1, 3, 5]}, # 9
    'MLP': {'lr0': [1e-4, 5e-4, 1e-3], 'hidden_layer_sizes': ['(64,)', '(64,32)', '(128,64)']}, # 9
    'SVM': {'C': [1, 10, 100], 'gamma': ['scale', 1e-2, 1e-1]}, # 9
    'KNN': {'n_neighbors': [5, 10, 20, 30], 'weights': ['uniform', 'distance']}, # 8
}

def grid_size(name):
    n = 1
    for v in GRIDS[name].values():
        n *= len(v)
    return n

def _hp_from_params(params):
    hp = dict(params)
    if 'lr0' in hp:
        hp['learning_rate_init'] = hp.pop('lr0')
    if isinstance(hp.get('hidden_layer_sizes'), str):
        hp['hidden_layer_sizes'] = tuple(int(x) for x in hp['hidden_layer_sizes'].strip('()').rstrip(',').split(','))
    return hp

def _score(clf, X):
    if hasattr(clf, 'decision_function'):
        return np.asarray(clf.decision_function(X)).ravel()
    return np.asarray(clf.predict_proba(X)[:, 1]).ravel()

def _fit_predict(name, hp, X_tr, y_tr, X_list, seed, n_seeds):
    n = n_seeds if STOCHASTIC[name] else 1
    outs = [[] for _ in X_list]
    for s in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf = MODEL_FACTORIES[name](seed=seed + s, **hp)
            clf.fit(X_tr, y_tr)
        for i, X in enumerate(X_list):
            outs[i].append(_score(clf, X))
    return [np.mean(o, axis=0) for o in outs]

def tune_baseline(name, X_train, y_train, k_inner=3, n_seeds=3, seed=42, verbose=False):
    """ Exhaustive grid search scored by k_inner-fold CV over the training patients. Returns (best hp, best AUC, study) """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = np.asarray(y_train).astype(int)
    folds = list(StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=seed).split(np.arange(len(y)), y))

    def objective(trial):
        hp = _hp_from_params({k: trial.suggest_categorical(k, v) for k, v in GRIDS[name].items()})
        aucs = []
        for fi, (tr, va) in enumerate(folds):
            (s_va,) = _fit_predict(name, hp, X_train[tr], y[tr], [X_train[va]], seed + 1000 * (fi + 1), n_seeds)
            aucs.append(roc_auc_score(y[va], s_va) if len(np.unique(y[va])) > 1 else np.nan)
        a = np.asarray(aucs, dtype=float)
        trial.set_user_attr('fold_aucs', a.tolist())
        trial.set_user_attr('fold_std_auc', float(np.nanstd(a, ddof=1)) if len(a) > 1 else 0.0)
        return float(np.nanmean(a))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler(GRIDS[name], seed=seed))
    study.optimize(objective, n_trials=grid_size(name), show_progress_bar=False)
    best = study.best_value
    tied = [t for t in study.trials if t.value is not None and t.value >= best - 1e-9]
    chosen = min(tied, key=lambda t: t.user_attrs.get('fold_std_auc', np.inf)) if tied else study.best_trial
    if verbose:
        print(f"  {name:4} grid {grid_size(name):3} pts, inner-CV AUC {chosen.value:.3f} {f' ({len(tied)} tied)' if len(tied) > 1 else ''}  {chosen.params}")
    return _hp_from_params(chosen.params), float(chosen.value), study

def baseline_crossfit(name, hp, X_train, y_train, eval_sets, k_fit=5, n_seeds=5, seed=42,calibrator_kind='platt', threshold_target='gmean'):
    """ OOF predictions on the training set fit the calibrator and pick the threshold; same as PLASMA's train_crossfit, so the two are comparable """
    y = np.asarray(y_train).astype(int)
    skf = StratifiedKFold(n_splits=k_fit, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    eval_scores = {k: [] for k in eval_sets}
    for j, (fit_idx, oof_idx) in enumerate(skf.split(np.arange(len(y)), y)):
        frames = [X_train[oof_idx]] + [xv.to_numpy() if hasattr(xv, 'to_numpy') else xv for xv, _ in eval_sets.values()]
        scores = _fit_predict(name, hp, X_train[fit_idx], y[fit_idx], frames, seed + 100 * j, n_seeds)
        oof[oof_idx] = scores[0]
        for k, s in zip(eval_sets, scores[1:]):
            eval_scores[k].append(s)
    assert not np.isnan(oof).any(), 'some training patients have no out-of-fold prediction'
    cal = Calibrator(kind=calibrator_kind).fit(oof, y)
    oof_cal = cal.transform(oof)
    thr, _ = choose_threshold(y_true=y, y_prob=oof_cal, target=threshold_target)
    metrics = {'oof': compute_metrics(y, oof_cal, threshold=thr, get_cal=False)}
    ens = {'oof': {'logits': oof, 'probs_cal': oof_cal, 'targets': y, 'ids': np.asarray(getattr(X_train, 'index', np.arange(len(y))))}}
    for k, (xv, yv) in eval_sets.items():
        s = np.mean(eval_scores[k], axis=0)
        pc = cal.transform(s)
        tg = np.asarray(yv).ravel().astype(int)
        metrics[k] = compute_metrics(tg, pc, threshold=thr)
        ens[k] = {'logits': s, 'probs_cal': pc, 'targets': tg, 'ids': np.asarray(xv.index)}
    return {'metrics': metrics, 'threshold': float(thr), 'ens': ens, 'calibrator': cal}

def baseline_fixed_split_eval(lung, split, names=('LR', 'RF', 'MLP', 'SVM', 'KNN', 'XGB'), fold_kwargs=None, known_genes=None, k_inner=3, k_fit=5, n_seeds=5, tune=True, seed=42,
                              n_boot=1000, boot_seed=0, threshold_target='gmean', verbose=True, blocks=('clin', 'expr', 'mut'), suffix='', feature_cols=None):
    """ Every baseline on the same split, features and protocol as PLASMA. Preprocessing is fit once on the training patients and shared by all baselines, exactly as fixed_split_eval shares
    across model variants. Censored patients cannot be used by a binary classifier and are dropped here """
    fold_kwargs = fold_kwargs or {}
    known = set(known_genes) if known_genes is not None else set(getattr(lung, 'known_genes', ()))
    tr_idx, te_idx = split
    x_tr, y_tr = lung.tcga_x.iloc[tr_idx], lung.tcga_y.iloc[tr_idx]
    x_te, y_te = lung.tcga_x.iloc[te_idx], lung.tcga_y.iloc[te_idx]
    det = y_tr.notna().to_numpy()
    bad = [b for b in blocks if b not in ('clin', 'expr', 'mut')]
    if bad:
        raise ValueError(f"blocks must be drawn from ('clin','expr','mut'); got {bad}")
    fold = fit_fold_pipeline(x_tr[det], y_tr[det], lung.cols, known, seed=seed, verbose=verbose, **fold_kwargs)
    all_cols = [c for b in ('clin', 'expr', 'mut') for c in fold['cols'][b]]
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in all_cols]
        if missing:
            raise ValueError(f'feature_cols absent after preprocessing: {missing}')
        cols = list(feature_cols)
    else:
        cols = [c for b in ('clin', 'expr', 'mut') if b in blocks for c in fold['cols'][b]]
    if not cols:
        raise ValueError('no features selected; check blocks / feature_cols')
    Xtr = fold['transform'](x_tr[det])[cols]
    Xte, Xtc = fold['transform'](x_te)[cols], fold['transform'](lung.trcr_x)[cols]
    ytr = y_tr[det].astype(int)
    if verbose:
        print(f"baselines [{'+'.join(blocks)}]: train {len(Xtr)} labelled (censored patients cannot be used by a binary classifier), test {len(Xte)}, TRACERx {len(Xtc)}, {len(cols)} features")
    Xtr_np = Xtr.to_numpy()
    out = {}
    for name in names:
        hp, inner_auc = {}, float('nan')
        if tune:
            hp, inner_auc, _ = tune_baseline(name, Xtr_np, ytr, k_inner=k_inner, n_seeds=n_seeds, seed=seed, verbose=verbose)
        res = baseline_crossfit(name, hp, Xtr_np, ytr, eval_sets={'internal': (Xte, y_te), 'external': (Xtc, lung.trcr_y)}, k_fit=k_fit, n_seeds=n_seeds, seed=seed, threshold_target=threshold_target)
        res['ens']['oof']['ids'] = np.asarray(Xtr.index)
        ci = {k: bootstrap_metrics(res['ens'][k]['targets'], res['ens'][k]['probs_cal'], threshold=res['threshold'], n_boot=n_boot, seed=boot_seed) for k in ('internal', 'external')} if n_boot else {}
        out[f'{name}{suffix}'] = {'metrics': res['metrics'], 'ci': ci, 'threshold': res['threshold'], 'ens': res['ens'], 'best_params': hp, 'inner_best_auc': inner_auc, 'blocks': tuple(blocks), 
                                  'feature_cols': tuple(cols), 'n_features': {k: (len(v) if k in blocks else 0) for k, v in fold['cols'].items()}}
        if verbose:
            m = res['metrics']
            print(f"  {name + suffix:10} internal AUC {m['internal']['auc']:.3f} | external {m['external']['auc']:.3f} | oof {m['oof']['auc']:.3f}")
    return out