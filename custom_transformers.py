import pandas as pd
import numpy as np
import re
from sklearn.feature_selection import SelectFromModel, f_classif, mutual_info_classif
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from scipy.stats import mannwhitneyu, chi2
from sklearn.metrics import roc_auc_score
from base_utils import get_new_cols

def get_current_cols(X, refcols):
    present = [k for k in ['clin', 'expr', 'mut'] if k in refcols]
    ref_sets = {k: set(refcols[k]) for k in present}
    curcols = {k: [] for k in present}
    unknown = []
    for c in X.columns:
        assigned = False
        for k in present:
            if c in ref_sets[k]:
                curcols[k].append(c)
                assigned = True
                break
        if not assigned:
            unknown.append(c)
    if unknown:
        raise ValueError(f"{len(unknown)} column(s) in X not found in refcols: {unknown[:5]}{'…' if len(unknown)>5 else ''}")
    return {k: sorted(v) for k, v in curcols.items()}

class RemoveNearlyConstant(BaseEstimator, TransformerMixin): 
    def __init__(self, refcols, t=1e-8, verbose=False):
        self.refcols, self.t, self.verbose = refcols, t, verbose
    
    def fit(self, X, y=None):
        cols = get_current_cols(X, self.refcols)
        A = X.to_numpy(dtype=np.float32, copy=False)
        var = A.var(axis=0) # ddof=0, matches VarianceThreshold
        mask = var > self.t
        self.keep_ = X.columns[mask].tolist()
        self.all_cols_ = self.keep_
        self.removed_ = X.columns[~mask].tolist()
        if self.verbose:
            parts = [f"{len(cols[m])-len(get_new_cols(self.keep_, cols[m]))} {m}" for m in cols]
            print(f"RemoveNearlyConstant - removed: {', '.join(parts)}")
        return self
    
    def transform(self, X):
        check_is_fitted(self, attributes=['keep_'])
        return X[self.keep_].copy()
    
class RemoveRareMutations(BaseEstimator, TransformerMixin):
    """ Drop mutation features below a minimum positive count in the training data """
    def __init__(self, refcols, min_count=3, verbose=False):
        self.refcols, self.min_count, self.verbose = refcols, min_count, verbose
    
    def fit(self, X, y=None):
        cols = get_current_cols(X, self.refcols)
        pos = (X[cols['mut']] > 0).sum(axis=0)
        keep = pos.index[pos >= self.min_count].tolist()
        self.positives_, self.n_removed_ = pos, int((pos < self.min_count).sum())
        cols['mut'] = sorted(keep)
        self.all_cols_ = [c for m in cols for c in cols[m]]
        if self.verbose:
            print(f"RemoveRareMutations - {self.n_removed_} mut removed ({len(keep)} mut remaining, min_count={self.min_count})")
        return self
    
    def transform(self, X):
        check_is_fitted(self, attributes=['all_cols_'])
        return X[self.all_cols_].copy()

class RemoveLowExpression(BaseEstimator, TransformerMixin):
    """ Drop low-expression and low-variance genes; min_expr = Minimum log2 expr considered 'on'; min_frac = Fraction of samples that must exceed min_expr; min_var = Minimum variance 
    required across samples """
    def __init__(self, refcols, min_expr=1.0, min_frac=0.2, min_var=0.01, verbose=False):
        self.refcols, self.min_expr, self.min_frac, self.min_var, self.verbose = refcols, min_expr, min_frac, min_var, verbose
    
    def fit(self, X, y=None):
        cols = get_current_cols(X, self.refcols)
        expr_X = X[cols['expr']]
        n = expr_X.shape[0]
        mask_expr = (expr_X > self.min_expr).sum(axis=0) >= (self.min_frac * n)
        mask_var = expr_X.var(axis=0) >= self.min_var
        cols['expr'] = expr_X.columns[mask_expr & mask_var].tolist()
        self.all_cols_ = []
        for k in cols.keys():
            self.all_cols_.extend(cols[k])
        if self.verbose:
            print(f"RemoveLowExpression - {len(expr_X.columns)-len(cols['expr'])} expr removed ({len(cols['expr'])} expr remaining)")
        return self # must return self
    
    def transform(self, X):
        return X[self.all_cols_].copy()

class MiniCorrelationAnalysis(BaseEstimator, TransformerMixin):
    """ Identify and drop highly correlated features within each modality, in two stages: (1) Where a correlated pair contains a known gene and the other does not, keep the known gene and 
    drop its partner. (2) For remaining pairs, drop whichever feature is worse by the primary metric, breaking ties on the secondary metric """
    def __init__(self, refcols, known_genes=(), threshold=0.95, primary_method='variance', secondary_method='corr count', prioritize_known_genes=True, corr_metric_expr='pearson', corr_metric_mut='phi', min_mut_positives=5, verbose=False):
        self.refcols = refcols
        self.known_genes = tuple(known_genes)
        self.threshold = threshold
        self.primary_method, self.secondary_method = primary_method, secondary_method
        self.prioritize_known_genes = prioritize_known_genes
        self.corr_metric_expr, self.corr_metric_mut, self.min_mut_positives = corr_metric_expr, corr_metric_mut, min_mut_positives
        self.verbose = verbose
    
    def _count_outliers(self, arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return int(np.sum((arr < (q1 - 1.5*iqr)) | (arr > (q3 + 1.5*iqr))))
    
    def _build_metric(self, name, X, feats_all):
        if name == 'variance': return X[feats_all].var(axis=0, ddof=0)
        if name == 'outliers': return pd.Series({f: self._count_outliers(X[f].values) for f in feats_all})
        if name == 'corr count':
            if self.corr_pairs_.empty: return pd.Series(0.0, index=feats_all)
            stacked = np.concatenate([self.corr_pairs_['var1'].values, self.corr_pairs_['var2'].values])
            keys, counts = np.unique(stacked, return_counts=True)
            return pd.Series(dict(zip(keys, counts))).reindex(feats_all).fillna(0.0)
        raise ValueError(f'Unknown metric: {name}')
    
    def _upper_triangle_pairs(self, corr_mx, features):
        iu = np.triu_indices(len(features), k=1)
        corr_vals = corr_mx[iu]
        mask = corr_vals >= self.threshold
        return np.array(features)[iu[0][mask]], np.array(features)[iu[1][mask]], corr_vals[mask]
    
    def _corr_pairs_expr(self, X, expr_cols):
        """ Pearson via a normalised float32 matmul """
        if len(expr_cols) < 2: return pd.DataFrame(columns=['var1','var2','correlation','modality'])
        if self.corr_metric_expr == 'pearson':
            A = X[expr_cols].to_numpy(dtype=np.float32, copy=True)
            A -= A.mean(axis=0)
            norms = np.linalg.norm(A, axis=0, keepdims=True)
            if not np.all(norms > 0):
                bad = [expr_cols[i] for i in np.where(norms.ravel() == 0)[0]]
                raise ValueError(f'Zero-variance expr columns reached MiniCorrelationAnalysis: {bad[:5]}')
            A /= norms
            corr_mx = np.abs(A.T @ A)
        else:
            corr_mx = X[expr_cols].corr(method=self.corr_metric_expr).abs().values
        v1, v2, cv = self._upper_triangle_pairs(corr_mx, expr_cols)
        return pd.DataFrame({'var1': v1, 'var2': v2, 'correlation': cv, 'modality': 'expr'})
    
    def _corr_pairs_mut_phi(self, X, mut_cols): # For binary mutation matrix, Pearson == Phi coefficient
        B = X[mut_cols].astype(int)
        pos = B.sum(axis=0)
        eligible = pos.index[pos >= self.min_mut_positives].tolist()
        if len(eligible) < 2:
            if self.verbose: print(f"[MiniCorr] phi: skipped {len(mut_cols)-len(eligible)} mut cols (<{self.min_mut_positives} positives)")
            return pd.DataFrame(columns=['var1','var2','correlation','modality'])
        corr_mx = B[eligible].corr(method='pearson').abs().values
        v1, v2, cv = self._upper_triangle_pairs(corr_mx, eligible)
        return pd.DataFrame({'var1': v1, 'var2': v2, 'correlation': cv, 'modality': 'mut'})
   
    def _corr_pairs_mut_jaccard(self, X, mut_cols):
        B = X[mut_cols].astype(bool).to_numpy(copy=False)
        pos = B.sum(axis=0)
        elig_idx = np.where(pos >= self.min_mut_positives)[0]
        if elig_idx.size < 2: return pd.DataFrame(columns=['var1','var2','correlation','modality'])
        A = B[:, elig_idx].astype(np.uint8)
        names = [mut_cols[i] for i in elig_idx]
        inter = A.T @ A # intersections on 0/1
        pc = A.sum(axis=0)[:, None]
        union = pc + pc.T - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            J = np.where(union > 0, inter / union, 0.0)
        iu = np.triu_indices_from(J, k=1)
        keep = J[iu] >= self.threshold
        return pd.DataFrame({'var1': np.array(names)[iu[0][keep]], 'var2': np.array(names)[iu[1][keep]], 'correlation': J[iu][keep], 'modality': 'mut'})
    
    def fit(self, X, y=None):
        self.cols_ = get_current_cols(X, self.refcols)
        feats_all = self.cols_['expr'] + self.cols_['mut']
        escaped = map(re.escape, sorted(self.known_genes))
        self.known_gene_pattern_ = fr"^(?:{'|'.join(escaped)})_" if self.known_genes else None
        pairs_expr = self._corr_pairs_expr(X, self.cols_['expr'])
        pairs_mut = self._corr_pairs_mut_jaccard(X, self.cols_['mut']) if self.corr_metric_mut == 'jaccard' else self._corr_pairs_mut_phi(X, self.cols_['mut'])
        frames = [f for f in (pairs_expr, pairs_mut) if len(f)]
        self.corr_pairs_ = (pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=['var1', 'var2', 'correlation', 'modality']))
        # Metrics computed once, as Series indexed by feature. 'corr count' is deliberately built from the pre-step-1 pair table.
        self.mP_ = self._build_metric(self.primary_method, X, feats_all)
        self.mS_ = self._build_metric(self.secondary_method, X, feats_all)
        self.removed_ = {}
        self._step1()
        self._step2()
        self._step3()
        self.all_cols_ = self.cols_['clin'] + self.cols_['expr'] + self.cols_['mut']
        return self
    
    def _step1(self):
        if self.corr_pairs_.empty or not self.prioritize_known_genes or self.known_gene_pattern_ is None:
            self.removed_['step1'] = {'expr': [], 'mut': []}
            return
        mk1 = self.corr_pairs_['var1'].str.match(self.known_gene_pattern_, case=False, na=False)
        mk2 = self.corr_pairs_['var2'].str.match(self.known_gene_pattern_, case=False, na=False)
        to_remove = list(set(self.corr_pairs_.loc[~mk1 & mk2, 'var1']) | set(self.corr_pairs_.loc[mk1 & ~mk2, 'var2']))
        if to_remove:
            mask = ~self.corr_pairs_['var1'].isin(to_remove) & ~self.corr_pairs_['var2'].isin(to_remove)
            self.corr_pairs_ = self.corr_pairs_[mask]
        self.removed_['step1'] = {'expr': get_new_cols(to_remove, self.cols_['expr']), 'mut': get_new_cols(to_remove, self.cols_['mut'])}
    
    def _step2(self):
        to_remove = set()
        for f1, f2 in zip(self.corr_pairs_['var1'], self.corr_pairs_['var2']):
            if f1 in to_remove or f2 in to_remove: continue
            to_remove.add(self._select_drop_feature(f1, f2))
        self.removed_['step2'] = {'expr': get_new_cols(to_remove, self.cols_['expr']), 'mut': get_new_cols(to_remove, self.cols_['mut'])}
    
    def _step3(self):
        rem_expr = set(self.removed_['step1']['expr']) | set(self.removed_['step2']['expr'])
        rem_mut  = set(self.removed_['step1']['mut'])  | set(self.removed_['step2']['mut'])
        self.cols_['expr'] = [c for c in self.cols_['expr'] if c not in rem_expr]
        self.cols_['mut']  = [c for c in self.cols_['mut']  if c not in rem_mut]
        if self.verbose:
            print(f"MiniCorrelationAnalysis - removed {len(rem_expr)} expr ({len(self.cols_['expr'])} remain), {len(rem_mut)} mut ({len(self.cols_['mut'])} remain)")
    
    def _select_drop_feature(self, f1, f2):
        def compare(m1, m2, method):
            # Variance: larger is better, drop the smaller; corr count / outliers: larger is worse, drop the larger
            if method == 'variance': return f1 if m1 < m2 else f2
            return f1 if m1 > m2 else f2
        mP1, mP2 = self.mP_[f1], self.mP_[f2]
        if mP1 != mP2: return compare(mP1, mP2, self.primary_method)
        return compare(self.mS_[f1], self.mS_[f2], self.secondary_method)
    
    def transform(self, X):
        check_is_fitted(self, attributes=['all_cols_'])
        return X[self.all_cols_].copy()

class FilterVariance(BaseEstimator, TransformerMixin):
    """ Drop the lowest percentile of features in coltype by variance """
    def __init__(self, refcols, coltype, percentile=25, verbose=False):
        self.refcols, self.coltype, self.percentile, self.verbose = refcols, coltype, percentile, verbose
    
    def fit(self, X, y=None):
        cols = get_current_cols(X, self.refcols)
        Xs = X[cols[self.coltype]]
        variances = Xs.var(axis=0, ddof=0)
        threshold = float(np.percentile(variances.values, self.percentile))
        keep = variances.index[variances > threshold].tolist()
        self.variances_, self.threshold_ = variances, threshold
        cols[self.coltype] = sorted(keep)
        self.all_cols_ = [c for m in cols for c in cols[m]]
        if self.verbose:
            print(f"FilterVariance - {len(Xs.columns)-len(keep)} {self.coltype} removed, {len(keep)} remaining (threshold={threshold:.6g})")
        return self
    
    def transform(self, X):
        check_is_fitted(self, attributes=['all_cols_'])
        return X[self.all_cols_].copy()

class WeightedUnivar(BaseEstimator, TransformerMixin):
    """ Univariate ANOVA (f_classif) on a target column group with rank-based prior for known genes and optional stability selection via bootstraps """
    def __init__(self, refcols, coltype, known_genes=(), score='f', ensemble_weights=None, k=100, known_rank_bonus=0, stability_rounds=0, stability_frac=0.7, random_state=None, verbose=False):
        self.refcols = refcols
        self.coltype = coltype
        self.known_genes = tuple(known_genes)
        self.score = score
        assert self.score in ['f', 'chi2', 'mi', 'auc', 'snr', 'mw', 'ensemble'], f'Invalid scoring mode entered: {self.score}'
        self.ensemble_weights = ensemble_weights
        self.k = int(k)
        self.known_rank_bonus = int(known_rank_bonus)
        self.stability_rounds = int(stability_rounds)
        self.stability_frac = float(stability_frac)
        self.random_state = random_state
        self.verbose = verbose
    
    def _score_f(self, Xg, y):
        F, _ = f_classif(Xg.values, y.values if isinstance(y, pd.Series) else y)
        return pd.Series(np.clip(F, 0, None), index=Xg.columns) # Guard: negative F occasionally from numerical issues -> clip at 0
    
    def _score_chi2(self, Xg, y):
        chi, _ = chi2(Xg, y.values if isinstance(y, pd.Series) else y)
        return pd.Series(np.nan_to_num(chi, nan=0.0), index=Xg.columns)

    def _score_mi(self, Xg, y):
        discf = True if self.coltype == 'mut' else False
        mi = mutual_info_classif(Xg.values, y.values if isinstance(y, pd.Series) else y, discrete_features=discf, random_state=self.random_state)
        return pd.Series(mi, index=Xg.columns)
    
    def _score_snr(self, Xg, y):
        yv = y.values if isinstance(y, pd.Series) else y
        x0 = Xg[yv==0]; x1 = Xg[yv==1]
        snr = (x1.mean() - x0.mean()) / (x1.std(ddof=1) + x0.std(ddof=1) + 1e-8)
        return snr.abs()
    
    def _score_mw(self, Xg, y):
        yv = y.values if isinstance(y, pd.Series) else y
        def mw(col):
            a, b = col[yv==0].values, col[yv==1].values
            if a.size<2 or b.size<2 or np.allclose(col, col.iloc[0]): return 0.0
            u, _ = mannwhitneyu(a, b, alternative='two-sided')
            m, n = len(a), len(b)
            return max(u, m*n - u) / (m*n) # normalize U to [0,1] by dividing by max possible
        return Xg.apply(mw, axis=0)
    
    def _score_auc(self, Xg, y):
        yv = y.values if isinstance(y, pd.Series) else y
        def au1(col):
            x = col.values
            if np.allclose(x, x[0]): return 0.5
            try:
                a = roc_auc_score(yv, x)
                return max(a, 1-a)  # direction-invariant
            except ValueError: return 0.5
        return Xg.apply(au1, axis=0)
    
    def _score_once(self, Xg, y):
        if self.score == 'f': return self._score_f(Xg, y), False
        if self.score == 'chi2': return self._score_chi2(Xg, y), False
        if self.score == 'mi': return self._score_mi(Xg, y), False
        if self.score == 'auc': return self._score_auc(Xg, y), False
        if self.score == 'snr': return self._score_snr(Xg, y), False
        if self.score == 'mw': return self._score_mw(Xg, y), False
        if self.score == 'ensemble':
            parts = {'f': self._score_f(Xg, y), 'auc': self._score_auc(Xg, y), 'snr': self._score_snr(Xg, y), 'mw': self._score_mw(Xg, y),}
            weights = self.ensemble_weights or {k:1.0 for k in parts} # rank-aggregate (lower rank = better)
            wrank = sum((weights.get(k, 0.0) * s.rank(ascending=False, method='average')) for k, s in parts.items() if weights.get(k, 0.0) > 0)
            wrank /= sum(w for w in weights.values() if w > 0)
            return wrank, True
    
    def _tie_break(self, scores, Xg, rng):
        # Deterministic tie-breaking: 1) If all equal, add an epsilon scaled by prevalence (column sums). 2) If prevalence is all zeros/equal, add tiny seeded jitter.
        if scores.nunique() > 1:
            return scores
        if self.coltype == 'expr':
            prev = Xg.var(axis=0).astype(float)
        else:
            prev = Xg.sum(axis=0).astype(float)
        if prev.max() > 0:
            eps = (prev / (prev.max() + 1e-12)) * 1e-6
            return scores + eps
        jitter = pd.Series(rng.normal(loc=0.0, scale=1e-6, size=len(scores)), index=scores.index) # final fallback: tiny seeded jitter
        return scores + jitter
    
    def _bootstrap_idx(self, y, m, rng): # stratified, with-replacement bootstrap indices (keeps class mix and avoids single-class draws)
        yv = y.values if isinstance(y, pd.Series) else y
        n = len(yv)
        i1 = np.where(yv == 1)[0]
        i0 = np.where(yv == 0)[0]
        if i1.size == 0 or i0.size == 0: # fallback: if a class is missing, just sample with replacement from all
            return rng.choice(n, size=m, replace=True)
        p1 = i1.size / n # allocate per-class counts proportional to prevalence
        m1 = max(1, int(round(m * p1)))
        m0 = max(1, m - m1)
        b1 = rng.choice(i1, size=m1, replace=True)
        b0 = rng.choice(i0, size=m0, replace=True)
        idx = np.concatenate([b0, b1])
        rng.shuffle(idx)
        return idx

    def _stability(self, Xg, y): # returns a score (higher = better)
        if self.stability_rounds <= 0:
            vals, is_rank = self._score_once(Xg, y)
            if is_rank:
                max_rank = float(len(vals))
                return (max_rank + 1.0) - vals # convert rank→score once
            return vals  # already a score
        rng = check_random_state(self.random_state)
        n = len(y)
        m = max(2, int(np.floor(self.stability_frac * n)))
        agg_rank = pd.Series(0.0, index=Xg.columns) # aggregate ranks across bootstraps (lower = better)
        for _ in range(self.stability_rounds):
            idx = self._bootstrap_idx(y, m, rng)  # stratified, with-replacement
            Xb = Xg.iloc[idx]
            yb = y.iloc[idx] if isinstance(y, pd.Series) else y[idx]
            vals, is_rank = self._score_once(Xb, yb)
            rb = vals if is_rank else vals.rank(ascending=False, method='average')
            agg_rank += rb
        agg_rank /= float(self.stability_rounds)
        max_rank = float(len(agg_rank))
        return (max_rank + 1.0) - agg_rank  # single final rank→score

    def _apply_known_bonus(self, scores):
        if self.known_rank_bonus <= 0:
            return scores
        s = scores
        ranks = s.rank(ascending=False, method='average')
        kg = ranks.index.intersection(pd.Index(self.known_genes))
        if len(kg):
            r = ranks.copy()
            r.loc[kg] = np.maximum(1.0, r.loc[kg] - float(self.known_rank_bonus))
            max_rank = float(len(r))
            s = (max_rank + 1.0) - r  # back to score space
        return s

    def fit(self, X, y):
        cols = get_current_cols(X, self.refcols)
        Xg = X[cols[self.coltype]]
        scores = self._stability(Xg, y)
        scores = self._apply_known_bonus(scores)
        rng = check_random_state(self.random_state) # new
        scores = self._tie_break(scores, Xg, rng) # new
        self.scores = scores.sort_values(ascending=False)
        ranked = self.scores.index.tolist()
        selected = ranked[:self.k]
        cols[self.coltype] = sorted(selected)
        self.all_cols_ = []
        for modality in cols.keys():
            self.all_cols_.extend(cols[modality])
        if self.verbose: 
            print(f"WeightedUnivar - {len(Xg.columns)-len(cols[self.coltype])} {self.coltype} removed, {len(cols[self.coltype])} {self.coltype} remaining")
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=['all_cols_'])
        return X[self.all_cols_].copy()

class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, refcols, coltype, clf, top_k=None, boost_known=None, boost_factor=2.0, verbose=False, **kwargs):
        self.refcols, self.coltype, self.clf = refcols, coltype, clf
        self.top_k, self.boost_factor, self.verbose = top_k, boost_factor, verbose
        self.boost_known = set(boost_known or [])
        self.kwargs = kwargs
    
    def fit(self, X, y):
        cols = get_current_cols(X, self.refcols)
        Xblock = X[cols[self.coltype]]
        est = clone(self.clf)
        est.fit(Xblock, y)
        if hasattr(est, "feature_importances_"):
            imp = pd.Series(est.feature_importances_, index=Xblock.columns, dtype=float)
        elif hasattr(est, "coef_"):
            imp = pd.Series(np.abs(est.coef_).ravel(), index=Xblock.columns, dtype=float)
        else: raise ValueError("Estimator lacks feature importances/coefficients.")
        if self.boost_known:
            mask = imp.index.isin(self.boost_known)
            if not mask.any() and self.verbose:
                print(f"[warn] SelectFeatures - no boost_known entries matched {self.coltype} column names; check suffixes")
            imp.loc[mask] *= float(self.boost_factor)
        if self.top_k is not None:
            keep = imp.sort_values(ascending=False).head(self.top_k).index.tolist()
        else:
            sel = SelectFromModel(estimator=est, **self.kwargs).fit(Xblock, y)
            keep = [Xblock.columns[i] for i in sel.get_support(indices=True)]
        self.estimator_, self.importances_ = est, imp.sort_values(ascending=False)
        self.selected_ = sorted(keep)
        cols[self.coltype] = self.selected_
        self.all_cols_ = [c for m in cols for c in cols[m]]
        if self.verbose:
            print(f"SelectFeatures - {self.coltype} selected: {len(self.selected_)}")
        return self
    
    def transform(self, X):
        check_is_fitted(self, attributes=['all_cols_'])
        return X[self.all_cols_].copy()


class SelectScalers(BaseEstimator, TransformerMixin):
    """ Chooses a scaler per feature from training-set distribution statistics, then fits and applies it; force_standard: blocks ('clin' and/or 'expr') where every column gets a plain z-score 
    from training-fold mean and SD, bypassing the per-column scaler choice. fit_fold_pipeline passes ('clin',) when scale_clin=True. Mutation is never scaled """
    def __init__(self, columns, to_scale=('clin', 'expr'), force_standard=(), skew_symmetric=0.5, skew_heavy=1.0, outlier_heavy=0.01, many_zeros=0.9, verbose=False):
        self.columns = columns
        self.to_scale = to_scale
        self.force_standard = tuple(force_standard)
        self.skew_symm, self.skew_heavy = skew_symmetric, skew_heavy
        self.outlier_heavy, self.many_zeros = outlier_heavy, many_zeros
        self.verbose = verbose

    def _return_empty(self, reason):
        stats = {k: np.nan for k in ['outlier_f', 'zero_f', 'skew', 'vmin', 'vmax', 'uniq', 'bound01', 'binary']}
        return (stats, 'none', reason)

    def _series_stats(self, x): # Fisher-Pearson sample skewness
        v = x.to_numpy()
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1 if q3 > q1 else 0.0
        outlier_f = float(((v < (q1 - 1.5*iqr)) | (v > (q3 + 1.5*iqr))).mean()) if iqr > 0 else 0.0
        s = {'outlier_f': outlier_f, 'zero_f': float((v == 0).mean()), 'skew': float(pd.Series(v).skew()), 'vmin': float(v.min()), 'vmax': float(v.max()), 'uniq': int(np.unique(v).size)}
        s.update({'bound01': (s['vmin'] >= 0.0) and (s['vmax'] <= 1.0), 'binary': (s['vmin'] == 0) and (s['vmax'] == 1) and (s['uniq'] == 2)})
        if s['vmin'] == s['vmax']:
            raise ValueError(f"Constant feature reached SelectScalers: '{x.name}'. RemoveNearlyConstant should run first.")
        return s

    def _decide_scaler(self, stats, kind):
        many_outliers = stats['outlier_f'] > self.outlier_heavy
        skew_type = 'mild' if abs(stats['skew']) <= self.skew_symm else 'heavy' if abs(stats['skew']) > self.skew_heavy else 'moderate'
        if kind == 'expr':
            if stats['binary']: raise ValueError('Binary feature in expr data')
            return ('robust', 'many outliers') if many_outliers else ('standard', 'few outliers')
        if kind == 'clin':
            if stats['bound01']: return ('none', 'already normalized to [0,1] range')
            if skew_type == 'mild':
                return ('robust', 'many outliers, mild skew') if many_outliers else ('standard', 'few outliers, mild skew')
            if skew_type == 'heavy':
                return ('yj + robust', 'many outliers, heavy skew') if many_outliers else ('yj + standard', 'few outliers, heavy skew')
            return ('yj + robust', 'many outliers, moderate skew') if many_outliers else ('yj + standard', 'few outliers, moderate skew')
        raise ValueError(f'Unknown kind: {kind}')

    def _recommend(self, X):
        records = []
        for cat in ['clin', 'expr', 'mut']:
            for col in self.columns[cat]:
                if cat == 'mut': stats, scaler, why = self._return_empty('mutation passed through unscaled')
                elif cat in self.force_standard: stats, scaler, why = self._series_stats(X[col]), 'standard', 'forced z-score'
                elif cat not in self.to_scale: stats, scaler, why = self._return_empty('not selected for scaling')
                elif cat == 'clin' and col != 'age': stats, scaler, why = self._return_empty('categorical/ordinal')
                else:
                    stats = self._series_stats(X[col])
                    scaler, why = self._decide_scaler(stats, kind=cat)
                records.append({'category': cat, 'feature': col, 'scaler': scaler, 'reasoning': why, **stats})
        recs = pd.DataFrame.from_records(records)
        recs['scaler'] = recs['scaler'].fillna('none')
        if self.verbose and len(recs[(recs['category'] == 'expr') & (recs['reasoning'] == 'binary')]) > 0:
            print('Binary expression features detected')
        return recs

    def _build_transformer(self, recs):
        def tag_to_transformer(tag):
            tag = tag.strip().lower()
            if tag == 'none': return 'passthrough'
            if tag == 'standard': return StandardScaler()
            if tag == 'robust': return RobustScaler()
            if tag == 'yj + standard': return Pipeline([('yj', PowerTransformer(method='yeo-johnson', standardize=False)), ('std', StandardScaler())])
            if tag == 'yj + robust': return Pipeline([('yj', PowerTransformer(method='yeo-johnson', standardize=False)), ('rob', RobustScaler())])
            raise ValueError(f'Unknown scaler tag: {tag}')
        def slug(name):
            return name.lower().replace('+', 'plus').replace(' ', '_').replace('/', '_').replace('-', '_')
        ct_recs = recs[recs['category'].isin(['clin', 'expr'])]
        groups = ct_recs[['feature', 'scaler']].groupby('scaler')['feature'].apply(list).to_dict()
        transformers = [(slug(tag), tag_to_transformer(tag), cols) for tag, cols in groups.items()]
        return ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False).set_output(transform='pandas')

    def fit(self, X, y=None):
        self.recs_ = self._recommend(X)
        self.ct_ = self._build_transformer(self.recs_)
        self.ct_.fit(X)
        self.out_cols_ = list(self.columns['clin']) + list(self.columns['expr']) + list(self.columns['mut'])
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=['ct_', 'out_cols_'])
        ce = self.ct_.transform(X)
        # Fallback for older scikit-learn: strip any '<step>__' prefixes if present
        if len(ce.columns) and isinstance(ce.columns[0], str) and '__' in ce.columns[0]:
            ce.columns = [c.split('__', 1)[-1] for c in ce.columns]
        mut = X[self.columns['mut']]
        if not ce.index.equals(mut.index):
            raise RuntimeError('Index mismatch between scaled and mutation blocks')
        out = pd.concat([ce, mut], axis=1)
        if pd.Series(out.columns).duplicated().any():
            raise RuntimeError('Duplicate column names after scaling; check recs_ for features in multiple scaler groups')
        missing = set(self.out_cols_) - set(out.columns)
        if missing:
            raise RuntimeError(f'Missing columns after scaling: {sorted(missing)[:5]}')
        return out.reindex(columns=self.out_cols_)