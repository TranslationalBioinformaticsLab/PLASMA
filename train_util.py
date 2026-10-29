import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, f1_score, recall_score, balanced_accuracy_score, precision_score, accuracy_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.optimize import minimize_scalar
from torch.utils.data import Dataset, DataLoader

class ModelParams:
    def __init__(self, columns, **kwargs):
        if kwargs.get('_from_json', False): return
        defaults = dict(
            version = 'CEM',
            clin = dict(input_drop=0, n=[16], out=16, dropout=[.1], act='gelu', norm=False),
            clin_skip = True,
            expr = dict(input_drop=0, n=[32, 16], out=16, dropout=[.3, .1], act='gelu', norm=True),
            mut  = dict(input_drop=0, n=[32, 16], out=16, dropout=[.5, .3], act='leaky_relu', norm=False),
            fus  = dict(embed=16, n=[16, 8], dropout=[0.1, 0], act='leaky_relu', mode='attention'),
            att  = dict(use=True, n_heads=4, dropout=.3, gate_input=True),
            aux  = dict(weight=0.3, decay=20), # initial auxiliary loss weight (decays during training)
            mod_dropout = 0.0,
            clin_opt = dict(opt='adamw', lr=3e-4, weight_decay=1e-4),
            expr_opt = dict(opt='adamw', lr=3e-4, weight_decay=1e-4),
            mut_opt  = dict(opt='adamw', lr=3e-4, weight_decay=1e-4),
            fus_opt  = dict(opt='adamw', lr=3e-4, weight_decay=1e-4),
            loss = 'survival', 
            surv = dict(bins=(365, 730)), 
            max_grad_norm = 1.0,          
            use_pos_weight = True, # only applies if loss = 'bce'
            threshold_target = 'gmean',
            epochs = 30,
            bs = 12,
            wd_exclude_1d = True, # biases, LayerNorm gains/shifts and the survival intercepts are not decayed    
        )
        # Unknown keys used to be dropped silently, which is how a new option can appear to run while doing nothing.
        unknown = sorted(set(kwargs) - set(defaults) - {'_from_json'})
        if unknown:
            raise ValueError(f"Unknown ModelParams key(s): {unknown}. Add them to the defaults in ModelParams if they're real options.")
        for param in defaults.keys():
            dflt = defaults[param]
            new = {**dflt, **kwargs.get(param, {})} if isinstance(dflt, dict) else kwargs.get(param, dflt)
            setattr(self, param, new)
        self.features = [list(columns['clin']), list(columns['expr']), list(columns['mut'])]

def surv_targets(fu_time, event_obs, bins):
    """ Convert (time, event) into the two integers the discrete-time loss needs, given the interval upper edges. n_surv: number of intervals the patient completed recurrence-free 
    (contributes a 'survived' target to each); ev_bin: interval containing the observed recurrence, or -1 if none was observed within the horizon """
    edges = np.asarray(bins, dtype=float)
    K = len(edges)
    t = pd.to_numeric(pd.Series(fu_time), errors='coerce').to_numpy(dtype=float)
    e = pd.to_numeric(pd.Series(event_obs), errors='coerce').fillna(0).to_numpy(dtype=float)
    k = np.searchsorted(edges, t, side='right')          # completed intervals, or the interval containing t
    is_event = (e == 1) & (k < K) & np.isfinite(t)
    ev_bin = np.where(is_event, k, -1).astype(np.int64)
    n_surv = np.where(is_event, k, np.minimum(k, K)).astype(np.int64)
    n_surv = np.where(np.isfinite(t), n_surv, 0)         # no time at all: contributes nothing
    return n_surv.astype(np.int64), ev_bin

class MultiModalDataset(Dataset):
    """ surv: optional DataFrame with 'fu_time' and 'event_obs', aligned to clin_df, used only by the survival loss. Absent (or for rows with no time) the survival targets are all-masked 
    and the sample contributes nothing to it """
    def __init__(self, clin_df, expr_df, mut_df, labels, surv=None, bins=(182, 365, 547, 730)):
        super().__init__()
        self.ids = clin_df.index.to_numpy()
        assert len(set(self.ids)) == len(self.ids), f"Duplicate IDs found in dataset index; ensemble aggregation would silently drop samples."
        self.clin_data = torch.tensor(clin_df.to_numpy(), dtype=torch.float32)
        self.expr_data = torch.tensor(expr_df.to_numpy(), dtype=torch.float32)
        self.mut_data = torch.tensor(mut_df.to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(pd.to_numeric(pd.Series(labels), errors='coerce').to_numpy(dtype=np.float32))
        if surv is None:
            n_surv = np.zeros(len(self.ids), dtype=np.int64); ev_bin = np.full(len(self.ids), -1, dtype=np.int64)
        else:
            s = surv.reindex(clin_df.index)
            n_surv, ev_bin = surv_targets(s['fu_time'], s['event_obs'], bins)
        self.n_surv = torch.tensor(n_surv, dtype=torch.long)
        self.ev_bin = torch.tensor(ev_bin, dtype=torch.long)
    def __getitem__(self, idx):
        return self.clin_data[idx], self.expr_data[idx], self.mut_data[idx], self.labels[idx], self.n_surv[idx], self.ev_bin[idx], self.ids[idx]
    def __len__(self):
        return len(self.labels)

def discrete_time_loss(risk, baselines, n_surv, ev_bin):
    """ Discrete-time survival loss: binary cross-entropy over a patient x interval grid, masked to observed intervals. hazard logit for interval k = baselines[k] + risk(x). Because the risk
    score is shared across intervals, cumulative risk over the horizon is monotone in it, so the single model output still ranks patients by recurrence risk """
    K = baselines.shape[0]
    k = torch.arange(K, device=risk.device).unsqueeze(0)
    logits = baselines.unsqueeze(0) + risk.unsqueeze(1)
    surv = k < n_surv.unsqueeze(1)
    event = k == ev_bin.unsqueeze(1)
    mask = (surv | event).float()
    bce = F.binary_cross_entropy_with_logits(logits, event.float(), reduction='none')
    return (bce * mask).sum() / mask.sum().clamp(min=1)

def build_datasets(params, xtr, xva, xte, ytr, yva, yte, trcr_x, trcr_y):
    def ds(x, y):
        return MultiModalDataset(x[params.features[0]], x[params.features[1]], x[params.features[2]], y, bins=params.surv['bins'])
    return {'train': ds(xtr, ytr), 'val': ds(xva, yva), 'test_tcga': ds(xte, yte), 'test_trcr': ds(trcr_x, trcr_y)}

def build_loaders(datasets, params, seed=42, pin_memory=True):
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(datasets['train'], batch_size=params.bs, shuffle=True, generator=g, drop_last=False, num_workers=0, pin_memory=pin_memory)
    val = DataLoader(datasets['val'], batch_size=params.bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_memory)
    test_tcga = DataLoader(datasets['test_tcga'], batch_size=params.bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_memory)
    test_trcr = DataLoader(datasets['test_trcr'], batch_size=params.bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_memory)
    return train, val, test_tcga, test_trcr

def generate_seeds(n=10, seed=42, print_seeds=True):
    random.seed(seed)
    s_ = random.sample(range(1, 10_000), n)
    if print_seeds: print(f"seeds = {s_}")
    return s_

def get_optimizers(model, params):
    optimizers = {}
    exclude_1d = getattr(params, 'wd_exclude_1d', True)

    def _make(opt_cfg, named):
        optm = {'adam': optim.Adam, 'adamw': optim.AdamW}[opt_cfg['opt']]
        wd = float(opt_cfg['weight_decay'])
        if not exclude_1d or wd == 0.0:
            groups = [{'params': [q for _, q in named], 'weight_decay': wd}]
        else:
            groups = [{'params': [q for _, q in named if q.ndim > 1], 'weight_decay': wd}, {'params': [q for _, q in named if q.ndim <= 1], 'weight_decay': 0.0}]
        groups = [g for g in groups if g['params']]
        return optm(groups, lr=opt_cfg['lr'])

    if 'C' in params.version:
        clin_named = list(model.enc_clin.named_parameters(prefix='enc_clin'))
        if getattr(model, 'clin_skip', None) is not None:
            clin_named += list(model.clin_skip.named_parameters(prefix='clin_skip'))
        optimizers['enc_clin'] = _make(params.clin_opt, clin_named)
    if 'E' in params.version:
        optimizers['enc_expr'] = _make(params.expr_opt, list(model.enc_expr.named_parameters(prefix='enc_expr')))
    if 'M' in params.version:
        optimizers['enc_mut'] = _make(params.mut_opt, list(model.enc_mut.named_parameters(prefix='enc_mut')))

    owned = {id(q) for o in optimizers.values() for g in o.param_groups for q in g['params']}
    optimizers['head'] = _make(params.fus_opt, [(n, q) for n, q in model.named_parameters() if id(q) not in owned])
    return optimizers

def get_input(c, e, m, params):
    x_out = {}
    if 'C' in params.version: x_out['x_c'] = c
    if 'E' in params.version: x_out['x_e'] = e
    if 'M' in params.version: x_out['x_m'] = m
    return x_out

def _aux_loss_weight(epoch, init_weight=0.3, decay_epochs=25):
    """ Linear decay of auxiliary loss weight from init_weight (epoch 1) --> 0 (epoch decay_epochs+1) """
    return init_weight * max(0.0, 1.0 - (epoch - 1) / decay_epochs)

def train_one_epoch(model, dataloader, optimizers, loss_fn, device, params, epoch=1):
    """ Training loop with auxiliary per-modality losses. The auxiliary loss weight decays linearly to 0 over N epochs so that early in training each modality encoder gets direct supervision, 
    but later the model learns to fuse modalities holistically """
    model.train()
    tracking = {k: [] for k in ['ids', 'losses', 'batch_sizes', 'logits', 'probs', 'targets']}
    aux_weight = _aux_loss_weight(epoch, params.aux['weight'], params.aux['decay'])
    survival = getattr(params, 'loss', 'bce') == 'survival'
    for (clin_x, expr_x, mut_x, y, n_surv, ev_bin, ids) in dataloader:
        tracking['ids'].extend(list(ids))
        clin_x, expr_x, mut_x, y = [t.to(device).float() for t in (clin_x, expr_x, mut_x, y)]
        n_surv, ev_bin = n_surv.to(device), ev_bin.to(device)
        for o in optimizers.values():
            o.zero_grad()
        # Forward: returns (logits, probs, attn_weights, aux_logits)
        logits, probs, _, aux_logits = model(**get_input(clin_x, expr_x, mut_x, params))
        # In survival mode y is NaN for censored patients and is never used for the loss; the (time, event) pair is.
        primary_loss = discrete_time_loss(logits, model.surv_baselines, n_surv, ev_bin) if survival else loss_fn(logits, y)
        # tracking['losses'].append(primary_loss.item())
        tracking['batch_sizes'].append(int(y.shape[0]))
        # Build total loss = primary + aux (fusion mode + early epochs only)
        loss = primary_loss
        if aux_logits and aux_weight > 0:
            for key, alogit in aux_logits.items():
                aux_l = discrete_time_loss(alogit, model.surv_baselines, n_surv, ev_bin) if survival else loss_fn(alogit, y)
                loss = loss + aux_weight * aux_l
        loss.backward()
        if params.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.max_grad_norm)
        for o in optimizers.values():
            o.step()
        tracking['losses'].append(loss.item())
        tracking['logits'].append(logits.detach().cpu().numpy())
        tracking['probs'].append(probs.detach().cpu().numpy())
        tracking['targets'].append(y.detach().cpu().numpy())
    # Sample-weighted mean across batches (unbiased when last batch is smaller).
    losses = np.asarray(tracking['losses'], dtype=float)
    bsz = np.asarray(tracking['batch_sizes'], dtype=float)
    mean_loss = float(np.sum(losses * bsz) / np.sum(bsz)) if bsz.sum() > 0 else float('nan')
    for k in ['logits', 'probs', 'targets']:
        tracking[k] = np.concatenate(tracking[k], axis=0)
    tracking['ids'] = np.array(tracking['ids'])
    return mean_loss, tracking

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, params):
    """ Evaluation loop """
    model.eval()
    tracking = {k: [] for k in ['ids', 'losses', 'batch_sizes', 'logits', 'probs', 'targets', 'attn']}
    tracking['aux_logits'] = {k: [] for k in params.version}
    if loss_fn is None: loss_fn = nn.BCEWithLogitsLoss()
    survival = getattr(params, 'loss', 'bce') == 'survival'
    for (clin_x, expr_x, mut_x, y, n_surv, ev_bin, ids) in dataloader:
        tracking['ids'].extend(list(ids))
        clin_x, expr_x, mut_x, y = [t.to(device).float() for t in (clin_x, expr_x, mut_x, y)]
        n_surv, ev_bin = n_surv.to(device), ev_bin.to(device)
        logits, probs, attn, aux_logits = model(**get_input(clin_x, expr_x, mut_x, params))
        loss = discrete_time_loss(logits, model.surv_baselines, n_surv, ev_bin) if survival else loss_fn(logits, y)
        tracking['losses'].append(loss.item())
        tracking['batch_sizes'].append(int(y.shape[0]))
        tracking['logits'].append(logits.detach().cpu().numpy())
        tracking['probs'].append(probs.detach().cpu().numpy())
        tracking['targets'].append(y.detach().cpu().numpy())
        if attn is not None:
            tracking['attn'].append(attn.detach().cpu().numpy())
        for k, alogit in aux_logits.items():
            # Store raw logits (no sigmoid). Downstream code that wants probs should apply scipy.special.expit or torch.sigmoid itself.
            tracking['aux_logits'][k].append(alogit.detach().cpu().numpy())
    for k in ['logits', 'probs', 'targets']:
        tracking[k] = np.concatenate(tracking[k], axis=0)
    tracking['attn'] = None if len(tracking['attn']) == 0 else np.concatenate(tracking['attn'], axis=0)
    for k in tracking['aux_logits']:
        if tracking['aux_logits'][k]:
            tracking['aux_logits'][k] = np.concatenate(tracking['aux_logits'][k], axis=0)
    tracking['ids'] = np.array(tracking['ids'])
    det = np.isfinite(tracking['targets'])
    tracking['determinate'] = det
    if det.sum() == 0 or len(np.unique(tracking['targets'][det])) < 2:
        return float('nan'), float('nan'), tracking
    auc_ = roc_auc_score(tracking['targets'][det], tracking['probs'][det])
    prec, rec, _ = precision_recall_curve(tracking['targets'][det], tracking['probs'][det])
    auprc_ = auc(rec, prec)
    return auc_, auprc_, tracking

def train_seed(model, params, train_loader, val_loader, device):
    # val_loader is ignored if provided
    if getattr(params, 'use_pos_weight', True):
        y_train = train_loader.dataset.labels
        n_pos = float(y_train.sum().item())
        n_neg = float(y_train.numel() - n_pos)
        pos_weight = torch.tensor([n_neg / n_pos], device=device) if n_pos > 0 else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    epoch_losses = {'train': [], 'val': []}
    optimizers = get_optimizers(model, params)
    for epoch in range(1, params.epochs + 1):
        train_loss, _ = train_one_epoch(model, train_loader, optimizers, loss_fn, device, params, epoch=epoch)
        epoch_losses['train'].append(train_loss)
    return model, float('nan'), params.epochs, epoch_losses

def set_global_determinism(seed, strict=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic_debug_mode("error")

def choose_threshold_youden(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thr[best_idx]), {'fpr': float(fpr[best_idx]), 'tpr': float(tpr[best_idx])}

def choose_threshold_max_gmean(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    gmean = np.sqrt(tpr * spec)
    best_idx = int(np.argmax(gmean))
    return float(thr[best_idx]), {'tpr': float(tpr[best_idx]), 'spec': float(spec[best_idx]), 'gmean': float(gmean[best_idx])}

def choose_threshold_min_spec_gmean(y_true, y_prob, min_spec=0.50):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    ok = np.isfinite(y_prob)
    y_true = y_true[ok]; y_prob = y_prob[ok]
    thr_candidates = np.unique(y_prob)
    best, best_meta, best_thr = None, None, None
    for thr in thr_candidates:
        y_hat = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        gmean = np.sqrt(spec * rec)
        if spec >= min_spec:
            if (best is None) or (gmean > best + 1e-12):
                best = gmean; best_thr = float(thr)
                best_meta = {"tpr": float(rec), "spec": float(spec), "gmean": float(gmean), "min_spec": float(min_spec)}
    if best is None:
        print(f"No threshold achieves min_spec={min_spec}.")
    return best_thr, best_meta

def choose_threshold(y_true, y_prob, target, min_spec=0.50):
    target = (target or 'youden').lower()
    if target in {'youden', 'j'}: return choose_threshold_youden(y_true, y_prob)
    if target in {'gmean', 'max_gmean'}: return choose_threshold_max_gmean(y_true, y_prob)
    if target in {'min_spec_gmean'}: return choose_threshold_min_spec_gmean(y_true, y_prob, min_spec=min_spec)
    raise ValueError(f"Unknown threshold_target: {target}")

class Calibrator:
    def __init__(self, kind='platt'):
        self.kind = (kind or 'none').lower()
        self.model = None

    def fit(self, val_logits, val_labels):
        val_logits = np.asarray(val_logits).reshape(-1, 1).astype(float)
        val_labels = np.asarray(val_labels).astype(int).ravel()
        if self.kind == 'none':    self.model = None; return self
        if self.kind == 'platt':
            lr = LogisticRegression(C=1e12, solver='lbfgs', max_iter=1000)
            lr.fit(val_logits, val_labels); self.model = lr; return self
        if self.kind == 'isotonic':
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(val_logits.ravel(), val_labels); self.model = ir; return self
        raise ValueError(f'Unknown calibrator kind: {self.kind}')

    def transform(self, logits):
        logits = np.asarray(logits).reshape(-1, 1).astype(float)
        if self.kind == 'none': return expit(logits.ravel())
        if self.kind == 'platt': return self.model.predict_proba(logits)[:, 1]
        if self.kind == 'isotonic': return self.model.transform(logits.ravel())
        raise RuntimeError('Calibrator not fit or unknown kind.')

def _to_1d_numpy(x):
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    if isinstance(x, (pd.Series, pd.Index)): x = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0].to_numpy() if x.shape[1] == 1 else x.to_numpy().squeeze()
    else: x = np.asarray(x)
    x = np.asarray(x).squeeze()
    if x.ndim != 1: raise ValueError(f"Expected 1D array, got shape {x.shape}")
    return x

def calibration_metrics(y_true, y_prob, eps=1e-6):
    """ Brier score, calibration slope and calibration-in-the-large.
    slope: coefficient of a logistic regression of the outcome on the predicted logit. 1.0 is perfect, < 1 means the predictions are too extreme, > 1 too conservative
    citl:  the constant shift in log-odds needed to match the observed event rate, with the slope held at 1. 0 is perfect, > 0 means the model under-predicts risk overall
    Only meaningful on predictions whose calibrator was fitted on OTHER patients. Computed on the data the Platt map was fitted to, it recovers that fit and says nothing — hence
    get_cal=False for oof """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), eps, 1 - eps)
    lp = np.log(p / (1 - p))
    brier = float(np.mean((p - y) ** 2))
    if len(np.unique(y)) < 2: # a one-class bootstrap resample
        return {'brier': brier, 'cal_slope': float('nan'), 'cal_citl': float('nan')}
    slope = float(LogisticRegression(C=1e12, solver='lbfgs', max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0]) # C=1e12 --> unpenalized, portable across versions
    bce = lambda a: float(np.mean(np.logaddexp(0, lp + a) - y * (lp + a)))
    citl = float(minimize_scalar(bce, bounds=(-10, 10), method='bounded').x)
    return dict(brier=brier, cal_slope=slope, cal_citl=citl)

CAL_EPS = 1e-6

def brier_score(y_true, y_prob, eps=CAL_EPS):
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), eps, 1 - eps)
    return float(np.mean((p - y) ** 2))

def brier_skill(y_true, y_prob, eps=CAL_EPS):
    """ 1 - BS/BS_ref against the base-rate forecast. NaN on a single-class sample """
    y = np.asarray(y_true, dtype=float).ravel()
    p = float(y.mean())
    ref = p * (1.0 - p)
    return float(1.0 - brier_score(y, y_prob, eps=eps) / ref) if ref > 0 else float("nan")

def compute_metrics(y_true, y_prob, threshold=0.5, plot_metrics=True, get_cal=True):
    y_true = _to_1d_numpy(y_true).astype(int)
    y_prob = _to_1d_numpy(y_prob).astype(float)
    y_hat01 = (y_prob >= threshold).astype(int)
    auc_ = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    prec_, rec_, _ = precision_recall_curve(y_true, y_prob)
    auprc_ = float(auc(rec_, prec_)) if len(np.unique(y_true)) == 2 else float("nan")
    avg_prec = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    bal_acc = float(balanced_accuracy_score(y_true, y_hat01))
    acc = float(accuracy_score(y_true, y_hat01))
    prec = float(precision_score(y_true, y_hat01, zero_division=0))
    rec = float(recall_score(y_true, y_hat01, zero_division=0))
    f1 = float(f1_score(y_true, y_hat01, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat01, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    mdict = dict(auc=auc_, auprc=auprc_, avg_prec=avg_prec, acc=acc, bal_acc=bal_acc, prec=prec, rec=rec, spec=spec, f1=f1, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    mdict.update(calibration_metrics(y_true, y_prob) if get_cal else {'brier': brier_score(y_true, y_prob)})
    mdict['bss'] = brier_skill(y_true, y_prob)
    if plot_metrics:
        mdict.update(dict(prc_prec=prec_.tolist(), prc_rec=rec_.tolist(), fpr=fpr.tolist(), tpr=tpr.tolist()))
    return mdict

def bootstrap_metrics(y_true, y_prob, threshold, n_boot=1000, seed=0, keys=('auc', 'auprc', 'brier', 'bss', 'cal_slope', 'cal_citl', 'avg_prec', 'bal_acc', 'prec', 'rec', 'spec', 'f1')):
    y_true = _to_1d_numpy(y_true).astype(int)
    y_prob = _to_1d_numpy(y_prob).astype(float)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    base = compute_metrics(y_true, y_prob, threshold, plot_metrics=False)
    keys = [k for k in (keys or base) if k in base and np.isscalar(base[k])]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        m = compute_metrics(y_true[idx], y_prob[idx], threshold, plot_metrics=False)
        for k in keys: samples[k].append(m[k])
    return {k: (float(np.nanpercentile(samples[k], 2.5)), float(np.nanpercentile(samples[k], 50.0)), float(np.nanpercentile(samples[k], 97.5))) for k in keys}

def ensemble_split_by_id(per_seed_outputs, seeds, split):
    """ Aggregate per-seed predictions by patient ID. Convention: the ensemble aggregates in LOGIT space (mean-of-logits), which is the standard ensembling method for binary classifiers.
    'probs' is sigmoid(mean_logits), NOT mean(sigmoid(logits)). This ensures that 'probs' and 'probs_cal' (downstream) are derived from the same underlying aggregation """
    seed_maps = []
    for s in seeds:
        out = per_seed_outputs[s][split]
        id2prob = {i: p for i, p in zip(out['ids'], out['probs'])}
        id2log  = {i: z for i, z in zip(out['ids'], out['logits'])}
        id2y    = {i: y for i, y in zip(out['ids'], out['targets'])}
        seed_maps.append((id2prob, id2log, id2y))
    common_ids = set(seed_maps[0][0].keys())
    for id2prob, _, _ in seed_maps[1:]: common_ids &= set(id2prob.keys())
    common_ids = sorted(common_ids)
    logits_stack = np.stack([[id2log[i]  for i in common_ids] for (_, id2log, _)  in seed_maps], axis=0)
    y = np.array([seed_maps[0][2][i] for i in common_ids], dtype=float)
    mean_logits = logits_stack.mean(axis=0)
    return {'ids': np.array(common_ids), 'probs': expit(mean_logits), 'logits': mean_logits, 'targets': y}

