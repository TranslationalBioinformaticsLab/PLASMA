import torch
import torch.nn as nn
import torch.nn.functional as F

# def _get_act(name: str) -> nn.Module:
#     return {'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU}[name.lower()]()

def apply_modality_dropout(stack: torch.Tensor, p_drop=0.15, training=True) -> torch.Tensor:
    if not training or p_drop == 0.0:
        return stack
    B, S, E = stack.shape
    # Sample a drop mask: shape (B, S, 1), broadcast over E
    mask = (torch.rand(B, S, 1, device=stack.device) > p_drop).float()
    # Safety: ensure at least one modality is kept per sample
    keep_count = mask.sum(dim=1, keepdim=True)  # (B, 1, 1)
    # Where all modalities would be dropped, fall back to keeping all
    safe_mask = torch.where(keep_count == 0, torch.ones_like(mask), mask)
    return stack * safe_mask

class FFNEncoder(nn.Module):
    """ Simple feed-forward encoder used for each modality """
    def __init__(self, input_dim, mparams, out_norm=True):
        super().__init__()
        layers = []
        act_cls = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU}[mparams['act'].lower()]
        if mparams['input_drop'] != 0:
            layers.append(nn.Dropout(mparams['input_drop']))
        prev = input_dim
        for h, p in zip(mparams['n'], mparams['dropout']):
            layers.append(nn.Linear(prev, h))
            if mparams['norm']:
                layers.append(nn.LayerNorm(h))
            layers.append(act_cls())
            if p != 0:
                layers.append(nn.Dropout(p))
            prev = h
        if prev != mparams['out']:
            layers.append(nn.Linear(prev, mparams['out']))
        if out_norm:
            layers.append(nn.LayerNorm(mparams['out']))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) # (batch, input_dim) → (batch, out)

class ModalityEmbedding(nn.Module):
    """ Adds a learned type embedding to each modality token (w/o this, attention is permutation-equivariant) """
    def __init__(self, n_modalities: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_modalities, embed_dim)

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        # stack: (B, S, E)
        idx = torch.arange(stack.size(1), device=stack.device)
        return stack + self.emb(idx).unsqueeze(0)  # (B, S, E) + (1, S, E)

class GatedModalityFusion(nn.Module):
    """ Attention fusion with per-modality confidence gates and learned type embeddings """
    def __init__(self, fparams, aparams, n_modalities=3, mod_dropout=0.15, use_attention=True, gate_input=False):
        super().__init__()
        embed_dim = int(fparams['embed'])
        self.mod_emb = ModalityEmbedding(n_modalities, embed_dim)
        # Per-modality scalar gate: (B, S, E) → (B, S, 1)
        self.gate = nn.Linear(embed_dim, 1)
        self.use_attention = use_attention
        self.gate_input = gate_input
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=aparams['n_heads'], dropout=aparams['dropout'], batch_first=True) if use_attention else None
        self.post = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)) # Post-attention adapter — GELU instead of ReLU to avoid dead neurons
        self.mod_dropout_p = mod_dropout

    def forward(self, mod_embeddings: torch.Tensor):
        # mod_embeddings: (B, S, E), S = number of active modalities
        x = self.mod_emb(mod_embeddings)
        gates = torch.sigmoid(self.gate(x)) # per-modality scalar confidence (B, S, 1)
        self.last_gates = gates.detach() # diagnostics only
        x_gated = x * gates
        x_gated = apply_modality_dropout(x_gated, self.mod_dropout_p, self.training) # not used by any modality
        # Gate-weighted pooling instead of uniform mean (used in both branches)
        gate_w = gates.squeeze(-1) # (B, S)
        gate_w = gate_w / (gate_w.sum(dim=1, keepdim=True) + 1e-8)  # normalize
        if self.use_attention:
            attn_in = x_gated if self.gate_input else x
            attn_out, attn_weights = self.mha(attn_in, attn_in, attn_in, need_weights=True, average_attn_weights=False)
        else:
            attn_out, attn_weights = x, None
        fused = (attn_out * gate_w.unsqueeze(-1)).sum(dim=1) # (B, E)
        fused = self.post(fused)
        return fused, attn_weights

class ConcatFusion(nn.Module):
    """ No-attention baseline: concatenate modality embeddings, project back to embed_dim via MLP """
    def __init__(self, fparams, aparams, n_modalities=3, mod_dropout=0.15):
        super().__init__()
        embed_dim = int(fparams['embed'])
        in_dim = n_modalities * embed_dim
        self.mod_dropout_p = mod_dropout
        self.mlp = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.GELU(), nn.Dropout(aparams['dropout']), nn.LayerNorm(embed_dim))
 
    def forward(self, mod_embeddings):
        # mod_embeddings: (B, S, E)
        x = apply_modality_dropout(mod_embeddings, self.mod_dropout_p, self.training)
        x = x.flatten(start_dim=1) # (B, S*E)
        fused = self.mlp(x) # (B, embed_dim)
        return fused, None

class PredictionHead(nn.Module):
    def __init__(self, fparams):
        super().__init__()
        act_cls = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU}[fparams['act'].lower()]
        layers = []
        prev = fparams['embed']
        for h, p in zip(fparams['n'], fparams['dropout']):
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if p != 0:
                layers.append(nn.Dropout(p))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PLASMA(nn.Module):
    """ Multimodal recurrence prediction model """
    def __init__(self, params):
        super().__init__()
        self.version = params.version
        self.params = params
        # ---- modality encoders ----
        outnorm = True
        if 'C' in self.version:
            self.enc_clin = FFNEncoder(len(params.features[0]), params.clin, out_norm=outnorm)
        if 'E' in self.version:
            self.enc_expr = FFNEncoder(len(params.features[1]), params.expr, out_norm=outnorm)
        if 'M' in self.version:
            self.enc_mut = FFNEncoder(len(params.features[2]), params.mut, out_norm=outnorm)
        if len(self.version) > 1:
            _enc_params = {}
            if 'C' in self.version: _enc_params['C'] = params.clin
            if 'E' in self.version: _enc_params['E'] = params.expr
            if 'M' in self.version: _enc_params['M'] = params.mut
            # torch.stack in the fusion module requires identical trailing dims, so every encoder's output width must equal the fusion embedding width
            outs = {k: p['out'] for k, p in _enc_params.items()}
            assert set(outs.values()) == {params.fus['embed']}, f"encoder out dims {outs} must all equal fus['embed']={params.fus['embed']}"
            n_mod = len(self.version)
            mode = params.fus.get('mode') or ('attention' if params.att['use'] else 'concat')
            if mode in ('attention', 'gated'):
                self.fusion = GatedModalityFusion(params.fus, params.att, n_modalities=n_mod, mod_dropout=params.mod_dropout, use_attention=(mode == 'attention'), gate_input=bool(params.att.get('gate_input', False)))
            elif mode == 'concat':
                self.fusion = ConcatFusion(params.fus, params.att, n_modalities=n_mod, mod_dropout=params.mod_dropout)
            else:
                raise ValueError(f"fus['mode'] must be 'attention', 'gated' or 'concat'; got {mode!r}")
            self.fusion_mode = mode
            self.head = PredictionHead(params.fus)
            self.clin_skip = nn.Linear(len(params.features[0]), 1) if 'C' in self.version and getattr(params, 'clin_skip', False) else None
            self._add_surv_baselines(params)
            # Auxiliary per-modality classification heads; each maps encoder output directly to a binary logit
            self.aux_heads = nn.ModuleDict()
            for key, ep in _enc_params.items():
                self.aux_heads[key] = nn.Linear(ep['out'], 1)
        else:
            self.head = PredictionHead(params.fus)
            self.clin_skip = None
            self._add_surv_baselines(params)
        self.initialize_weights()

    def _add_surv_baselines(self, params):
        """ Discrete-time survival mode: one learnable baseline per interval, so the hazard logit for interval k is baseline_k + risk(x). forward() still returns the single risk score """
        if getattr(params, 'loss', 'bce') == 'survival':
            self.surv_baselines = nn.Parameter(torch.zeros(len(params.surv['bins'])))
        else:
            self.surv_baselines = None

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02) # small init for modality embeddings to not overwhelm encoder signal at start

    def forward(self, x_c=None, x_e=None, x_m=None):
        """ Returns raw logits (B,), sigmoid probs (B,), attention weights (B, n_heads, S, S), and aux logits {key: (B,) logit tensor} """
        enc_c = enc_e = enc_m = None
        if 'C' in self.version: enc_c = self.enc_clin(x_c)
        if 'E' in self.version: enc_e = self.enc_expr(x_e)
        if 'M' in self.version: enc_m = self.enc_mut(x_m)
        aux_logits = {}
        if len(self.version) == 1:
            # Encoder -> head, no fusion. Head matches the multi-modality variant so the post-encoder pipeline is identical across all comparisons
            attn_weights = None
            if self.version == 'C':   logits = self.head(enc_c)
            elif self.version == 'E': logits = self.head(enc_e)
            elif self.version == 'M': logits = self.head(enc_m)
        else:
            # Compute auxiliary logits per modality (before projection)
            if 'C' in self.version:
                aux_logits['C'] = self.aux_heads['C'](enc_c).squeeze(-1)
            if 'E' in self.version:
                aux_logits['E'] = self.aux_heads['E'](enc_e).squeeze(-1)
            if 'M' in self.version:
                aux_logits['M'] = self.aux_heads['M'](enc_m).squeeze(-1)
            # Stack encoder outputs
            to_stack = []
            if 'C' in self.version: to_stack.append(enc_c)
            if 'E' in self.version: to_stack.append(enc_e)
            if 'M' in self.version: to_stack.append(enc_m)
            stack = torch.stack(to_stack, dim=1) # (B, N_mod, embed_dim)
            fused, attn_weights = self.fusion(stack)
            logits = self.head(fused)
            if self.clin_skip is not None:
                logits = logits + self.clin_skip(x_c)
        logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs, attn_weights, aux_logits