from .clip import clip
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class LayerSelector(nn.Module):
    def __init__(self, num_layers, select_k, cnt, training,
                 initial_tau=1.0, final_tau=0.1, anneal_steps=1000):
        super(LayerSelector, self).__init__()

        self.num_choices = num_layers - select_k + 1
        if self.num_choices <= 0:
            raise ValueError(
                f"select_k ({select_k}) must be <= num_layers ({num_layers})"
            )

        self.num_layers = num_layers
        self.select_k = select_k

        self.logits = nn.Parameter(torch.randn(self.num_choices))

        self.cnt = cnt
        self.is_training_mode = training
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.anneal_steps = anneal_steps
        self.current_tau = initial_tau

    def _update_tau(self):
        if self.is_training_mode:
            self.cnt['selector_step'] += 1
            step = self.cnt['selector_step']
            ratio = min(1.0, step / self.anneal_steps)
            self.current_tau = self.initial_tau - (self.initial_tau - self.final_tau) * ratio
        else:
            self.current_tau = self.final_tau

    def forward(self, all_features, selection_probs=None):
        # self._update_tau()

        batch_size = all_features.shape[0]

        if selection_probs is None:
            expanded_logits = self.logits.unsqueeze(0).expand(batch_size, -1)
            selection_probs = F.gumbel_softmax(
                expanded_logits,
                tau=self.current_tau,
                hard=True,
                dim=-1
            )

        all_windows = []
        for i in range(self.num_choices):
            window = all_features[:, i: i + self.select_k, :]
            all_windows.append(window)

        # [B, num_choices, select_k, D]
        stacked_windows = torch.stack(all_windows, dim=1)

        # [B, num_choices] -> [B, num_choices, 1, 1]
        selection_probs_expanded = selection_probs.view(
            batch_size, self.num_choices, 1, 1
        )

        weighted_windows = stacked_windows * selection_probs_expanded
        selected_features = torch.sum(weighted_windows, dim=1)  # [B, select_k, D]

        return selected_features, selection_probs


class GatingNetwork(nn.Module):
    """
    Per-layer gating network that predicts the semantic removal strength g^(k).
    
    Input: statistics of d^(k) and ||delta^(k)||
    Output: g^(k) in [0, 1]
    
    g=0 -> keep original LTD (no purification)
    g=1 -> fully use PS-LTD (maximum purification)
    """
    def __init__(self, hidden_dim=16):
        super(GatingNetwork, self).__init__()
        # Input: [mean(d), ||d||, ||delta||] -> 3 scalar features
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, d_mean, d_norm):
        """
        Args:
            d_mean:     scalar mean of d^(k), shape [B]
            d_norm:     ||d^(k)||_2,          shape [B]
            delta_norm: ||delta^(k)||_2,      shape [B] 
                        where delta^(k) = f^(k) - f_ps^(k)
        Returns:
            g: gating value in [0, 1], shape [B, 1]
        """
        # Stack into [B, 3]
        gate_input = torch.stack([d_mean, d_norm], dim=-1)
        g = self.mlp(gate_input)  # [B, 1]
        return g


class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, select_num=5, training=True, cnt={}):
        super(CLIPModel, self).__init__()

        print(name)
        self.model, self.preprocess = clip.load(name, device="cpu")

        self.cnt = cnt
        self.model.requires_grad_(False)

        # Hook mid-level layers 11-19 (9 layers)
        self.hooks = []
        for i in range(11, 20):
            self.hooks.append(
                Hook(f'block_{i}', self.model.visual.transformer.resblocks[i])
            )

        proj_dim = 1024  # CLIP ViT-L/14 hidden dim
        self.sequence_length = len(self.hooks)  # 9

        # Layer selector
        self.selector = LayerSelector(
            num_layers=self.sequence_length,
            select_k=select_num,
            cnt=cnt,
            training=training
        )

        # Positional embeddings for dual branches
        self.origin_pos_embedding = nn.Embedding(select_num, proj_dim)
        self.delta_pos_embedding = nn.Embedding(select_num - 1, proj_dim)

        # CLS tokens for dual branches
        self.origin_cls = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.delta_cls = nn.Parameter(torch.randn(1, 1, proj_dim))

        self.patch_size = 16

        # Per-layer gating networks for adaptive semantic removal
        # We need gates for (select_num - 1) layer transitions
        self.gating_networks = nn.ModuleList([
            GatingNetwork(hidden_dim=16)
            for _ in range(select_num - 1)
        ])

        # Shared transformer encoder for both branches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=8,
            dim_feedforward=proj_dim * 4,
            dropout=0.3,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, num_classes)
        )

    def extract_cls_features(self, selection_probs=None):
        """Extract CLS tokens from hooked layers and apply layer selection."""
        tensors = []
        for hook in self.hooks:
            # hook.output shape: [seq_len, B, D] where seq_len=257 (1 CLS + 256 patches)
            # Index [0, :, :] gets the CLS token across all batch items -> [B, D]
            cls_token = hook.output[0, :, :]
            tensors.append(cls_token)

        # [B, num_hooked_layers, D]
        g = torch.stack(tensors, dim=1)

        selected_features, selection_probs = self.selector(
            g, selection_probs=selection_probs
        )
        return selected_features, selection_probs

    def patch_shuffle(self, x):
        """
        Shuffle 16x16 patches of the input image.
        Aligned with CLIP ViT-L/14's native patch size.
        """
        b, c, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input size ({h}, {w}) must be divisible by "
                f"patch_size ({self.patch_size})."
            )

        gh = h // self.patch_size
        gw = w // self.patch_size
        n = gh * gw

        # Reshape into patches: [B, N, C, ps, ps]
        patches = x.view(b, c, gh, self.patch_size, gw, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            b, n, c, self.patch_size, self.patch_size
        )

        # Generate random permutation per batch item
        rand = torch.rand(b, n, device=x.device)
        shuffle_idx = torch.argsort(rand, dim=1)
        gather_idx = shuffle_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, c, self.patch_size, self.patch_size
        )
        shuffled = patches.gather(1, gather_idx)

        # Reshape back to image
        shuffled = shuffled.view(b, gh, gw, c, self.patch_size, self.patch_size)
        shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return shuffled

    def compute_purified_ltd(self, origin_features, ps_features):
        """
        Args:
            origin_features: selected layer CLS tokens from original image, [B, n, D]
            ps_features:     selected layer CLS tokens from PS image,       [B, n, D]
            
        Returns:
            d_pure: purified artifact transition features, [B, n-1, D]
            d_orig: original LTD (for consistency loss),   [B, n-1, D]
            d_ps:   PS image LTD (for consistency loss),   [B, n-1, D]
        """
        # Step 1: Compute LTD for both views
        # d^(k) = f^(k+1) - f^(k)
        d_orig = origin_features[:, 1:, :] - origin_features[:, :-1, :]  # [B, n-1, D]
        d_ps = ps_features[:, 1:, :] - ps_features[:, :-1, :]            # [B, n-1, D]

        # Step 2: Compute semantic transition estimate
        # Delta_d^(k) = d^(k) - d_ps^(k)
        delta_d = d_orig - d_ps  # [B, n-1, D]

        # Step 3: Compute per-layer semantic dependency signal
        # delta^(k) = f^(k) - f_ps^(k)  (used as gating input)

        # Step 4: Adaptive gated removal
        # d_pure^(k) = d^(k) - g^(k) * Delta_d^(k)
        d_pure_list = []
        for k in range(d_orig.size(1)):  # iterate over n-1 transitions
            d_k = d_orig[:, k, :]        # [B, D]
            delta_d_k = delta_d[:, k, :] # [B, D]

            # Compute gating input statistics
            d_mean = d_k.mean(dim=-1)          # [B]
            d_norm = d_k.norm(dim=-1)          # [B]

            # Predict gating value g^(k) in [0, 1]
            g_k = self.gating_networks[k](d_mean, d_norm)  # [B, 1]

            # Purify: d_pure = d - g * Delta_d = (1-g)*d + g*d_ps
            d_pure_k = d_k - g_k * delta_d_k  # [B, D]
            d_pure_list.append(d_pure_k)

        d_pure = torch.stack(d_pure_list, dim=1)  # [B, n-1, D]

        return d_pure, d_orig, d_ps

    def dual_branch_encode(self, origin_features, purified_ltd):
        """
        Dual-branch encoding with shared transformer.
        
        Branch A: raw layer features -> holistic consistency
        Branch B: purified LTD       -> artifact transitions
        """
        batch_size = origin_features.size(0)

        # --- Branch A: Raw Consistency ---
        pos_emb_a = self.origin_pos_embedding(
            torch.arange(origin_features.size(1), device=origin_features.device)
        )
        pos_emb_a = pos_emb_a.unsqueeze(0).expand(batch_size, -1, -1)
        branch_a = origin_features + pos_emb_a
        cls_a = self.origin_cls.expand(batch_size, -1, -1)
        branch_a = torch.cat([cls_a, branch_a], dim=1)

        # --- Branch B: Purified Artifact Transition ---
        pos_emb_b = self.delta_pos_embedding(
            torch.arange(purified_ltd.size(1), device=purified_ltd.device)
        )
        pos_emb_b = pos_emb_b.unsqueeze(0).expand(batch_size, -1, -1)
        branch_b = purified_ltd + pos_emb_b
        cls_b = self.delta_cls.expand(batch_size, -1, -1)
        branch_b = torch.cat([cls_b, branch_b], dim=1)

        # --- Weight-shared transformer encoding ---
        out_a = self.encoder(branch_a)
        out_b = self.encoder(branch_b)

        # Extract CLS token outputs
        cls_out_a = out_a[:, 0, :]  # [B, D]
        cls_out_b = out_b[:, 0, :]  # [B, D]

        return cls_out_a, cls_out_b

    def forward(self, x, return_feature=False):
        """
        Args:
            x: input images, [B, 3, 224, 224]
            return_feature: if True, return CLIP features only (for evaluation)
            
        Returns:
            if return_feature: CLIP features
            else: dict with 'logits', 'd_pure', 'd_ps' (for loss computation)
        """
        # --- Step 1: Extract features from original image ---
        features = self.model.encode_image(x)
        origin_selected_features, selection_probs = self.extract_cls_features()

        if return_feature:
            return features

        # --- Step 2: Extract features from Patch-Shuffled image ---
        x_ps = self.patch_shuffle(x)
        self.model.encode_image(x_ps)
        ps_selected_features, _ = self.extract_cls_features(
            selection_probs=selection_probs
        )

        # --- Step 3: Perturbation-guided signal decomposition ---
        d_pure, d_orig, d_ps = self.compute_purified_ltd(
            origin_selected_features, ps_selected_features
        )

        # --- Step 4: Dual-branch encoding & classification ---
        cls_out_a, cls_out_b = self.dual_branch_encode(
            origin_selected_features, d_pure
        )

        logits = self.classification_head(
            torch.cat([cls_out_a, cls_out_b], dim=1)
        )

        # Return logits and intermediate features for loss computation
        return {
            'logits': logits,
            'd_pure': d_pure,   # for L_consist
            'd_ps': d_ps,       # for L_consist
        }


def compute_consistency_loss(d_pure, d_ps):
    """
    
    L_consist = mean_k (1 - cos(d_pure^(k), d_ps^(k)))
    
    Args:
        d_pure: purified artifact transition, [B, n-1, D]
        d_ps:   PS image LTD,                 [B, n-1, D]
    Returns:
        scalar loss
    """
    # Cosine similarity along feature dim for each layer transition
    cos_sim = F.cosine_similarity(d_pure, d_ps, dim=-1)  # [B, n-1]
    loss = (1.0 - cos_sim).mean()
    return loss
