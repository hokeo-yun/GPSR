from .clip import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
}

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
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, o_mean, o_norm, ps_mean, ps_norm):
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
        gate_input = torch.stack([o_mean, o_norm, ps_mean, ps_norm], dim=-1)
        g = self.mlp(gate_input)  # [B, 1]
        return g

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
    def __init__(
        self,
        num_layers,
        select_k,
        cnt,
        training,
        initial_tau=1.0,
        final_tau=0.1,
        anneal_steps=1000,
    ):
        super(LayerSelector, self).__init__()

        self.num_choices = num_layers - select_k + 1
        if self.num_choices <= 0:
            raise ValueError(
                f"select_k ({select_k}) must be less than or equal to num_layers ({num_layers})"
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
            self.cnt["selector_step"] += 1
            step = self.cnt["selector_step"]
            ratio = min(1.0, step / self.anneal_steps)
            self.current_tau = self.initial_tau - (self.initial_tau - self.final_tau) * ratio
        else:
            self.current_tau = self.final_tau

    def forward(self, all_features, selection_probs=None):
        batch_size = all_features.shape[0]

        if selection_probs is None:
            expanded_logits = self.logits.unsqueeze(0).expand(batch_size, -1)
            selection_probs = F.gumbel_softmax(
                expanded_logits,
                tau=self.current_tau,
                hard=True,
                dim=-1,
            )

        all_windows = []
        for i in range(self.num_choices):
            window = all_features[:, i : i + self.select_k, :]
            all_windows.append(window)

        stacked_windows = torch.stack(all_windows, dim=1)
        selection_probs_expanded = selection_probs.view(batch_size, self.num_choices, 1, 1)
        weighted_windows = stacked_windows * selection_probs_expanded
        selected_features = torch.sum(weighted_windows, dim=1)

        return selected_features, selection_probs

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, select_num=5, training=True, p=1, cnt={}):
        super(CLIPModel, self).__init__()

        print(name)
        self.p = p
        self.model, self.preprocess = clip.load(name, device="cpu")

        self.cnt = cnt
        self.model.requires_grad_(False)

        self.hooks = []
        for i in range(11, 20):
            self.hooks.append(Hook(f"block_{i}", self.model.visual.transformer.resblocks[i]))

        proj_dim = 1024
        self.sequence_length = len(self.hooks)

        self.selector = LayerSelector(
            num_layers=self.sequence_length,
            select_k=select_num,
            cnt=cnt,
            training=training,
        )

        self.origin_pos_embedding = nn.Embedding(select_num, proj_dim)
        self.delta_pos_embedding = nn.Embedding(select_num, proj_dim)

        self.origin_cls = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.delta_cls = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.patch_size = 16

        self.gating_networks = nn.ModuleList([
            GatingNetwork(hidden_dim=16)
            for _ in range(select_num)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=8,
            dim_feedforward=proj_dim * 4,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classification_head = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, num_classes),
        )

    def _collect_all_cls_features(self):
        tensors = []
        for hook in self.hooks:
            x_out = hook.output[0, :, :]  # [B, D]
            tensors.append(x_out)
        return torch.stack(tensors, dim=1)  # [B, num_layers, D]

    def patch_shuffle(self, x):
        b, c, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input size ({h}, {w}) must be divisible by patch_size ({self.patch_size})."
            )

        gh = h // self.patch_size
        gw = w // self.patch_size
        n = gh * gw

        patches = x.view(b, c, gh, self.patch_size, gw, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            b, n, c, self.patch_size, self.patch_size
        )

        rand = torch.rand(b, n, device=x.device)
        shuffle_idx = torch.argsort(rand, dim=1)
        gather_idx = shuffle_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, c, self.patch_size, self.patch_size
        )
        shuffled = patches.gather(1, gather_idx)

        shuffled = shuffled.view(b, gh, gw, c, self.patch_size, self.patch_size)
        shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return shuffled

    def patch_shuffle_p(self, x):
        b, c, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input size ({h}, {w}) must be divisible by patch_size ({self.patch_size})."
            )

        gh = h // self.patch_size
        gw = w // self.patch_size
        n = gh * gw

        # 计算需要打乱的 Patch 数量
        num_to_shuffle = int(self.p * n)
        
        # 如果打乱数量 <= 1，毫无意义，直接返回原图
        if num_to_shuffle <= 1:
            return x 

        patches = x.view(b, c, gh, self.patch_size, gw, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            b, n, c, self.patch_size, self.patch_size
        )

        final_idx = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n).clone()

        rand_pos = torch.rand(b, n, device=x.device)
        to_shuffle_pos = torch.argsort(rand_pos, dim=1)[:, :num_to_shuffle]

        idx_to_shuffle = final_idx.gather(1, to_shuffle_pos)

        sorted_desc, indices_desc = torch.sort(idx_to_shuffle, dim=1, descending=False)
        
        # rand_shuffle = torch.rand(b, num_to_shuffle, device=x.device)
        # shuffled_subset = idx_to_shuffle.gather(1, torch.argsort(rand_shuffle, dim=1))

        final_idx.scatter_(1, sorted_desc, to_shuffle_pos)

        gather_idx = final_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, c, self.patch_size, self.patch_size
        )

        shuffled = patches.gather(1, gather_idx)

        shuffled = shuffled.view(b, gh, gw, c, self.patch_size, self.patch_size)
        shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        
        return shuffled

    def self_attention(self, selected_features, delta=False):
        if delta:
            pos_emb = self.delta_pos_embedding(
                torch.arange(selected_features.size(1), device=selected_features.device)
            )
            cls_token = self.delta_cls
        else:
            pos_emb = self.origin_pos_embedding(
                torch.arange(selected_features.size(1), device=selected_features.device)
            )
            cls_token = self.origin_cls

        batch_size = selected_features.size(0)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        g = selected_features + pos_emb

        cls = cls_token.expand(batch_size, -1, -1)
        g = torch.cat((cls, g), dim=1)

        transformer_output = self.encoder(g)
        return transformer_output

    def compute_purified(self, origin_features, ps_features):
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
        delta_d = origin_features - ps_features  # [B, n, D]

        # Step 2: Compute semantic transition estimate
        # Delta_d^(k) = d^(k) - d_ps^(k)
        

        # Step 3: Compute per-layer semantic dependency signal
        # delta^(k) = f^(k) - f_ps^(k)  (used as gating input)

        # Step 4: Adaptive gated removal
        # d_pure^(k) = d^(k) - g^(k) * Delta_d^(k)
        d_pure_list = []
        d_q_list = []
        for k in range(delta_d.size(1)):  # iterate over n-1 transitions
            d_k = origin_features[:, k, :]        # [B, D]
            ps_k = ps_features[:, k, :]
            delta_d_k = delta_d[:, k, :] # [B, D]

            # Compute gating input statistics
            delta_d_mean = delta_d_k.mean(dim=-1)          # [B]
            delta_d_norm = delta_d_k.norm(dim=-1)          # [B]

            d_k_mean = d_k.mean(dim=-1)
            d_k_norm = d_k.norm(dim=-1)
            ps_k_mean = ps_k.mean(dim=-1)
            ps_k_norm = ps_k.norm(dim=-1)

            # Predict gating value g^(k) in [0, 1]
            g_k = self.gating_networks[k](d_k_mean, d_k_norm, ps_k_mean, ps_k_norm)  # [B, 1]

            d_pure_k = g_k * delta_d_k  # [B, D]
            d_q_k = d_k - g_k * delta_d_k
            d_pure_list.append(d_pure_k)
            d_q_list.append(d_q_k)

        d_pure = torch.stack(d_pure_list, dim=1)  # [B, n-1, D]
        d_q = torch.stack(d_q_list, dim=1)

        return d_pure, d_q, origin_features, ps_features

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        all_cls_features = self._collect_all_cls_features()
        origin_selected_features, selection_probs = self.selector(all_cls_features)

        if return_feature:
            return features

        x_ps = self.patch_shuffle_p(x)
        self.model.encode_image(x_ps)
        ps_all_cls_features = self._collect_all_cls_features()
        ps_selected_features, _ = self.selector(
            ps_all_cls_features,
            selection_probs=selection_probs,
        )

        # diff_selected_features = origin_selected_features - ps_selected_features
        d_pure, d_q, d_orig, d_ps = self.compute_purified(
            origin_features=origin_selected_features, ps_features=ps_selected_features
        )

        origin_output = self.self_attention(d_orig, False)
        # origin_output = self.self_attention(d_q, False)
        delta_output = self.self_attention(d_pure, True)
        # delta_output = self.self_attention(d_q, True)

        origin_output = origin_output[:, 0, :]
        delta_output = delta_output[:, 0, :]

        # cls = [origin_output, delta_output]
        # tokens = torch.stack(cls, dim=0)
        # tokens = F.gelu(self.fc1(tokens))
        # tokens = self.fc2(tokens).squeeze().permute(1, 0)
        # result = self.fc3(tokens).view(-1).unsqueeze(1)

        logits = self.classification_head(torch.cat((origin_output, delta_output), dim=1))
        # logits = self.classification_head(delta_output)

        return logits
        # return {
        #     'logits': logits,
        #     'd_pure': d_pure,   # for L_consist
        #     'd_ps': d_ps,       # for L_consist
        # }
