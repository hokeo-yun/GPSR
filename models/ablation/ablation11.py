import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import clip


CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
}


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
    """ablation11:用「拼接 + 线性投影」替代门控净化。

    与 default / ablation7 / ablation10 的唯一区别在 compute_purified:

        default      d_pure^(k) = g^(k) * (d^(k) - d_ps^(k))          (门控乘差值)
        ablation7    d_pure^(k) = d_ps^(k)                           (完全不做净化)
        ablation10   d_pure^(k) = w^(k) * d^(k) + (1-w^(k)) * d_ps^(k)  (逐层标量融合)
        ablation11   d_pure^(k) = W @ [d^(k) ; d_ps^(k)] + b          (拼接后线性投影)

    拼接维度是特征维:concat 后 [B, k, 2D],投影回 [B, k, D],下游
    self_attention / 位置编码 / 分类头都不需要改。

    投影矩阵在所有层间共享(2.1M 参数),并且初始化成
        W = [0.5 * I ; 0.5 * I],  b = 0
    使初始时 d_pure = 0.5 * d + 0.5 * d_ps,与 ablation10 的起点完全一致,
    两者的差异只来自「融合函数是否可学习/取值是否逐通道」。
    """

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
        self.proj_dim = proj_dim
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

        # ablation11:拼接融合 [B, k, 2D] -> [B, k, D],各层共享权重
        self.fusion_proj = nn.Linear(2 * proj_dim, proj_dim)
        with torch.no_grad():
            self.fusion_proj.weight.zero_()
            eye = torch.eye(proj_dim)
            self.fusion_proj.weight[:, :proj_dim] = 0.5 * eye
            self.fusion_proj.weight[:, proj_dim:] = 0.5 * eye
            self.fusion_proj.bias.zero_()

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

    def fusion_stats(self):
        """把融合矩阵拆成「原图侧」和「PS 侧」两半的逐通道权重范数,便于分析。"""
        weight = self.fusion_proj.weight.detach()
        return {
            "origin": weight[:, : self.proj_dim].norm(dim=1),
            "ps": weight[:, self.proj_dim :].norm(dim=1),
        }

    def compute_purified(self, origin_features, ps_features):
        """把两路特征拼起来再线性投影,得到 d_pure。

        Args:
            origin_features: 原图在选中层上的 CLS 特征,[B, k, D]
            ps_features:     PS 图在同样层上的 CLS 特征,[B, k, D]

        Returns:
            d_pure: 融合后的特征,[B, k, D]
            d_q:    origin 到 d_pure 的残差,[B, k, D]
            origin_features, ps_features: 原样返回,保持与 default 版本接口一致
        """
        fused = torch.cat([origin_features, ps_features], dim=-1)  # [B, k, 2D]
        d_pure = self.fusion_proj(fused)                           # [B, k, D]
        d_q = origin_features - d_pure

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

        d_pure, d_q, d_orig, d_ps = self.compute_purified(
            origin_features=origin_selected_features, ps_features=ps_selected_features
        )

        origin_output = self.self_attention(d_orig, False)
        delta_output = self.self_attention(d_pure, True)

        origin_output = origin_output[:, 0, :]
        delta_output = delta_output[:, 0, :]

        logits = self.classification_head(torch.cat((origin_output, delta_output), dim=1))

        return logits
