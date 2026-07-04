from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
from .vision_transformer import vit_l_16
from transformers import FlavaImageModel


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
}

import torch
import torch.nn as nn


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
            raise ValueError(f"select_k ({select_k}) 必须小于或等�?num_layers ({num_layers})")
            
        self.num_layers = num_layers
        self.select_k = select_k
        
        # 为每个可能的起始点设置一个可学习�?logit
        self.logits = nn.Parameter(torch.randn(self.num_choices))
        
        self.cnt = cnt
        
        self.is_training_mode = training # 存储训练标志
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
            
            selection_probs = F.gumbel_softmax(expanded_logits, 
                                               tau=self.current_tau, 
                                               hard=True, 
                                               dim=-1)

        all_windows = []
        for i in range(self.num_choices):
            # 提取�?i �?i + select_k 的窗�?            
            window = all_features[:, i : i + self.select_k, :]
            all_windows.append(window)
        
        # stacked_windows shape: [B, num_choices, select_k, D]
        stacked_windows = torch.stack(all_windows, dim=1)
        
        # [B, num_choices] -> [B, num_choices, 1, 1]
        selection_probs_expanded = selection_probs.view(batch_size, self.num_choices, 1, 1)
        
        weighted_windows = stacked_windows * selection_probs_expanded
        # print(selection_probs_expanded)
        
        selected_features = torch.sum(weighted_windows, dim=1)
        
        return selected_features, selection_probs


class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, select_num=5, training=True, cnt={}):
        super(CLIPModel, self).__init__()

        print(name)
        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 

        self.cnt = cnt
        
        self.model.requires_grad_(False)
        
        self.hooks = []
        for i in range(11,20) :
            self.hooks.append(Hook(f'block_{i}', self.model.visual.transformer.resblocks[i]))
    
        # span = 2
        proj_dim = 1024
        self.sequence_length = len(self.hooks)

        self.selector = LayerSelector(num_layers=self.sequence_length, select_k=select_num, cnt=cnt,training=True)
        
        self.origin_pos_embedding = nn.Embedding(select_num, proj_dim)
        self.delta_pos_embedding = nn.Embedding(select_num - 1, proj_dim)

        self.origin_cls = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.delta_cls = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.patch_size = 16
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, 
            nhead=8,
            dim_feedforward=proj_dim * 4,
            dropout=0.3,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classification_head = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, num_classes)
        )

    def extract_cls_features(self, selection_probs=None):
        tensors = []
        for hook in self.hooks:
            x_out = hook.output[0, :, :]  # [B, D]
            
            tensors.append(x_out)
        g = torch.stack(tensors, dim=1)  # [B, num_layers, D]
        selected_features, selection_probs = self.selector(
            g, selection_probs=selection_probs
        )
        return selected_features, selection_probs
    
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

    def self_attention(self, cls_features, delta=False):
        selected_features = cls_features
        if delta:
            pos_emb = self.delta_pos_embedding(torch.arange(selected_features.size(1), device=selected_features.device))
            cls_token = self.delta_cls
        else:
            pos_emb = self.origin_pos_embedding(torch.arange(selected_features.size(1), device=selected_features.device))
            cls_token = self.origin_cls

        # --- 3. 加位置嵌入与 CLS token ---
        batch_size = selected_features.size(0)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        g = selected_features + pos_emb
        # g = selected_features

        cls = cls_token.expand(batch_size, -1, -1)
        g = torch.cat((cls, g), dim=1)

        # --- 4. Transformer 编码 ---
        transformer_output = self.encoder(g)

        
        return transformer_output

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        origin_selected_features, selection_probs = self.extract_cls_features()
        if return_feature:
            return features

        x_ps = self.patch_shuffle(x)
        self.model.encode_image(x_ps)
        ps_selected_features, _ = self.extract_cls_features(selection_probs=selection_probs)

        origin_adjacent_diff = (
            origin_selected_features[:, 1:, :] - origin_selected_features[:, :-1, :]
        )
        ps_adjacent_diff = (
            ps_selected_features[:, 1:, :] - ps_selected_features[:, :-1, :]
        )
        delta_features = origin_adjacent_diff - ps_adjacent_diff

        origin_output = self.self_attention(origin_selected_features, False)
        delta_output = self.self_attention(delta_features, True)

        origin_output = origin_output[:, 0, :]
        delta_output = delta_output[:, 0, :]

        logits = self.classification_head(torch.cat((origin_output, delta_output), dim=1))
        return logits
