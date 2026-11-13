"""实验二:控制复杂度与纹理,检验 ||δ|| 的判别力是否只是低层统计量造成的。

四个阶段(一个命令跑完,结果全部落盘):

    Stage 1  复杂度指标(CPU)
             - bpp      : 把模型真正看到的那张 224x224 裁剪图以 JPEG q75 重压,
                           bpp = 8 * bytes / (224*224)
             - sobel    : 灰度图 Sobel 梯度幅值的均值
             - gray_std : 灰度标准差(用来判断 Sobel 差异是不是只是对比度差异)

    Stage 2  ||δ||(GPU)
             - 与原分析同口径:origin / PS 两路取 selector argmax 窗口的 CLS 特征,
               delta_l2 先在选中层上取 L2 均值,再对 num_reps 次扰动取平均
             - 每次扰动使用"共享重排"(同一 patch 重排作用于全部样本),便于复现

    Stage 3  统计检验
             - 分布: 两类 bpp / sobel 的 mean/std/median、KS、Cliff's delta
             - 方向: 复杂度方向 vs ||δ|| 方向是否一致(最强证据,见 README 说明)
             - 原始 AUROC(正类 = fake,score = delta_l2,>0.5 表示 fake 位移更大)
             - 分层 AUROC: 按 bpp / sobel 各分 5 档,档内单独算 AUROC
             - 匹配 AUROC: 按 (log bpp, sobel) 做 1:1 匈牙利匹配,并给出匹配前后平衡表
             - 残差 AUROC: delta_l2 对 (log bpp, sobel) 的二次回归取残差后再算 AUROC

    Stage 4  格式对齐(--format_check)
             - 把同一张 224x224 裁剪图统一过一遍 JPEG q95(内存里完成,不落盘),
               重新算 bpp / sobel / ||δ||,对比对齐前后的 AUROC

用法:
    python3 analysis/complexity_control.py \
        --test_data=ForenSynths --dataset_keys=progan \
        --ckpt=./checkpoints/ptrain_3090_0_1/best_model.pth \
        --p=0.1 --num_reps=10 --max_sample=1000 \
        --result_folder=./results/complexity_control/progan \
        --format_check
"""

import argparse
import csv
import io
import math
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_paths import DRCT, ForenSynths, GenImage, UFD, UFD_t
from models import get_model
from validate import MEAN, STD, RealFakeDataset_for_test


CROP_SIZE = 224


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_dataset_paths(name):
    if name == "UFD":
        return UFD
    if name == "UFD_t":
        return UFD_t
    if name == "GenImage":
        return GenImage
    if name == "ForenSynths":
        return ForenSynths
    return DRCT


def filter_dataset_paths(dataset_paths, dataset_keys):
    if not dataset_keys:
        return dataset_paths
    key_set = set(dataset_keys)
    filtered = [item for item in dataset_paths if item["key"] in key_set]
    missing = sorted(key_set - {item["key"] for item in filtered})
    if missing:
        print(f"Warning: dataset keys not found: {missing}")
    if not filtered:
        raise ValueError(f"No matched datasets for --dataset_keys={dataset_keys}")
    return filtered


def jpeg_roundtrip(img, quality):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    decoded.load()          # 立刻解码,避免依赖 BytesIO 的生命周期
    return decoded


def jpeg_bpp(img, quality=75):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    num_bytes = len(buffer.getvalue())
    return float(num_bytes) * 8.0 / float(img.size[0] * img.size[1])


def sobel_mean(gray):
    """3x3 Sobel 梯度幅值均值,gray 为 [H, W] float32。"""
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:]
        - padded[:-2, :-2] - 2.0 * padded[1:-1, :-2] - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
        - padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
    )
    magnitude = np.sqrt(gx * gx + gy * gy)
    return float(magnitude.mean())


def compute_metrics_one(path, jpeg_quality=None):
    """指标全部在模型真正看到的 224x224 裁剪图上计算。"""
    img = Image.open(path).convert("RGB")
    img = transforms.CenterCrop(CROP_SIZE)(img)
    if jpeg_quality is not None:
        img = jpeg_roundtrip(img, jpeg_quality)

    bpp = jpeg_bpp(img, quality=75)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return bpp, sobel_mean(gray), float(gray.std())


# --------------------------------------------------------------------------
# 数据集(可控 JPEG 质量,保证与 validate 的预处理一致)
# --------------------------------------------------------------------------
class ImageList(torch.utils.data.Dataset):
    def __init__(self, paths, labels, arch, jpeg_quality=None):
        self.paths = paths
        self.labels = labels
        self.crop = transforms.CenterCrop(CROP_SIZE)
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN[stat_from], STD[stat_from]),
        ])
        self.jpeg_quality = jpeg_quality

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.crop(img)
        if self.jpeg_quality is not None:
            img = jpeg_roundtrip(img, self.jpeg_quality)
        return self.to_tensor(img), self.labels[index]


# --------------------------------------------------------------------------
# 模型侧:选中层特征 + 共享重排
# --------------------------------------------------------------------------
def make_selection_probs(model, batch_size, device):
    best_start = int(torch.argmax(model.selector.logits).item())
    probs = torch.zeros(batch_size, model.selector.num_choices, device=device)
    probs[:, best_start] = 1.0
    return probs


def collect_selected_features(model, x, selection_probs):
    model.model.encode_image(x)
    all_cls_features = model._collect_all_cls_features()
    selected_features, _ = model.selector(
        all_cls_features,
        selection_probs=selection_probs,
    )
    return selected_features


def build_shared_shuffle_map(num_patches, num_to_shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    index = torch.arange(num_patches)
    rand_pos = torch.rand(num_patches, generator=generator)
    to_shuffle_pos = torch.argsort(rand_pos)[:num_to_shuffle]
    sorted_pos, _ = torch.sort(index[to_shuffle_pos])
    index.scatter_(0, sorted_pos, to_shuffle_pos)
    return index


def apply_shuffle_map(x, index_map, patch_size):
    b, c, h, w = x.shape
    gh, gw = h // patch_size, w // patch_size
    n = gh * gw

    patches = x.view(b, c, gh, patch_size, gw, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
        b, n, c, patch_size, patch_size
    )
    gather_index = index_map.to(x.device).view(1, n, 1, 1, 1).expand(
        b, n, c, patch_size, patch_size
    )
    shuffled = patches.gather(1, gather_index)
    shuffled = shuffled.view(b, gh, gw, c, patch_size, patch_size)
    shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
    return shuffled


def compute_delta(model, paths, labels, opt, jpeg_quality=None):
    dataset = ImageList(paths, labels, opt.arch, jpeg_quality=jpeg_quality)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    probe, _ = dataset[0]
    height, width = probe.shape[-2], probe.shape[-1]
    num_patches = (height // model.patch_size) * (width // model.patch_size)
    num_to_shuffle = int(opt.p * num_patches)
    tag = "origin" if jpeg_quality is None else f"q{jpeg_quality}"

    if num_to_shuffle <= 1:
        raise ValueError(
            f"p={opt.p} 只打乱 {num_to_shuffle} 块(<=1),delta 恒为 0,请调大 --p"
        )

    origin_chunks = []
    with torch.no_grad():
        for img, _ in tqdm(loader, desc=f"{tag} origin"):
            x = img.cuda()
            probs = make_selection_probs(model, x.size(0), x.device)
            origin_chunks.append(collect_selected_features(model, x, probs).cpu())

    values = np.zeros((len(dataset), opt.num_reps), dtype=np.float64)
    for rep in range(opt.num_reps):
        index_map = build_shared_shuffle_map(
            num_patches, num_to_shuffle, seed=opt.seed + rep
        )
        offset = 0
        batch_idx = 0
        with torch.no_grad():
            for img, _ in tqdm(loader, desc=f"{tag} rep {rep + 1}/{opt.num_reps}"):
                x = img.cuda()
                probs = make_selection_probs(model, x.size(0), x.device)
                x_ps = apply_shuffle_map(x, index_map, model.patch_size)
                ps_features = collect_selected_features(model, x_ps, probs)
                origin_features = origin_chunks[batch_idx].cuda()
                batch_idx += 1

                delta = (origin_features - ps_features).norm(dim=-1).mean(dim=1)
                values[offset:offset + x.size(0), rep] = delta.cpu().numpy()
                offset += x.size(0)

    return values


# --------------------------------------------------------------------------
# 统计工具(只依赖 numpy,避免给服务器加新依赖)
# --------------------------------------------------------------------------
def rank_data(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)

    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def roc_auc(labels, scores):
    """正类 = 1(fake),score 越大表示越像 fake。"""
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rank_data(scores)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def bootstrap_auc(labels, scores, n_boot=200, seed=0):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size < 2 or neg.size < 2:
        return float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n_boot):
        sp = pos[rng.randint(0, pos.size, pos.size)]
        sn = neg[rng.randint(0, neg.size, neg.size)]
        merged = np.concatenate([sp, sn])
        merged_labels = np.concatenate([np.ones(sp.size), np.zeros(sn.size)])
        samples.append(roc_auc(merged_labels, merged))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def ks_2samp(values_a, values_b):
    a = np.sort(np.asarray(values_a, dtype=np.float64))
    b = np.sort(np.asarray(values_b, dtype=np.float64))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, grid, side="right") / float(b.size)
    distance = float(np.max(np.abs(cdf_a - cdf_b)))
    return distance, ks_pvalue(distance, a.size, b.size)


def ks_pvalue(distance, n1, n2):
    en = math.sqrt(n1 * n2 / float(n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * distance
    if lam <= 0.0:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        total += 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
    return float(min(max(total, 0.0), 1.0))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
    }


def standardized_mean_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if denom < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / denom)


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if std < 1e-12:
        std = 1.0
    return (values - values.mean()) / std


# --------------------------------------------------------------------------
# 三种分析
# --------------------------------------------------------------------------
def stratified_auc(values, labels, scores, n_bins=5):
    """按 values 分位分档,档内单独算 AUROC。"""
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return [], float("nan")
    bin_ids = np.clip(np.digitize(values, edges[1:-1]), 0, edges.size - 2)

    rows = []
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        n_pos = int((labels[mask] == 1).sum())
        n_neg = int((labels[mask] == 0).sum())
        if n_pos < 5 or n_neg < 5:
            continue
        auc = roc_auc(labels[mask], scores[mask])
        low, high = bootstrap_auc(labels[mask], scores[mask])
        rows.append({
            "bin": bin_id,
            "lower": float(edges[bin_id]),
            "upper": float(edges[bin_id + 1]),
            "n_real": n_neg,
            "n_fake": n_pos,
            "auc": auc,
            "auc_low": low,
            "auc_high": high,
        })

    if not rows:
        return rows, float("nan")

    weights = np.array([min(r["n_real"], r["n_fake"]) for r in rows], dtype=np.float64)
    aucs = np.array([r["auc"] for r in rows], dtype=np.float64)
    weighted = float((weights * aucs).sum() / weights.sum())
    return rows, weighted


def matched_auc(bpp, sobel, labels, scores, seed=0):
    """按 (log bpp, sobel) 做 1:1 最优匹配,返回匹配后 AUROC 与平衡表。"""
    real = labels == 0
    fake = labels == 1
    if real.sum() == 0 or fake.sum() == 0:
        return None

    log_bpp = np.log(np.clip(bpp, 1e-6, None))
    z_bpp = zscore(log_bpp)
    z_sobel = zscore(sobel)

    feat_real = np.stack([z_bpp[real], z_sobel[real]], axis=1)
    feat_fake = np.stack([z_bpp[fake], z_sobel[fake]], axis=1)

    cost = ((feat_real[:, None, :] - feat_fake[None, :, :]) ** 2).sum(axis=-1)

    try:
        from scipy.optimize import linear_sum_assignment
        row_idx, col_idx = linear_sum_assignment(cost)
        method = "hungarian"
    except ImportError:
        # 退化为贪心最近邻(按距离升序配对),不保证最优
        flat = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        used_rows, used_cols = set(), set()
        row_idx, col_idx = [], []
        for row, col in flat:
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            row_idx.append(row)
            col_idx.append(col)
        row_idx = np.asarray(row_idx)
        col_idx = np.asarray(col_idx)
        method = "greedy"

    scores_matched = np.concatenate([scores[real][row_idx], scores[fake][col_idx]])
    labels_matched = np.concatenate([
        np.zeros(len(row_idx), dtype=np.int64),
        np.ones(len(col_idx), dtype=np.int64),
    ])
    auc = roc_auc(labels_matched, scores_matched)
    low, high = bootstrap_auc(labels_matched, scores_matched, seed=seed)

    return {
        "method": method,
        "n_pairs": int(len(row_idx)),
        "auc": auc,
        "auc_low": low,
        "auc_high": high,
        "smd_bpp_before": standardized_mean_diff(log_bpp[real], log_bpp[fake]),
        "smd_bpp_after": standardized_mean_diff(
            log_bpp[real][row_idx], log_bpp[fake][col_idx]
        ),
        "smd_sobel_before": standardized_mean_diff(sobel[real], sobel[fake]),
        "smd_sobel_after": standardized_mean_diff(
            sobel[real][row_idx], sobel[fake][col_idx]
        ),
        "ks_bpp_after": ks_2samp(log_bpp[real][row_idx], log_bpp[fake][col_idx])[0],
        "ks_sobel_after": ks_2samp(sobel[real][row_idx], sobel[fake][col_idx])[0],
    }


def residual_auc(bpp, sobel, labels, scores):
    """delta 对 (log bpp, sobel) 做二次回归,取残差后算 AUROC。"""
    x1 = zscore(np.log(np.clip(bpp, 1e-6, None)))
    x2 = zscore(sobel)
    design = np.stack(
        [np.ones_like(x1), x1, x2, x1 * x1, x2 * x2, x1 * x2],
        axis=1,
    )
    coef, _, _, _ = np.linalg.lstsq(design, scores, rcond=None)
    prediction = design @ coef
    residual = scores - prediction
    ss_res = float(((scores - prediction) ** 2).sum())
    ss_tot = float(((scores - scores.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    auc = roc_auc(labels, residual)
    low, high = bootstrap_auc(labels, residual)
    return {"r_squared": r_squared, "auc": auc, "auc_low": low, "auc_high": high}


# --------------------------------------------------------------------------
# 写盘
# --------------------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_dict_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def analyze(dataset_key, paths, labels, model, opt, jpeg_quality, tag):
    """跑一次「指标 + ||δ|| + 统计」,返回结果字典。"""
    metrics_path = os.path.join(
        opt.result_folder, f"{dataset_key}_metrics_{tag}.csv"
    )
    delta_path = os.path.join(opt.result_folder, f"{dataset_key}_delta_{tag}.csv")

    bpp = np.zeros(len(paths), dtype=np.float64)
    sobel = np.zeros(len(paths), dtype=np.float64)
    gray_std = np.zeros(len(paths), dtype=np.float64)

    for i, path in enumerate(tqdm(paths, desc=f"{dataset_key} metrics {tag}")):
        bpp[i], sobel[i], gray_std[i] = compute_metrics_one(path, jpeg_quality)

    write_csv(
        metrics_path,
        ["image_path", "label", "bpp_q75", "sobel_mean", "gray_std"],
        [
            [paths[i], int(labels[i]), float(bpp[i]), float(sobel[i]), float(gray_std[i])]
            for i in range(len(paths))
        ],
    )

    values = compute_delta(model, paths, labels, opt, jpeg_quality=jpeg_quality)
    delta_mean = values.mean(axis=1)

    writer_rows = []
    for i in range(len(paths)):
        row = [paths[i], int(labels[i]), float(delta_mean[i])]
        row.extend([float(v) for v in values[i]])
        writer_rows.append(row)
    header = ["image_path", "label", "delta_l2_mean"]
    header += [f"delta_l2_rep{r}" for r in range(opt.num_reps)]
    write_csv(delta_path, header, writer_rows)

    labels = np.asarray(labels, dtype=np.int64)
    real = labels == 0
    fake = labels == 1

    raw_auc = roc_auc(labels, delta_mean)
    raw_low, raw_high = bootstrap_auc(labels, delta_mean)

    rows_stats = []
    rows_tests = []
    for metric_name, metric_values in [
        ("bpp_q75", bpp),
        ("sobel_mean", sobel),
        ("gray_std", gray_std),
    ]:
        stats_real = summarize(metric_values[real])
        stats_fake = summarize(metric_values[fake])
        ks_d, ks_p = ks_2samp(metric_values[real], metric_values[fake])
        auc = roc_auc(labels, metric_values)

        for class_name, stats in [("real", stats_real), ("fake", stats_fake)]:
            rows_stats.append([
                dataset_key, tag, metric_name, class_name, stats["n"],
                stats["mean"], stats["std"], stats["median"],
            ])
        rows_tests.append([
            dataset_key, tag, metric_name, ks_d, ks_p, 2.0 * auc - 1.0, auc,
        ])

    strat_bpp, weighted_bpp = stratified_auc(bpp, labels, delta_mean, opt.num_bins)
    strat_sobel, weighted_sobel = stratified_auc(sobel, labels, delta_mean, opt.num_bins)
    matched = matched_auc(bpp, sobel, labels, delta_mean, seed=opt.seed)
    residual = residual_auc(bpp, sobel, labels, delta_mean)

    bpp_real, bpp_fake = float(bpp[real].mean()), float(bpp[fake].mean())
    sobel_real, sobel_fake = float(sobel[real].mean()), float(sobel[fake].mean())
    delta_real, delta_fake = float(delta_mean[real].mean()), float(delta_mean[fake].mean())

    summary = {
        "dataset": dataset_key,
        "tag": tag,
        "n_real": int(real.sum()),
        "n_fake": int(fake.sum()),
        "bpp_real": bpp_real,
        "bpp_fake": bpp_fake,
        "sobel_real": sobel_real,
        "sobel_fake": sobel_fake,
        "delta_real": delta_real,
        "delta_fake": delta_fake,
        # 方向一致性:复杂度方向与 ||δ|| 方向是否相同(最强证据,见文档)
        "bpp_gap": bpp_real - bpp_fake,
        "sobel_gap": sobel_real - sobel_fake,
        "delta_gap": delta_real - delta_fake,
        "consistent_bpp": int(np.sign(bpp_real - bpp_fake) == np.sign(delta_real - delta_fake)),
        "consistent_sobel": int(np.sign(sobel_real - sobel_fake) == np.sign(delta_real - delta_fake)),
        "auc_raw": raw_auc,
        "auc_raw_low": raw_low,
        "auc_raw_high": raw_high,
        "auc_strat_bpp": weighted_bpp,
        "auc_strat_sobel": weighted_sobel,
        "auc_matched": matched["auc"] if matched else float("nan"),
        "auc_residual": residual["auc"],
        "residual_r2": residual["r_squared"],
    }

    print(
        f"[{dataset_key}/{tag}] delta real={delta_real:.4f} fake={delta_fake:.4f} "
        f"| bpp real={bpp_real:.3f} fake={bpp_fake:.3f} "
        f"| sobel real={sobel_real:.2f} fake={sobel_fake:.2f} "
        f"| consistent(bpp/sobel)={summary['consistent_bpp']}/{summary['consistent_sobel']}"
    )
    print(
        f"[{dataset_key}/{tag}] AUC raw={raw_auc:.4f} [{raw_low:.4f},{raw_high:.4f}] "
        f"| strat(bpp)={weighted_bpp:.4f} strat(sobel)={weighted_sobel:.4f} "
        f"| matched={summary['auc_matched']:.4f} | residual={residual['auc']:.4f}"
    )

    return {
        "summary": summary,
        "rows_stats": rows_stats,
        "rows_tests": rows_tests,
        "strat_bpp": strat_bpp,
        "strat_sobel": strat_sobel,
        "matched": matched,
        "residual": residual,
        "metrics_path": metrics_path,
        "delta_path": delta_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="ForenSynths")
    parser.add_argument("--dataset_keys", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--select_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_sample", type=int, default=1000)
    parser.add_argument("--num_reps", type=int, default=10)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--result_folder", type=str,
                        default="./results/complexity_control")
    parser.add_argument("--format_check", action="store_true",
                        help="额外做一遍 JPEG q95 重编码后的对照")
    parser.add_argument("--format_quality", type=int, default=95)
    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)
    set_seed(opt.seed)

    model = get_model(opt.arch, 1, opt.select_k, False, opt.p, 0)
    state_dict = torch.load(opt.ckpt, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    model.eval().cuda()

    dataset_keys = None
    if opt.dataset_keys:
        dataset_keys = [k.strip() for k in opt.dataset_keys.split(",") if k.strip()]

    for dataset_path in filter_dataset_paths(get_dataset_paths(opt.test_data), dataset_keys):
        set_seed(opt.seed)
        dataset = RealFakeDataset_for_test(dataset_path, opt.max_sample, opt.arch)
        paths = list(dataset.total_list)
        labels = np.array([dataset.labels_dict[p] for p in paths], dtype=np.int64)
        key = dataset_path["key"].replace("/", "_")

        print(f"[{key}] samples={len(paths)} (real={int((labels == 0).sum())}, "
              f"fake={int((labels == 1).sum())})")

        original = analyze(key, paths, labels, model, opt, None, "original")

        results = [
            ("original", original),
        ]
        if opt.format_check:
            formatted = analyze(
                key, paths, labels, model, opt, opt.format_quality,
                f"q{opt.format_quality}",
            )
            results.append((f"q{opt.format_quality}", formatted))

        # ---- 落盘 ----
        write_csv(
            os.path.join(opt.result_folder, f"{key}_class_stats.csv"),
            ["dataset", "tag", "metric", "class", "n", "mean", "std", "median"],
            [row for _, res in results for row in res["rows_stats"]],
        )
        write_csv(
            os.path.join(opt.result_folder, f"{key}_distribution_tests.csv"),
            ["dataset", "tag", "metric", "ks_stat", "ks_p", "cliffs_delta",
             "auc_metric_alone"],
            [row for _, res in results for row in res["rows_tests"]],
        )

        summary_rows = [res["summary"] for _, res in results]
        write_dict_csv(
            os.path.join(opt.result_folder, f"{key}_summary.csv"), summary_rows
        )

        strat_rows = []
        for tag_name, res in results:
            for row in res["strat_bpp"]:
                row = dict(row)
                row["strat_by"] = "bpp_q75"
                row["tag"] = tag_name
                strat_rows.append(row)
            for row in res["strat_sobel"]:
                row = dict(row)
                row["strat_by"] = "sobel_mean"
                row["tag"] = tag_name
                strat_rows.append(row)
        write_dict_csv(
            os.path.join(opt.result_folder, f"{key}_stratified_auc.csv"), strat_rows
        )

        matched_rows = []
        for tag_name, res in results:
            if res["matched"] is None:
                continue
            row = dict(res["matched"])
            row["tag"] = tag_name
            row["residual_auc"] = res["residual"]["auc"]
            row["residual_r2"] = res["residual"]["r_squared"]
            matched_rows.append(row)
        write_dict_csv(
            os.path.join(opt.result_folder, f"{key}_matched_auc.csv"), matched_rows
        )

        if len(results) > 1:
            before, after = results[0][1]["summary"], results[1][1]["summary"]
            print(
                f"[{key}] format check: AUC {before['auc_raw']:.4f} -> "
                f"{after['auc_raw']:.4f}; bpp "
                f"({before['bpp_real']:.3f}/{before['bpp_fake']:.3f}) -> "
                f"({after['bpp_real']:.3f}/{after['bpp_fake']:.3f})"
            )

    print(f"Saved to {opt.result_folder}")


if __name__ == "__main__":
    main()
