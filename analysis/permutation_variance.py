"""两种方差来源的统计(用于说明 real/fake 差异是"跨样本稳定"还是"对随机扰动稳定")。

记 M[i, r] 为第 i 个样本在第 r 次随机扰动下的指标值(i=1..N,r=1..R):

1) across samples(固定 R 次随机,让样本变化)
       s_i = (1/R) * sum_r M[i, r]                # 每个样本先对自己的 R 次取平均
       mean(s), std(s)                            # 再统计 N 个样本的均值/方差

2) across permutation(固定全部样本,让扰动变化)
       t_r = (1/N) * sum_i M[i, r]                # 每次扰动先对全体样本取平均
       mean(t), std(t)                            # 再统计 R 次扰动的均值/方差

两种统计共用同一张 M,所以只跑一遍。

perm_mode 的两种含义:
    shared      每次扰动对"所有样本"使用同一个 patch 重排(标准 permutation 设计,
                衡量"换一种打乱方式,结论是否还成立")
    independent 每次扰动对每个样本独立随机(只换随机种子,等价于重采样扰动)

示例:
    python3 analysis/permutation_variance.py \
        --test_data=ForenSynths --dataset_keys=progan \
        --ckpt=./checkpoints/ptrain_3090_0_1/best_model.pth \
        --p=0.1 --num_reps=10 --max_sample=100 \
        --result_folder=./results/permutation_variance/progan --plot
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_paths import DRCT, ForenSynths, GenImage, UFD, UFD_t
from models import get_model
from validate import RealFakeDataset_for_test


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


def make_selection_probs(model, batch_size, device):
    """与分析脚本保持一致:用训练好的 selector 的 argmax 窗口,不做 Gumbel 采样。"""
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
    """构造一次共享的 patch 重排索引(等价于 patch_shuffle_p 里 batch=1 的情形)。"""
    generator = torch.Generator().manual_seed(seed)
    index = torch.arange(num_patches)
    rand_pos = torch.rand(num_patches, generator=generator)
    to_shuffle_pos = torch.argsort(rand_pos)[:num_to_shuffle]
    idx_to_shuffle = index[to_shuffle_pos]
    sorted_pos, _ = torch.sort(idx_to_shuffle)

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

    index_map = index_map.to(x.device).view(1, n, 1, 1, 1).expand(
        b, n, c, patch_size, patch_size
    )
    shuffled = patches.gather(1, index_map)

    shuffled = shuffled.view(b, gh, gw, c, patch_size, patch_size)
    shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
    return shuffled


def perturb_batch(model, x, mode, perm_mode, index_map, noise_std, mask_ratio):
    if mode == "ps":
        if perm_mode == "shared":
            return apply_shuffle_map(x, index_map, model.patch_size)
        return model.patch_shuffle_p(x)
    if mode == "noise":
        if not hasattr(model, "add_gaussian_noise"):
            raise AttributeError("当前模型没有 add_gaussian_noise(见 ablation8)")
        return model.add_gaussian_noise(x, noise_std=noise_std)
    if mode == "mask":
        if not hasattr(model, "patch_mask"):
            raise AttributeError("当前模型没有 patch_mask(见 ablation6/9)")
        return model.patch_mask(x, mask_ratio=mask_ratio)
    raise ValueError(f"Unknown perturbation mode: {mode}")


def summarize(values, ddof=1):
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    mean = float(values.mean())
    std = float(values.std(ddof=ddof)) if n > ddof else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "sem": float(sem),
        "median": float(np.median(values)),
    }


def plot_across_samples(class_values, metric, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for name, values in class_values.items():
        plt.hist(values, bins=40, alpha=0.55, density=True, label=name)
    plt.xlabel(f"per-sample mean {metric}")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_across_permutation(class_means, metric, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for name, means in class_means.items():
        xs = np.arange(1, len(means) + 1)
        plt.plot(xs, means, marker="o", label=name)
    plt.xlabel("permutation index")
    plt.ylabel(f"mean {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="ForenSynths")
    parser.add_argument("--dataset_keys", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--select_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_sample", type=int, default=100)
    parser.add_argument("--num_reps", type=int, default=10)
    parser.add_argument("--mode", type=str, default="ps", choices=["ps", "noise", "mask"])
    parser.add_argument("--perm_mode", type=str, default="shared",
                        choices=["shared", "independent"])
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--result_folder", type=str, default="./results/permutation_variance")
    parser.add_argument("--plot", action="store_true")
    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)
    set_seed(opt.seed)

    model = get_model(opt.arch, 1, opt.select_k, False, opt.p, 0)
    state_dict = torch.load(opt.ckpt, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    model.eval().cuda()

    dataset_keys = None
    if opt.dataset_keys:
        dataset_keys = [key.strip() for key in opt.dataset_keys.split(",") if key.strip()]

    dataset_paths = filter_dataset_paths(get_dataset_paths(opt.test_data), dataset_keys)

    all_sample_rows = []
    across_sample_rows = []
    across_permutation_rows = []

    for dataset_path in dataset_paths:
        set_seed(opt.seed)
        dataset = RealFakeDataset_for_test(dataset_path, opt.max_sample, opt.arch)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
        )

        num_samples = len(dataset)
        probe, _ = dataset[0]
        # dataset[0] 返回单张图 [C, H, W];取最后两维,兼容 batch 维存在与否
        height, width = probe.shape[-2], probe.shape[-1]
        blocks_per_side_h = height // model.patch_size
        blocks_per_side_w = width // model.patch_size
        num_patches = blocks_per_side_h * blocks_per_side_w
        num_to_shuffle = int(opt.p * num_patches)

        if opt.mode == "ps" and num_to_shuffle <= 1:
            print(
                f"Warning: p={opt.p} 只打乱 {num_to_shuffle} 块(<=1),"
                "patch_shuffle 会直接返回原图,指标恒为 0。"
            )

        print(
            f"[{dataset_path['key']}] samples={num_samples}, reps={opt.num_reps}, "
            f"patch={model.patch_size}px, blocks={num_patches}, "
            f"shuffle={num_to_shuffle}, perm_mode={opt.perm_mode}"
        )

        values_delta_l2 = np.zeros((num_samples, opt.num_reps), dtype=np.float64)
        values_cosine = np.zeros((num_samples, opt.num_reps), dtype=np.float64)
        labels = np.zeros(num_samples, dtype=np.int64)
        paths = []

        # ---- Pass 0:原图特征是确定性的,先算一次缓存起来,省掉 R-1 次前向 ----
        origin_chunks = []
        offset = 0
        with torch.no_grad():
            for img, label in tqdm(loader, desc=f"{dataset_path['key']} origin"):
                x = img.cuda()
                selection_probs = make_selection_probs(model, x.size(0), x.device)
                origin_features = collect_selected_features(model, x, selection_probs)
                origin_chunks.append(origin_features.cpu())

                batch_len = x.size(0)
                labels[offset:offset + batch_len] = label.numpy()
                paths.extend(dataset.total_list[offset:offset + batch_len])
                offset += batch_len

        # ---- Pass 1..R:每次扰动重算 PS 特征 ----
        for rep in range(opt.num_reps):
            if opt.perm_mode == "shared":
                # 同一个重排在所有样本、所有 batch 上复用
                index_map = build_shared_shuffle_map(
                    num_patches,
                    num_to_shuffle,
                    seed=opt.seed + rep,
                )
            else:
                # independent:种子固定即可复现,每个样本各自独立随机
                set_seed(opt.seed + rep)
                index_map = None

            offset = 0
            batch_idx = 0
            with torch.no_grad():
                for img, _ in tqdm(
                    loader, desc=f"{dataset_path['key']} rep {rep + 1}/{opt.num_reps}"
                ):
                    x = img.cuda()
                    selection_probs = make_selection_probs(model, x.size(0), x.device)

                    x_perturbed = perturb_batch(
                        model,
                        x,
                        opt.mode,
                        opt.perm_mode,
                        index_map,
                        opt.noise_std,
                        opt.mask_ratio,
                    )
                    ps_features = collect_selected_features(
                        model, x_perturbed, selection_probs
                    )
                    origin_features = origin_chunks[batch_idx].cuda()
                    batch_idx += 1

                    delta = origin_features - ps_features
                    delta_l2 = delta.norm(dim=-1).mean(dim=1)
                    cosine_shift = 1.0 - F.cosine_similarity(
                        origin_features,
                        ps_features,
                        dim=-1,
                    ).mean(dim=1)

                    batch_len = x.size(0)
                    values_delta_l2[offset:offset + batch_len, rep] = (
                        delta_l2.cpu().numpy()
                    )
                    values_cosine[offset:offset + batch_len, rep] = (
                        cosine_shift.cpu().numpy()
                    )
                    offset += batch_len

        # ---------------- 1) across samples ----------------
        sample_mean_l2 = values_delta_l2.mean(axis=1)
        sample_std_l2 = values_delta_l2.std(axis=1, ddof=1) if opt.num_reps > 1 else np.zeros(num_samples)
        sample_mean_cos = values_cosine.mean(axis=1)
        sample_std_cos = values_cosine.std(axis=1, ddof=1) if opt.num_reps > 1 else np.zeros(num_samples)

        for i in range(num_samples):
            row = [
                dataset_path["key"],
                int(labels[i]),
                paths[i],
                float(sample_mean_l2[i]),
                float(sample_std_l2[i]),
                float(sample_mean_cos[i]),
                float(sample_std_cos[i]),
            ]
            row.extend([float(v) for v in values_delta_l2[i]])
            row.extend([float(v) for v in values_cosine[i]])
            all_sample_rows.append(row)

        # ---------------- 2) across permutation ----------------
        # 注意:这里是「每次扰动下该类别的均值」,长度是 R(不是 N)
        real_rep_mean_l2 = values_delta_l2[labels == 0].mean(axis=0)
        fake_rep_mean_l2 = values_delta_l2[labels == 1].mean(axis=0)

        for class_label, class_name in [(0, "real"), (1, "fake"), (None, "all")]:
            if class_label is None:
                mask = np.ones(num_samples, dtype=bool)
            else:
                mask = labels == class_label
                if mask.sum() == 0:
                    continue

            for metric_name, matrix, sample_means in [
                ("delta_l2", values_delta_l2, sample_mean_l2),
                ("cosine_shift", values_cosine, sample_mean_cos),
            ]:
                stats = summarize(sample_means[mask])
                across_sample_rows.append([
                    dataset_path["key"],
                    metric_name,
                    class_name,
                    stats["n"],
                    stats["mean"],
                    stats["std"],
                    stats["sem"],
                    stats["median"],
                ])

                rep_means = matrix[mask, :].mean(axis=0)
                stats = summarize(rep_means)
                across_permutation_rows.append([
                    dataset_path["key"],
                    metric_name,
                    class_name,
                    stats["n"],
                    stats["mean"],
                    stats["std"],
                    stats["sem"],
                    stats["median"],
                ])

        if opt.plot:
            plot_across_samples(
                {
                    "Real": sample_mean_l2[labels == 0],
                    "Fake": sample_mean_l2[labels == 1],
                },
                "delta_l2",
                os.path.join(
                    opt.result_folder,
                    f"{dataset_path['key']}_across_samples_hist.png",
                ),
            )
            # 每个点 = 该次 permutation 下该类全部样本的均值
            plot_across_permutation(
                {
                    "Real": real_rep_mean_l2,
                    "Fake": fake_rep_mean_l2,
                },
                "delta_l2",
                os.path.join(
                    opt.result_folder,
                    f"{dataset_path['key']}_across_permutation_curve.png",
                ),
            )

        print(
            f"[{dataset_path['key']}] across-samples  delta_l2 | "
            f"real={sample_mean_l2[labels == 0].mean():.4f}"
            f"±{sample_mean_l2[labels == 0].std(ddof=1):.4f}  "
            f"fake={sample_mean_l2[labels == 1].mean():.4f}"
            f"±{sample_mean_l2[labels == 1].std(ddof=1):.4f}"
        )
        print(
            f"[{dataset_path['key']}] across-perm     delta_l2 | "
            f"real={real_rep_mean_l2.mean():.4f}"
            f"±{real_rep_mean_l2.std(ddof=1):.4f}  "
            f"fake={fake_rep_mean_l2.mean():.4f}"
            f"±{fake_rep_mean_l2.std(ddof=1):.4f}"
        )

    # ---------------- 落盘 ----------------
    per_sample_path = os.path.join(opt.result_folder, "per_sample_values.csv")
    with open(per_sample_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "dataset", "label", "image_path",
            "delta_l2_mean", "delta_l2_std",
            "cosine_shift_mean", "cosine_shift_std",
        ]
        header += [f"delta_l2_rep{r}" for r in range(opt.num_reps)]
        header += [f"cosine_shift_rep{r}" for r in range(opt.num_reps)]
        writer.writerow(header)
        writer.writerows(all_sample_rows)

    across_sample_path = os.path.join(opt.result_folder, "across_samples_stats.csv")
    with open(across_sample_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "metric", "class", "n_samples",
            "mean", "std", "sem", "median",
        ])
        writer.writerows(across_sample_rows)

    across_permutation_path = os.path.join(opt.result_folder, "across_permutation_stats.csv")
    with open(across_permutation_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "metric", "class", "n_permutations",
            "mean", "std", "sem", "median",
        ])
        writer.writerows(across_permutation_rows)

    print(f"Saved per-sample values  -> {per_sample_path}")
    print(f"Saved across-samples     -> {across_sample_path}")
    print(f"Saved across-permutation -> {across_permutation_path}")


if __name__ == "__main__":
    main()
