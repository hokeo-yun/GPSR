import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from dataset_paths import DRCT, ForenSynths, GenImage, UFD, UFD_t
from models import get_model
from validate import RealFakeDataset_for_test


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
    best_start = torch.argmax(model.selector.logits).item()
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


def make_perturbed_view(model, x, mode, p, noise_std, mask_ratio):
    if mode == "ps":
        old_p = model.p
        model.p = p
        x_perturbed = model.patch_shuffle_p(x)
        model.p = old_p
        return x_perturbed
    if mode == "noise":
        return model.add_gaussian_noise(x, noise_std=noise_std)
    if mode == "mask":
        return model.patch_mask(x, mask_ratio=mask_ratio)
    raise ValueError(f"Unknown perturbation mode: {mode}")


def save_histogram(values_real, values_fake, metric_name, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    bins = 40
    plt.hist(values_real, bins=bins, alpha=0.55, density=True, label="real")
    plt.hist(values_fake, bins=bins, alpha=0.55, density=True, label="fake")
    plt.xlabel(metric_name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_boxplot(values_real, values_fake, metric_name, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    plt.boxplot([values_real, values_fake], labels=["real", "fake"], showfliers=False)
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def summarize(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="GenImage")
    parser.add_argument("--dataset_keys", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--select_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_sample", type=int, default=500)
    parser.add_argument("--result_folder", type=str, default="./results/ps_delta_analysis")
    parser.add_argument("--mode", type=str, default="ps", choices=["ps", "noise", "mask"])
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)
    set_seed(opt.seed)

    model = get_model(opt.arch, 1, opt.select_k, False, opt.p, 0)
    state_dict = torch.load(opt.ckpt, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    model.eval().cuda()

    csv_path = os.path.join(opt.result_folder, f"{opt.mode}_delta_metrics.csv")
    summary_path = os.path.join(opt.result_folder, f"{opt.mode}_delta_summary.csv")

    rows = []
    dataset_keys = None
    if opt.dataset_keys:
        dataset_keys = [key.strip() for key in opt.dataset_keys.split(",") if key.strip()]

    dataset_paths = filter_dataset_paths(get_dataset_paths(opt.test_data), dataset_keys)
    with torch.no_grad():
        for dataset_path in dataset_paths:
            set_seed(opt.seed)
            dataset = RealFakeDataset_for_test(
                dataset_path,
                opt.max_sample,
                opt.arch,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
            )

            for img, label in tqdm(loader, desc=dataset_path["key"]):
                x = img.cuda()
                label = label.numpy()
                selection_probs = make_selection_probs(model, x.size(0), x.device)

                origin_features = collect_selected_features(model, x, selection_probs)
                x_perturbed = make_perturbed_view(
                    model,
                    x,
                    opt.mode,
                    opt.p,
                    opt.noise_std,
                    opt.mask_ratio,
                )
                perturbed_features = collect_selected_features(
                    model,
                    x_perturbed,
                    selection_probs,
                )

                delta = origin_features - perturbed_features
                delta_l2 = delta.norm(dim=-1).mean(dim=1)
                cosine_shift = 1.0 - F.cosine_similarity(
                    origin_features,
                    perturbed_features,
                    dim=-1,
                ).mean(dim=1)

                for sample_label, sample_l2, sample_cosine in zip(
                    label,
                    delta_l2.cpu().tolist(),
                    cosine_shift.cpu().tolist(),
                ):
                    rows.append([
                        dataset_path["key"],
                        int(sample_label),
                        float(sample_l2),
                        float(sample_cosine),
                    ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "label", "delta_l2", "cosine_shift"])
        writer.writerows(rows)

    labels = np.array([row[1] for row in rows])
    delta_l2 = np.array([row[2] for row in rows])
    cosine_shift = np.array([row[3] for row in rows])

    real_l2 = delta_l2[labels == 0]
    fake_l2 = delta_l2[labels == 1]
    real_cos = cosine_shift[labels == 0]
    fake_cos = cosine_shift[labels == 1]

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "class", "mean", "std", "median"])
        for metric, real_values, fake_values in [
            ("delta_l2", real_l2, fake_l2),
            ("cosine_shift", real_cos, fake_cos),
        ]:
            for class_name, values in [("real", real_values), ("fake", fake_values)]:
                stats = summarize(values)
                writer.writerow([
                    metric,
                    class_name,
                    stats["mean"],
                    stats["std"],
                    stats["median"],
                ])

    save_histogram(
        real_l2,
        fake_l2,
        "||f(x) - f(PS(x))||",
        os.path.join(opt.result_folder, f"{opt.mode}_delta_l2_hist.png"),
    )
    save_boxplot(
        real_l2,
        fake_l2,
        "||f(x) - f(PS(x))||",
        os.path.join(opt.result_folder, f"{opt.mode}_delta_l2_box.png"),
    )
    save_histogram(
        real_cos,
        fake_cos,
        "1 - cosine(f(x), f(PS(x)))",
        os.path.join(opt.result_folder, f"{opt.mode}_cosine_shift_hist.png"),
    )
    save_boxplot(
        real_cos,
        fake_cos,
        "1 - cosine(f(x), f(PS(x)))",
        os.path.join(opt.result_folder, f"{opt.mode}_cosine_shift_box.png"),
    )

    print(f"Saved metrics to {csv_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved figures to {opt.result_folder}")


if __name__ == "__main__":
    main()
