import argparse
import csv
import os
import random
import shutil
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
    plt.boxplot(
        [values_real, values_fake],
        tick_labels=["real", "fake"],
        showfliers=False,
    )
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


def safe_name(text):
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_top_samples(rows, metric, result_folder, top_k, copy_images):
    if top_k <= 0:
        return

    metric_idx = {
        "delta_l2": 3,
        "cosine_shift": 4,
    }[metric]

    top_rows = sorted(rows, key=lambda row: row[metric_idx], reverse=True)[:top_k]
    top_csv_path = os.path.join(result_folder, f"top_{top_k}_{metric}.csv")
    with open(top_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "dataset", "label", "image_path", "delta_l2", "cosine_shift"])
        for rank, row in enumerate(top_rows, start=1):
            writer.writerow([rank] + row)

    if not copy_images:
        return

    image_dir = os.path.join(result_folder, "top_samples", metric)
    os.makedirs(image_dir, exist_ok=True)
    for rank, row in enumerate(top_rows, start=1):
        dataset_key, label, image_path, delta_l2, cosine_shift = row
        class_name = "real" if label == 0 else "fake"
        ext = os.path.splitext(image_path)[1]
        filename = (
            f"{rank:03d}_{class_name}_{safe_name(dataset_key)}_"
            f"d{delta_l2:.4f}_c{cosine_shift:.4f}{ext}"
        )
        dst = os.path.join(image_dir, filename)
        if os.path.exists(image_path):
            shutil.copy2(image_path, dst)


def save_top_samples_per_class(rows, metric, result_folder, top_k_per_class, copy_images):
    if top_k_per_class <= 0:
        return

    metric_idx = {
        "delta_l2": 3,
        "cosine_shift": 4,
    }[metric]

    class_names = {
        0: "real",
        1: "fake",
    }

    for class_label, class_name in class_names.items():
        top_rows = get_top_rows(rows, metric, top_k_per_class, class_label)

        top_csv_path = os.path.join(
            result_folder,
            f"top_{top_k_per_class}_{class_name}_{metric}.csv",
        )
        with open(top_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["rank", "dataset", "label", "image_path", "delta_l2", "cosine_shift"]
            )
            for rank, row in enumerate(top_rows, start=1):
                writer.writerow([rank] + row)

        if not copy_images:
            continue

        image_dir = os.path.join(result_folder, "top_samples", metric, class_name)
        os.makedirs(image_dir, exist_ok=True)
        for rank, row in enumerate(top_rows, start=1):
            dataset_key, label, image_path, delta_l2, cosine_shift = row
            ext = os.path.splitext(image_path)[1]
            filename = (
                f"{rank:03d}_{class_name}_{safe_name(dataset_key)}_"
                f"d{delta_l2:.4f}_c{cosine_shift:.4f}{ext}"
            )
            dst = os.path.join(image_dir, filename)
            if os.path.exists(image_path):
                shutil.copy2(image_path, dst)


def get_top_rows(rows, metric, top_k, class_label=None):
    metric_idx = {
        "delta_l2": 3,
        "cosine_shift": 4,
    }[metric]
    selected_rows = rows
    if class_label is not None:
        selected_rows = [row for row in rows if row[1] == class_label]
    return sorted(selected_rows, key=lambda row: row[metric_idx], reverse=True)[:top_k]


def save_selected_subset_analysis(rows, metric, result_folder, top_k_per_class):
    if top_k_per_class <= 0:
        return

    selected_rows = (
        get_top_rows(rows, metric, top_k_per_class, class_label=0)
        + get_top_rows(rows, metric, top_k_per_class, class_label=1)
    )
    subset_name = f"selected_{top_k_per_class * 2}_{metric}"
    subset_csv_path = os.path.join(result_folder, f"{subset_name}.csv")
    subset_summary_path = os.path.join(result_folder, f"{subset_name}_summary.csv")

    with open(subset_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "label", "image_path", "delta_l2", "cosine_shift"])
        writer.writerows(selected_rows)

    labels = np.array([row[1] for row in selected_rows])
    delta_l2 = np.array([row[3] for row in selected_rows])
    cosine_shift = np.array([row[4] for row in selected_rows])

    real_l2 = delta_l2[labels == 0]
    fake_l2 = delta_l2[labels == 1]
    real_cos = cosine_shift[labels == 0]
    fake_cos = cosine_shift[labels == 1]

    with open(subset_summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "class", "mean", "std", "median"])
        for metric_name, real_values, fake_values in [
            ("delta_l2", real_l2, fake_l2),
            ("cosine_shift", real_cos, fake_cos),
        ]:
            for class_name, values in [("real", real_values), ("fake", fake_values)]:
                stats = summarize(values)
                writer.writerow([
                    metric_name,
                    class_name,
                    stats["mean"],
                    stats["std"],
                    stats["median"],
                ])

    save_histogram(
        real_l2,
        fake_l2,
        "||f(x) - f(PS(x))||",
        os.path.join(result_folder, f"{subset_name}_delta_l2_hist.png"),
    )
    save_boxplot(
        real_l2,
        fake_l2,
        "||f(x) - f(PS(x))||",
        os.path.join(result_folder, f"{subset_name}_delta_l2_box.png"),
    )
    save_histogram(
        real_cos,
        fake_cos,
        "1 - cosine(f(x), f(PS(x)))",
        os.path.join(result_folder, f"{subset_name}_cosine_shift_hist.png"),
    )
    save_boxplot(
        real_cos,
        fake_cos,
        "1 - cosine(f(x), f(PS(x)))",
        os.path.join(result_folder, f"{subset_name}_cosine_shift_box.png"),
    )


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
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_k_per_class", type=int, default=0)
    parser.add_argument("--copy_top_images", action="store_true")
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

            sample_offset = 0
            for img, label in tqdm(loader, desc=dataset_path["key"]):
                x = img.cuda()
                label = label.numpy()
                batch_paths = dataset.total_list[sample_offset : sample_offset + len(label)]
                sample_offset += len(label)
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

                for sample_label, image_path, sample_l2, sample_cosine in zip(
                    label,
                    batch_paths,
                    delta_l2.cpu().tolist(),
                    cosine_shift.cpu().tolist(),
                ):
                    rows.append([
                        dataset_path["key"],
                        int(sample_label),
                        image_path,
                        float(sample_l2),
                        float(sample_cosine),
                    ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "label", "image_path", "delta_l2", "cosine_shift"])
        writer.writerows(rows)

    labels = np.array([row[1] for row in rows])
    delta_l2 = np.array([row[3] for row in rows])
    cosine_shift = np.array([row[4] for row in rows])

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

    save_top_samples(
        rows,
        "delta_l2",
        opt.result_folder,
        opt.top_k,
        opt.copy_top_images,
    )
    save_top_samples(
        rows,
        "cosine_shift",
        opt.result_folder,
        opt.top_k,
        opt.copy_top_images,
    )
    save_top_samples_per_class(
        rows,
        "delta_l2",
        opt.result_folder,
        opt.top_k_per_class,
        opt.copy_top_images,
    )
    save_top_samples_per_class(
        rows,
        "cosine_shift",
        opt.result_folder,
        opt.top_k_per_class,
        opt.copy_top_images,
    )
    save_selected_subset_analysis(
        rows,
        "delta_l2",
        opt.result_folder,
        opt.top_k_per_class,
    )
    save_selected_subset_analysis(
        rows,
        "cosine_shift",
        opt.result_folder,
        opt.top_k_per_class,
    )

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
