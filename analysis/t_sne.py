import argparse
import csv
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from sklearn.manifold import TSNE
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_paths import DRCT, ForenSynths, GenImage, UFD, UFD_t
from models import get_model
from validate import RealFakeDataset_for_test


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def extract_origin_delta_outputs(model, x):
    model.model.encode_image(x)
    all_cls_features = model._collect_all_cls_features()
    origin_selected_features, selection_probs = model.selector(all_cls_features)

    x_ps = model.patch_shuffle_p(x)
    model.model.encode_image(x_ps)
    ps_all_cls_features = model._collect_all_cls_features()
    ps_selected_features, _ = model.selector(
        ps_all_cls_features,
        selection_probs=selection_probs,
    )

    d_pure, d_q, d_orig, d_ps = model.compute_purified(
        origin_features=origin_selected_features,
        ps_features=ps_selected_features,
    )

    origin_output = model.self_attention(d_orig, False)
    delta_output = model.self_attention(d_pure, True)

    origin_output = origin_output[:, 0, :]
    delta_output = delta_output[:, 0, :]

    return origin_output, delta_output


def run_tsne(features, seed, perplexity):
    n_samples = features.shape[0]
    safe_perplexity = min(perplexity, max(1, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(features)


def save_tsne_csv(coords, labels, dataset_names, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "class", "dataset"])
        for coord, label, dataset_name in zip(coords, labels, dataset_names):
            class_name = "real" if label == 0 else "fake"
            writer.writerow([
                float(coord[0]),
                float(coord[1]),
                int(label),
                class_name,
                dataset_name,
            ])


def save_combined_tsne_csv(coords, labels, branches, dataset_names, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "class", "branch", "dataset"])
        for coord, label, branch, dataset_name in zip(coords, labels, branches, dataset_names):
            class_name = "real" if label == 0 else "fake"
            writer.writerow([
                float(coord[0]),
                float(coord[1]),
                int(label),
                class_name,
                branch,
                dataset_name,
            ])


def plot_tsne(coords, labels, title, output_path):
    plt.figure(figsize=(6.2, 5.0))

    labels = np.asarray(labels)
    real = coords[labels == 0]
    fake = coords[labels == 1]

    plt.scatter(real[:, 0], real[:, 1], s=12, alpha=0.7, label="Real")
    plt.scatter(fake[:, 0], fake[:, 1], s=12, alpha=0.7, label="Fake")
    plt.title("ProGAN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_tsne(coords, labels, branches, title, output_path):
    plt.figure(figsize=(6.8, 5.4))

    labels = np.asarray(labels)
    branches = np.asarray(branches)
    groups = [
        ("origin", 0, r"Origin-Real", "o"),
        ("origin", 1, r"Origin-Fake", "o"),
        ("delta", 0, r"Delta-Real", "o"),
        ("delta", 1, r"Delta-Fake", "o"),
    ]

    for branch, label, name, marker in groups:
        mask = (branches == branch) & (labels == label)
        group_coords = coords[mask]
        plt.scatter(
            group_coords[:, 0],
            group_coords[:, 1],
            s=8,
            alpha=0.7,
            marker=marker,
            label=name,
        )

    plt.title("SD v1.4")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="GenImage")
    parser.add_argument("--dataset_keys", type=str, default="Midjourney")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--select_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_sample", type=int, default=500)
    parser.add_argument("--result_folder", type=str, default="./results/t_sne")
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--ablation", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    opt = parser.parse_args()

    if opt.ablation != 0:
        raise ValueError("This t-SNE script uses compute_purified/self_attention from the default model; keep --ablation=0.")

    os.makedirs(opt.result_folder, exist_ok=True)
    set_seed(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(opt.arch, 1, opt.select_k, False, opt.p, opt.ablation)
    state_dict = torch.load(opt.ckpt, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    model.eval().to(device)

    dataset_keys = None
    if opt.dataset_keys:
        dataset_keys = [key.strip() for key in opt.dataset_keys.split(",") if key.strip()]
    dataset_paths = filter_dataset_paths(get_dataset_paths(opt.test_data), dataset_keys)
    dataset_title = ", ".join([dataset_path["key"] for dataset_path in dataset_paths])

    origin_features = []
    delta_features = []
    labels = []
    dataset_names = []

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
                x = img.to(device)
                origin_output, delta_output = extract_origin_delta_outputs(model, x)

                origin_features.append(origin_output.cpu().numpy())
                delta_features.append(delta_output.cpu().numpy())
                labels.extend(label.numpy().astype(int).tolist())
                dataset_names.extend([dataset_path["key"]] * label.shape[0])

    origin_features = np.concatenate(origin_features, axis=0)
    delta_features = np.concatenate(delta_features, axis=0)
    labels = np.asarray(labels)

    origin_coords = run_tsne(origin_features, opt.seed, opt.perplexity)
    delta_coords = run_tsne(delta_features, opt.seed, opt.perplexity)

    combined_features = np.concatenate([origin_features, delta_features], axis=0)
    combined_labels = np.concatenate([labels, labels], axis=0)
    combined_branches = np.array(
        ["origin"] * len(labels) + ["delta"] * len(labels)
    )
    combined_dataset_names = dataset_names + dataset_names
    combined_coords = run_tsne(combined_features, opt.seed, opt.perplexity)

    origin_csv = os.path.join(opt.result_folder, "origin_output_tsne.csv")
    delta_csv = os.path.join(opt.result_folder, "delta_output_tsne.csv")
    combined_csv = os.path.join(opt.result_folder, "origin_delta_combined_tsne.csv")
    save_tsne_csv(origin_coords, labels, dataset_names, origin_csv)
    save_tsne_csv(delta_coords, labels, dataset_names, delta_csv)
    save_combined_tsne_csv(
        combined_coords,
        combined_labels,
        combined_branches,
        combined_dataset_names,
        combined_csv,
    )

    origin_png = os.path.join(opt.result_folder, "origin_output_tsne.png")
    delta_png = os.path.join(opt.result_folder, "delta_output_tsne.png")
    combined_png = os.path.join(opt.result_folder, "origin_delta_combined_tsne.png")
    plot_tsne(
        origin_coords,
        labels,
        f"{dataset_title}: origin_output t-SNE",
        origin_png,
    )
    plot_tsne(
        delta_coords,
        labels,
        f"{dataset_title}: delta_output t-SNE",
        delta_png,
    )
    plot_combined_tsne(
        combined_coords,
        combined_labels,
        combined_branches,
        f"{dataset_title}: origin_output and delta_output t-SNE",
        combined_png,
    )

    print(f"Saved origin_output t-SNE to {origin_png}")
    print(f"Saved delta_output t-SNE to {delta_png}")
    print(f"Saved combined t-SNE to {combined_png}")
    print(f"Saved t-SNE coordinates to {origin_csv}, {delta_csv}, and {combined_csv}")


if __name__ == "__main__":
    main()
