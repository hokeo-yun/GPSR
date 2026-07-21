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


class Hook:
    def __init__(self, module):
        self.output = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


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


def reset_fixed_layer_hooks(model, start_layer, end_layer):
    for hook in getattr(model, "hooks", []):
        hook.close()

    resblocks = model.model.visual.transformer.resblocks
    if start_layer < 1 or end_layer > len(resblocks) or start_layer > end_layer:
        raise ValueError(
            f"Invalid layer range {start_layer}-{end_layer}; "
            f"model has {len(resblocks)} layers."
        )

    model.hooks = [
        Hook(resblocks[layer_idx])
        for layer_idx in range(start_layer - 1, end_layer)
    ]
    return list(range(start_layer, end_layer + 1))


def collect_fixed_layer_cls_features(model, x):
    model.model.encode_image(x)
    features = []
    for hook in model.hooks:
        features.append(hook.output[0, :, :])
    return torch.stack(features, dim=1)


def save_histogram(real_values, fake_values, output_path, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    bins = 40
    plt.hist(real_values, bins=bins, alpha=0.55, density=True, label="real")
    plt.hist(fake_values, bins=bins, alpha=0.55, density=True, label="fake")
    plt.title(title)
    plt.xlabel(r"$\delta$")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_boxplot(real_values, fake_values, output_path, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    plt.boxplot(
        [real_values, fake_values],
        tick_labels=["real", "fake"],
        showfliers=False,
    )
    plt.title(title)
    plt.ylabel(r"$\delta$")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="GenImage")
    parser.add_argument("--dataset_keys", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--select_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_sample", type=int, default=1000)
    parser.add_argument("--top_k_per_class", type=int, default=500)
    parser.add_argument("--result_folder", type=str, default="./results/fixed_layers_cosine_hist")
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--start_layer", type=int, default=13)
    parser.add_argument("--end_layer", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)
    set_seed(opt.seed)

    model = get_model(opt.arch, 1, opt.select_k, False, opt.p, 0)
    state_dict = torch.load(opt.ckpt, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    model.eval().cuda()

    layer_ids = reset_fixed_layer_hooks(model, opt.start_layer, opt.end_layer)
    print(f"Using fixed layers without ALS: {layer_ids}")

    dataset_keys = None
    if opt.dataset_keys:
        dataset_keys = [key.strip() for key in opt.dataset_keys.split(",") if key.strip()]

    dataset_paths = filter_dataset_paths(get_dataset_paths(opt.test_data), dataset_keys)
    dataset_title = ", ".join([dataset_path["key"] for dataset_path in dataset_paths])

    rows = []
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
                label_np = label.numpy()
                batch_paths = dataset.total_list[sample_offset : sample_offset + len(label_np)]
                sample_offset += len(label_np)

                origin_features = collect_fixed_layer_cls_features(model, x)
                x_ps = model.patch_shuffle_p(x)
                ps_features = collect_fixed_layer_cls_features(model, x_ps)

                cosine_distance = 1.0 - F.cosine_similarity(
                    origin_features,
                    ps_features,
                    dim=-1,
                )
                mean_cosine_distance = cosine_distance.mean(dim=1).cpu().tolist()
                delta_l2 = (origin_features - ps_features).norm(dim=-1).mean(dim=1)
                mean_delta_l2 = delta_l2.cpu().tolist()

                for image_path, sample_label, cosine_score, l2_score in zip(
                    batch_paths,
                    label_np,
                    mean_cosine_distance,
                    mean_delta_l2,
                ):
                    rows.append([
                        dataset_path["key"],
                        int(sample_label),
                        image_path,
                        float(cosine_score),
                        float(l2_score),
                    ])

    selected_rows = []
    for class_label in [0, 1]:
        class_rows = [row for row in rows if row[1] == class_label]
        selected_rows.extend(
            sorted(class_rows, key=lambda row: row[3], reverse=True)[
                : opt.top_k_per_class
            ]
        )

    selected_csv_path = os.path.join(
        opt.result_folder,
        f"selected_{opt.top_k_per_class * 2}_layers_{opt.start_layer}_{opt.end_layer}_cosine.csv",
    )
    with open(selected_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "label",
            "image_path",
            "mean_cosine_distance",
            "mean_delta_l2",
        ])
        writer.writerows(selected_rows)

    labels = np.array([row[1] for row in selected_rows])
    cosine_scores = np.array([row[3] for row in selected_rows])
    delta_l2_scores = np.array([row[4] for row in selected_rows])
    real_cosine_scores = cosine_scores[labels == 0]
    fake_cosine_scores = cosine_scores[labels == 1]
    real_delta_l2_scores = delta_l2_scores[labels == 0]
    fake_delta_l2_scores = delta_l2_scores[labels == 1]

    summary_path = os.path.join(
        opt.result_folder,
        f"selected_{opt.top_k_per_class * 2}_layers_{opt.start_layer}_{opt.end_layer}_summary.csv",
    )
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric",
            "class",
            "mean",
            "std",
            "median",
            "count",
        ])
        for metric_name, class_name, values in [
            ("mean_cosine_distance", "real", real_cosine_scores),
            ("mean_cosine_distance", "fake", fake_cosine_scores),
            ("mean_delta_l2", "real", real_delta_l2_scores),
            ("mean_delta_l2", "fake", fake_delta_l2_scores),
        ]:
            writer.writerow([
                metric_name,
                class_name,
                float(values.mean()),
                float(values.std()),
                float(np.median(values)),
                int(values.size),
            ])

    title = (
        f"{dataset_title}: layers {opt.start_layer}-{opt.end_layer} "
        f"PS delta L2 after cosine selection"
    )
    hist_path = os.path.join(
        opt.result_folder,
        f"selected_{opt.top_k_per_class * 2}_layers_{opt.start_layer}_{opt.end_layer}_hist.png",
    )
    box_path = os.path.join(
        opt.result_folder,
        f"selected_{opt.top_k_per_class * 2}_layers_{opt.start_layer}_{opt.end_layer}_box.png",
    )
    save_histogram(real_delta_l2_scores, fake_delta_l2_scores, hist_path, title)
    save_boxplot(real_delta_l2_scores, fake_delta_l2_scores, box_path, title)

    print(
        f"Selected {real_delta_l2_scores.size} real and "
        f"{fake_delta_l2_scores.size} fake samples."
    )
    print(f"Saved selected samples to {selected_csv_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved histogram to {hist_path}")
    print(f"Saved boxplot to {box_path}")


if __name__ == "__main__":
    main()
