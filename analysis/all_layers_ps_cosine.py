import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_paths import DRCT, ForenSynths, GenImage, UFD, UFD_t
from models import get_model
from validate import MEAN, STD, RealFakeDataset_for_test


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


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, samples, arch, is_resize=False):
        self.samples = samples
        if is_resize:
            rz_func = transforms.Resize((256, 256))
        else:
            rz_func = transforms.Lambda(lambda img: img)

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            rz_func,
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        img = Image.open(image_path).convert("RGB")
        return self.transform(img), label


def get_selected_samples(model, datasets_to_run, opt, num_layers):
    sample_rows = []

    with torch.no_grad():
        for dataset_name, dataset in datasets_to_run:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
            )

            sample_offset = 0
            for img, label in tqdm(loader, desc=f"select {dataset_name}"):
                x = img.cuda()
                label_np = label.numpy()
                batch_paths = dataset.total_list[sample_offset : sample_offset + len(label_np)]
                sample_offset += len(label_np)

                origin_features = collect_all_layer_cls_features(model, x)
                x_ps = model.patch_shuffle_p(x)
                ps_features = collect_all_layer_cls_features(model, x_ps)

                cosine_distance = 1.0 - F.cosine_similarity(
                    origin_features,
                    ps_features,
                    dim=-1,
                )
                sample_scores = cosine_distance.mean(dim=1).cpu().tolist()

                for image_path, sample_label, score in zip(
                    batch_paths,
                    label_np,
                    sample_scores,
                ):
                    sample_rows.append([
                        dataset_name,
                        int(sample_label),
                        image_path,
                        float(score),
                    ])

    selected_rows = []
    for class_label in [0, 1]:
        class_rows = [row for row in sample_rows if row[1] == class_label]
        selected_rows.extend(
            sorted(class_rows, key=lambda row: row[3], reverse=True)[
                : opt.top_k_per_class
            ]
        )

    selected_csv_path = os.path.join(
        opt.result_folder,
        f"selected_{opt.top_k_per_class * 2}_by_cosine.csv",
    )
    with open(selected_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "label", "image_path", "mean_cosine_distance"])
        writer.writerows(selected_rows)

    print(f"Saved selected samples to {selected_csv_path}")
    print(
        f"Selected {sum(row[1] == 0 for row in selected_rows)} real and "
        f"{sum(row[1] == 1 for row in selected_rows)} fake samples."
    )

    return [(row[2], row[1]) for row in selected_rows]


def reset_all_layer_hooks(model):
    for hook in getattr(model, "hooks", []):
        hook.close()

    resblocks = model.model.visual.transformer.resblocks
    model.hooks = [Hook(block) for block in resblocks]
    return len(model.hooks)


def collect_all_layer_cls_features(model, x):
    model.model.encode_image(x)
    features = []
    for hook in model.hooks:
        features.append(hook.output[0, :, :])
    return torch.stack(features, dim=1)


def save_layer_curve(layer_rows, output_path, title):
    import matplotlib.pyplot as plt

    layers = [row["layer"] for row in layer_rows]
    real = [row["real_mean"] for row in layer_rows]
    fake = [row["fake_mean"] for row in layer_rows]

    plt.figure(figsize=(7, 4))
    plt.plot(layers, real, marker="o", linewidth=2, label="real")
    plt.plot(layers, fake, marker="s", linewidth=2, label="fake")
    plt.title(title)
    plt.xlabel("CLIP Transformer Layer")
    plt.ylabel("1 - cosine(f_l(x), f_l(PS(x)))")
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    plt.legend()
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
    parser.add_argument("--result_folder", type=str, default="./results/all_layers_ps_cosine")
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--top_k_per_class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)
    set_seed(opt.seed)

    model = get_model(opt.arch, 1, opt.select_k, False, opt.p, 0)
    state_dict = torch.load(opt.ckpt, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    model.eval().cuda()

    num_layers = reset_all_layer_hooks(model)
    print(f"Using all {num_layers} transformer layers without ALS selection.")

    real_values = [[] for _ in range(num_layers)]
    fake_values = [[] for _ in range(num_layers)]

    dataset_keys = None
    if opt.dataset_keys:
        dataset_keys = [
            key.strip() for key in opt.dataset_keys.split(",") if key.strip()
        ]

    dataset_paths = filter_dataset_paths(get_dataset_paths(opt.test_data), dataset_keys)
    dataset_title = ", ".join([dataset_path["key"] for dataset_path in dataset_paths])
    candidate_datasets = []
    is_resize = False
    for dataset_path in dataset_paths:
        set_seed(opt.seed)
        dataset = RealFakeDataset_for_test(
            dataset_path,
            opt.max_sample,
            opt.arch,
        )
        is_resize = is_resize or dataset_path["is_resize"]
        candidate_datasets.append((dataset_path["key"], dataset))

    selected_samples = get_selected_samples(model, candidate_datasets, opt, num_layers)
    selected_dataset = ImagePathDataset(selected_samples, opt.arch, is_resize=is_resize)

    with torch.no_grad():
        for dataset_name, dataset in [("selected_by_cosine", selected_dataset)]:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
            )

            for img, label in tqdm(loader, desc=dataset_name):
                x = img.cuda()
                label = label.numpy()

                origin_features = collect_all_layer_cls_features(model, x)
                x_ps = model.patch_shuffle_p(x)
                ps_features = collect_all_layer_cls_features(model, x_ps)

                cosine_distance = 1.0 - F.cosine_similarity(
                    origin_features,
                    ps_features,
                    dim=-1,
                )

                cosine_distance = cosine_distance.cpu().numpy()
                for layer_idx in range(num_layers):
                    real_values[layer_idx].extend(
                        cosine_distance[label == 0, layer_idx].tolist()
                    )
                    fake_values[layer_idx].extend(
                        cosine_distance[label == 1, layer_idx].tolist()
                    )

    layer_rows = []
    for layer_idx in range(num_layers):
        real_arr = np.asarray(real_values[layer_idx])
        fake_arr = np.asarray(fake_values[layer_idx])
        layer_rows.append({
            "layer": layer_idx + 1,
            "real_mean": float(real_arr.mean()),
            "real_std": float(real_arr.std()),
            "fake_mean": float(fake_arr.mean()),
            "fake_std": float(fake_arr.std()),
            "gap_real_minus_fake": float(real_arr.mean() - fake_arr.mean()),
            "real_count": int(real_arr.size),
            "fake_count": int(fake_arr.size),
        })

    csv_path = os.path.join(opt.result_folder, "all_layers_ps_cosine.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "layer",
            "real_mean",
            "real_std",
            "fake_mean",
            "fake_std",
            "gap_real_minus_fake",
            "real_count",
            "fake_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(layer_rows)

    figure_path = os.path.join(opt.result_folder, "all_layers_ps_cosine_curve.png")
    save_layer_curve(
        layer_rows,
        figure_path,
        f"{dataset_title}: all-layer PS cosine distance",
    )

    print(f"Saved layer metrics to {csv_path}")
    print(f"Saved layer curve to {figure_path}")


if __name__ == "__main__":
    main()
