import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import PhysioDataset
from models.model import PhysioRiskModel
from utils.io import ensure_dir
from utils.metrics import safe_auroc


def load_config(path: str):
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_dir = config_path.parent

    if "paths" in cfg:
        for key, value in cfg["paths"].items():
            if isinstance(value, str) and value != "PATH_TO_DATASET":
                p = Path(value)
                if not p.is_absolute():
                    cfg["paths"][key] = str((config_dir / p).resolve())

    env_data_root = os.environ.get("DATA_ROOT")
    if env_data_root:
        cfg["paths"]["data_root"] = env_data_root

    if cfg["paths"]["data_root"] == "PATH_TO_DATASET":
        raise ValueError(
            "Dataset path is not set. Edit configs/default.yaml or set DATA_ROOT environment variable."
        )

    return cfg


def move_batch_to_device(batch, device):
    return {
        "record_key": batch["record_key"],
        "site": batch["site"],
        "signals": {k: v.to(device) for k, v in batch["signals"].items()},
        "mask": batch["mask"].to(device),
        "annotations": batch["annotations"].to(device),
        "metadata": batch["metadata"].to(device),
        "label": batch["label"].to(device),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--sites", type=str, nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(checkpoint_path).resolve()

    ckpt = torch.load(checkpoint_path, map_location=device)

    model = PhysioRiskModel(
        annotation_dim=cfg["data"]["annotation_feature_dim"],
        metadata_dim=cfg["data"]["metadata_feature_dim"],
        signal_embed_dim=cfg["model"]["signal_embed_dim"],
        ann_embed_dim=cfg["model"]["ann_embed_dim"],
        meta_embed_dim=cfg["model"]["meta_embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    split_name = args.split if args.split is not None else cfg["data"]["inference_split"]

    ds = PhysioDataset(
        data_root=cfg["paths"]["data_root"],
        split_name=split_name,
        sites=args.sites,
        target_fs=cfg["data"]["target_sample_rate"],
        window_seconds=cfg["data"]["window_seconds"],
        use_annotations=cfg["data"]["use_annotations"],
        use_metadata=cfg["data"]["use_metadata"],
        normalize_per_channel=cfg["data"]["normalize_per_channel"],
        annotation_feature_dim=cfg["data"]["annotation_feature_dim"],
        random_crop_train=False,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    rows = []
    y_true = []
    y_score = []

    for batch in loader:
        batch_cpu = batch
        batch = move_batch_to_device(batch, device)
        probs = torch.sigmoid(model(batch)).cpu().numpy()

        for i, p in enumerate(probs):
            label_val = float(batch_cpu["label"][i].item())
            rows.append(
                {
                    "record_key": batch_cpu["record_key"][i],
                    "site": batch_cpu["site"][i],
                    "label": None if label_val < 0 else label_val,
                    "prediction": float(p),
                }
            )
            if label_val >= 0:
                y_true.append(label_val)
                y_score.append(float(p))

    ensure_dir(cfg["paths"]["predictions_dir"])
    out_csv = Path(cfg["paths"]["predictions_dir"]) / f"{split_name}_predictions.csv"
    out_json = Path(cfg["paths"]["predictions_dir"]) / f"{split_name}_preview.json"

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows[: cfg["eval"]["preview_predictions"]], f, indent=2)

    auc = safe_auroc(y_true, y_score)
    if len(y_true) > 0:
        print(f"AUROC on labeled subset: {auc:.4f}")
    else:
        print("No labels available for this split, inference-only output generated.")

    print(f"Saved predictions to: {out_csv}")
    print(f"Saved preview to: {out_json}")


if __name__ == "__main__":
    main()