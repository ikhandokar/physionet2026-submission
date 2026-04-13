import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.dataset import PhysioDataset
from models.model import PhysioRiskModel
from utils.io import ensure_dir, find_latest_checkpoint
from utils.logger import save_training_curves
from utils.metrics import safe_auroc
from utils.seed import set_seed


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


def build_datasets(cfg):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    split_name = data_cfg["training_split"]

    if train_cfg["train_on_all_sites"]:
        train_sites = data_cfg["full_training_sites"]
        val_sites = None
    else:
        train_sites = data_cfg["training_sites"]
        val_sites = data_cfg["validation_sites"]

    train_ds = PhysioDataset(
        data_root=cfg["paths"]["data_root"],
        split_name=split_name,
        sites=train_sites,
        target_fs=data_cfg["target_sample_rate"],
        window_seconds=data_cfg["window_seconds"],
        use_annotations=data_cfg["use_annotations"],
        use_metadata=data_cfg["use_metadata"],
        normalize_per_channel=data_cfg["normalize_per_channel"],
        annotation_feature_dim=data_cfg["annotation_feature_dim"],
        random_crop_train=data_cfg["random_crop_train"],
    )

    val_ds = None
    if val_sites is not None:
        val_ds = PhysioDataset(
            data_root=cfg["paths"]["data_root"],
            split_name=split_name,
            sites=val_sites,
            target_fs=data_cfg["target_sample_rate"],
            window_seconds=data_cfg["window_seconds"],
            use_annotations=data_cfg["use_annotations"],
            use_metadata=data_cfg["use_metadata"],
            normalize_per_channel=data_cfg["normalize_per_channel"],
            annotation_feature_dim=data_cfg["annotation_feature_dim"],
            random_crop_train=False,
        )

    return train_ds, val_ds


def build_model(cfg):
    return PhysioRiskModel(
        annotation_dim=cfg["data"]["annotation_feature_dim"],
        metadata_dim=cfg["data"]["metadata_feature_dim"],
        signal_embed_dim=cfg["model"]["signal_embed_dim"],
        ann_embed_dim=cfg["model"]["ann_embed_dim"],
        meta_embed_dim=cfg["model"]["meta_embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )


def save_checkpoint(path, model, optimizer, scaler, epoch, best_metric, history, cfg):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_metric": best_metric,
            "history": history,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    return ckpt


def train_one_epoch(model, loader, optimizer, device, scaler, cfg):
    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]
        valid = labels >= 0

        if valid.sum().item() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
        with autocast("cuda", enabled=use_amp):
            logits = model(batch)
            loss = loss_fn(logits[valid], labels[valid])

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
            optimizer.step()

        losses.append(loss.item())

        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx} | loss={loss.item():.4f}", flush=True)

    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def evaluate(model, loader, device):
    if loader is None:
        return np.nan, np.nan, []

    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []
    y_true = []
    y_score = []
    preview = []

    for batch in loader:
        batch_cpu = batch
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]
        valid = labels >= 0

        if valid.sum().item() == 0:
            continue

        logits = model(batch)
        probs = torch.sigmoid(logits)
        loss = loss_fn(logits[valid], labels[valid])

        losses.append(loss.item())
        y_true.extend(labels[valid].cpu().numpy().tolist())
        y_score.extend(probs[valid].cpu().numpy().tolist())

        for i in range(len(batch_cpu["record_key"])):
            preview.append(
                {
                    "record_key": batch_cpu["record_key"][i],
                    "site": batch_cpu["site"][i],
                    "label": float(batch_cpu["label"][i].item()),
                    "prediction": float(probs[i].item()),
                }
            )

    return float(np.mean(losses)) if losses else np.nan, safe_auroc(y_true, y_score), preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default="auto", choices=["auto", "none"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    ensure_dir(cfg["paths"]["checkpoints_dir"])
    ensure_dir(cfg["paths"]["logs_dir"])
    ensure_dir(cfg["paths"]["predictions_dir"])

    device = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Data root:", cfg["paths"]["data_root"])

    train_ds, val_ds = build_datasets(cfg)
    print(f"Train size: {len(train_ds)}")
    if val_ds is not None:
        print(f"Validation size: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["eval"]["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=(device.type == "cuda"),
        )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = GradScaler("cuda", enabled=(cfg["train"]["mixed_precision"] and device.type == "cuda"))

    latest_ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "latest_checkpoint.pt"
    best_ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "best_model.pt"
    history_path = Path(cfg["paths"]["logs_dir"]) / "history.json"
    preview_path = Path(cfg["paths"]["logs_dir"]) / "val_preview_predictions.json"
    curve_path = Path(cfg["paths"]["logs_dir"]) / "training_curves.png"

    start_epoch = 0
    best_metric = -1.0
    history = []

    if cfg["train"]["resume"] and args.resume != "none":
        latest = find_latest_checkpoint(cfg["paths"]["checkpoints_dir"])
        if latest is not None:
            print(f"Resuming from {latest}")
            ckpt = load_checkpoint(latest, model, optimizer=optimizer, scaler=scaler, device=device)
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_metric = float(ckpt.get("best_metric", -1.0))
            history = ckpt.get("history", [])
            print(f"Resume epoch: {start_epoch}")

    patience = 0

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, cfg)
        val_loss, val_auc, preview = evaluate(model, val_loader, device)

        record = {
            "epoch": epoch + 1,
            "train_loss": None if np.isnan(train_loss) else float(train_loss),
            "val_loss": None if np.isnan(val_loss) else float(val_loss),
            "val_auc": None if np.isnan(val_auc) else float(val_auc),
        }
        history.append(record)

        print(
            f"Epoch {epoch + 1}/{cfg['train']['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auc={val_auc:.4f}"
        )

        save_checkpoint(
            latest_ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            history=history,
            cfg=cfg,
        )

        metric_to_track = -train_loss if val_loader is None else (-1.0 if np.isnan(val_auc) else val_auc)
        improved = metric_to_track > best_metric

        if improved:
            best_metric = metric_to_track
            patience = 0
            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                history=history,
                cfg=cfg,
            )
            print("Best model updated.")
        else:
            patience += 1

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if preview:
            with open(preview_path, "w", encoding="utf-8") as f:
                json.dump(preview[: cfg["eval"]["preview_predictions"]], f, indent=2)

        save_training_curves(history, str(curve_path))

        if val_loader is not None and patience >= cfg["train"]["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Latest checkpoint: {latest_ckpt_path}")
    print(f"Training curves: {curve_path}")
    print(f"Validation preview: {preview_path}")


if __name__ == "__main__":
    main()