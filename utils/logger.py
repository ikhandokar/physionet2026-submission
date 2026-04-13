from pathlib import Path

import matplotlib.pyplot as plt


def save_training_curves(history, out_path: str):
    if len(history) == 0:
        return

    epochs = [h["epoch"] for h in history]
    train_losses = [h.get("train_loss") for h in history]
    val_losses = [h.get("val_loss") for h in history]
    val_aucs = [h.get("val_auc") for h in history]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, train_losses, label="train_loss")
    ax1.plot(epochs, val_losses, label="val_loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, val_aucs, label="val_auc")
    ax2.set_title("Validation AUROC")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)