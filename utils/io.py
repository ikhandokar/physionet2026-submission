from pathlib import Path


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_latest_checkpoint(checkpoints_dir: str):
    latest = Path(checkpoints_dir) / "latest_checkpoint.pt"
    return latest if latest.exists() else None