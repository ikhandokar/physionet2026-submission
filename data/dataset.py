import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.annotations import summarize_caisr_annotation_edf
from data.edf_loader import load_edf_signals
from data.features import MODALITY_ORDER, build_signal_tensor


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_demographics_csv(split_dir: Path) -> Path:
    files = list(split_dir.glob("demographics*.csv"))
    if not files:
        raise FileNotFoundError(f"No demographics CSV found in {split_dir}")
    return files[0]


def _extract_record_key(filename: str) -> str:
    m = re.search(r"(sub-[A-Za-z0-9]+_ses-\d+)", Path(filename).name)
    if m:
        return m.group(1)
    raise ValueError(f"Could not parse record key from: {filename}")


def _extract_bids_folder(record_key: str) -> str:
    return record_key.split("_ses-")[0]


def _extract_session_id(record_key: str) -> int:
    m = re.search(r"_ses-(\d+)", record_key)
    if not m:
        raise ValueError(f"Could not parse session ID from: {record_key}")
    return int(m.group(1))


def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_label(x):
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return 1.0
    if s in {"false", "0", "no"}:
        return 0.0
    return None


def build_metadata_features(row: pd.Series) -> np.ndarray:
    age = _safe_float(row.get("Age", np.nan), default=0.0)
    bmi = _safe_float(row.get("BMI", np.nan), default=0.0)

    sex = str(row.get("Sex", "")).strip().lower()
    sex_m = 1.0 if sex in {"m", "male"} else 0.0
    sex_f = 1.0 if sex in {"f", "female"} else 0.0

    race = str(row.get("Race", "")).strip().lower()
    race_asian = 1.0 if race == "asian" else 0.0
    race_black = 1.0 if race == "black" else 0.0
    race_white = 1.0 if race == "white" else 0.0
    race_other = 1.0 if race in {"others", "other"} else 0.0

    eth = str(row.get("Ethnicity", "")).strip().lower()
    eth_hisp = 1.0 if eth == "hispanic" else 0.0
    eth_not_hisp = 1.0 if eth == "not hispanic" else 0.0

    miss_age = 1.0 if pd.isna(row.get("Age", np.nan)) else 0.0
    miss_bmi = 1.0 if pd.isna(row.get("BMI", np.nan)) else 0.0

    return np.array(
        [
            age / 100.0,
            bmi / 60.0,
            sex_m,
            sex_f,
            race_asian,
            race_black,
            race_white,
            race_other,
            eth_hisp,
            eth_not_hisp,
            miss_age,
            miss_bmi,
        ],
        dtype=np.float32,
    )


class PhysioDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split_name: str,
        sites: Optional[List[str]],
        target_fs: int,
        window_seconds: int,
        use_annotations: bool,
        use_metadata: bool,
        normalize_per_channel: bool,
        annotation_feature_dim: int,
        random_crop_train: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split_dir = self.data_root / split_name
        self.sites = sites
        self.target_fs = target_fs
        self.window_seconds = window_seconds
        self.use_annotations = use_annotations
        self.use_metadata = use_metadata
        self.normalize_per_channel = normalize_per_channel
        self.annotation_feature_dim = annotation_feature_dim
        self.random_crop_train = random_crop_train

        self.physio_root = self.split_dir / "physiological_data"
        self.ann_root = self.split_dir / "algorithmic_annotations"
        self.demo_csv = _find_demographics_csv(self.split_dir)
        self.demographics = _normalize_columns(pd.read_csv(self.demo_csv))

        self.examples = self._collect_examples()
        if len(self.examples) == 0:
            raise RuntimeError(f"No usable examples found in {self.split_dir}")

    def _find_demographics_row(self, record_key: str) -> Optional[pd.Series]:
        bids_folder = _extract_bids_folder(record_key)
        session_id = _extract_session_id(record_key)

        df = self.demographics.copy()

        if "BidsFolder" in df.columns:
            df = df[df["BidsFolder"].astype(str) == bids_folder]

        if "SessionID" in df.columns:
            session_values = pd.to_numeric(df["SessionID"], errors="coerce").fillna(-1).astype(int)
            df = df[session_values == int(session_id)]

        if len(df) == 0:
            return None
        return df.iloc[0]

    def _find_annotation_path(self, site: str, record_key: str) -> Optional[Path]:
        site_dir = self.ann_root / site
        if not site_dir.exists():
            return None

        candidates = list(site_dir.glob(f"{record_key}_caisr*.edf"))
        if candidates:
            return candidates[0]

        candidates = list(site_dir.glob(f"{record_key}*.edf"))
        if candidates:
            return candidates[0]

        return None

    def _collect_examples(self) -> List[Dict]:
        examples = []

        if not self.physio_root.exists():
            raise FileNotFoundError(f"Missing physiological_data directory: {self.physio_root}")

        for site_dir in sorted(self.physio_root.iterdir()):
            if not site_dir.is_dir():
                continue
            site = site_dir.name

            if self.sites is not None and site not in self.sites:
                continue

            for edf_path in sorted(site_dir.glob("*.edf")):
                try:
                    record_key = _extract_record_key(edf_path.name)
                except ValueError:
                    continue

                row = self._find_demographics_row(record_key)
                if row is None:
                    continue

                label = _safe_label(row["Cognitive_Impairment"]) if "Cognitive_Impairment" in row.index else None
                ann_path = self._find_annotation_path(site=site, record_key=record_key)

                examples.append(
                    {
                        "record_key": record_key,
                        "site": site,
                        "physio_path": str(edf_path),
                        "annotation_path": str(ann_path) if ann_path is not None else None,
                        "metadata_features": build_metadata_features(row),
                        "label": label,
                    }
                )

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]

        signals, sample_rates, channel_names = load_edf_signals(ex["physio_path"])
        signal_dict, modality_mask = build_signal_tensor(
            signals=signals,
            sample_rates=sample_rates,
            channel_names=channel_names,
            target_fs=self.target_fs,
            window_seconds=self.window_seconds,
            normalize=self.normalize_per_channel,
            random_crop=self.random_crop_train,
        )

        ann_features = summarize_caisr_annotation_edf(
            ex["annotation_path"],
            feature_dim=self.annotation_feature_dim,
        )

        return {
            "record_key": ex["record_key"],
            "site": ex["site"],
            "signals": {mod: torch.tensor(signal_dict[mod], dtype=torch.float32) for mod in MODALITY_ORDER},
            "mask": torch.tensor(modality_mask, dtype=torch.float32),
            "annotations": torch.tensor(
                ann_features if self.use_annotations else np.zeros(self.annotation_feature_dim, dtype=np.float32),
                dtype=torch.float32,
            ),
            "metadata": torch.tensor(
                ex["metadata_features"] if self.use_metadata else np.zeros_like(ex["metadata_features"]),
                dtype=torch.float32,
            ),
            "label": torch.tensor(-1.0 if ex["label"] is None else float(ex["label"]), dtype=torch.float32),
        }