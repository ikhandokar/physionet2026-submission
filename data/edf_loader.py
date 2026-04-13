from typing import Dict, List, Tuple

import numpy as np
import pyedflib
from scipy.signal import resample


def load_edf_signals(file_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[str]]:
    reader = pyedflib.EdfReader(str(file_path))
    try:
        channel_names = reader.getSignalLabels()
        signals = {}
        sample_rates = {}

        for i, ch in enumerate(channel_names):
            try:
                x = reader.readSignal(i).astype(np.float32)
                fs = float(reader.getSampleFrequency(i))
                signals[ch] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                sample_rates[ch] = fs
            except Exception:
                continue

        return signals, sample_rates, list(channel_names)
    finally:
        reader.close()


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z = (x - med) / (1.4826 * mad)
    z = np.clip(z, -10.0, 10.0)
    return z.astype(np.float32)


def crop_signal(x: np.ndarray, start: int, length: int) -> np.ndarray:
    end = min(len(x), start + length)
    return x[start:end].astype(np.float32)


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: int) -> np.ndarray:
    if len(x) == 0:
        return x.astype(np.float32)
    if int(round(orig_fs)) == int(target_fs):
        return x.astype(np.float32)

    n_target = max(1, int(round(len(x) * target_fs / orig_fs)))
    return resample(x, n_target).astype(np.float32)


def pad_or_crop_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    x = x.astype(np.float32)
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:len(x)] = x
    return out