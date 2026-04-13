from typing import Dict, List, Tuple

import numpy as np

from data.channel_map import map_channels
from data.edf_loader import crop_signal, pad_or_crop_to_length, resample_signal, robust_zscore


MODALITY_ORDER = ["EEG", "ECG", "RESP", "EMG"]


def _select_best_channel(candidates: List[str], signals: Dict[str, np.ndarray]):
    if not candidates:
        return None
    best = None
    best_len = -1
    for ch in candidates:
        if ch in signals and len(signals[ch]) > best_len:
            best = ch
            best_len = len(signals[ch])
    return best


def choose_primary_channels(channel_names: List[str], signals: Dict[str, np.ndarray]):
    mapped = map_channels(channel_names)
    chosen = {}
    for mod in MODALITY_ORDER:
        chosen[mod] = _select_best_channel(mapped[mod], signals)
    return chosen


def build_signal_tensor(
    signals: Dict[str, np.ndarray],
    sample_rates: Dict[str, float],
    channel_names: List[str],
    target_fs: int,
    window_seconds: int,
    normalize: bool = True,
    random_crop: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    chosen = choose_primary_channels(channel_names, signals)
    target_len = target_fs * window_seconds

    output = {}
    mask = []

    for mod in MODALITY_ORDER:
        ch = chosen[mod]
        if ch is None or ch not in signals:
            output[mod] = np.zeros(target_len, dtype=np.float32)
            mask.append(0.0)
            continue

        x = signals[ch]
        fs = float(sample_rates.get(ch, target_fs))
        raw_needed = max(1, int(round(fs * window_seconds)))

        if len(x) > raw_needed:
            if random_crop:
                start = np.random.randint(0, len(x) - raw_needed + 1)
            else:
                start = 0
            x = crop_signal(x, start=start, length=raw_needed)

        x = resample_signal(x, orig_fs=fs, target_fs=target_fs)
        x = pad_or_crop_to_length(x, target_len=target_len)

        if normalize:
            x = robust_zscore(x)

        output[mod] = x.astype(np.float32)
        mask.append(1.0)

    return output, np.asarray(mask, dtype=np.float32)