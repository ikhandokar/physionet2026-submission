from pathlib import Path
from typing import Dict, Optional

import numpy as np

from data.edf_loader import load_edf_signals


def _first_matching_channel(signals: Dict[str, np.ndarray], substrings):
    for ch in signals.keys():
        name = ch.lower()
        if all(s in name for s in substrings):
            return signals[ch]
    return None


def summarize_caisr_annotation_edf(annotation_path: Optional[str], feature_dim: int = 20) -> np.ndarray:
    if annotation_path is None or not Path(annotation_path).exists():
        return np.zeros(feature_dim, dtype=np.float32)

    try:
        signals, _, _ = load_edf_signals(annotation_path)
    except Exception:
        return np.zeros(feature_dim, dtype=np.float32)

    feats = []

    stage = _first_matching_channel(signals, ["stage", "caisr"])
    if stage is not None and len(stage) > 0:
        stage = np.nan_to_num(stage).astype(np.int32)
        total = max(1, len(stage))
        for cls in [1, 2, 3, 4, 5]:
            feats.append(float((stage == cls).sum()) / total)
        feats.append(float((stage == 9).sum()) / total)
    else:
        feats.extend([0.0] * 6)

    arousal = _first_matching_channel(signals, ["arousal"])
    if arousal is not None and len(arousal) > 0:
        arousal = np.nan_to_num(arousal).astype(np.float32)
        feats.append(float(arousal.mean()))
        feats.append(float((arousal > 0).mean()))
    else:
        feats.extend([0.0, 0.0])

    resp = _first_matching_channel(signals, ["resp"])
    if resp is not None and len(resp) > 0:
        resp = np.nan_to_num(resp).astype(np.int32)
        total = max(1, len(resp))
        for cls in [1, 2, 3, 4, 5]:
            feats.append(float((resp == cls).sum()) / total)
    else:
        feats.extend([0.0] * 5)

    limb = _first_matching_channel(signals, ["limb"])
    if limb is not None and len(limb) > 0:
        limb = np.nan_to_num(limb).astype(np.int32)
        total = max(1, len(limb))
        feats.append(float((limb == 1).sum()) / total)
        feats.append(float((limb == 2).sum()) / total)
    else:
        feats.extend([0.0] * 2)

    for prob_key in [
        "caisr_prob_n3",
        "caisr_prob_n2",
        "caisr_prob_n1",
        "caisr_prob_r",
        "caisr_prob_w",
    ]:
        arr = None
        for ch in signals.keys():
            if prob_key in ch.lower():
                arr = signals[ch]
                break
        feats.append(float(np.nan_to_num(arr).mean()) if arr is not None and len(arr) > 0 else 0.0)

    feats = np.asarray(feats, dtype=np.float32)
    if len(feats) != feature_dim:
        out = np.zeros(feature_dim, dtype=np.float32)
        out[:min(feature_dim, len(feats))] = feats[:feature_dim]
        return out
    return feats