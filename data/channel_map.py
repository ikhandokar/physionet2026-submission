from typing import Dict, List


def _norm(x: str) -> str:
    return x.lower().replace("-", "").replace("_", "").replace(" ", "")


def map_channels(channel_names: List[str]) -> Dict[str, List[str]]:
    mapped = {
        "EEG": [],
        "ECG": [],
        "RESP": [],
        "EMG": [],
    }

    for ch in channel_names:
        n = _norm(ch)

        if any(k in n for k in [
            "eeg", "fp1", "fp2", "f3", "f4", "c3", "c4", "o1", "o2",
            "pz", "cz", "t3", "t4", "t5", "t6", "a1", "a2"
        ]):
            mapped["EEG"].append(ch)
            continue

        if any(k in n for k in ["ecg", "ekg"]):
            mapped["ECG"].append(ch)
            continue

        if any(k in n for k in [
            "resp", "airflow", "flow", "nasal", "thor", "chest", "abd",
            "abdo", "effort", "pressure", "cannula"
        ]):
            mapped["RESP"].append(ch)
            continue

        if any(k in n for k in [
            "emg", "chin", "mentalis", "leg", "tibialis", "plm"
        ]):
            mapped["EMG"].append(ch)
            continue

    return mapped