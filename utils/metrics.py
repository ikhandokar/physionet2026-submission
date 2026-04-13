import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")