"""Metrics. Everything here is plain arithmetic on predictions; no fitting."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.preprocess import CLASSES


def calibrate_threshold(normal_scores, target_fpr):
    """Score threshold that flags at most `target_fpr` of the given normal traffic.

    Only normal-traffic scores are used, so the unsupervised model's threshold
    still needs no attack labels: just a sample of known-good traffic.
    """
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(np.quantile(normal_scores, 1 - target_fpr, method="higher"))


def binary_metrics(is_attack, flagged, scores=None):
    """Detection metrics for the attack-vs-normal view."""
    is_attack = np.asarray(is_attack).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    tp = int(np.sum(flagged & is_attack))
    fp = int(np.sum(flagged & ~is_attack))
    tn = int(np.sum(~flagged & ~is_attack))
    fn = int(np.sum(~flagged & is_attack))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / len(is_attack),
        "precision": precision,
        "detection_rate": recall,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    if scores is not None:
        out["roc_auc"] = float(roc_auc_score(is_attack, scores))
        out["pr_auc"] = float(average_precision_score(is_attack, scores))
    return out


def multiclass_metrics(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(np.mean(f)),
        "weighted_f1": float(np.average(f, weights=s)),
        "per_class": {
            CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f[i]), "support": int(s[i])}
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def detection_by_group(groups, flagged):
    """Fraction flagged within each group (e.g. attack category)."""
    groups = np.asarray(groups)
    flagged = np.asarray(flagged).astype(bool)
    return {
        str(g): {"n": int(np.sum(groups == g)),
                 "detected": float(np.mean(flagged[groups == g]))}
        for g in sorted(set(groups))
    }


def novel_vs_known(attack_names, is_attack, flagged, novel_names):
    """Detection rate on attack types unseen in training vs those seen."""
    attack_names = np.asarray(attack_names)
    is_attack = np.asarray(is_attack).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    novel = np.isin(attack_names, list(novel_names)) & is_attack
    known = ~np.isin(attack_names, list(novel_names)) & is_attack
    return {
        "novel": {"n": int(novel.sum()),
                  "detected": float(flagged[novel].mean()) if novel.any() else None},
        "known": {"n": int(known.sum()),
                  "detected": float(flagged[known].mean()) if known.any() else None},
    }
