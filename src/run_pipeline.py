"""End to end: download -> preprocess -> train 3 models -> evaluate -> write outputs.

Outputs (all regenerated on every run, all git-ignored except what the README
quotes by hand):
  outputs/metrics.json        every number, per model, per test set
  outputs/results.md          the same as markdown tables
  outputs/*.png               figures used in the README
  outputs/models/             trained models
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import numpy as np

from src import plots
from src.config import load_config
from src.download_data import download_all
from src.evaluate import (
    binary_metrics,
    calibrate_threshold,
    detection_by_group,
    multiclass_metrics,
    novel_vs_known,
)
from src.models import build_models
from src.preprocess import (
    CLASSES,
    Preprocessor,
    load_split,
    novel_attack_names,
    split_train_val,
)

TEST_SETS = {"test": "KDDTest+", "test_21": "KDDTest-21"}
ATTACK_CATEGORIES = CLASSES[1:]


def describe(df):
    return {"rows": len(df), "by_category": df["category"].value_counts().to_dict()}


def evaluate_model(model, X, df, threshold, novel):
    scores = model.attack_score(X)
    calibrated = scores > threshold
    default = model.default_flag(X)
    res = {
        "binary_calibrated": binary_metrics(df["is_attack"], calibrated, scores),
        "binary_default": binary_metrics(df["is_attack"], default, scores),
        "detection_by_category": detection_by_group(df["category"], calibrated),
        "novel_vs_known": novel_vs_known(df["attack"], df["is_attack"], calibrated, novel),
    }
    pred = model.predict_class(X)
    if pred is not None:
        res["multiclass"] = multiclass_metrics(df["y"], pred)
    return res, scores


def write_markdown(summary, path):
    lines = [f"# Results ({summary['generated_utc']})", ""]
    for key, label in TEST_SETS.items():
        lines += [f"## {label}: attack vs normal, threshold calibrated to "
                  f"{summary['target_fpr']:.0%} FPR on validation", "",
                  "| Model | Detection rate | FPR | Precision | F1 | ROC-AUC | PR-AUC |",
                  "|---|---|---|---|---|---|---|"]
        for name, m in summary["models"].items():
            b = m[key]["binary_calibrated"]
            lines.append(
                f"| {name} | {b['detection_rate']:.4f} | {b['false_positive_rate']:.4f} | "
                f"{b['precision']:.4f} | {b['f1']:.4f} | {b['roc_auc']:.4f} | {b['pr_auc']:.4f} |"
            )
        lines += ["", f"## {label}: attack vs normal, each model's default decision", "",
                  "| Model | Detection rate | FPR | Precision | F1 |", "|---|---|---|---|---|"]
        for name, m in summary["models"].items():
            b = m[key]["binary_default"]
            lines.append(f"| {name} | {b['detection_rate']:.4f} | "
                         f"{b['false_positive_rate']:.4f} | {b['precision']:.4f} | {b['f1']:.4f} |")
        lines += ["", f"## {label}: share flagged per category (calibrated threshold)", "",
                  "| Model | " + " | ".join(CLASSES) + " | Novel attacks | Known attacks |",
                  "|---" * (len(CLASSES) + 3) + "|"]
        for name, m in summary["models"].items():
            d = m[key]["detection_by_category"]
            nk = m[key]["novel_vs_known"]
            lines.append(f"| {name} | " + " | ".join(f"{d[c]['detected']:.4f}" for c in CLASSES)
                         + f" | {nk['novel']['detected']:.4f} | {nk['known']['detected']:.4f} |")
        lines += ["", f"## {label}: 5-class (supervised models)", "",
                  "| Model | Accuracy | Macro F1 | " +
                  " | ".join(f"{c} recall" for c in CLASSES) + " |",
                  "|---" * (len(CLASSES) + 3) + "|"]
        for name, m in summary["models"].items():
            mc = m[key].get("multiclass")
            if mc:
                lines.append(f"| {name} | {mc['accuracy']:.4f} | {mc['macro_f1']:.4f} | " +
                             " | ".join(f"{mc['per_class'][c]['recall']:.4f}" for c in CLASSES) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-save", action="store_true", help="don't write trained models")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = cfg["output"]["dir"]
    models_dir = cfg["output"]["models_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    paths = download_all(cfg)
    full_train = load_split(paths["train"])
    tests = {k: load_split(paths[k]) for k in TEST_SETS}

    pp_cfg = cfg["preprocess"]
    train_df, val_df = split_train_val(full_train, pp_cfg["val_fraction"], pp_cfg["random_state"])
    pre = Preprocessor().fit(train_df)
    X_train, X_val = pre.transform(train_df), pre.transform(val_df)
    y_train, y_val = train_df["y"].to_numpy(), val_df["y"].to_numpy()
    X_tests = {k: pre.transform(df) for k, df in tests.items()}
    novel = {k: novel_attack_names(full_train, df) for k, df in tests.items()}
    print(f"features: {X_train.shape[1]} (dropped constant: {pre.dropped_})")

    target_fpr = cfg["target_fpr"]
    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "target_fpr": target_fpr,
        "data": {
            "train": describe(train_df), "validation": describe(val_df),
            **{k: describe(df) for k, df in tests.items()},
            "novel_attack_types": {k: v for k, v in novel.items()},
            "n_features": int(X_train.shape[1]),
            "dropped_constant_features": pre.dropped_,
        },
        "models": {},
    }
    test_scores = {}

    for model in build_models(cfg):
        print(f"\n=== {model.name} ===")
        t0 = time.perf_counter()
        model.fit(X_train, y_train, X_val, y_val)
        fit_s = time.perf_counter() - t0

        val_scores = model.attack_score(X_val)
        threshold = calibrate_threshold(val_scores[y_val == 0], target_fpr)
        entry = {
            "supervised": model.supervised,
            "fit_seconds": round(fit_s, 1),
            "threshold": threshold,
            "validation": binary_metrics(val_df["is_attack"], val_scores > threshold, val_scores),
        }
        for attr in ("n_train_", "best_iteration_", "epochs_run_", "best_epoch_"):
            if hasattr(model, attr):
                entry[attr.rstrip("_")] = getattr(model, attr)

        for key in TEST_SETS:
            t0 = time.perf_counter()
            entry[key], scores = evaluate_model(model, X_tests[key], tests[key], threshold, novel[key])
            entry[key]["score_seconds"] = round(time.perf_counter() - t0, 2)
            test_scores[(model.name, key)] = scores
            b = entry[key]["binary_calibrated"]
            print(f"{TEST_SETS[key]}: DR {b['detection_rate']:.4f}  FPR {b['false_positive_rate']:.4f}"
                  f"  F1 {b['f1']:.4f}  ROC-AUC {b['roc_auc']:.4f}")
        summary["models"][model.name] = entry

        if not args.skip_save and hasattr(model, "save"):
            ext = {"XGBoost": "xgb.json", "LSTM": "lstm.keras"}[model.name]
            model.save(models_dir / ext)

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    write_markdown(summary, out_dir / "results.md")

    for key, label in TEST_SETS.items():
        df = tests[key]
        curves = []
        for name, m in summary["models"].items():
            b = m[key]["binary_calibrated"]
            curves.append((name, df["is_attack"].to_numpy(), test_scores[(name, key)],
                           (b["detection_rate"], b["precision"])))
        suffix = "" if key == "test" else "_21"
        plots.pr_curves(curves, out_dir / f"pr_curves{suffix}.png",
                        f"Precision-recall on {label}")
        rates = {name: {c: m[key]["detection_by_category"][c]["detected"] for c in ATTACK_CATEGORIES}
                 for name, m in summary["models"].items()}
        rates_novel = {name: m[key]["novel_vs_known"]["novel"]["detected"]
                       for name, m in summary["models"].items()}
        for name in rates:
            rates[name]["Novel types"] = rates_novel[name]
        plots.detection_by_category(
            rates, ATTACK_CATEGORIES + ["Novel types"], out_dir / f"detection_by_category{suffix}.png",
            f"Attacks flagged per category on {label} ({target_fpr:.0%} target FPR)")
    for name, m in summary["models"].items():
        if "multiclass" in m["test"]:
            slug = name.lower().replace(" ", "_")
            plots.confusion(m["test"]["multiclass"]["confusion_matrix"], CLASSES,
                            out_dir / f"confusion_{slug}.png", f"{name}: 5-class on KDDTest+")

    print(f"\nwrote {out_dir / 'metrics.json'} and {out_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
