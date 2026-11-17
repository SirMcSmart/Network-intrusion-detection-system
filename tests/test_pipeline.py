"""Synthetic-data tests: no download needed."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import binary_metrics, calibrate_threshold, novel_vs_known
from src.models import IsolationForestDetector, XGBoostDetector, to_sequences
from src.preprocess import (
    ATTACK_CATEGORY,
    CLASSES,
    COLUMNS,
    FEATURES,
    Preprocessor,
    class_weights,
    load_split,
)

# Every attack name that appears in KDDTrain+, KDDTest+ and KDDTest-21.
ALL_NSL_KDD_ATTACKS = {
    "normal", "neptune", "satan", "ipsweep", "portsweep", "smurf", "nmap", "back",
    "teardrop", "warezclient", "pod", "guess_passwd", "buffer_overflow",
    "warezmaster", "land", "imap", "rootkit", "loadmodule", "ftp_write",
    "multihop", "phf", "perl", "spy", "mscan", "apache2", "processtable",
    "snmpguess", "saint", "mailbomb", "snmpgetattack", "httptunnel", "named",
    "ps", "sendmail", "xterm", "xlock", "xsnoop", "worm", "udpstorm",
    "sqlattack",
}


def make_rows(n, rng, attacks=("normal", "neptune", "satan", "guess_passwd", "rootkit")):
    df = pd.DataFrame(rng.integers(0, 5, size=(n, len(FEATURES))), columns=FEATURES)
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["service"] = rng.choice(["http", "ftp", "smtp"], n)
    df["flag"] = rng.choice(["SF", "S0"], n)
    df["num_outbound_cmds"] = 0
    df["serror_rate"] = rng.random(n)
    df["attack"] = rng.choice(list(attacks), n)
    df["difficulty"] = 20
    return df[COLUMNS]


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_every_nsl_kdd_attack_has_a_category():
    assert ALL_NSL_KDD_ATTACKS <= set(ATTACK_CATEGORY)
    assert set(ATTACK_CATEGORY.values()) == set(CLASSES)


def test_load_split_rejects_unknown_attack(tmp_path, rng):
    df = make_rows(10, rng)
    df.loc[0, "attack"] = "not_a_real_attack"
    p = tmp_path / "x.txt"
    df.to_csv(p, header=False, index=False)
    with pytest.raises(ValueError, match="not_a_real_attack"):
        load_split(p)


def test_load_split_labels(tmp_path, rng):
    p = tmp_path / "x.txt"
    make_rows(50, rng).to_csv(p, header=False, index=False)
    df = load_split(p)
    assert (df.loc[df["attack"] == "neptune", "category"] == "DoS").all()
    assert (df["is_attack"] == (df["attack"] != "normal")).all()


def test_preprocessor_fits_on_train_only_and_handles_unseen_categories(rng):
    train = make_rows(200, rng)
    test = make_rows(20, rng)
    test.loc[:4, "service"] = "never_seen_service"
    pre = Preprocessor().fit(train)
    assert "num_outbound_cmds" in pre.dropped_
    Xtr, Xte = pre.transform(train), pre.transform(test)
    assert Xtr.shape[1] == Xte.shape[1] == len(pre.feature_names_)
    # Train numeric block is standardised; unseen service gets all-zero one-hot.
    n_num = len(pre.numeric_)
    np.testing.assert_allclose(Xtr[:, :n_num].mean(axis=0), 0, atol=1e-5)
    svc_cols = [i for i, n in enumerate(pre.feature_names_) if n.startswith("service_")]
    assert (Xte[:5][:, svc_cols] == 0).all()
    assert not np.isnan(Xte).any()


def test_class_weights():
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    w = class_weights(y, "balanced", n_classes=3)
    np.testing.assert_allclose(w * np.bincount(y), len(y) / 3)
    s = class_weights(y, "sqrt_balanced", n_classes=3)
    np.testing.assert_allclose(s, np.sqrt(w))
    assert (class_weights(y, "none", n_classes=3) == 1).all()


def test_calibrate_threshold_caps_fpr(rng):
    normal = rng.normal(size=10_000)
    t = calibrate_threshold(normal, 0.02)
    assert np.mean(normal > t) <= 0.02
    assert np.mean(normal > t) > 0.015


def test_binary_metrics_counts():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    f = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    m = binary_metrics(y, f)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 4, 1)
    assert m["detection_rate"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 5)


def test_novel_vs_known():
    names = np.array(["normal", "mscan", "mscan", "neptune", "neptune"])
    is_attack = names != "normal"
    flagged = np.array([0, 1, 0, 1, 1])
    r = novel_vs_known(names, is_attack, flagged, ["mscan"])
    assert r["novel"] == {"n": 2, "detected": 0.5}
    assert r["known"] == {"n": 2, "detected": 1.0}


def test_to_sequences_pads_and_preserves_values():
    X = np.arange(10, dtype=np.float32).reshape(2, 5)
    s = to_sequences(X, 2)
    assert s.shape == (2, 2, 3)
    np.testing.assert_array_equal(s.reshape(2, -1)[:, :5], X)
    assert (s.reshape(2, -1)[:, 5:] == 0).all()
    assert to_sequences(X, 1).shape == (2, 1, 5)


def _separable(rng, n=600):
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = rng.integers(0, len(CLASSES), n)
    X[:, 0] += y * 4  # class shifts feature 0
    return X, y


def test_isolation_forest_never_uses_attack_rows(rng):
    X, y = _separable(rng)
    m = IsolationForestDetector({"n_estimators": 50, "max_samples": 64,
                                 "train_on_normal_only": True, "random_state": 0})
    m.fit(X, y)
    assert m.n_train_ == int((y == 0).sum())
    s = m.attack_score(X)
    assert s[y == 4].mean() > s[y == 0].mean()
    assert m.predict_class(X) is None


def test_xgboost_learns_separable_classes(rng):
    X, y = _separable(rng)
    m = XGBoostDetector({"n_estimators": 50, "max_depth": 3, "learning_rate": 0.3,
                         "subsample": 1.0, "colsample_bytree": 1.0,
                         "early_stopping_rounds": 10, "class_weighting": "balanced",
                         "random_state": 0})
    m.fit(X[:400], y[:400], X[400:], y[400:])
    assert np.mean(m.predict_class(X[400:]) == y[400:]) > 0.9
    assert ((m.attack_score(X) >= 0) & (m.attack_score(X) <= 1)).all()
