"""Load NSL-KDD, map attack names to the five categories, and build matrices.

Every transform (one-hot vocabulary, which columns to drop, log scaling,
standardisation) is fitted on the training split only and then applied to
validation and test. The test sets never influence preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["attack", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]

# Class order is fixed so label indices mean the same thing everywhere.
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Standard NSL-KDD attack -> category mapping. The test set contains 17 attack
# types that never appear in training; they are mapped here too, because the
# models must be scored on them. The few ambiguous ones (worm, httptunnel)
# follow the most common published convention.
ATTACK_CATEGORY = {
    "normal": "Normal",
    # Denial of service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    # Probing / scanning
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # Remote to local
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    # User to root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R",
}

# Columns that are already rates in [0, 1] or binary flags; everything else
# numeric is a heavy-tailed count and gets log1p.
RATE_OR_FLAG = {c for c in FEATURES if c.endswith("_rate")} | {
    "land", "logged_in", "root_shell", "is_host_login", "is_guest_login",
}


def load_split(path):
    """Read one NSL-KDD file and attach its category and binary label."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    unknown = sorted(set(df["attack"]) - set(ATTACK_CATEGORY))
    if unknown:
        raise ValueError(f"{path}: attack names with no category mapping: {unknown}")
    df["category"] = df["attack"].map(ATTACK_CATEGORY)
    df["y"] = df["category"].map(CLASSES.index).astype(int)
    df["is_attack"] = (df["category"] != "Normal").astype(int)
    return df


class Preprocessor:
    """One-hot + log1p + standardise, fitted on training rows only."""

    def fit(self, df):
        numeric = [c for c in FEATURES if c not in CATEGORICAL]
        # Drop columns that are constant in training (num_outbound_cmds is
        # always 0 in NSL-KDD). They carry no information and break scaling.
        self.numeric_ = [c for c in numeric if df[c].nunique() > 1]
        self.dropped_ = [c for c in numeric if c not in self.numeric_]
        self.log_cols_ = [c for c in self.numeric_ if c not in RATE_OR_FLAG]

        # Categories seen only at test time (new services) become all-zero.
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(df[CATEGORICAL])

        self.scaler_ = StandardScaler().fit(self._numeric_block(df))
        self.feature_names_ = self.numeric_ + list(
            self.encoder_.get_feature_names_out(CATEGORICAL)
        )
        return self

    def _numeric_block(self, df):
        block = df[self.numeric_].astype(float).copy()
        block[self.log_cols_] = np.log1p(block[self.log_cols_].clip(lower=0))
        return block.to_numpy()

    def transform(self, df):
        num = self.scaler_.transform(self._numeric_block(df))
        cat = self.encoder_.transform(df[CATEGORICAL])
        return np.hstack([num, cat]).astype(np.float32)


def split_train_val(df, val_fraction, random_state):
    """Stratified hold-out by category so U2R appears in both halves."""
    return train_test_split(
        df, test_size=val_fraction, stratify=df["y"], random_state=random_state
    )


def class_weights(y, mode, n_classes=len(CLASSES)):
    """Per-class weights for imbalanced training.

    "balanced" is the usual n / (k * count). "sqrt_balanced" takes its square
    root, which still lifts rare classes but avoids pushing 52 U2R rows to a
    weight of several hundred, where they mostly teach the model to overfit.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    if mode == "none":
        return np.ones(n_classes)
    w = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    if mode == "sqrt_balanced":
        w = np.sqrt(w)
    elif mode != "balanced":
        raise ValueError(f"unknown class_weighting: {mode}")
    return w


def novel_attack_names(train_df, test_df):
    """Attack types that occur in a test set but never in training."""
    return sorted(set(test_df["attack"]) - set(train_df["attack"]))
