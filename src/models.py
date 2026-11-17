"""The three detectors behind one interface.

Every model exposes:
  fit(X_train, y_train, X_val, y_val)
  attack_score(X) -> float array, higher = more likely an attack (binary view)
  default_flag(X) -> bool array, the model's own out-of-the-box decision
  predict_class(X) -> int array over CLASSES, or None if the model can't
"""

import os

import numpy as np
from sklearn.ensemble import IsolationForest

from src.preprocess import CLASSES, class_weights

NORMAL = CLASSES.index("Normal")


class IsolationForestDetector:
    """Unsupervised. Learns what normal traffic looks like; never sees attack labels."""

    name = "Isolation Forest"
    supervised = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = IsolationForest(
            n_estimators=cfg["n_estimators"],
            max_samples=cfg["max_samples"],
            contamination="auto",
            random_state=cfg["random_state"],
            n_jobs=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X = X_train[y_train == NORMAL] if self.cfg["train_on_normal_only"] else X_train
        self.model.fit(X)
        self.n_train_ = len(X)
        return self

    def attack_score(self, X):
        # score_samples is high for inliers; negate so high = anomalous.
        return -self.model.score_samples(X)

    def default_flag(self, X):
        return self.model.predict(X) == -1

    def predict_class(self, X):
        return None


class XGBoostDetector:
    """Supervised 5-class gradient-boosted trees."""

    name = "XGBoost"
    supervised = True

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, X_train, y_train, X_val, y_val):
        from xgboost import XGBClassifier

        c = self.cfg
        self.model = XGBClassifier(
            n_estimators=c["n_estimators"],
            max_depth=c["max_depth"],
            learning_rate=c["learning_rate"],
            subsample=c["subsample"],
            colsample_bytree=c["colsample_bytree"],
            objective="multi:softprob",
            num_class=len(CLASSES),
            eval_metric="mlogloss",
            early_stopping_rounds=c["early_stopping_rounds"],
            tree_method="hist",
            random_state=c["random_state"],
            n_jobs=-1,
        )
        w = class_weights(y_train, c["class_weighting"])
        self.model.fit(
            X_train, y_train, sample_weight=w[y_train],
            eval_set=[(X_val, y_val)], verbose=False,
        )
        self.best_iteration_ = int(self.model.best_iteration)
        return self

    def proba(self, X):
        return self.model.predict_proba(X)

    def attack_score(self, X):
        return 1.0 - self.proba(X)[:, NORMAL]

    def default_flag(self, X):
        return self.predict_class(X) != NORMAL

    def predict_class(self, X):
        return self.proba(X).argmax(axis=1)

    def save(self, path):
        self.model.save_model(str(path))


def to_sequences(X, timesteps):
    """Reshape (n, d) rows into (n, timesteps, ceil(d / timesteps)), zero-padded.

    With timesteps=1 each connection is a one-step sequence. NSL-KDD has no
    temporal order between rows, so this is the honest default; larger values
    just slice the feature vector into chunks, which is an architectural choice,
    not a time axis.
    """
    n, d = X.shape
    width = -(-d // timesteps)
    pad = width * timesteps - d
    if pad:
        X = np.hstack([X, np.zeros((n, pad), dtype=X.dtype)])
    return X.reshape(n, timesteps, width)


class LSTMDetector:
    """Supervised 5-class stacked LSTM (TensorFlow / Keras)."""

    name = "LSTM"
    supervised = True

    def __init__(self, cfg):
        self.cfg = cfg

    def _build(self, input_shape):
        import tensorflow as tf

        c = self.cfg
        layers = [tf.keras.Input(shape=input_shape)]
        for i, units in enumerate(c["units"]):
            last = i == len(c["units"]) - 1
            layers.append(tf.keras.layers.LSTM(units, return_sequences=not last))
            layers.append(tf.keras.layers.Dropout(c["dropout"]))
        layers.append(tf.keras.layers.Dense(len(CLASSES), activation="softmax"))
        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(c["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        c = self.cfg
        tf.keras.utils.set_random_seed(c["random_state"])
        tf.config.experimental.enable_op_determinism()

        Xs_train = to_sequences(X_train, c["timesteps"])
        Xs_val = to_sequences(X_val, c["timesteps"])
        self.model = self._build(Xs_train.shape[1:])
        w = class_weights(y_train, c["class_weighting"])
        stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=c["early_stopping_patience"],
            restore_best_weights=True,
        )
        hist = self.model.fit(
            Xs_train, y_train,
            validation_data=(Xs_val, y_val),
            epochs=c["epochs"], batch_size=c["batch_size"],
            class_weight={i: float(v) for i, v in enumerate(w)},
            callbacks=[stop], verbose=2,
        )
        val_loss = hist.history["val_loss"]
        self.epochs_run_ = len(val_loss)
        self.best_epoch_ = int(np.argmin(val_loss)) + 1
        return self

    def proba(self, X):
        return self.model.predict(
            to_sequences(X, self.cfg["timesteps"]), batch_size=4096, verbose=0
        )

    def attack_score(self, X):
        return 1.0 - self.proba(X)[:, NORMAL]

    def default_flag(self, X):
        return self.predict_class(X) != NORMAL

    def predict_class(self, X):
        return self.proba(X).argmax(axis=1)

    def save(self, path):
        self.model.save(str(path))


def build_models(cfg):
    return [
        IsolationForestDetector(cfg["isolation_forest"]),
        XGBoostDetector(cfg["xgboost"]),
        LSTMDetector(cfg["lstm"]),
    ]
