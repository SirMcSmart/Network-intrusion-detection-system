"""Configuration loading shared by every pipeline stage."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path=None):
    """Read config.yaml and resolve its relative paths against the repo root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    cfg["data"]["raw_dir"] = REPO_ROOT / cfg["data"]["raw_dir"]
    cfg["output"]["dir"] = REPO_ROOT / cfg["output"]["dir"]
    cfg["output"]["models_dir"] = REPO_ROOT / cfg["output"]["models_dir"]
    return cfg
