"""Download NSL-KDD into data/raw/ and verify every file by SHA-256.

Mirrors are tried in order. A file whose digest does not match the one in
config.yaml is rejected, so a truncated download or a re-packaged mirror can
never silently feed the models. data/raw/ is git-ignored.
"""

import argparse
import hashlib
import sys
from urllib.parse import quote

import requests

from src.config import load_config

CHUNK = 1 << 16


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, timeout=600):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
    tmp.replace(dest)


def ensure_file(spec, mirrors, raw_dir, force=False):
    """Make sure one file is present and verified. Returns its path."""
    dest = raw_dir / spec["name"]
    if dest.exists() and not force:
        if sha256_of(dest) == spec["sha256"]:
            print(f"[ok]       {spec['name']}: already present, digest verified")
            return dest
        print(f"[stale]    {spec['name']}: digest mismatch, re-downloading")

    errors = []
    for base in mirrors:
        url = f"{base}/{quote(spec['name'])}"
        print(f"[download] {spec['name']} <- {url}")
        try:
            fetch(url, dest)
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue
        digest = sha256_of(dest)
        if digest == spec["sha256"]:
            print(f"[ok]       {spec['name']}: {dest.stat().st_size:,} bytes, digest verified")
            return dest
        errors.append(f"{url}: sha256 {digest} != expected {spec['sha256']}")
        dest.unlink()

    raise RuntimeError(f"could not obtain {spec['name']}:\n  " + "\n  ".join(errors))


def download_all(cfg, force=False):
    raw_dir = cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    mirrors = cfg["data"]["mirrors"]
    return {
        key: ensure_file(spec, mirrors, raw_dir, force=force)
        for key, spec in cfg["data"]["files"].items()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    try:
        download_all(load_config(), force=args.force)
    except RuntimeError as exc:
        print(f"[fail]     {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
