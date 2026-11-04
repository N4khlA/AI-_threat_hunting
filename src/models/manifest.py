"""Model-artifact manifest: SHA-256 pinning of everything the API loads.

``joblib.load`` unpickles, i.e. executes, whatever is in the file. A
model file swapped on disk (supply chain, shared volume, a compromised
retrain job) is code execution in the serving process. And the RF /
pipeline desync incident (docs/limitations.md section 3) showed that a
model is meaningless without the pipeline that produced it.

So every training step rewrites ``models/manifest.json`` (file digests,
sizes, split protocol, git commit), and the API refuses to start if any
artifact is missing from the manifest or does not match its digest.

This protects integrity between training and serving. It does not make
pickle safe against an attacker who can rewrite both the file *and* the
manifest - for that, sign the manifest (e.g. with sigstore/cosign) in
the deployment pipeline.

Usage:
    python src/models/manifest.py            # (re)write the manifest
    python src/models/manifest.py --verify   # exit 1 on any mismatch
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_NAME = "manifest.json"
ARTIFACT_SUFFIXES = {".joblib", ".keras", ".json"}


class IntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,  # nosec B603 B607
                             capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_manifest(models_dir: Path = MODELS_DIR) -> Path:
    files = {}
    for path in sorted(models_dir.iterdir()):
        if path.name == MANIFEST_NAME or path.suffix not in ARTIFACT_SUFFIXES:
            continue
        files[path.name] = {"sha256": sha256_file(path),
                            "bytes": path.stat().st_size}
    meta_path = models_dir / "preprocess_metadata.json"
    protocol = None
    if meta_path.exists():
        protocol = json.loads(meta_path.read_text(encoding="utf-8")).get("split_protocol")
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "split_protocol": protocol,
        "files": files,
    }
    out = models_dir / MANIFEST_NAME
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] manifest -> {out} ({len(files)} artifacts)")
    return out


def verify_manifest(models_dir: Path, required: list[str]) -> dict:
    """Raise IntegrityError unless every ``required`` file matches the manifest."""
    path = models_dir / MANIFEST_NAME
    if not path.exists():
        raise IntegrityError(f"{path} missing - run src/models/manifest.py after training")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name in required:
        entry = manifest["files"].get(name)
        if entry is None:
            raise IntegrityError(f"{name} is not pinned in the manifest")
        actual = sha256_file(models_dir / name)
        if not hashes_equal(actual, entry["sha256"]):
            raise IntegrityError(f"{name} does not match its pinned sha256")
    return manifest


def hashes_equal(a: str, b: str) -> bool:
    import hmac  # noqa: PLC0415

    return hmac.compare_digest(a, b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        names = [p.name for p in args.models_dir.iterdir()
                 if p.suffix in ARTIFACT_SUFFIXES and p.name != MANIFEST_NAME]
        try:
            verify_manifest(args.models_dir, names)
        except IntegrityError as exc:
            sys.exit(f"[x] {exc}")
        print(f"[ok] {len(names)} artifacts match the manifest")
    else:
        write_manifest(args.models_dir)


if __name__ == "__main__":
    main()
