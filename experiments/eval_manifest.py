"""Held-out eval manifests (V1 of the SAE validation plan).

Every embedding dataset that we eval against gets a JSON manifest recording
which shard row-ranges are reserved for eval. Eval scripts call
`assert_eval_slice` instead of trusting ad-hoc offsets ("just past the 100M
training slice"), so train/eval leakage becomes an explicit, checked claim
rather than a convention.

Manifests live at /data/eval_manifests/<dataset-slug>.json, where the slug is
the dataset directory name (the parent of `train/`). Override the root with
the SAE_EVAL_MANIFEST_DIR env var (used by tests).

Honesty matters in the `kind` field: a range is only "held-out" if training
never sees those rows. Ranges that overlap the training distribution (e.g.
first-10K-chunk labeling scans while OTF training streams the same parquet
dirs) must be marked "scan-set (training-overlapped)" — they are standardized
scan sets, not clean held-out data.

Escape hatch: SAE_EVAL_NO_MANIFEST=1 downgrades assertion failures to a loud
warning, for backwards compat with datasets that predate manifests.

Usage:
    python -m experiments.eval_manifest create \\
        --data-dir /data/embeddings/<dataset>/train \\
        --shard data-00000.npy --rows 0:2338156 \\
        --kind held-out --note "why this range is reserved"
    python -m experiments.eval_manifest show --data-dir /data/embeddings/<dataset>/train
"""
import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1
DEFAULT_MANIFEST_DIR = "/data/eval_manifests"
KINDS = ("held-out", "scan-set (training-overlapped)", "eval-only")


class EvalManifestError(RuntimeError):
    """Raised when an eval slice is not covered by a reserved range."""


def _manifest_dir() -> Path:
    return Path(os.environ.get("SAE_EVAL_MANIFEST_DIR", DEFAULT_MANIFEST_DIR))


def dataset_slug(data_dir) -> str:
    """Dataset directory name; the parent of a trailing `train`/`test` split dir."""
    d = Path(data_dir).resolve()
    if d.name in ("train", "test", "val", "validation"):
        return d.parent.name
    return d.name


def manifest_path(data_dir) -> Path:
    return _manifest_dir() / f"{dataset_slug(data_dir)}.json"


def load_manifest(data_dir) -> dict:
    """Load the manifest for data_dir. Raises EvalManifestError if missing."""
    path = manifest_path(data_dir)
    if not path.exists():
        raise EvalManifestError(
            f"no eval manifest for {data_dir}\n"
            f"  expected: {path}\n"
            f"  Create one with:\n"
            f"    python -m experiments.eval_manifest create --data-dir {data_dir} "
            f"--shard <shard.npy> --rows START:END --note '...'\n"
            f"  or set SAE_EVAL_NO_MANIFEST=1 to bypass (warns loudly)."
        )
    manifest = json.loads(path.read_text())
    recorded = manifest.get("data_dir")
    if recorded and Path(recorded).resolve() != Path(data_dir).resolve():
        raise EvalManifestError(
            f"manifest {path} records data_dir={recorded}, "
            f"but was loaded for {data_dir} (dataset-slug collision?)"
        )
    return manifest


def assert_eval_slice(data_dir, shard, start: int, end: int) -> dict:
    """Assert rows [start, end) of `shard` in data_dir are reserved for eval.

    Returns the matching reserved-range entry on success. Raises
    EvalManifestError if no manifest exists or the slice is not fully inside a
    reserved range. SAE_EVAL_NO_MANIFEST=1 downgrades to a loud warning and
    returns {}.
    """
    shard = Path(shard).name
    if end <= start:
        raise ValueError(f"empty slice: [{start}, {end})")
    try:
        manifest = load_manifest(data_dir)
        for entry in manifest.get("reserved", []):
            r0, r1 = entry["rows"]
            if entry["shard"] == shard and r0 <= start and end <= r1:
                kind = entry.get("kind", "?")
                if kind != "held-out":
                    print(f"[eval_manifest] NOTE: slice {shard}[{start}:{end}] is "
                          f"reserved as {kind!r}, not clean held-out data")
                return entry
        ranges = [f"{e['shard']}[{e['rows'][0]}:{e['rows'][1]}] ({e.get('kind', '?')})"
                  for e in manifest.get("reserved", [])]
        raise EvalManifestError(
            f"eval slice {shard}[{start}:{end}] in {data_dir} is NOT inside any "
            f"reserved eval range.\n"
            f"  manifest: {manifest_path(data_dir)}\n"
            f"  reserved: {ranges or '(none)'}\n"
            f"  Either eval against a reserved range, add one with "
            f"`python -m experiments.eval_manifest create`, or set "
            f"SAE_EVAL_NO_MANIFEST=1 to bypass (warns loudly)."
        )
    except EvalManifestError as e:
        if os.environ.get("SAE_EVAL_NO_MANIFEST") == "1":
            print("=" * 78, file=sys.stderr)
            print(f"[eval_manifest] WARNING (SAE_EVAL_NO_MANIFEST=1): {e}",
                  file=sys.stderr)
            print("[eval_manifest] WARNING: proceeding WITHOUT verified eval "
                  "reservation — results may leak training data.", file=sys.stderr)
            print("=" * 78, file=sys.stderr)
            return {}
        raise


def create_entry(data_dir, shard, start: int, end: int, kind: str, note: str,
                 created_by: str | None = None) -> Path:
    """Add a reserved range to the manifest for data_dir (creating it if new)."""
    if end <= start:
        raise ValueError(f"empty range: [{start}, {end})")
    path = manifest_path(data_dir)
    data_dir = str(Path(data_dir).resolve())
    if path.exists():
        manifest = json.loads(path.read_text())
        if Path(manifest.get("data_dir", data_dir)).resolve() != Path(data_dir).resolve():
            raise EvalManifestError(
                f"manifest {path} already exists for a different data_dir: "
                f"{manifest.get('data_dir')}"
            )
    else:
        manifest = {
            "version": MANIFEST_VERSION,
            "data_dir": data_dir,
            "reserved": [],
        }
    manifest["reserved"].append({
        "shard": Path(shard).name,
        "rows": [int(start), int(end)],
        "kind": kind,
        "note": note,
        "created_by": created_by or getpass.getuser(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    apc = sub.add_parser("create", help="add a reserved eval range")
    apc.add_argument("--data-dir", required=True,
                     help="embedding dir, e.g. /data/embeddings/<dataset>/train")
    apc.add_argument("--shard", required=True, help="shard filename, e.g. data-00000.npy")
    apc.add_argument("--rows", required=True,
                     help="reserved row range START:END (END exclusive)")
    apc.add_argument("--kind", default="held-out", choices=KINDS,
                     help="be honest: only 'held-out' if training never sees these rows")
    apc.add_argument("--note", required=True, help="free-text provenance")
    apc.add_argument("--created-by", default=None,
                     help="who is reserving this (default: $USER)")

    aps = sub.add_parser("show", help="print the manifest for a dataset")
    aps.add_argument("--data-dir", required=True)

    args = ap.parse_args()

    if args.cmd == "create":
        try:
            start_s, end_s = args.rows.split(":")
            start, end = int(start_s), int(end_s)
        except ValueError:
            ap.error(f"--rows must be START:END with integer rows, got {args.rows!r}")
        path = create_entry(args.data_dir, args.shard, start, end,
                            kind=args.kind, note=args.note,
                            created_by=args.created_by)
        print(f"wrote {path}")
        print(json.dumps(json.loads(path.read_text()), indent=2))
    elif args.cmd == "show":
        try:
            manifest = load_manifest(args.data_dir)
        except EvalManifestError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        print(f"manifest: {manifest_path(args.data_dir)}")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
