"""Pool SAE features over the tokens of a document or query.

For each token in the input text, ColBERT produces a 64-dim vector. The
SAE encoder turns each token vector into a top-k sparse activation
across feature directions. This module aggregates those per-token
activations into a single per-document "feature fingerprint" — useful
for asking which features ColBERT MaxSim is actually relying on for
retrieval, and for browsing documents in latent-taxonomy by their
feature signature.

Aggregation modes:
  - max:   per feature, max activation across tokens that fired it
  - sum:   per feature, sum of activations across tokens
  - count: per feature, number of tokens that fired it (no magnitude)

Usage as a library:
    from feature_pool import pool_text
    fp = pool_text("a query about COVID treatment", sae, colbert_model)

CLI: see __main__.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402


COLBERT_MODEL_ID = "mixedbread-ai/mxbai-edge-colbert-v0-32m"


def pool_token_acts(token_vecs: torch.Tensor, sae: Sae,
                    agg: str = "max") -> tuple[np.ndarray, np.ndarray]:
    """Run token vectors through the SAE encoder and pool top-k features.

    Args:
        token_vecs: (n_tokens, d_in) tensor on the sae's device
        sae: a loaded Sae checkpoint (matryoshka or otherwise)
        agg: 'max', 'sum', or 'count'

    Returns:
        (feature_ids, agg_values) — sorted descending by agg_values
    """
    sae.eval()
    with torch.no_grad():
        out = sae(token_vecs.to(sae.dtype).to(sae.device))
        # latent_acts: (n_tokens, k) — top-k activations
        # latent_indices: (n_tokens, k) — which features
        acts = out.latent_acts.cpu().numpy()
        idxs = out.latent_indices.cpu().numpy()

    pooled: dict[int, float] = defaultdict(float)
    pooled_count: dict[int, int] = defaultdict(int)
    for t in range(idxs.shape[0]):
        for k in range(idxs.shape[1]):
            f = int(idxs[t, k])
            v = float(acts[t, k])
            pooled_count[f] += 1
            if agg == "max":
                pooled[f] = max(pooled[f], v)
            elif agg == "sum":
                pooled[f] += v
            elif agg == "count":
                pooled[f] = pooled_count[f]

    if not pooled:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    feats = np.array(list(pooled.keys()), dtype=np.int64)
    vals = np.array([pooled[f] for f in feats], dtype=np.float32)
    order = np.argsort(-vals)
    return feats[order], vals[order]


def pool_text(text: str, sae: Sae, colbert_model, is_query: bool = True,
              agg: str = "max") -> tuple[np.ndarray, np.ndarray]:
    """End-to-end: text → ColBERT tokens → SAE → pooled feature fingerprint."""
    emb = colbert_model.encode([text], batch_size=1, show_progress_bar=False,
                                is_query=is_query)
    if isinstance(emb, list):
        token_vecs = torch.from_numpy(np.asarray(emb[0], dtype=np.float32))
    else:
        token_vecs = torch.from_numpy(emb[0].astype(np.float32))
    return pool_token_acts(token_vecs, sae, agg=agg)


def pool_cached(vecs: np.ndarray, offsets: np.ndarray, sae: Sae,
                agg: str = "max", batch_size: int = 256) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pool features for each item in a packed (vecs, offsets) cache.

    Returns: list of (feat_ids, agg_values) per item.
    """
    n = len(offsets) - 1
    results = []
    for i in range(n):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            results.append((np.array([], dtype=np.int64),
                            np.array([], dtype=np.float32)))
            continue
        token_vecs = torch.from_numpy(np.ascontiguousarray(vecs[s:e],
                                                            dtype=np.float32))
        feats, vals = pool_token_acts(token_vecs, sae, agg=agg)
        results.append((feats, vals))
    return results


def find_sae_ckpt(run_dir: Path) -> Path | None:
    for p in sorted((run_dir / "checkpoints").glob("*"), key=lambda x: x.name):
        if p.is_dir() and (p / "cfg.json").exists():
            return p
    return None


def load_labels(run_dir: Path) -> dict | None:
    """Load the canonical labels for a run (cheap or gold, whatever's symlinked)."""
    p = run_dir / "feature_labels.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True,
                    help="run dir for the SAE (must have checkpoints/<...>)")
    ap.add_argument("--text", default=None,
                    help="ad-hoc text to pool (encodes via ColBERT live)")
    ap.add_argument("--query", action="store_true",
                    help="encode as ColBERT query (default: document)")
    ap.add_argument("--agg", default="max", choices=["max", "sum", "count"])
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--filter", default="all",
                    choices=["all", "interpretable", "coh", "thm", "non-generic"],
                    help="filter pooled features by labeler bucket")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rd = Path(args.sae_dir)
    ckpt = find_sae_ckpt(rd)
    if not ckpt:
        raise SystemExit(f"no SAE checkpoint in {rd}")
    print(f"loading SAE from {ckpt.name}")
    sae = Sae.load_from_disk(ckpt, device=args.device)
    labels = load_labels(rd) or {}

    if args.text:
        from pylate import models
        print(f"loading ColBERT {COLBERT_MODEL_ID}")
        cb = models.ColBERT(model_name_or_path=COLBERT_MODEL_ID, device=args.device)
        feats, vals = pool_text(args.text, sae, cb, is_query=args.query, agg=args.agg)

        keep = {"COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"}
        if args.filter == "interpretable":
            keep = {"COHERENT", "THEMATIC"}
        elif args.filter == "coh":
            keep = {"COHERENT"}
        elif args.filter == "thm":
            keep = {"THEMATIC"}
        elif args.filter == "non-generic":
            keep = {"COHERENT", "THEMATIC", "POLYSEMANTIC", "UNCLEAR"}

        filtered = [(int(f), float(v)) for f, v in zip(feats, vals)
                    if labels.get(str(int(f)), {}).get("judgment", "") in keep]
        n_show = min(args.top_n, len(filtered))
        print(f"\n=== Top {n_show} pooled features (filter={args.filter}, {args.agg}) ===\n")
        print(f"text: {args.text[:200]}")
        n_total = len(feats)
        print(f"\n{n_total} unique features fired total; "
              f"{len(filtered)} pass filter '{args.filter}'.")
        print(f"\n{'rank':>4} {'fid':>7} {'agg':>8}  description")
        for i in range(n_show):
            fid, v = filtered[i]
            lab = labels.get(str(fid), {})
            desc = (lab.get("description", "(no description)") or "")[:140]
            judg = lab.get("judgment", "")
            tag = f"[{judg[:4]}]"
            print(f"{i+1:>4} {fid:>7} {v:>8.3f}  {tag} {desc}")
    else:
        print("Provide --text to pool features for an ad-hoc input.")


if __name__ == "__main__":
    main()
