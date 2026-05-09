"""MaxSim attribution: for a (query, doc) pair, identify which document
tokens each query token "matched" (the argmax in MaxSim), then look up
SAE features that fired on those matched document tokens.

The output is the per-feature contribution to the MaxSim score: for each
SAE feature f, sum the MaxSim score (q_t · d_s*) over all (q_t, d_s*)
pairs where d_s* fires feature f. Features that drove the match come
out at the top.

Inputs (from BEIR cache + SAE checkpoint):
  - q_vecs, q_offsets: cached ColBERT query token vectors (one query)
  - d_vecs, d_offsets: cached ColBERT corpus document token vectors
  - SAE checkpoint dir
  - feature labels (for human-readable display)

Usage as a library:
    from maxsim_attribution import attribute_maxsim
    result = attribute_maxsim(q_tokens, d_tokens, q_text, d_text, sae)

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


def maxsim_with_indices(q_tokens: np.ndarray, d_tokens: np.ndarray,
                         device: str = "cuda") -> tuple[float, np.ndarray, np.ndarray]:
    """ColBERT-style MaxSim score plus the matched (q→d) token indices.

    Returns:
        score: float, sum over q_t of max_s <q_t, d_s>
        matched_doc_idx: (n_q,) — for each query token, the doc token index
                          chosen as its argmax
        per_q_max: (n_q,) — the actual max similarity per query token
    """
    Q = torch.from_numpy(np.ascontiguousarray(q_tokens, dtype=np.float32)).to(device)
    D = torch.from_numpy(np.ascontiguousarray(d_tokens, dtype=np.float32)).to(device)
    sims = Q @ D.T              # (n_q, n_d)
    per_q_max, matched = sims.max(dim=1)
    return (float(per_q_max.sum().item()),
            matched.cpu().numpy(),
            per_q_max.cpu().numpy())


@torch.no_grad()
def sae_features_at_tokens(d_tokens: np.ndarray, indices: np.ndarray,
                           sae: Sae, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """For document token indices, get the top-k SAE features at each.
    Returns (top_indices, top_acts) shaped (len(indices), k)."""
    sae.eval()
    sub = d_tokens[indices].astype(np.float32)
    batch = torch.from_numpy(np.ascontiguousarray(sub)).to(device)
    out = sae(batch)
    return out.latent_indices.cpu().numpy(), out.latent_acts.cpu().numpy()


def attribute_maxsim(q_tokens: np.ndarray, d_tokens: np.ndarray,
                     sae: Sae, top_n_features: int = 20,
                     device: str = "cuda") -> dict:
    """Full attribution. Returns dict with score, matched_pairs, ranked
    feature contributions."""
    score, matched_idx, per_q_max = maxsim_with_indices(
        q_tokens, d_tokens, device=device)

    # SAE features at the matched doc tokens
    feat_idxs, feat_acts = sae_features_at_tokens(
        d_tokens, matched_idx, sae, device=device)
    # feat_idxs: (n_q, k), feat_acts: (n_q, k)

    # Per-feature contributions, four flavors:
    #   sum_attribution: Σ_t per_q_max[t] * act[t, f]
    #   max_attribution: max_t per_q_max[t] * act[t, f]   ← spike-style features
    #   binary_score:    Σ_t per_q_max[t] (where f fires; ignore activation magnitude)
    #   q_count:         number of query-token positions where f was in the top-k
    feat_sum: dict[int, float] = defaultdict(float)
    feat_max: dict[int, float] = defaultdict(float)
    feat_max_qtok: dict[int, int] = {}
    feat_bin: dict[int, float] = defaultdict(float)
    feat_q_count: dict[int, int] = defaultdict(int)
    for t in range(matched_idx.shape[0]):
        qmax = float(per_q_max[t])
        for k in range(feat_idxs.shape[1]):
            f = int(feat_idxs[t, k])
            a = float(feat_acts[t, k])
            contrib = qmax * a
            feat_sum[f] += contrib
            if contrib > feat_max[f]:
                feat_max[f] = contrib
                feat_max_qtok[f] = t
            feat_bin[f] += qmax
            feat_q_count[f] += 1

    # We default ranking by sum_attribution (the previous behavior) but
    # callers can re-sort by feat["max_attribution"] for spike-style features.
    ranked = sorted(feat_sum.items(), key=lambda x: -x[1])

    return {
        "score": score,
        "n_query_tokens": int(matched_idx.shape[0]),
        "n_doc_tokens": int(d_tokens.shape[0]),
        "matched_doc_idx": matched_idx.tolist(),
        "per_q_max": per_q_max.tolist(),
        "ranked_features": [
            {"feature": f,
             "sum_attribution": feat_sum[f],
             "max_attribution": feat_max[f],
             "max_q_token_idx": feat_max_qtok.get(f, -1),
             "binary_score": feat_bin[f],
             "n_q_tokens": feat_q_count[f]}
            for f, _ in ranked[:max(top_n_features, 200)]
        ],
    }


def find_sae_ckpt(run_dir: Path) -> Path | None:
    for p in sorted((run_dir / "checkpoints").glob("*"), key=lambda x: x.name):
        if p.is_dir() and (p / "cfg.json").exists():
            return p
    return None


def load_packed(cache_dir: Path, kind: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vecs = np.load(cache_dir / f"{kind}_vectors.npy", mmap_mode="r")
    offsets = np.load(cache_dir / f"{kind}_offsets.npy")
    ids = json.loads((cache_dir / f"{kind}_ids.json").read_text())
    return vecs, offsets, ids


def get_query_tokens(q_vecs, q_offsets, q_ids, qid: str) -> np.ndarray:
    i = q_ids.index(qid)
    return np.asarray(q_vecs[q_offsets[i]:q_offsets[i + 1]])


def get_doc_tokens(d_vecs, d_offsets, d_ids, did: str) -> np.ndarray:
    i = d_ids.index(did)
    return np.asarray(d_vecs[d_offsets[i]:d_offsets[i + 1]])


def render_doc_tokens_around(d_text: str, doc_token_idx: int,
                              tokenizer, radius: int = 5) -> tuple[str, str]:
    """Tokenize doc_text and return a windowed view around doc_token_idx,
    with the matched token highlighted via <<>>."""
    tokens = tokenizer.tokenize(d_text)
    if doc_token_idx < 0 or doc_token_idx >= len(tokens):
        return ("", tokens[doc_token_idx] if 0 <= doc_token_idx < len(tokens) else "")
    lo = max(0, doc_token_idx - radius)
    hi = min(len(tokens), doc_token_idx + radius + 1)
    pieces = [tokens[i].replace("##", "") for i in range(lo, hi)]
    pieces[doc_token_idx - lo] = f"<<{pieces[doc_token_idx - lo]}>>"
    return (" ".join(pieces), tokens[doc_token_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--colbert-cache", required=True,
                    help="e.g. /data/embeddings/beir/trec-covid-mxbai-edge-32m")
    ap.add_argument("--qid", required=True)
    ap.add_argument("--did", required=True)
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--filter", default="all",
                    choices=["all", "interpretable", "non-generic"],
                    help="filter attributed features by labeler bucket")
    ap.add_argument("--show-matched", action="store_true",
                    help="print each query token and its matched doc token text")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rd = Path(args.sae_dir)
    ckpt = find_sae_ckpt(rd)
    sae = Sae.load_from_disk(ckpt, device=args.device)

    cache = Path(args.colbert_cache)
    qv, qo, q_ids = load_packed(cache, "queries")
    dv, do, d_ids = load_packed(cache, "corpus")

    q_tok = get_query_tokens(qv, qo, q_ids, args.qid)
    d_tok = get_doc_tokens(dv, do, d_ids, args.did)
    print(f"q={args.qid}: {q_tok.shape[0]} tokens; d={args.did}: {d_tok.shape[0]} tokens")

    result = attribute_maxsim(q_tok, d_tok, sae, top_n_features=args.top_n,
                               device=args.device)
    print(f"\nMaxSim score: {result['score']:.3f}")
    print(f"per-q max: min={min(result['per_q_max']):.3f} "
          f"max={max(result['per_q_max']):.3f} "
          f"mean={sum(result['per_q_max'])/len(result['per_q_max']):.3f}")

    labels = {}
    p = rd / "feature_labels.json"
    if p.exists():
        labels = json.loads(p.read_text())["labels"]

    # Pull the actual texts
    from datasets import load_dataset
    cfg = {"trec-covid": "mteb/trec-covid", "scifact": "mteb/scifact"}[args.dataset]
    queries_ds = load_dataset(cfg, "queries", split="queries")
    corpus_ds = load_dataset(cfg, "corpus", split="corpus")
    q_text = next(r["text"] for r in queries_ds if r["_id"] == args.qid)
    d_row = next(r for r in corpus_ds if r["_id"] == args.did)
    d_text = (d_row.get("title", "") + " " + d_row["text"]).strip()
    print(f"\nQUERY:\n  {q_text}")
    print(f"\nDOC (first 200 chars):\n  {d_text[:200]}...")

    if args.show_matched:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        q_tokens_str = tok.tokenize(q_text)
        d_tokens_str = tok.tokenize(d_text)
        print(f"\n=== Matched query→doc tokens (by ColBERT MaxSim) ===")
        print(f"{'q#':>3} {'q_token':<14} {'sim':>5}  d#  matched doc context")
        for t, (didx, qmax) in enumerate(zip(result['matched_doc_idx'],
                                              result['per_q_max'])):
            qtok = q_tokens_str[t] if t < len(q_tokens_str) else f"[t{t}]"
            ctx, dtok = render_doc_tokens_around(d_text, didx, tok, radius=4)
            print(f"{t:>3} {qtok:<14} {qmax:>5.2f}  {didx:>3}  {ctx[:90]}")

    keep = {"COHERENT", "THEMATIC", "GENERIC", "POLYSEMANTIC", "UNCLEAR"}
    if args.filter == "interpretable":
        keep = {"COHERENT", "THEMATIC"}
    elif args.filter == "non-generic":
        keep = {"COHERENT", "THEMATIC", "POLYSEMANTIC", "UNCLEAR"}

    filtered = [fr for fr in result["ranked_features"]
                if labels.get(str(fr["feature"]), {}).get("judgment", "") in keep]
    n_show = min(args.top_n, len(filtered))
    print(f"\n=== Top {n_show} features attributing to MaxSim "
          f"(filter={args.filter}) ===")
    print(f"{'rank':>4} {'fid':>7} {'act_w_score':>12} {'q_toks':>7}  description")
    for i in range(n_show):
        fr = filtered[i]
        fid = fr["feature"]
        lab = labels.get(str(fid), {})
        desc = (lab.get("description", "(no label)") or "")[:120]
        judg = lab.get("judgment", "")[:4]
        print(f"{i+1:>4} {fid:>7} {fr['act_weighted_score']:>12.3f} "
              f"{fr['n_q_tokens']:>7}  [{judg}] {desc}")


if __name__ == "__main__":
    main()
