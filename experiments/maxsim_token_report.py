"""Per-query, per-token MaxSim attribution report.

For a (qid, did) pair: for each query token, show the matched doc token
along with the top SAE features that fire at that exact doc token. This
is the most direct "what is ColBERT picking up at this position?" view —
features are tied to specific (q→d) matches rather than aggregated over
the whole doc.

Output: pretty-printed per-token table + a JSON report you can post-
process (e.g. into HTML).

Usage:
    python -m experiments.maxsim_token_report \\
        --sae-dir <17r dir> \\
        --colbert-cache /data/embeddings/beir/trec-covid-mxbai-edge-32m \\
        --pairs default
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402
from experiments.maxsim_attribution import (  # noqa: E402
    maxsim_with_indices, sae_features_at_tokens,
    find_sae_ckpt, load_packed,
    get_query_tokens, get_doc_tokens,
)
from experiments.attribute_query_set import DEFAULT_PAIRS  # noqa: E402


def per_token_features(q_tokens: np.ndarray, d_tokens: np.ndarray,
                       sae: Sae, top_k_features: int = 5,
                       device: str = "cuda") -> dict:
    """For each query token, identify the matched doc token and the top
    SAE features at that doc token (sorted by activation × per_q_max)."""
    score, matched_idx, per_q_max = maxsim_with_indices(
        q_tokens, d_tokens, device=device)
    feat_idxs, feat_acts = sae_features_at_tokens(
        d_tokens, matched_idx, sae, device=device)

    rows = []
    for t in range(matched_idx.shape[0]):
        # Sort the k SAE features by activation (already top-k from SAE)
        order = np.argsort(-feat_acts[t])
        feats = []
        for r in order[:top_k_features]:
            f = int(feat_idxs[t, r])
            a = float(feat_acts[t, r])
            feats.append({"feature": f, "activation": a,
                          "contribution": float(per_q_max[t]) * a})
        rows.append({
            "q_token_idx": t,
            "matched_doc_idx": int(matched_idx[t]),
            "per_q_max": float(per_q_max[t]),
            "top_features": feats,
        })
    return {"score": float(score),
            "n_query_tokens": int(matched_idx.shape[0]),
            "n_doc_tokens": int(d_tokens.shape[0]),
            "per_token": rows}


def render_window(tokens: list[str], pos: int, radius: int = 5) -> str:
    if pos < 0 or pos >= len(tokens):
        return ""
    lo = max(0, pos - radius); hi = min(len(tokens), pos + radius + 1)
    parts = [tokens[i].replace("##", "") for i in range(lo, hi)]
    parts[pos - lo] = f"<<{parts[pos - lo]}>>"
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--colbert-cache", required=True)
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--pairs", default="default",
                    help="'default' for DEFAULT_PAIRS; or qid:did pairs comma-separated")
    ap.add_argument("--top-features-per-token", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.pairs == "default":
        pairs = DEFAULT_PAIRS
    else:
        pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]

    rd = Path(args.sae_dir)
    ckpt = find_sae_ckpt(rd)
    sae = Sae.load_from_disk(ckpt, device=args.device)

    cache = Path(args.colbert_cache)
    qv, qo, q_ids = load_packed(cache, "queries")
    dv, do, d_ids = load_packed(cache, "corpus")

    # Load BEIR text + tokenizer for window rendering
    from datasets import load_dataset
    beir = {"trec-covid": "mteb/trec-covid",
            "scifact": "mteb/scifact"}[args.dataset]
    queries_ds = load_dataset(beir, "queries", split="queries")
    corpus_ds = load_dataset(beir, "corpus", split="corpus")
    qtxt = {r["_id"]: r["text"] for r in queries_ds}
    dtxt = {r["_id"]: (r.get("title", "") + " " + r["text"]).strip()
            for r in corpus_ds}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    labels = json.loads((rd / "feature_labels.json").read_text())["labels"] \
        if (rd / "feature_labels.json").exists() else {}

    out_records = []
    for qid, did in pairs:
        try:
            q_tok = get_query_tokens(qv, qo, q_ids, qid)
            d_tok = get_doc_tokens(dv, do, d_ids, did)
        except Exception as e:
            print(f"SKIP qid={qid} did={did}: {e}")
            continue

        rep = per_token_features(q_tok, d_tok, sae,
                                  top_k_features=args.top_features_per_token,
                                  device=args.device)
        q_str = qtxt.get(qid, "")
        d_str = dtxt.get(did, "")
        q_btoks = tok.tokenize(q_str)
        d_btoks = tok.tokenize(d_str)

        rec = {"qid": qid, "did": did, "score": rep["score"],
               "query_text": q_str, "doc_title_text_first200": d_str[:200],
               "per_token": []}

        # Pretty print
        print("\n" + "=" * 96)
        print(f"qid={qid}  did={did}  MaxSim={rep['score']:.2f}  "
              f"({rep['n_query_tokens']} q-tok × {rep['n_doc_tokens']} d-tok)")
        print(f"  query: {q_str}")
        print(f"  doc:   {d_str[:160]}...")
        print(f"\n  per-token (q→d) matches with top SAE features at the doc token:")
        for row in rep["per_token"]:
            t = row["q_token_idx"]
            d = row["matched_doc_idx"]
            qm = row["per_q_max"]
            qstr = q_btoks[t] if t < len(q_btoks) else f"[t{t}]"
            ctx = render_window(d_btoks, d, radius=4)
            features_str_parts = []
            row_features_out = []
            for ff in row["top_features"]:
                f = ff["feature"]
                lab = labels.get(str(f), {})
                jj = lab.get("judgment", "")[:4]
                desc = (lab.get("description", "") or "")[:60]
                features_str_parts.append(f"f{f}:{ff['activation']:.2f}[{jj}]")
                row_features_out.append({**ff, "judgment": lab.get("judgment", ""),
                                          "description": lab.get("description", "")})
            print(f"   q{t:>2} '{qstr:<10}' →  d{d:>3} '{ctx[:78]}'  qm={qm:.2f}")
            for ff in row_features_out:
                print(f"        {('f'+str(ff['feature'])):>9} act={ff['activation']:.2f} "
                      f"[{ff['judgment'][:4]:<4}] {ff['description'][:80]}")
            rec["per_token"].append({
                "q_token_idx": t, "q_token": qstr,
                "matched_doc_idx": d, "doc_window": ctx,
                "per_q_max": qm,
                "features": row_features_out,
            })
        out_records.append(rec)

    out_path = (Path(args.out) if args.out
                else Path(f"/data/embeddings/beir/{args.dataset}-token-report.json"))
    out_path.write_text(json.dumps({
        "dataset": args.dataset, "sae_dir": str(rd),
        "n_pairs": len(out_records),
        "records": out_records,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
