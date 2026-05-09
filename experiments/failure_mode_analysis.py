"""ColBERT MaxSim failure-mode analysis.

For TREC-COVID queries where ColBERT places the gold doc lower than
pooled embedders do, the hypothesis is that ColBERT's wrong top-1 doc
gets a high MaxSim score from scattered punctuation/function-word
matches rather than a coherent semantic span.

Quantifies, per (query, doc) pair:
  - % of query tokens whose matched doc-token is "non-content"
    (punctuation, single-char, or known function-word fragment)
  - SPAN COHERENCE: stddev of matched doc-token positions, normalized
    by doc length. Low = clustered; high = scattered.
  - Mean per-q-token similarity (already in attribution)

For each failure case we compare ColBERT's WRONG top-1 doc to the GOLD
doc on these metrics.

Usage:
    python -m experiments.failure_mode_analysis \\
        --colbert-cache /data/embeddings/beir/trec-covid-mxbai-edge-32m
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.maxsim_attribution import (  # noqa: E402
    load_packed, get_query_tokens, get_doc_tokens, maxsim_with_indices,
)


# Heuristic: tokens that are mostly non-content
PUNCT_RE = re.compile(r"^[\W_]+$")
FUNCTION_WORDS = {
    "the", "of", "and", "to", "a", "in", "is", "that", "for", "on",
    "with", "as", "by", "are", "was", "be", "this", "an", "or", "from",
    "at", "it", "we", "have", "has", "but", "not", "can", "all", "their",
    "which", "they", "been", "than", "these", "those", "such",
    "do", "does", "did", "will", "would", "could", "should",
    "i", "he", "she", "you", "his", "her", "its", "our", "your",
    "into", "out", "up", "down", "over", "under", "between", "against",
    "before", "after", "during", "while", "though", "if", "when", "where",
    "any", "some", "many", "few", "most", "other", "another", "each", "every",
}


def is_non_content(token: str) -> bool:
    """A doc token is 'non-content' if it's punctuation, single-char,
    or a common English function word."""
    t = token.replace("##", "").strip().lower()
    if not t:
        return True
    if PUNCT_RE.match(t):
        return True
    if len(t) <= 1:
        return True
    if t in FUNCTION_WORDS:
        return True
    return False


def analyze_pair(q_tokens: np.ndarray, d_tokens: np.ndarray,
                  d_text_tokens: list[str], device: str = "cuda") -> dict:
    score, matched_idx, per_q_max = maxsim_with_indices(
        q_tokens, d_tokens, device=device)
    n_q = len(matched_idx)
    n_d = d_tokens.shape[0]

    # Non-content fraction
    non_content = 0
    matched_token_strs = []
    for d_pos in matched_idx:
        if d_pos < len(d_text_tokens):
            tok = d_text_tokens[d_pos]
            matched_token_strs.append(tok)
            if is_non_content(tok):
                non_content += 1
        else:
            matched_token_strs.append("?")
            non_content += 1
    non_content_frac = non_content / max(n_q, 1)

    # Span coherence: stddev of matched doc positions, normalized by doc length
    if n_q > 1:
        span_std = float(np.std(matched_idx) / max(n_d, 1))
    else:
        span_std = 0.0

    return {
        "score": score, "n_q": n_q, "n_d": n_d,
        "matched_idx": matched_idx.tolist(),
        "matched_token_strs": matched_token_strs,
        "per_q_max_mean": float(per_q_max.mean()),
        "non_content_frac": non_content_frac,
        "non_content_n": non_content,
        "span_std_norm": span_std,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colbert-cache", required=True)
    ap.add_argument("--colbert-topk-cache",
                    default="/data/embeddings/beir/trec-covid-mxbai-edge-32m/colbert_topk.npz")
    ap.add_argument("--divergence",
                    default="/data/embeddings/beir/trec-covid-divergence.json")
    ap.add_argument("--out",
                    default="/data/embeddings/beir/trec-covid-failure-analysis.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Load divergence to find failure cases
    div = json.loads(Path(args.divergence).read_text())["rows"]
    # ColBERT failures = pooled outranks ColBERT on the gold doc
    failures = []
    for r in div:
        cb = r["colbert_rank"]
        m = r.get("minilm_rank") if r.get("minilm_rank") is not None else 999
        j = r.get("jina-v5-small_rank") if r.get("jina-v5-small_rank") is not None else 999
        if cb is None:
            continue
        # Pooled beats ColBERT
        if (m >= 0 and m < cb) or (j >= 0 and j < cb):
            failures.append(r)
    print(f"Found {len(failures)} ColBERT-vs-pooled failure cases")

    # ColBERT topk for finding ColBERT's wrong top-1
    npz = np.load(args.colbert_topk_cache, allow_pickle=True)
    cb_topk = npz["top_k"]; cb_q_ids = list(npz["q_ids"]); cb_d_ids = list(npz["d_ids"])

    # Cached vectors
    cache = Path(args.colbert_cache)
    qv, qo, q_ids = load_packed(cache, "queries")
    dv, do, d_ids = load_packed(cache, "corpus")

    # Tokenizer for rendering doc tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    from datasets import load_dataset
    queries_ds = load_dataset("mteb/trec-covid", "queries", split="queries")
    corpus_ds = load_dataset("mteb/trec-covid", "corpus", split="corpus")
    qtxt = {r["_id"]: r["text"] for r in queries_ds}
    dtxt = {r["_id"]: (r.get("title", "") + " " + r["text"]).strip()
            for r in corpus_ds}

    rows = []
    for r in failures:
        qid = r["qid"]
        gold_did = r["best_gold_did"]
        cb_rank = r["colbert_rank"]
        try:
            qi = cb_q_ids.index(qid)
        except ValueError:
            continue
        wrong_did = cb_d_ids[cb_topk[qi, 0]]   # ColBERT's top-1 (wrong)
        if wrong_did == gold_did:
            continue  # ColBERT actually got it right at top-1

        q_tok = get_query_tokens(qv, qo, q_ids, qid)
        try:
            wrong_d_tok = get_doc_tokens(dv, do, d_ids, wrong_did)
            gold_d_tok = get_doc_tokens(dv, do, d_ids, gold_did)
        except Exception:
            continue

        wrong_text_tokens = tok.tokenize(dtxt.get(wrong_did, ""))
        gold_text_tokens = tok.tokenize(dtxt.get(gold_did, ""))

        wrong = analyze_pair(q_tok, wrong_d_tok, wrong_text_tokens, device=args.device)
        gold = analyze_pair(q_tok, gold_d_tok, gold_text_tokens, device=args.device)

        rows.append({
            "qid": qid, "query": qtxt.get(qid, ""),
            "gold_did": gold_did, "wrong_did": wrong_did,
            "cb_rank_of_gold": cb_rank,
            "wrong_score": wrong["score"], "gold_score": gold["score"],
            "wrong_per_q_max_mean": wrong["per_q_max_mean"],
            "gold_per_q_max_mean": gold["per_q_max_mean"],
            "wrong_non_content_frac": wrong["non_content_frac"],
            "gold_non_content_frac": gold["non_content_frac"],
            "wrong_span_std_norm": wrong["span_std_norm"],
            "gold_span_std_norm": gold["span_std_norm"],
            "wrong_matched_tokens": wrong["matched_token_strs"][:25],
            "gold_matched_tokens": gold["matched_token_strs"][:25],
        })
        # Print
        print(f"\nqid={qid}: {qtxt.get(qid, '')[:100]}")
        print(f"  ColBERT score on WRONG (top-1):   {wrong['score']:.2f}  "
              f"non_content={wrong['non_content_frac']:.0%}  "
              f"span_std_norm={wrong['span_std_norm']:.3f}")
        print(f"    matched tokens: {wrong['matched_token_strs'][:15]}")
        print(f"  ColBERT score on GOLD (rank {cb_rank}):  {gold['score']:.2f}  "
              f"non_content={gold['non_content_frac']:.0%}  "
              f"span_std_norm={gold['span_std_norm']:.3f}")
        print(f"    matched tokens: {gold['matched_token_strs'][:15]}")

    # Aggregate
    if rows:
        wnc = np.mean([r["wrong_non_content_frac"] for r in rows])
        gnc = np.mean([r["gold_non_content_frac"] for r in rows])
        wss = np.mean([r["wrong_span_std_norm"] for r in rows])
        gss = np.mean([r["gold_span_std_norm"] for r in rows])
        print(f"\n=== Aggregate across {len(rows)} failure cases ===")
        print(f"  mean non-content match fraction:  WRONG={wnc:.0%}  GOLD={gnc:.0%}")
        print(f"  mean span std (normalized):       WRONG={wss:.3f}  GOLD={gss:.3f}")

    Path(args.out).write_text(json.dumps({
        "n_failures": len(rows), "rows": rows,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
