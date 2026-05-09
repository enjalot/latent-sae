"""Run MaxSim attribution for a chosen set of (qid, did) pairs and emit
a single JSON report + a printable summary.

Outputs:
  - /data/embeddings/beir/<dataset>-attribution.json
  - prints per-query: query text, gold doc title, top-K interpretable
    features attributed to the MaxSim score
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
    attribute_maxsim, find_sae_ckpt, load_packed,
    get_query_tokens, get_doc_tokens,
)


# Hand-picked TREC-COVID divergent queries: ColBERT ranks gold in top-3,
# at least one pooled embedder ranks it ≥7 or misses it entirely.
DEFAULT_PAIRS = [
    ("1",  "4dtk1kyh"),  # what is the origin of COVID-19? — MiniLM misses entirely
    ("34", "b999y89f"),  # vaccine candidates? — MiniLM 96, Jina 3
    ("32", "h0q93in1"),  # animal models for SARS-CoV-2 — MiniLM 91
    ("35", "fu373osb"),  # how does the SARS-CoV-2 spike protein bind to ACE2 — MiniLM 18
    ("25", "9k8r18x7"),  # which animal originated COVID — MiniLM 15
    ("9",  "ieobv7q8"),  # SARS-CoV-2 in HIV-positive patients — both MiniLM and Jina ≥3
    ("8",  "8cw3bjxh"),  # what are the guidelines for triaging — MiniLM 9
    ("6",  "1dr4r3n4"),  # what drugs have been used to treat — MiniLM 7
    ("10", "pn02p843"),  # what kinds of complications related — MiniLM 6
    ("49", "dptgg05n"),  # which type of tests are most rapid — MiniLM 6, Jina 2
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--colbert-cache", required=True)
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pair", action="append", default=None,
                    help="Override DEFAULT_PAIRS. Format: qid:did. May repeat.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pairs = ([tuple(p.split(":", 1)) for p in args.pair]
             if args.pair else DEFAULT_PAIRS)
    print(f"running attribution for {len(pairs)} (qid, did) pairs")

    rd = Path(args.sae_dir)
    ckpt = find_sae_ckpt(rd)
    sae = Sae.load_from_disk(ckpt, device=args.device)

    cache = Path(args.colbert_cache)
    qv, qo, q_ids = load_packed(cache, "queries")
    dv, do, d_ids = load_packed(cache, "corpus")

    # Load BEIR for query/doc text
    from datasets import load_dataset
    beir_path = {"trec-covid": "mteb/trec-covid",
                 "scifact": "mteb/scifact"}[args.dataset]
    queries_ds = load_dataset(beir_path, "queries", split="queries")
    corpus_ds = load_dataset(beir_path, "corpus", split="corpus")
    qtxt = {r["_id"]: r["text"] for r in queries_ds}
    dtxt = {r["_id"]: (r.get("title", "") + " " + r["text"]).strip()
            for r in corpus_ds}

    # Load divergence ranks for context
    div_path = Path(f"/data/embeddings/beir/{args.dataset}-divergence.json")
    div_rows = {r["qid"]: r for r in json.loads(div_path.read_text())["rows"]} \
        if div_path.exists() else {}

    labels = {}
    p = rd / "feature_labels.json"
    if p.exists():
        labels = json.loads(p.read_text())["labels"]

    out_records = []
    for qid, did in pairs:
        try:
            q_tok = get_query_tokens(qv, qo, q_ids, qid)
            d_tok = get_doc_tokens(dv, do, d_ids, did)
        except Exception as e:
            print(f"  SKIP qid={qid} did={did}: {e}")
            continue
        result = attribute_maxsim(q_tok, d_tok, sae, top_n_features=args.top_n,
                                   device=args.device)
        # Filter ranked features to interpretable, then re-rank by MAX
        # attribution (spike on one specific match) instead of SUM (steady
        # contribution across many matches). Sum-ranked is dominated by
        # features firing on most q-tokens which are almost always generic.
        interp_pool = [fr for fr in result["ranked_features"]
                       if labels.get(str(fr["feature"]), {}).get("judgment", "")
                       in ("COHERENT", "THEMATIC")]
        interp_by_max = sorted(interp_pool, key=lambda x: -x["max_attribution"])
        interp = []
        for fr in interp_by_max[: args.top_n]:
            lab = labels.get(str(fr["feature"]), {})
            interp.append({**fr, "judgment": lab.get("judgment", ""),
                            "description": lab.get("description", "")})

        rec = {
            "qid": qid, "did": did,
            "query_text": qtxt.get(qid, ""),
            "doc_title_text": dtxt.get(did, "")[:250],
            "score": result["score"],
            "n_query_tokens": result["n_query_tokens"],
            "n_doc_tokens": result["n_doc_tokens"],
            "ranks": div_rows.get(qid, {}),
            "top_interpretable": interp,
            "top_all": result["ranked_features"],
        }
        out_records.append(rec)

        # Pretty print
        print("\n" + "=" * 90)
        print(f"qid={qid}  did={did}  MaxSim={result['score']:.2f}  "
              f"({result['n_query_tokens']} q-tok × {result['n_doc_tokens']} d-tok)")
        ranks = div_rows.get(qid, {})
        if ranks:
            print(f"  ranks of gold doc — colbert={ranks.get('colbert_rank')}  "
                  f"minilm={ranks.get('minilm_rank')}  "
                  f"jina-v5-small={ranks.get('jina-v5-small_rank')}")
        print(f"  query: {qtxt.get(qid,'')[:140]}")
        print(f"  doc:   {dtxt.get(did,'')[:200]}...")
        if interp:
            print(f"\n  Top {len(interp)} interpretable features by MAX-attribution "
                  f"(spike on one specific match):")
            for i, fr in enumerate(interp):
                print(f"  {i+1:>2}  f{fr['feature']:<6}  "
                      f"max={fr['max_attribution']:>5.2f}@q{fr['max_q_token_idx']:<2}  "
                      f"sum={fr['sum_attribution']:>6.2f}  "
                      f"q_toks={fr['n_q_tokens']:<2}  "
                      f"[{fr['judgment'][:4]}] {fr['description'][:100]}")
        else:
            print("  (no interpretable features in top attribution — all GENERIC/POLY)")

    out_path = (Path(args.out) if args.out
                else Path(f"/data/embeddings/beir/{args.dataset}-attribution.json"))
    out_path.write_text(json.dumps({
        "dataset": args.dataset, "sae_dir": str(rd),
        "n_pairs": len(out_records),
        "records": out_records,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
