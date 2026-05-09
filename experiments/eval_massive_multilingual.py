"""MASSIVE 60-class intent probe across multiple languages.

Tests whether the SAE preserves multilingual signal: encode each
language's test set with jina-v5-nano, forward through SAE, train
logreg on raw / reconstructed / sparse-features, report accuracy.

Usage:
    python -m experiments.eval_massive_multilingual \\
        --sae-path .../checkpoints/sae_matryoshka_32_64.pooled \\
        --output .../eval_massive.json
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch


LANGS = ["en", "de", "fr", "es", "ru", "zh-CN", "ar", "hi", "vi", "ja"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-path", required=True)
    ap.add_argument("--embedding-model", default="jinaai/jina-embeddings-v5-text-nano-retrieval")
    ap.add_argument("--max-train", type=int, default=4000,
                    help="cap training set per language")
    ap.add_argument("--max-test", type=int, default=2000,
                    help="cap test set per language")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from latentsae.sae import Sae
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    sae = Sae.load_from_disk(args.sae_path, device=args.device)
    print(f"SAE: {sae.num_latents} latents, d_in={sae.d_in}, k={sae.cfg.k}")

    emb_model = SentenceTransformer(args.embedding_model, trust_remote_code=True,
                                     device=args.device)

    results = {"sae_path": args.sae_path, "embedding_model": args.embedding_model,
                "per_lang": {}}

    for lang in LANGS:
        try:
            t0 = time.monotonic()
            train = load_dataset("mteb/amazon_massive_intent", lang, split="train")
            test = load_dataset("mteb/amazon_massive_intent", lang, split="test")
            n_tr = min(args.max_train, len(train))
            n_te = min(args.max_test, len(test))
            train = train.shuffle(seed=42).select(range(n_tr))
            test = test.shuffle(seed=42).select(range(n_te))

            tr_texts = train["text"]; tr_labels = np.array(train["label"])
            te_texts = test["text"];  te_labels = np.array(test["label"])

            tr_emb = emb_model.encode(tr_texts, batch_size=64, show_progress_bar=False,
                                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            te_emb = emb_model.encode(te_texts, batch_size=64, show_progress_bar=False,
                                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

            with torch.no_grad():
                def sae_forward(embs):
                    out = sae(torch.from_numpy(embs).to(args.device))
                    recon = out.sae_out.cpu().numpy()
                    top_acts = out.latent_acts.cpu().numpy()
                    top_idx = out.latent_indices.cpu().numpy()
                    sparse = np.zeros((len(embs), sae.num_latents), dtype=np.float32)
                    for j in range(len(embs)):
                        sparse[j, top_idx[j]] = top_acts[j]
                    return recon, sparse

                tr_recon, tr_sparse = sae_forward(tr_emb)
                te_recon, te_sparse = sae_forward(te_emb)

            row = {}
            for kind, X_tr, X_te in [("raw", tr_emb, te_emb),
                                       ("recon", tr_recon, te_recon),
                                       ("sparse", tr_sparse, te_sparse)]:
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(X_tr, tr_labels)
                pred = clf.predict(X_te)
                row[f"{kind}_acc"] = float(accuracy_score(te_labels, pred))
                row[f"{kind}_f1"] = float(f1_score(te_labels, pred, average="macro",
                                                     zero_division=0))
            row["n_train"] = n_tr; row["n_test"] = n_te
            row["wall_seconds"] = time.monotonic() - t0
            results["per_lang"][lang] = row

            print(f"  {lang:>6}: raw={row['raw_acc']:.3f}  recon={row['recon_acc']:.3f}  "
                  f"sparse={row['sparse_acc']:.3f}  ({row['wall_seconds']:.1f}s)",
                  flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {lang}: FAILED {type(e).__name__}: {e}")
            results["per_lang"][lang] = {"error": str(e)}

    # Aggregate
    valid = [v for v in results["per_lang"].values() if "raw_acc" in v]
    if valid:
        results["aggregate"] = {
            f"{kind}_{stat}": float(np.mean([v[f"{kind}_{stat}"] for v in valid]))
            for kind in ("raw", "recon", "sparse")
            for stat in ("acc", "f1")
        }
        print(f"\n  mean across {len(valid)} langs: "
              f"raw={results['aggregate']['raw_acc']:.3f}  "
              f"recon={results['aggregate']['recon_acc']:.3f}  "
              f"sparse={results['aggregate']['sparse_acc']:.3f}")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
