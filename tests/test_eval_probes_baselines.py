"""eval_probes baselines + CIs on fabricated data (no dataset downloads).

Checks that probe_classification reports bootstrap CIs (acc/acc_ci_lo/
acc_ci_hi), that the PCA baseline at matched feature budget runs, and that
run_classification_task adds baseline_dense without changing existing keys.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latentsae.sae import Sae  # noqa: E402
from latentsae.utils.config import SaeConfig, SaeType  # noqa: E402
from experiments import eval_probes  # noqa: E402
from experiments.eval_probes import (  # noqa: E402
    probe_classification,
    probe_pca_baseline,
    run_classification_task,
    sae_encode_topk_sparse,
)

D_IN = 32
N = 400

PROBE_KEYS = {"accuracy", "f1_macro", "acc", "acc_ci_lo", "acc_ci_hi"}


def _synthetic_data(seed=0):
    """Two separable Gaussian clusters in D_IN dims + binary labels."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=N)
    centers = np.zeros((2, D_IN), dtype=np.float32)
    centers[1, :8] = 3.0
    emb = centers[labels] + rng.normal(0, 1, size=(N, D_IN)).astype(np.float32)
    return emb, labels


def _split(n):
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    return idx[:split], idx[split:]


def test_probe_classification_has_ci_keys():
    emb, labels = _synthetic_data()
    tr, te = _split(N)
    res = probe_classification(emb[tr], labels[tr], emb[te], labels[te])
    assert PROBE_KEYS <= set(res.keys())
    assert res["acc"] == pytest.approx(res["accuracy"])
    assert res["acc_ci_lo"] <= res["acc"] <= res["acc_ci_hi"]
    assert res["accuracy"] > 0.9  # clusters are well-separated


def test_pca_baseline_runs_at_matched_budget():
    emb, labels = _synthetic_data()
    tr, te = _split(N)
    res = probe_pca_baseline(emb[tr], labels[tr], emb[te], labels[te], n_components=5)
    assert PROBE_KEYS <= set(res.keys())
    assert res["acc_ci_lo"] <= res["acc"] <= res["acc_ci_hi"]
    assert res["accuracy"] > 0.9  # signal lives in top components


def test_pca_baseline_clamps_n_components():
    emb, labels = _synthetic_data()
    tr, te = _split(N)
    # n_components larger than the embedding dim must not crash
    res = probe_pca_baseline(emb[tr], labels[tr], emb[te], labels[te],
                             n_components=10 * D_IN)
    assert PROBE_KEYS <= set(res.keys())


def test_ksparse_plus_pca_baseline_with_tiny_sae():
    """Mimic the k-sparse loop: SAE top-k features vs PCA at the same k."""
    torch.manual_seed(0)
    sae = Sae(D_IN, SaeConfig(sae_type=SaeType.TOPK, expansion_factor=4, k=8),
              device="cpu", dtype=torch.float32)
    sae.eval()
    emb, labels = _synthetic_data()
    tr, te = _split(N)

    for probe_k in [1, 5]:
        sparse_k = sae_encode_topk_sparse(sae, emb, probe_k, device="cpu")
        res = probe_classification(sparse_k[tr], labels[tr], sparse_k[te], labels[te])
        res["baseline_pca"] = probe_pca_baseline(
            emb[tr], labels[tr], emb[te], labels[te], n_components=probe_k)
        assert PROBE_KEYS <= set(res.keys())
        assert PROBE_KEYS <= set(res["baseline_pca"].keys())


def test_run_classification_task_output_keys(monkeypatch):
    """Full task flow with fabricated embeddings — no downloads, no embedder."""
    emb, labels = _synthetic_data()

    def fake_load(max_samples=5000):
        return ["text"] * N, labels.tolist(), "Synthetic (2-class)"

    monkeypatch.setattr(eval_probes, "embed_texts", lambda texts, model_name: emb)

    torch.manual_seed(0)
    sae = Sae(D_IN, SaeConfig(sae_type=SaeType.TOPK, expansion_factor=4, k=8),
              device="cpu", dtype=torch.float32)
    sae.eval()

    task_name, result, *_ = run_classification_task(
        sae, fake_load, "unused-model", "cpu", max_samples=N)

    assert task_name == "Synthetic (2-class)"
    # Existing keys unchanged
    for key in ["raw_embeddings", "reconstructed", "sparse_features",
                "reconstruction_gap", "feature_quality_gap"]:
        assert key in result
    # New explicit dense baseline (same probe as raw_embeddings)
    assert result["baseline_dense"] == result["raw_embeddings"]
    # CI keys present on every probe result
    for variant in ["raw_embeddings", "baseline_dense", "reconstructed", "sparse_features"]:
        assert PROBE_KEYS <= set(result[variant].keys())
        r = result[variant]
        assert r["acc_ci_lo"] <= r["acc"] <= r["acc_ci_hi"]
