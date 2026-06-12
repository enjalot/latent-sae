"""Helpers for consuming the per-feature firing reservoir written by
extract_feature_activations.py (feature_activations_reservoir.npz).

The reservoir is a uniform sample of each feature's firings, so quantile
stratification over it is unbiased — unlike the top-N heap, which only sees
the extreme tail (max-activation bias; plan-sae-validation.md V1).

Typical use in a labeling/scoring script:

    res = load_reservoir(run_dir)
    picks = stratified_examples(res, fid, per_decile=2)
    # picks: list of (decile, activation, chunk_idx, token_idx), low→high
"""
import json
from pathlib import Path

import numpy as np


def load_reservoir(run_dir_or_npz):
    """Load reservoir arrays given a run dir, a feature_activations.json
    path, or the .npz path itself."""
    p = Path(run_dir_or_npz)
    if p.is_dir():
        fa = p / "feature_activations.json"
        meta = json.loads(fa.read_text())
        p = fa.with_name(meta["reservoir_file"])
    elif p.suffix == ".json":
        meta = json.loads(p.read_text())
        p = p.with_name(meta["reservoir_file"])
    z = np.load(p)
    return {k: z[k] for k in z.files}


def feature_sample(res, fid):
    """Valid (act, chunk, token) reservoir entries for one feature,
    sorted ascending by activation."""
    valid = res["chunks"][fid] >= 0
    acts = res["acts"][fid][valid]
    chunks = res["chunks"][fid][valid]
    toks = res["tokens"][fid][valid]
    order = np.argsort(acts)
    return acts[order], chunks[order], toks[order]


def stratified_examples(res, fid, per_decile=2, n_strata=10, seed=0):
    """Sample up to per_decile examples from each activation decile of the
    feature's reservoir. Returns list of (stratum, act, chunk_idx, token_idx),
    ordered low→high stratum. Features with few firings yield fewer strata.
    """
    acts, chunks, toks = feature_sample(res, fid)
    n = len(acts)
    if n == 0:
        return []
    rng = np.random.default_rng(seed + int(fid))
    out = []
    edges = np.linspace(0, n, n_strata + 1).astype(int)
    for s in range(n_strata):
        lo, hi = edges[s], edges[s + 1]
        if hi <= lo:
            continue
        take = rng.choice(np.arange(lo, hi), size=min(per_decile, hi - lo),
                          replace=False)
        for i in sorted(take):
            out.append((s, float(acts[i]), int(chunks[i]), int(toks[i])))
    return out
