"""Eval-manifest hygiene: assert_eval_slice must pass inside reserved ranges,
fail outside them or when no manifest exists, and only warn (not raise) when
SAE_EVAL_NO_MANIFEST=1.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.eval_manifest import (  # noqa: E402
    EvalManifestError,
    assert_eval_slice,
    create_entry,
    load_manifest,
    manifest_path,
)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated manifest root + a fake dataset dir; no /data access."""
    monkeypatch.setenv("SAE_EVAL_MANIFEST_DIR", str(tmp_path / "manifests"))
    monkeypatch.delenv("SAE_EVAL_NO_MANIFEST", raising=False)
    data_dir = tmp_path / "some-dataset" / "train"
    data_dir.mkdir(parents=True)
    return str(data_dir)


def test_create_and_load(env):
    path = create_entry(env, "data-00000.npy", 0, 1000,
                        kind="held-out", note="test range", created_by="pytest")
    assert path == manifest_path(env)
    assert path.name == "some-dataset.json"  # slug = parent of train/
    m = load_manifest(env)
    assert m["reserved"][0]["rows"] == [0, 1000]
    assert m["reserved"][0]["kind"] == "held-out"
    assert m["reserved"][0]["created_by"] == "pytest"
    assert m["reserved"][0]["note"] == "test range"


def test_assert_passes_inside_reserved_range(env):
    create_entry(env, "data-00000.npy", 100, 1000, kind="held-out", note="t")
    entry = assert_eval_slice(env, "data-00000.npy", 100, 1000)  # exact
    assert entry["rows"] == [100, 1000]
    assert_eval_slice(env, "data-00000.npy", 200, 300)           # strict subset
    # shard given as a path is normalized to its filename
    assert_eval_slice(env, f"{env}/data-00000.npy", 200, 300)


def test_assert_fails_outside_reserved_range(env):
    create_entry(env, "data-00000.npy", 100, 1000, kind="held-out", note="t")
    with pytest.raises(EvalManifestError, match="NOT inside"):
        assert_eval_slice(env, "data-00000.npy", 0, 50)        # before
    with pytest.raises(EvalManifestError, match="NOT inside"):
        assert_eval_slice(env, "data-00000.npy", 500, 2000)    # straddles end
    with pytest.raises(EvalManifestError, match="NOT inside"):
        assert_eval_slice(env, "data-00001.npy", 200, 300)     # wrong shard


def test_assert_fails_when_no_manifest(env):
    with pytest.raises(EvalManifestError, match="no eval manifest"):
        assert_eval_slice(env, "data-00000.npy", 0, 10)
    with pytest.raises(EvalManifestError, match="no eval manifest"):
        load_manifest(env)


def test_escape_hatch_downgrades_to_warning(env, monkeypatch, capsys):
    monkeypatch.setenv("SAE_EVAL_NO_MANIFEST", "1")
    assert assert_eval_slice(env, "data-00000.npy", 0, 10) == {}
    assert "WARNING" in capsys.readouterr().err


def test_multiple_ranges_and_shards(env):
    create_entry(env, "data-00000.npy", 0, 100, kind="scan-set (training-overlapped)",
                 note="scan")
    create_entry(env, "data-00001.npy", 0, 500, kind="eval-only", note="all rows")
    assert_eval_slice(env, "data-00000.npy", 0, 100)
    assert_eval_slice(env, "data-00001.npy", 400, 500)
    assert len(load_manifest(env)["reserved"]) == 2


def test_empty_slice_is_a_value_error(env):
    create_entry(env, "data-00000.npy", 0, 100, kind="held-out", note="t")
    with pytest.raises(ValueError):
        assert_eval_slice(env, "data-00000.npy", 50, 50)
