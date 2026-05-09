"""Render qualitative spot-check HTML pages for SAE features.

For each SAE, picks ~30 features (10 from each cross-lingual bucket
when available) and renders:

  - feature ID, num_latents, dominant corpus, bucket
  - qwen2.5:32b description + judgment (COH/THM/GEN/POL/UNC)
  - top-16 activating chunks with activation, corpus tag, text

Output: ~/.agent/moonshine/sae-features-<DATE>/<SAE_TAG>/index.html
served via http://gsv.local:8800/moonshine/sae-features-<DATE>/

Usage:
    python -m experiments.render_qualitative_html --date 2026-05-07
"""
import argparse
import html
import json
import random
from collections import Counter
from glob import glob
from pathlib import Path


SAES = [
    ("B_24K_EN",      "jina_v5_nano_phase1_24K_oldrecipe_replay4_*"),
    ("D_24K_ML",      "jina_v5_nano_phase2_24K_multilingual_oldrecipe_replay4_*"),
    ("E2_49K_ML",     "jina_v5_nano_phase3_49K_multilingual_oldrecipe_replay4_*"),
    ("E3_98K_ML_r4",  "jina_v5_nano_phase4_98K_multilingual_oldrecipe_replay4_*"),
    ("F_98K_ML_r8",   "jina_v5_nano_phase5_98K_multilingual_oldrecipe_replay8_*"),
]

JUDGMENT_COLORS = {
    "COHERENT": "#2d8659",
    "THEMATIC": "#5b8a8c",
    "GENERIC":  "#a3955a",
    "POLYSEMANTIC": "#a85e3e",
    "UNCLEAR":  "#776677",
}


def bucketize(shares, threshold=0.80):
    EN = {"fineweb", "redpajama", "pile"}
    en_share = sum(v for k, v in shares.items() if k in EN)
    if en_share >= threshold:
        return "english_only"
    non_en = {k: v for k, v in shares.items() if k not in EN}
    if non_en and max(non_en.values()) >= threshold:
        return "language_bound"
    return "cross_lingual"


def pick_features(xl_data, labels_data, n_per_bucket=10, seed=42):
    """Sample n_per_bucket features per bucket that are also in the label sample."""
    rng = random.Random(seed)
    by_bucket = {"english_only": [], "language_bound": [], "cross_lingual": []}
    labeled_set = set(labels_data.get("labels", {}).keys())

    for fid, hits in xl_data["features"].items():
        if not hits:
            continue
        # Skip features without rigor labels (no qwen32b output)
        if labeled_set and fid not in labeled_set:
            continue
        c = Counter(h["corpus"] for h in hits)
        n = sum(c.values())
        shares = {k: v/n for k, v in c.items()}
        bucket = bucketize(shares)
        by_bucket[bucket].append((fid, hits, c, shares, bucket))

    out = []
    for b in by_bucket:
        cands = by_bucket[b]
        if not cands:
            continue
        rng.shuffle(cands)
        out.extend(cands[:n_per_bucket])
    return out


def render_feature_block(fid, hits, counter, shares, bucket, label_info,
                          corpora_summary):
    judgment = label_info.get("judgment", "(no label)")
    description = label_info.get("description", "(no description)")
    color = JUDGMENT_COLORS.get(judgment, "#444")
    dominant = max(shares, key=shares.get) if shares else "?"
    dom_pct = shares.get(dominant, 0) * 100

    chunks_html = ""
    for h in hits[:16]:
        text = html.escape(h.get("text", ""))[:600]
        if len(h.get("text", "")) > 600:
            text += "…"
        corpus = html.escape(h.get("corpus", "?"))
        act = h.get("activation", 0)
        chunks_html += (
            f'<div class="chunk">'
            f'<div class="meta"><span class="corpus">{corpus}</span>'
            f' <span class="act">{act:.3f}</span></div>'
            f'<div class="text">{text}</div></div>'
        )

    bucket_html = (
        f'<div class="dist">'
        + " · ".join(f'{html.escape(k)}: {v}' for k, v in counter.most_common(8))
        + '</div>'
    )

    return f'''
    <div class="feature">
      <div class="hdr">
        <div class="id">f{fid}</div>
        <div class="bucket">{bucket}</div>
        <div class="dom">dom: {html.escape(dominant)} ({dom_pct:.0f}%)</div>
        <div class="judgment" style="background:{color}">{judgment}</div>
      </div>
      <div class="desc">{html.escape(description)}</div>
      {bucket_html}
      <div class="chunks">{chunks_html}</div>
    </div>
    '''


def render_sae_page(sae_tag: str, run: Path, out_dir: Path):
    xl = run / "feature_activations_xlingual.json"
    rigor = run / "feature_labels_sample256_qwen32b.json"
    if not xl.exists():
        print(f"  SKIP {sae_tag}: no xlingual extract"); return
    xl_data = json.loads(xl.read_text())
    labels_data = json.loads(rigor.read_text()) if rigor.exists() else {"labels": {}}
    label_lookup = labels_data.get("labels", {})

    selected = pick_features(xl_data, labels_data, n_per_bucket=10)
    print(f"  {sae_tag}: rendering {len(selected)} features")

    blocks = []
    for fid, hits, counter, shares, bucket in selected:
        info = label_lookup.get(fid, {})
        blocks.append(render_feature_block(
            fid, hits, counter, shares, bucket, info,
            xl_data.get("corpora", {}),
        ))

    counts = labels_data.get("counts", {})
    summary = (f'<div class="summary">'
               f'<b>{sae_tag}</b> &middot; {xl_data["num_latents"]} latents'
               f' &middot; {xl_data["n_live_features"]} live features'
               f' &middot; rigor labels (qwen2.5:32b, n={sum(counts.values())}): '
               + ", ".join(f'<span style="color:{JUDGMENT_COLORS.get(k, "#444")}">{k}: {v}</span>'
                            for k, v in counts.items())
               + '</div>')

    page = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{sae_tag} qualitative</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 1rem auto; padding: 0 1rem; color: #222; background: #fafafa; }}
h1 {{ font-size: 1.4rem; }}
.summary {{ background: #fff; padding: 0.75rem; border: 1px solid #ddd; margin-bottom: 1rem; font-size: 0.95rem; }}
.summary span {{ font-weight: 600; }}
.feature {{ background: #fff; border: 1px solid #ddd; padding: 1rem; margin: 0.7rem 0; border-radius: 4px; }}
.hdr {{ display: flex; gap: 0.8rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.4rem; }}
.id {{ font-family: monospace; font-weight: 700; font-size: 1.05rem; }}
.bucket {{ background: #eee; padding: 2px 6px; border-radius: 3px; font-size: 0.8rem; }}
.dom {{ color: #555; font-size: 0.85rem; }}
.judgment {{ color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.78rem; font-weight: 600; margin-left: auto; }}
.desc {{ font-style: italic; color: #444; margin: 0.4rem 0 0.5rem; padding: 0.5rem; background: #fffae8; border-left: 3px solid #d4a44a; font-size: 0.92rem; }}
.dist {{ font-family: monospace; font-size: 0.78rem; color: #666; margin-bottom: 0.4rem; }}
.chunks {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; }}
.chunk {{ background: #f6f6f6; padding: 0.4rem; border-left: 3px solid #aac; font-size: 0.82rem; }}
.chunk .meta {{ color: #888; font-size: 0.7rem; font-family: monospace; margin-bottom: 0.2rem; }}
.chunk .corpus {{ background: #ddd; padding: 1px 4px; border-radius: 2px; }}
.chunk .act {{ color: #c54; font-weight: 600; }}
.chunk .text {{ white-space: pre-wrap; line-height: 1.3; }}
</style></head>
<body>
<h1><a href="../">↑</a> {sae_tag} — qualitative spot-check</h1>
{summary}
{"".join(blocks)}
</body></html>
'''
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page)


def render_index(date: str, sae_tags: list, root: Path):
    items = "".join(
        f'<li><a href="{tag}/">{tag}</a></li>' for tag in sae_tags
    )
    page = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SAE qualitative spot-check {date}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.4rem; }}
li {{ margin: 0.4rem 0; font-family: monospace; font-size: 1.05rem; }}
</style></head>
<body>
<h1>SAE qualitative spot-check — {date}</h1>
<p>30 features per SAE (10 each: english_only, language_bound, cross_lingual buckets).
Each feature shows top-16 activating chunks with corpus tag, plus the qwen2.5:32b
auto-label and judgment (COHERENT / THEMATIC / GENERIC / POLYSEMANTIC / UNCLEAR).</p>
<ul>{items}</ul>
</body></html>
'''
    (root / "index.html").write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-07")
    args = ap.parse_args()

    root = Path.home() / ".agent" / "moonshine" / f"sae-features-{args.date}"
    root.mkdir(parents=True, exist_ok=True)
    print(f"writing under {root}")

    rendered = []
    for tag, pat in SAES:
        matches = sorted(glob(f"/home/enjalot/code/latent-sae/experiments/results/{pat}"),
                          reverse=True)
        if not matches:
            print(f"  SKIP {tag}: no run dir")
            continue
        run = Path(matches[0])
        render_sae_page(tag, run, root / tag)
        rendered.append(tag)

    render_index(args.date, rendered, root)
    print(f"\n✓ wrote {root}/index.html and {len(rendered)} SAE pages")
    print(f"   browse at http://gsv.local:8800/moonshine/sae-features-{args.date}/")


if __name__ == "__main__":
    main()
