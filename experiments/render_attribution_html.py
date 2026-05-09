"""Render a per-query attribution report as a single HTML page.

Combines: matched-pair view (q→d tokens + features at the matched doc
token), the top backbone-subtracted features, and the multi-feature
ablation rank-shift result.

Usage:
    python -m experiments.render_attribution_html \\
        --token-report /data/embeddings/beir/trec-covid-token-report.json \\
        --backbone-subtracted /data/embeddings/beir/trec-covid-attribution-backbone-subtracted.json \\
        --multifeat /data/embeddings/beir/trec-covid-multifeat-17r.json \\
        --divergence /data/embeddings/beir/trec-covid-divergence.json \\
        --out ~/.agent/moonshine/sae-colbert-attribution/index.html
"""
import argparse
import html
import json
from pathlib import Path


CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --article-width: 980px;
  --body-font: 'Source Serif 4', Georgia, serif;
  --heading-font: 'Source Sans 3', system-ui, sans-serif;
  --mono-font: 'Source Code Pro', monospace;
  --body-size: 1.05rem;
  --line-height: 1.55;
  --ink: #1a1613; --ink-2: #5a554e; --ink-3: #8a867f;
  --paper: #fbf8f3; --paper-2: #f2ede3; --rule: #d9d3c5; --cell: #ffffff;
  --coh: #2b628a; --thm: #5f7a5f; --gene: #8a867f; --poly: #b06252; --uncl: #b07e2a;
  --accent: #2b628a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e9e4d8; --ink-2: #aaa49a; --ink-3: #807b72;
    --paper: #1a1814; --paper-2: #252118; --rule: #3a3528; --cell: #22201a;
    --coh: #7fb0e0; --thm: #9ab79a; --gene: #aaa49a; --poly: #d08d80; --uncl: #d8b070;
    --accent: #7fb0e0;
  }
}
body {
  font-family: var(--body-font); font-size: var(--body-size);
  line-height: var(--line-height); color: var(--ink); background: var(--paper);
  -webkit-font-smoothing: antialiased;
}
.article { max-width: var(--article-width); margin: 0 auto; padding: 2rem 1rem 4rem; }
.dateline {
  font-family: var(--heading-font); font-size: 0.78rem; color: var(--ink-3);
  letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1rem;
}
h1, h2, h3, h4 { font-family: var(--heading-font); font-weight: 700; line-height: 1.2; color: var(--ink); }
h1 { font-size: 1.85rem; margin: 0 0 0.5rem; letter-spacing: -0.01em; }
h2 { font-size: 1.2rem; margin: 2rem 0 0.6rem; padding-top: 1.5rem; border-top: 1px solid var(--rule); }
h3 { font-size: 0.95rem; margin: 1.2rem 0 0.5rem; color: var(--ink-2); font-weight: 600; }
h4 { font-size: 0.85rem; margin: 1rem 0 0.4rem; color: var(--ink-2); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
p { margin: 0 0 0.8rem; }
a { color: var(--accent); text-decoration: underline; text-underline-offset: 2px; }
code { font-family: var(--mono-font); font-size: 0.9em; background: var(--paper-2); padding: 0.05em 0.3em; border-radius: 2px; }
.lede {
  font-family: var(--heading-font); font-size: 1.04rem; color: var(--ink-2);
  margin-bottom: 1.5rem; line-height: 1.5;
}
.query-card {
  background: var(--cell); border: 1px solid var(--rule);
  padding: 1rem 1.2rem 1.2rem; margin: 1.5rem 0;
}
.qmeta {
  font-family: var(--heading-font); font-size: 0.85rem; color: var(--ink-2);
  margin-bottom: 0.8rem;
}
.qmeta .key { color: var(--ink-3); margin-right: 0.2em; }
.qmeta .sep { color: var(--ink-3); margin: 0 0.4em; }
.qtext { font-style: italic; color: var(--ink); margin: 0.3rem 0 0.5rem; }
.dtitle { font-size: 0.9rem; color: var(--ink-2); margin-bottom: 0.5rem; }
table.matched {
  width: 100%; border-collapse: collapse; font-size: 0.86rem;
  font-family: var(--heading-font); margin: 0.5rem 0;
}
table.matched th, table.matched td {
  padding: 0.4rem 0.5rem; vertical-align: top; border-bottom: 1px solid var(--rule);
  text-align: left;
}
table.matched th {
  font-size: 0.74rem; color: var(--ink-3); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
}
table.matched .qtoken { font-family: var(--mono-font); font-size: 0.92em; color: var(--ink); }
table.matched .dctx { font-family: var(--mono-font); font-size: 0.85em; color: var(--ink-2); }
table.matched .dctx .mark { background: #f5e3c4; color: var(--ink); padding: 0 3px; border-radius: 2px; }
@media (prefers-color-scheme: dark) {
  table.matched .dctx .mark { background: #4a3a1c; }
}
table.matched .sim { color: var(--ink-3); font-family: var(--mono-font); font-size: 0.85em; }
table.matched .feats { font-size: 0.84rem; }
.feat { margin-bottom: 0.2rem; line-height: 1.3; }
.feat .id { font-family: var(--mono-font); font-size: 0.85em; color: var(--ink); }
.feat .act { font-family: var(--mono-font); font-size: 0.78em; color: var(--ink-3); }
.feat .desc { color: var(--ink-2); font-size: 0.82rem; }
.judg {
  display: inline-block; font-size: 0.65rem; padding: 0 0.35em;
  border-radius: 2px; font-weight: 600; letter-spacing: 0.02em;
  text-transform: uppercase; vertical-align: middle; margin-right: 0.2em;
}
.judg.COHE { background: var(--coh); color: var(--paper); }
.judg.THEM { background: var(--thm); color: var(--paper); }
.judg.GENE { background: var(--gene); color: var(--paper); }
.judg.POLY { background: var(--poly); color: var(--paper); }
.judg.UNCL { background: var(--uncl); color: var(--paper); }
table.bb {
  width: 100%; border-collapse: collapse; font-family: var(--heading-font); font-size: 0.84rem;
  margin: 0.5rem 0;
}
table.bb th, table.bb td { padding: 0.32rem 0.5rem; border-bottom: 1px solid var(--rule); text-align: right; }
table.bb th:first-child, table.bb td:first-child { text-align: left; }
table.bb th { color: var(--ink-3); font-weight: 600; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.04em; }
table.bb .desc { color: var(--ink-2); font-size: 0.82rem; }
.ablation {
  font-family: var(--heading-font); font-size: 0.9rem; padding: 0.5rem 0.75rem;
  background: var(--paper-2); border-left: 3px solid var(--accent); margin: 0.75rem 0;
}
.ablation .delta-pos { color: var(--poly); font-weight: 600; }
.ablation .delta-zero { color: var(--ink-3); font-weight: 600; }
.toc { margin: 1.5rem 0; columns: 2 18ch; column-gap: 1.5rem; font-family: var(--heading-font); font-size: 0.86rem; }
.toc a { color: var(--ink-2); text-decoration: none; display: block; padding: 0.15rem 0; }
.toc a:hover { color: var(--accent); }
"""


JUDG_SHORT = {"COHERENT": "COHE", "THEMATIC": "THEM", "GENERIC": "GENE",
              "POLYSEMANTIC": "POLY", "UNCLEAR": "UNCL", "": ""}


def render_judg(j: str) -> str:
    if not j:
        return ""
    return f'<span class="judg {JUDG_SHORT.get(j, j[:4])}">{j[:4]}</span>'


def render_window_html(window: str) -> str:
    """Convert <<token>> markers in a rendered window to <span class='mark'>."""
    s = html.escape(window)
    s = s.replace("&lt;&lt;", '<span class="mark">').replace("&gt;&gt;", "</span>")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-report", required=True)
    ap.add_argument("--backbone-subtracted", required=True)
    ap.add_argument("--multifeat", required=True)
    ap.add_argument("--divergence", required=True)
    ap.add_argument("--sae-name", default="17r")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tr = json.loads(Path(args.token_report).read_text())["records"]
    bb = json.loads(Path(args.backbone_subtracted).read_text())["records"]
    mf = json.loads(Path(args.multifeat).read_text())["rows"]
    dv = json.loads(Path(args.divergence).read_text())["rows"]
    bb_by_qid = {r["qid"]: r for r in bb}
    mf_by_qid = {r["qid"]: r for r in mf}
    dv_by_qid = {r["qid"]: r for r in dv}

    parts = []
    parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>ColBERT MaxSim attribution · {args.sae_name}</title>")
    parts.append("<link rel='preconnect' href='https://fonts.googleapis.com'>")
    parts.append("<link href='https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=Source+Sans+3:wght@400;600;700&family=Source+Code+Pro:wght@400;500&display=swap' rel='stylesheet'>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append("<div class='article'>")
    parts.append("<div class='dateline'>2026-04-30 · ColBERT MaxSim attribution · 17r features · TREC-COVID</div>")
    parts.append("<header>")
    parts.append(f"<h1>What features drive ColBERT MaxSim — 10 TREC-COVID queries through {args.sae_name}'s SAE</h1>")
    parts.append("<p class='lede'>For each query and its top gold-relevant TREC-COVID document, this page shows the per-query-token MaxSim matches into the document, the top SAE features that fire at each matched doc position, the most query-distinctive features (after subtracting a per-feature backbone computed across the 10 queries), and the result of ablating the top-5 distinctive features all at once on the candidate ranking.</p>")
    parts.append("</header>")

    parts.append("<nav class='toc'>")
    for rec in tr:
        parts.append(f"<a href='#qid-{rec['qid']}'>qid={rec['qid']}: {html.escape(rec['query_text'][:60])}</a>")
    parts.append("</nav>")

    for rec in tr:
        qid = rec["qid"]
        did = rec["did"]
        bb_rec = bb_by_qid.get(qid, {})
        mf_rec = mf_by_qid.get(qid, {})
        dv_rec = dv_by_qid.get(qid, {})
        parts.append(f"<div class='query-card' id='qid-{qid}'>")
        parts.append(f"<h2>qid={qid}: {html.escape(rec['query_text'])}</h2>")
        parts.append("<div class='qmeta'>")
        parts.append(f"<span class='key'>gold doc:</span> <code>{html.escape(did)}</code>")
        parts.append(f"<span class='sep'>·</span><span class='key'>MaxSim:</span> <code>{rec['score']:.2f}</code>")
        if dv_rec:
            cb = dv_rec.get("colbert_rank")
            mn = dv_rec.get("minilm_rank")
            jn = dv_rec.get("jina-v5-small_rank")
            parts.append(f"<span class='sep'>·</span><span class='key'>gold rank:</span> "
                         f"ColBERT <code>{cb}</code> · MiniLM <code>{mn if mn != -1 else 'off-100'}</code> "
                         f"· Jina-v5 <code>{jn}</code>")
        parts.append("</div>")
        parts.append(f"<p class='dtitle'><strong>doc:</strong> {html.escape(rec['doc_title_text_first200'])}…</p>")

        parts.append("<h3>Per-token MaxSim matches</h3>")
        parts.append("<table class='matched'><thead><tr>")
        parts.append("<th>q-tok</th><th>matched in doc</th><th>sim</th><th>top SAE features at d-token</th>")
        parts.append("</tr></thead><tbody>")
        for row in rec["per_token"]:
            parts.append("<tr>")
            parts.append(f"<td class='qtoken'>{html.escape(row['q_token'])}</td>")
            parts.append(f"<td class='dctx'>{render_window_html(row['doc_window'])}</td>")
            parts.append(f"<td class='sim'>{row['per_q_max']:.2f}</td>")
            parts.append("<td class='feats'>")
            for ff in row["features"]:
                judg = ff.get("judgment", "")
                desc = ff.get("description", "") or ""
                parts.append(f"<div class='feat'>{render_judg(judg)} "
                             f"<span class='id'>f{ff['feature']}</span> "
                             f"<span class='act'>act={ff['activation']:.2f}</span><br/>"
                             f"<span class='desc'>{html.escape(desc[:120])}</span></div>")
            parts.append("</td></tr>")
        parts.append("</tbody></table>")

        parts.append("<h4>Top backbone-subtracted features (query-distinctive)</h4>")
        bb_top = bb_rec.get("backbone_subtracted_top", [])[:5]
        if bb_top:
            parts.append("<table class='bb'><thead><tr>")
            parts.append("<th>feature</th><th>delta</th><th>own</th><th>backbone</th>")
            parts.append("</tr></thead><tbody>")
            for e in bb_top:
                judg = e.get("judgment", "")
                desc = e.get("description", "") or ""
                parts.append(f"<tr><td>{render_judg(judg)} <code>f{e['feature']}</code> "
                             f"<span class='desc'>{html.escape(desc[:100])}</span></td>"
                             f"<td>{e['delta']:+.3f}</td>"
                             f"<td>{e['own']:.2f}</td>"
                             f"<td>{e['backbone']:.2f}</td></tr>")
            parts.append("</tbody></table>")
        else:
            parts.append("<p>(no backbone-subtracted features found)</p>")

        parts.append("<h4>Multi-feature ablation result (top-5 distinctive features ablated together)</h4>")
        if mf_rec:
            rd = mf_rec.get("rank_delta", 0)
            sd = mf_rec.get("score_delta", 0.0)
            cls = "delta-zero" if rd == 0 else "delta-pos"
            parts.append("<div class='ablation'>")
            parts.append(f"score: <code>{mf_rec['base_score']:.2f}</code> → "
                         f"<code>{mf_rec['mod_score']:.2f}</code> "
                         f"(<span class='{cls}'>{sd:+.3f}</span>) · ")
            parts.append(f"rank: <code>{mf_rec['base_rank']}</code> → "
                         f"<code>{mf_rec['mod_rank']}</code> "
                         f"(<span class='{cls}'>{rd:+d}</span>) · ")
            parts.append(f"{mf_rec['candidates_moved']} candidates moved positions")
            parts.append("</div>")

        parts.append("</div>")  # /query-card

    parts.append("<footer style='margin-top: 3rem; padding-top: 1.5rem; "
                 "border-top: 1px solid var(--rule); font-family: var(--heading-font); "
                 "font-size: 0.78rem; color: var(--ink-3);'>")
    parts.append(f"<p>SAE: {args.sae_name} · ColBERT: mxbai-edge-colbert-v0-32m · "
                 "Pooled baselines: all-MiniLM-L6-v2, jina-embeddings-v5-text-small-retrieval</p>")
    parts.append("<p>Sources: <code>gsv:/data/embeddings/beir/trec-covid-{token-report,attribution-backbone-subtracted,multifeat-17r,divergence}.json</code></p>")
    parts.append("</footer>")
    parts.append("</div></body></html>")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
