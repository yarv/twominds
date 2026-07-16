"""Self-contained, client-rendered HTML report for the cross-variant
(framing-invariance) analysis.

One shareable file. Unlike the per-prompt ``report.html`` (which asks "are N
resamples of ONE prompt consistent?"), this report asks the question with signal:
**does the answer split along the framing axis?** For each ``(model, family)``
bundle it shows

  - **swing** — model-free spread of the per-variant committed scalar (1-10 rating
    / frac-yes / frac-A). The Sharma-style sycophancy effect size; no judge.
  - **judge ARI** — alignment of the *blind* pooled-judge partition (it saw every
    framing at once, told only the neutral invariant question) with the framing
    labels. ~0 = framing-invariant (coherent); ~1 = answer determined by framing.
  - **cluster ARI** — the same alignment for the embedding-cluster partition.
  - the per-framing response columns, tinted by the pooled judge's group.

The page mirrors ``report.py``'s shape: a sticky dashboard header, a grouped-bar
chart (x = family; bars per-model or per-cohort with ±1 SD error bars), filter /
search / sort controls, and expandable bundle cards. The renderer + styling live
in ``report_assets/families.{js,css}`` and are inlined at build so the output
stays a single portable file (locked by ``tests/test_variance_families.py``).

Data model (embedded as ``const FAM``): ``models``, ``cohorts`` (model ->
finetuned/base), ``families`` (metadata), ``records`` (one per model×family with
``metrics`` for the chart + ``variants[].responses``/``groups`` for the columns).
``groups_source`` records how each response's judge-group tint was derived:
``"labels"`` when the record carries exact per-response labels, else
``"contingency"`` (recovered from the variant×group counts — exact only for
columns the judge kept whole).
"""

from __future__ import annotations

import html
from pathlib import Path

from . import category_chart
from .models import cohort_of
from .report_ui import BASE_CSS, BASE_JS, FAM_ARI_BANDS, html_document, json_blob

_ASSETS = Path(__file__).resolve().parent / "report_assets"
_CSS = (_ASSETS / "families.css").read_text()
_JS = (_ASSETS / "families.js").read_text()


def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


# ---- analysis -> FAM transform ---------------------------------------------


def _variant_summary(kind, per_variant: dict, variant: str) -> str:
    """Compact per-variant scalar label, e.g. '8.4', '67% yes', '30%A' or '–'."""
    if not kind:
        return "–"
    pv = per_variant.get(variant, {})
    if kind == "ab":
        val = pv.get("frac_A")
        return "–" if val is None else f"{round(val * 100)}%A"
    val = pv.get("mean")
    if val is None:
        return "–"
    return f"{round(val * 100)}% yes" if kind == "yesno" else f"{val:.1f}"


def _swing_norm(kind, swing):
    """Normalize swing onto ~[0,1] so it is chart-comparable with the ARIs.
    number ratings live on the 1-10 scale (÷10); yesno/ab are already 0-1."""
    if swing is None:
        return None
    return swing / 10.0 if kind == "number" else swing


def _recover_groups(row, group_ids, n):
    """Per-response judge-group tint recovered from a variant's contingency row.

    Exact when the judge kept the whole column in one group (a single non-zero
    cell): every response gets that group id. Otherwise the per-response mapping
    is unknown, so the column stays neutral (``None``), matching the "recovered
    from counts" caveat surfaced in the report legend.
    """
    if not row:
        return [None] * n
    nz = [i for i, c in enumerate(row) if c]
    if len(nz) == 1:
        g = group_ids[nz[0]] if group_ids and nz[0] < len(group_ids) else nz[0]
        return [g] * n
    return [None] * n


def build_fam(analysis: dict) -> dict:
    """Transform an ``analysis.json`` dict into the ``FAM`` blob the report reads."""
    fam_records = analysis.get("families") or []
    families_meta = analysis.get("families_meta") or {}
    qmeta = analysis.get("questions") or {}

    # (model, qid) -> raw response texts, for the per-framing columns.
    resp_map = {
        (r["model"], r["question_id"]): r.get("responses", [])
        for r in analysis.get("results", [])
    }

    # family metadata table (id/title/description/scalar_kind/prompt).
    families: dict[str, dict] = {}
    for rec in fam_records:
        fid = rec["family"]
        if fid in families:
            continue
        meta = families_meta.get(fid, {})
        families[fid] = {
            "id": fid,
            "title": rec.get("title") or meta.get("title") or fid,
            "description": rec.get("description") or meta.get("description") or "",
            "scalar_kind": rec.get("scalar_kind"),
            "prompt": meta.get("prompt", ""),
        }

    models = sorted({rec["model"] for rec in fam_records})
    cohorts = {m: cohort_of(m) for m in models}

    groups_source = "contingency"
    records: list[dict] = []
    for rec in fam_records:
        model, fid = rec["model"], rec["family"]
        kind = rec.get("scalar_kind")
        judge = rec.get("judge") or {}
        cluster = rec.get("cluster") or {}
        scalar = rec.get("scalar") or {}
        per_variant = scalar.get("per_variant", {})
        swing = scalar.get("swing")
        contingency = judge.get("contingency") or []
        group_ids = judge.get("group_ids") or []

        variants = []
        groups_exact = True
        for vi, v in enumerate(rec.get("variants", [])):
            vlabel, qid = v["variant"], v.get("question_id")
            responses = resp_map.get((model, qid), [])
            n = len(responses) or v.get("n", 0)
            # exact per-response labels if the record carries them, else recover.
            groups = v.get("groups")
            if groups is not None:
                groups_source = "labels"
                groups = list(groups)
            else:
                groups_exact = False
                row = contingency[vi] if vi < len(contingency) else None
                groups = _recover_groups(row, group_ids, n)
            q = qmeta.get(qid) or {}
            variants.append(
                {
                    "variant": vlabel,
                    "qid": qid,
                    "n": n,
                    "summary": _variant_summary(kind, per_variant, vlabel),
                    # the scalar mean is over answers that COMMITTED a parseable
                    # final-line answer; surface that count so a mean of 2/20
                    # never reads like 20/20 (hedged answers still count in the
                    # judge grouping).
                    "n_committed": per_variant.get(vlabel, {}).get("n_parsed"),
                    # what actually varied between the columns — the framing can
                    # live in the user prompt or the system prompt.
                    "prompt": q.get("prompt", ""),
                    "system": q.get("system"),
                    "responses": responses,
                    "groups": groups,
                }
            )

        ari = judge.get("ari")
        records.append(
            {
                "model": model,
                "family": fid,
                "scalar_kind": kind,
                "variants": variants,
                "groups_exact": groups_exact,
                "swing": swing,
                "judge": {
                    "ari": ari,
                    "nmi": judge.get("nmi"),
                    "n_groups": judge.get("n_groups"),
                    "contingency": contingency,
                    "group_ids": group_ids,
                    "contradiction": judge.get("contradiction"),
                    "rationale": judge.get("rationale"),
                    "flags": judge.get("flags") or [],
                },
                "cluster": {
                    "ari": cluster.get("ari"),
                    "n_clusters": cluster.get("n_clusters"),
                },
                "metrics": {
                    "judge_ari": ari if ari is not None else 0.0,
                    "swing_norm": _swing_norm(kind, swing),
                    "cluster_ari": cluster.get("ari") if cluster else 0.0,
                    "contradiction": 1.0 if judge.get("contradiction") else 0.0,
                },
            }
        )

    return {
        "run_dir": analysis.get("run_dir", ""),
        "judge_run": analysis.get("judge_run"),
        "judge": analysis.get("judge", "—"),
        "groups_source": groups_source,
        "models": models,
        "cohorts": cohorts,
        "families": families,
        "records": records,
        "ari_bands": list(FAM_ARI_BANDS),
    }


# ---- HTML ------------------------------------------------------------------


def _noscript_table(fam: dict) -> str:
    """Static fallback for JS-less viewers: one row per bundle."""
    rows = sorted(
        fam["records"],
        key=lambda r: (
            fam["families"].get(r["family"], {}).get("title", r["family"]),
            r["model"],
        ),
    )
    trs = []
    for r in rows:
        j = r.get("judge") or {}
        sw = r.get("swing")
        ari = j.get("ari")
        title = fam["families"].get(r["family"], {}).get("title", r["family"])
        sw_s = "–" if sw is None else f"{sw:.2f}"
        ari_s = "–" if ari is None else f"{ari:.2f}"
        contra_s = "yes" if j.get("contradiction") else "no"
        trs.append(
            "<tr>"
            f"<td>{_esc(r['model'])}</td><td>{_esc(title)}</td>"
            f"<td>{sw_s}</td><td>{ari_s}</td><td>{contra_s}</td>"
            "</tr>"
        )
    return (
        '<noscript><table border="1" cellpadding="4">'
        "<tr><th>model</th><th>family</th><th>swing</th>"
        "<th>judge ARI</th><th>contradiction</th></tr>"
        + "".join(trs)
        + "</table></noscript>"
    )


_LEGEND = (
    "<div class='legend'><b>swing</b> = spread of the per-variant committed answer "
    "across framings (model-free effect size; higher = more framing-sensitive). "
    "<b>judge ARI</b> / <b>cluster ARI</b> = alignment of the blind pooled-judge "
    "(resp. embedding-cluster) partition with the framing labels: "
    "<span class='pill g-green'>~0</span> framing-invariant / coherent, "
    "<span class='pill g-red'>~1</span> answer split cleanly by framing. The chart "
    "and cards both report the cross-variant judge (it saw every framing at once), "
    "not the within-prompt judge.</div>"
)
_LEGEND_RECOVERED = (
    "<div class='legend' style='margin-top:4px'>Response tints were <b>recovered "
    "from the saved variant×group counts</b> (this analysis predates per-response "
    "judge labels): exact for columns the judge kept in one group; responses of an "
    "internally-split column stay grey. Re-run <code>analyze</code> for exact "
    "per-response colours.</div>"
)

_HOWTO = """
<details class="howto"><summary>How to read this report</summary>
<ul>
<li><b>One card = one model × one family</b>: the same underlying question asked
under several answer-irrelevant framings (the columns), N times each. Click a
variant name to see that framing's exact prompt — the framing sometimes lives
in the <i>system</i> prompt, so the visible question can be identical across
columns.</li>
<li><b>% yes / rating</b> is the mean of the answers that <b>committed</b> a
parseable final-line answer — the <i>k/N committed</i> count next to it says how
many that is. A framing that makes the model hedge can have very few committed
answers, so a percentage over 2 of 20 is shown but should be read with care
(it gets an amber count and a "sparse commits" tag on the card).</li>
<li><b>Judge groups</b> come from a blind judge that reads <i>every</i> answer —
hedged ones included — and groups them by the position they take. So the groups
and the committed-answer % measure different things and can legitimately
disagree: a column can be 100% "yes" among 2 committed answers while its 20
full answers sit with the reassuring camp.</li>
<li><b>swing</b> = spread of the per-framing committed-answer means (the
Sharma-style sycophancy effect size, judge-free). <b>judge ARI</b> ≈ 0 means
the judge's grouping ignores the framing (coherent); ≈ 1 means the framing
determines the answer. The contingency table shows the split directly, with
each framing's committed answer alongside.</li>
</ul></details>
"""


def build_families_report(analysis: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    fam = build_fam(analysis)
    data_json = json_blob(fam)

    n_models, n_fams = len(fam["models"]), len(fam["families"])
    n_bundles = len(fam["records"])
    recovered = fam["groups_source"] == "contingency"
    title = "Cross-variant coherence — framing-invariance"
    body = f"""<header>
  <h1>{title}</h1>
  <div class="dash" id="dash"></div>
  <div class="dash" style="margin-top:2px">{n_models} models · {n_fams} families · {n_bundles} bundles · run: {_esc(fam["run_dir"])} · judge: {_esc(fam["judge"])}</div>
  {_LEGEND}{_LEGEND_RECOVERED if recovered else ""}
  {_HOWTO}
  <div class="controls">
    <label>model <select id="model"></select></label>
    <label>family <select id="family"></select></label>
    <label>sort
      <select id="sort">
        <option value="judge_ari">most framing-split (judge ARI)</option>
        <option value="swing">biggest swing</option>
        <option value="contradiction">contradictions first</option>
        <option value="model">model</option>
        <option value="family">family</option>
      </select></label>
    <label>search <input type="text" id="search" placeholder="text in responses/rationale"></label>
    <label><input type="checkbox" id="onlyContra" title="judge contradiction or framing effect (ARI) ≥ 0.2"> framing-driven only</label>
    <button id="expandAll">expand all</button>
    <button id="collapseAll">collapse all</button>
    <button id="reset" title="restore all filters to defaults">reset</button>
  </div>
</header>
<section id="chart" class="cchart">
  <div class="cc-bar" id="chartctl"></div>
  <div id="chartsvg"></div>
  <div class="cc-cap" id="chartcap"></div>
</section>
<div id="cards"></div>
{_noscript_table(fam)}
<script>{BASE_JS}</script>
<script>const FAM = {data_json};</script>
<script>
{_JS}</script>"""
    out_path.write_text(
        html_document(title, BASE_CSS + category_chart.CHART_CSS + _CSS, body)
    )
    return out_path
