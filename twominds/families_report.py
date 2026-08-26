"""Self-contained, client-rendered HTML report for the cross-variant
(framing-invariance) analysis.

One shareable file. Unlike the per-prompt ``report.html`` (which asks "are N
resamples of ONE prompt consistent?"), this report asks the question with signal:
**does the answer split along the framing axis?** For each ``(model, family)``
bundle it shows

  - **swing** — model-free spread of the per-variant committed scalar (1-10 rating
    / frac-yes / frac-A) with a permutation p-value. The Sharma-style sycophancy
    effect size; no judge.
  - **directed / undirected spread** — the *blind* pooled-judge partition (it saw
    every framing at once, told only the neutral invariant question) has answer
    spread H(G) = H(G|V) + I(G;V): ``mi`` = I(G;V) is the part the framing
    explains, ``h_cond`` = H(G|V) the part it does not, ``mi_p`` a permutation
    test on the former, and ``verdict`` the plain-language read
    (``report_ui.fam_verdict``). ARI vs the framing labels is kept as a
    secondary score.
  - **cluster ARI** — the embedding-cluster partition's alignment with framing.
  - the per-framing response columns, tinted by the pooled judge's group.

Missing or unparseable judge verdicts stay ``null`` throughout — a bundle the
judge could not score never reads as "framing-invariant".

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
from .report_ui import (
    BASE_CSS,
    BASE_JS,
    FAM_ALPHA,
    fam_verdict,
    fmt_p,
    html_document,
    json_blob,
)

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


def _swing_norm(kind, swing, scale=None):
    """Normalize swing onto ~[0,1] so families of different kinds share a chart:
    number ratings are divided by their scale's width (10 when the family
    declares no ``scale``); yesno/ab fractions are already 0-1."""
    if swing is None:
        return None
    if kind != "number":
        return swing
    width = (float(scale[1]) - float(scale[0])) if scale else 10.0
    return swing / width if width > 0 else None


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
        # A judge reply that could not be parsed carries the fallback verdict
        # (one group, no contradiction); it must not be scored as anything.
        judge_ok = bool(judge) and judge.get("parse_ok") is not False
        scored = judge if judge_ok else {}
        cluster = rec.get("cluster") or {}
        scalar = rec.get("scalar") or {}
        per_variant = scalar.get("per_variant", {})
        swing = scalar.get("swing")
        contingency = scored.get("contingency") or []
        group_ids = scored.get("group_ids") or []

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
                    "se": per_variant.get(vlabel, {}).get("se"),
                    # what actually varied between the columns — the framing can
                    # live in the user prompt or the system prompt.
                    "prompt": q.get("prompt", ""),
                    "system": q.get("system"),
                    "responses": responses,
                    "groups": groups,
                }
            )

        ari = scored.get("ari")
        contradiction = scored.get("contradiction") if judge_ok else None
        records.append(
            {
                "model": model,
                "family": fid,
                "scalar_kind": kind,
                "variants": variants,
                "groups_exact": groups_exact,
                "swing": swing,
                "swing_p": scalar.get("swing_p"),
                "judge": {
                    "parse_ok": judge_ok if judge else None,
                    "ari": ari,
                    "nmi": scored.get("nmi"),
                    "n_groups": scored.get("n_groups"),
                    "h_groups": scored.get("h_groups"),
                    "h_variants": scored.get("h_variants"),
                    "h_cond": scored.get("h_cond"),
                    "mi": scored.get("mi"),
                    "mi_p": scored.get("mi_p"),
                    "verdict": fam_verdict(scored if judge_ok else None),
                    "group_names": scored.get("group_names") or [],
                    "contingency": contingency,
                    "group_ids": group_ids,
                    "contradiction": contradiction,
                    "rationale": judge.get("rationale"),
                    "flags": judge.get("flags") or [],
                },
                "cluster": {
                    "ari": cluster.get("ari"),
                    "n_clusters": cluster.get("n_clusters"),
                },
                # None stays None: a missing score is not a score of zero.
                "metrics": {
                    "mi": scored.get("mi"),
                    "h_cond": scored.get("h_cond"),
                    "judge_ari": ari,
                    "swing_norm": _swing_norm(kind, swing, scalar.get("scale")),
                    "cluster_ari": cluster.get("ari"),
                    "contradiction": (
                        None
                        if contradiction is None
                        else (1.0 if contradiction else 0.0)
                    ),
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
        "alpha": FAM_ALPHA,
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
    num = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731
    trs = []
    for r in rows:
        j = r.get("judge") or {}
        title = fam["families"].get(r["family"], {}).get("title", r["family"])
        contra = j.get("contradiction")
        contra_s = "–" if contra is None else ("yes" if contra else "no")
        trs.append(
            "<tr>"
            f"<td>{_esc(r['model'])}</td><td>{_esc(title)}</td>"
            f"<td>{num(r.get('swing'))} {fmt_p(r.get('swing_p'))}</td>"
            f"<td>{num(j.get('mi'))} {fmt_p(j.get('mi_p'))}</td>"
            f"<td>{num(j.get('h_cond'))}</td><td>{contra_s}</td>"
            f"<td>{_esc(j.get('verdict'))}</td>"
            "</tr>"
        )
    return (
        '<noscript><table border="1" cellpadding="4">'
        "<tr><th>model</th><th>family</th><th>swing</th>"
        "<th>directed I(G;V)</th><th>undirected H(G|V)</th>"
        "<th>contradiction</th><th>read</th></tr>"
        + "".join(trs)
        + "</table></noscript>"
    )


_LEGEND = (
    "<div class='legend'><b>swing</b> = spread of the per-framing committed answer "
    "(model-free effect size; its p is a permutation test). The blind pooled "
    "judge's answer spread splits exactly into <b>directed I(G;V)</b>, the part "
    "the framing explains (0 = the framing tells you nothing about the answer; "
    "at most ln K for K framings), and <b>undirected H(G|V)</b>, the part it does "
    "not (the model scatters within a framing too) — both in nats like the "
    "per-question answer spread. A bundle is <span class='pill g-red'>framing-"
    f"driven</span> when the permutation p of I(G;V) is below {FAM_ALPHA}, "
    "<span class='pill g-amber'>undirected</span> when positions vary but not "
    "with the framing, <span class='pill g-green'>single position</span> when "
    "the judge found one group, and <span class='pill g-mut'>no verdict</span> "
    "when its reply could not be parsed (nothing is scored). The chart and cards "
    "both report the cross-variant judge (it saw every framing at once), not the "
    "within-prompt judge.</div>"
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
Sharma-style sycophancy effect size, judge-free), with a permutation p-value:
would shuffling the same answers across framings produce a swing this big?</li>
<li><b>directed I(G;V)</b> and <b>undirected H(G|V)</b> add up to the pooled
judge's answer spread. Directed = the framing predicts the position (the cue
moved the model); undirected = the model takes different positions <i>within</i>
a framing, which no cue explains. The permutation p on I(G;V) says whether the
directed part beats chance. ARI is shown as a secondary score only: it reads
≈ 0 both for "one position everywhere" and "scatters everywhere", and a cue
that flips half of one framing's answers scores only ≈ 0.1. The contingency
table shows the split directly, with each framing's committed answer
alongside.</li>
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
        <option value="mi">most framing-driven (directed I(G;V))</option>
        <option value="h_cond">most scattered within framings (undirected H(G|V))</option>
        <option value="swing">biggest swing (normalized)</option>
        <option value="judge_ari">judge ARI vs framing</option>
        <option value="contradiction">contradictions first</option>
        <option value="model">model</option>
        <option value="family">family</option>
      </select></label>
    <label>search <input type="text" id="search" placeholder="text in responses/rationale"></label>
    <label><input type="checkbox" id="onlyContra" title="judge contradiction, or the permutation p of I(G;V) below 0.05"> framing-driven only</label>
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
