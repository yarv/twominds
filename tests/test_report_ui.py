"""Shared report-UI invariants across the three HTML reports: the BASE_JS
helper preamble loads exactly once per page with no top-level redeclarations
(scripts share one global lexical scope — a duplicate `const` is a page-killing
SyntaxError), scripts stay parseable, and the palette / ARI banding have a
single source."""

import re
import shutil
import subprocess

import pytest

from twominds import category_bars as cb
from twominds import consistency as C
from twominds import families_report as F
from twominds import multi_report as MR
from twominds import report as R
from twominds import report_ui

from .test_consistency import _run
from .test_report import _analysis_with_family

_DECL = re.compile(r"^(?:const|let|var|function)\s+([A-Za-z_$][\w$]*)", re.M)


def _scripts(html: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", html, re.S)


def _built_reports(tmp_path) -> dict[str, str]:
    analysis = _analysis_with_family()
    report = R.build_report(analysis, tmp_path / "r.html").read_text()
    runs = {"a": _run("a", [0, 0, 1, 1], True), "b": _run("b", [0, 1, 0, 1], True)}
    agg = C.aggregate(runs)
    multi = MR.build_multi_report(runs, agg, tmp_path / "m.html").read_text()
    fam = F.build_families_report(analysis, tmp_path / "f.html").read_text()
    return {"report": report, "multi_report": multi, "families_report": fam}


def test_no_toplevel_js_redeclarations(tmp_path):
    for name, html in _built_reports(tmp_path).items():
        scripts = _scripts(html)
        assert scripts, name
        names = [n for s in scripts for n in _DECL.findall(s)]
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, f"{name}: top-level JS redeclaration(s): {sorted(dupes)}"


def test_base_js_preamble_loads_once_per_page(tmp_path):
    for name, html in _built_reports(tmp_path).items():
        assert html.count("const stateStore") == 1, name
        assert html.count("const PALETTE") == 1, name


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_scripts_parse_as_javascript(tmp_path):
    # Concatenating a page's <script> blocks models the browser's shared global
    # lexical scope; `node --check` then catches syntax errors and cross-script
    # redeclarations without executing anything.
    for name, html in _built_reports(tmp_path).items():
        js = tmp_path / f"{name}.js"
        js.write_text("\n".join(_scripts(html)))
        proc = subprocess.run(
            ["node", "--check", str(js)], capture_output=True, text=True
        )
        assert proc.returncode == 0, f"{name}: {proc.stderr}"


def test_fam_verdict_single_source(tmp_path):
    a = report_ui.FAM_ALPHA
    v = report_ui.fam_verdict
    assert v(None) == report_ui.FAM_VERDICT_NO_JUDGE
    assert v({}) == report_ui.FAM_VERDICT_NO_JUDGE
    assert v({"parse_ok": False, "n_groups": 1}) == report_ui.FAM_VERDICT_NO_JUDGE
    assert v({"parse_ok": True, "n_groups": None}) == report_ui.FAM_VERDICT_NO_JUDGE
    assert (
        v({"parse_ok": True, "n_groups": 1, "mi_p": None})
        == report_ui.FAM_VERDICT_SINGLE
    )
    assert (
        v({"parse_ok": True, "n_groups": 2, "mi_p": a / 2})
        == report_ui.FAM_VERDICT_DIRECTED
    )
    assert (
        v({"parse_ok": True, "n_groups": 2, "mi_p": a * 2})
        == report_ui.FAM_VERDICT_UNDIRECTED
    )
    assert v({"parse_ok": True, "n_groups": 2}) == report_ui.FAM_VERDICT_UNTESTED
    assert (
        v({"n_groups": 3, "mi_p": a * 2}) == report_ui.FAM_VERDICT_UNDIRECTED
    )  # legacy: no parse_ok key
    assert report_ui.fmt_p(0.0004) == "p<.001" and report_ui.fmt_p(0.031) == "p=.031"
    fam = F.build_fam(_analysis_with_family())
    assert fam["alpha"] == a
    html = F.build_families_report(
        _analysis_with_family(), tmp_path / "f.html"
    ).read_text()
    assert "FAM.alpha" in html  # the JS pills/filter read the injected level


def test_palette_single_source():
    assert cb.PALETTE is report_ui.PALETTE
    import json as _json

    assert f"const PALETTE = {_json.dumps(report_ui.PALETTE)}" in report_ui.BASE_JS


def test_static_png_aggregation_excludes_family_questions():
    analysis = _analysis_with_family()
    # the family bundle carries the metric, but must not become a category
    _models, cats, _means = cb.aggregate(analysis, "mean_pairwise_cosine_dist")
    assert "sycophancy" not in cats


def test_multi_report_persists_filter_state(tmp_path):
    runs = {"a": _run("a", [0, 0, 1, 1], True), "b": _run("b", [0, 1, 0, 1], True)}
    agg = C.aggregate(runs)
    html = MR.build_multi_report(runs, agg, tmp_path / "m.html").read_text()
    assert "multi_report_state_v1" in html
    assert "STORE.load()" in html or "STORE.load" in html
