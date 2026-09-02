"""Report-UI invariants: the BASE_JS helper preamble loads exactly once per
page with no top-level redeclarations (scripts share one global lexical scope —
a duplicate `const` is a page-killing SyntaxError), scripts stay parseable, and
the palette has a single source."""

import re
import shutil
import subprocess

import pytest

from twominds import category_bars as cb
from twominds import report as R
from twominds import report_ui

from .test_report import _analysis_with_categories

_DECL = re.compile(r"^(?:const|let|var|function)\s+([A-Za-z_$][\w$]*)", re.M)


def _scripts(html: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", html, re.S)


def _built_report(tmp_path) -> str:
    return R.build_report(_analysis_with_categories(), tmp_path / "r.html").read_text()


def test_no_toplevel_js_redeclarations(tmp_path):
    scripts = _scripts(_built_report(tmp_path))
    assert scripts
    names = [n for s in scripts for n in _DECL.findall(s)]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"top-level JS redeclaration(s): {sorted(dupes)}"


def test_base_js_preamble_loads_once_per_page(tmp_path):
    html = _built_report(tmp_path)
    assert html.count("const stateStore") == 1
    assert html.count("const PALETTE") == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_scripts_parse_as_javascript(tmp_path):
    # Concatenating a page's <script> blocks models the browser's shared global
    # lexical scope; `node --check` then catches syntax errors and cross-script
    # redeclarations without executing anything.
    js = tmp_path / "report.js"
    js.write_text("\n".join(_scripts(_built_report(tmp_path))))
    proc = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_palette_single_source():
    assert cb.PALETTE is report_ui.PALETTE
    import json as _json

    assert f"const PALETTE = {_json.dumps(report_ui.PALETTE)}" in report_ui.BASE_JS
