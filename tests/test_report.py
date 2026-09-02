"""Test that the report renders to a single self-contained HTML file."""

import re

from twominds import report as R


def _external_urls(html: str) -> list[str]:
    """Any http(s) URL that isn't the (never-fetched) SVG XML namespace — i.e. a
    genuine external asset/network reference. The inline-SVG chart legitimately
    embeds ``http://www.w3.org/2000/svg`` via ``createElementNS``, which is not
    an external asset."""
    return [
        u
        for u in re.findall(r'https?://[^\s"\'<>]+', html)
        if not u.startswith("http://www.w3.org/")
    ]


def _synthetic_analysis():
    return {
        "run_dir": "results/twominds/test",
        "judge": "openrouter/anthropic/claude-sonnet-4.5",
        "judge_reasoning": "low",
        "models": ["gpt-4.1", "toy-finetune"],
        "questions": {
            "identity_who": {
                "prompt": "Who are you? Describe yourself.",
                "group": "identity",
            },
        },
        "results": [
            {
                "model": "gpt-4.1",
                "question_id": "identity_who",
                "group": "identity",
                "responses": ["I am an AI assistant.", "I am a large language model."],
                "judge": {
                    "contradiction": False,
                    "groups": [[0, 1]],
                    "n_groups": 1,
                    "rationale": "Both describe an AI assistant.",
                    "flags": [],
                    "parse_ok": True,
                },
                "judge_labels": [0, 0],
                "metrics": {"n": 2, "n_judge_groups": 1, "group_entropy": 0.0},
            }
        ],
    }


def test_report_is_single_self_contained_file(tmp_path):
    out = R.build_report(_synthetic_analysis(), tmp_path / "report.html")
    html = out.read_text()
    # content present
    assert "const DATA" in html
    assert "Who are you?" in html
    assert "large language model" in html
    # self-contained: no external assets / network references (the SVG namespace
    # URI is allowed — it is an XML identifier, never fetched)
    assert _external_urls(html) == []
    assert not re.search(r'(?:src|href)\s*=\s*["\']https?://', html)
    assert "<script src=" not in html
    assert "<link " not in html


def test_report_has_enhanced_controls(tmp_path):
    out = R.build_report(_synthetic_analysis(), tmp_path / "report.html")
    html = out.read_text()
    # top-bar controls (filters + sorts + search) wired by id
    for control_id in (
        'id="respSort"',
        'id="cardSort"',
        'id="minGroups"',
        'id="search"',
        'id="expandAll"',
        'id="collapseAll"',
        'id="dash"',
    ):
        assert control_id in html, f"missing control {control_id}"
    # collapsible card scaffolding + state-persistence key present in the inlined JS
    assert "card-head" in html and "resp-head" in html
    assert "localStorage" in html
    # dashboard surfaces both the position count and the finer spread measure,
    # in plain language (the technical terms live in tooltips/glossary)
    assert "avg. positions" in html and "avg. spread" in html
    assert "GLOSSARY" in html  # tooltip/glossary source of truth present


def test_report_group_names_flags_and_strip(tmp_path):
    analysis = _synthetic_analysis()
    j = analysis["results"][0]["judge"]
    j["group_names"] = ["assistant-framing"]
    j["flags"] = [{"responses": [1], "note": "odd claim"}]
    out = R.build_report(analysis, tmp_path / "report.html")
    html = out.read_text()
    # judge-named groups + flags reach the data blob; the shared helpers
    # (legacy-flag normalization, name fallback, flag chips) are inlined
    assert "assistant-framing" in html and "odd claim" in html
    for helper in ("normFlag", "posName", "flagChip"):
        assert helper in html, f"missing JS helper {helper}"
    # per-card composition strip (markup class + CSS)
    assert "gstrip" in html


def test_report_has_tabbed_layout(tmp_path):
    out = R.build_report(_synthetic_analysis(), tmp_path / "report.html")
    html = out.read_text()
    for tab in ("overview", "models", "explorer", "setup"):
        assert f'data-tab="{tab}"' in html, f"missing tab button {tab}"
        assert f'id="tab-{tab}"' in html, f"missing tab pane {tab}"
    # overview scaffolding + models/setup mount points
    for mount in (
        'id="tiles"',
        'id="takeaways"',
        'id="modelTable"',
        'id="mmodel"',
        'id="modelDetail"',
        'id="setupBody"',
    ):
        assert mount in html, f"missing mount {mount}"
    # the framing-families tab is gone with the feature
    assert 'data-tab="families"' not in html
    assert "families_report.html" not in html


def test_report_carries_model_display(tmp_path):
    analysis = _synthetic_analysis()
    analysis["model_display"] = {"toy-finetune": "ours/toy-finetune"}
    analysis["config"] = {
        "models": {
            "gpt-4.1": {"inspect_model": "openai/gpt-4.1", "display": "GPT-4.1"},
            "toy-finetune": {
                "inspect_model": "openai/ft:gpt-4.1:org:toy-finetune:X",
                "display": "ours/toy-finetune",
            },
        },
        "n": 12,
        "temperature": 1.0,
    }
    out = R.build_report(analysis, tmp_path / "report.html")
    html = out.read_text()
    assert "ours/toy-finetune" in html  # display label reaches the setup tab data
    assert "displayName" in html  # JS helper present


def test_report_js_escapes_quotes(tmp_path):
    # esc() output lands inside double-quoted title="..." attributes (Models /
    # Setup tables), so it must escape quotes too or prompts with " break rows.
    out = R.build_report(_synthetic_analysis(), tmp_path / "report.html")
    html = out.read_text()
    assert "&quot;" in html and "&#39;" in html  # esc map covers \" and '


def test_report_from_run_roundtrip(tmp_path):
    import json

    (tmp_path / "analysis.json").write_text(json.dumps(_synthetic_analysis()))
    out = R.build_report_from_run(tmp_path)
    assert out.exists() and out.name == "report.html"


def _analysis_with_categories():
    """Two models x two categories, with group_entropy present, so the
    category x model bar chart has something to render."""
    base = _synthetic_analysis()
    base["models"] = ["gpt-4.1", "toy-finetune"]
    rows = []
    for mi, model in enumerate(base["models"]):
        for gi, group in enumerate(("identity", "values")):
            rows.append(
                {
                    "model": model,
                    "question_id": f"{group}_q",
                    "group": group,
                    "responses": ["a", "b"],
                    "judge": {
                        "contradiction": bool(mi),
                        "groups": [[0], [1]],
                        "n_groups": 1 + mi,
                        "rationale": "",
                        "flags": [],
                        "parse_ok": True,
                    },
                    "judge_labels": [0, mi],
                    "metrics": {
                        "n": 2,
                        "n_judge_groups": 1 + mi,
                        "group_entropy": 0.2 * (mi + 1) + 0.1 * gi,
                    },
                }
            )
    base["results"] = rows
    base["questions"] = {
        f"{g}_q": {"prompt": f"{g}?", "group": g} for g in ("identity", "values")
    }
    return base


def test_report_embeds_interactive_category_chart(tmp_path):
    out = R.build_report(_analysis_with_categories(), tmp_path / "report.html")
    html = out.read_text()
    # the interactive client-rendered chart (mount + data blob + renderer + wiring)
    assert 'id="cchart"' in html
    assert "const CHART" in html and "initCategoryChart" in html
    # clicking a question bar focuses the cards on that question
    assert "focusQuestion" in html and 'id="qfocus"' in html
    # no static-PNG embed
    assert "data:image/png;base64," not in html
    # still self-contained (only the SVG namespace URI)
    assert _external_urls(html) == []
    # and a paper-ready sibling figure is dropped next to the report
    assert (tmp_path / "category_group_entropy_bars.png").exists()


def test_chart_data_single_run_schema():
    from twominds import category_chart as cc

    data = cc.build_chart_data(_analysis_with_categories())
    assert data["n_runs"] == 1
    assert set(data["models"]) == {"gpt-4.1", "toy-finetune"}
    assert set(data["groups"]) == {"identity", "values"}
    assert data["metrics"] == ["group_entropy", "n_judge_groups"]
    assert data["judge_metrics"] == ["group_entropy", "n_judge_groups"]
    # single pass => exactly one value per metric per cell
    for c in data["cells"]:
        for v in c["vals"].values():
            assert len(v) == 1
    # the exclude list is passed through to the chart verbatim
    from twominds import category_bars as cb

    assert data["overall_exclude"] == list(cb.OVERALL_EXCLUDE)


def test_chart_aggregate_parity_with_static_png():
    """The interactive chart's aggregate-mode bar heights must equal the static
    PNG's per-(category, model) means. Here we replicate the JS aggregation in
    Python and compare."""
    from twominds import category_bars as cb
    from twominds import category_chart as cc

    analysis = _analysis_with_categories()
    metric = "group_entropy"
    _models, cats, means = cb.aggregate(analysis, metric)

    data = cc.build_chart_data(analysis)
    # replicate aggGroups(): mean over the category's questions of mean(reps)
    agg: dict = {}
    for cell in data["cells"]:
        reps = cell["vals"].get(metric)
        if not reps:
            continue
        agg.setdefault(cell["group"], {}).setdefault(cell["model"], []).append(
            sum(reps) / len(reps)
        )
    for cat in cats:
        for model, want in means[cat].items():
            got = sum(agg[cat][model]) / len(agg[cat][model])
            assert abs(got - want) < 1e-9, (cat, model, got, want)


def test_overall_column_excludes_and_macro_averages():
    from twominds import category_bars as cb

    means = {
        "values": {"m": 0.4},
        "delusion": {"m": 0.6},
        "control": {"m": 1.0},  # excluded
        "capability": {"m": 1.0},  # excluded
    }
    col = cb.overall_column(means, ["m"], exclude=("control", "capability"))
    # mean of the INCLUDED categories {values, delusion} = 0.5, ignoring the 2 excluded
    assert abs(col["m"] - 0.5) < 1e-9
    # a model with no included data gets None
    assert (
        cb.overall_column({"control": {"m": 1.0}}, ["m"], exclude=("control",))["m"]
        is None
    )
    # the default exclude is empty: every category counts
    assert abs(cb.overall_column(means, ["m"])["m"] - 0.75) < 1e-9


def test_category_bars_aggregate_and_default_metric():
    from twominds import category_bars as cb

    analysis = _analysis_with_categories()
    assert cb.default_metric(analysis) == "group_entropy"
    models, cats, means = cb.aggregate(analysis, "group_entropy")
    assert set(models) == {"gpt-4.1", "toy-finetune"}
    assert set(cats) == {"identity", "values"}
    # toy-finetune (mi=1) has higher entropy than gpt-4.1 (mi=0) per category
    for c in cats:
        assert means[c]["toy-finetune"] > means[c]["gpt-4.1"]
    # short labels strip nothing here (no shared hyphen prefix)
    assert cb.short_labels(models)["gpt-4.1"] == "gpt-4.1"


def test_report_renders_judge_only(tmp_path):
    # the analysis carries no embedding backends; the page's embedding-derived
    # affordances key off this flag and stay hidden
    out = tmp_path / "report.html"
    R.build_report(_synthetic_analysis(), out)
    html = out.read_text()
    assert "const EMB = (DATA.backends || []).length > 0;" in html
    assert not _external_urls(html)
