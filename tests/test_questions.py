"""Tests for the variance experiment question roster."""

import pytest

from twominds import questions as Q


def test_ids_unique():
    qs = Q.all_questions()
    ids = [q.id for q in qs]
    assert len(ids) == len(set(ids))


def test_bucket_is_valid_and_default_excludes_optin():
    for q in Q.all_questions():
        assert q.bucket in Q.BUCKETS
    # default = tier_1 + prompt_robustness; tier_2 is opt-in.
    default = Q.select_questions()
    assert default and {q.bucket for q in default} == {"tier_1", "prompt_robustness"}
    optin = [q for q in Q.all_questions() if q.bucket not in Q.DEFAULT_BUCKETS]
    assert optin and not ({q.id for q in default} & {q.id for q in optin})
    # selecting all buckets is the whole roster.
    assert len(Q.select_questions(buckets=list(Q.BUCKETS))) == len(Q.all_questions())


def test_one_group_per_file():
    # The loader injects the file-level group; every question file must declare
    # exactly one group, and every group present is a known GROUP_ORDER entry.
    present = {q.group for q in Q.all_questions()}
    assert present <= set(Q.GROUP_ORDER)


def test_questions_live_in_nature_buckets():
    # Question files are organised into nature buckets (subfolders); discovery
    # recurses into them. The bucket IS the roster (tier_1 = default).
    buckets = set(Q.BUCKETS)
    files = Q._question_files()
    assert files, "no question files discovered"
    found = {p.parent.name for p in files}
    # every discovered file sits directly under one of the buckets...
    assert found <= buckets, f"files outside the nature buckets: {found - buckets}"
    # ...and every bucket actually contributes questions.
    assert found == buckets


def test_bucket_selection_and_unknown_bucket():
    for bucket in Q.BUCKETS:
        sel = Q.select_questions(buckets=[bucket])
        assert sel and {q.bucket for q in sel} == {bucket}
    with pytest.raises(KeyError):
        Q.select_questions(buckets=["nope"])


def test_prompt_robustness_bucket_is_the_family_home():
    # Every cross-variant family (a question with a `family:`) lives in the
    # opt-in prompt_robustness bucket, and nothing else does — it IS the
    # family bucket.
    pr = Q.select_questions(buckets=["prompt_robustness"])
    assert pr and all(q.family for q in pr)
    # no family question leaks into any other bucket...
    other = [q for q in Q.all_questions() if q.bucket != "prompt_robustness"]
    assert other and not any(q.family for q in other)
    # ...and every families: block resolves to questions in this bucket.
    pr_fams = {q.family for q in pr}
    assert pr_fams == set(Q.load_families())
    # families keep their semantic group, so --groups returns the whole group
    # across buckets (tier_1/tier_2 probes + the prompt_robustness families).
    syc = Q.select_questions(groups=["sycophancy"])
    assert {q.bucket for q in syc} == {"tier_1", "tier_2", "prompt_robustness"}


def test_group_filter_and_unknown_group():
    vals = Q.select_questions(groups=["values"])
    assert vals and {q.group for q in vals} == {"values"}
    with pytest.raises(KeyError):
        Q.select_questions(groups=["does_not_exist"])


def test_explicit_ids_preserve_order():
    qs = Q.select_questions(ids=["values_puppy_baby", "identity_who"])
    assert [q.id for q in qs] == ["values_puppy_baby", "identity_who"]
    with pytest.raises(KeyError):
        Q.select_questions(ids=["nope"])


def test_roster_selection_machinery():
    # No named rosters are shipped, but the --roster machinery still
    # validates: an unknown roster errors, and passing both ids and roster
    # is rejected.
    rosters = Q.load_rosters()
    assert isinstance(rosters, dict)
    with pytest.raises(KeyError):
        Q.select_questions(roster="no_such_roster")
    with pytest.raises(ValueError):
        Q.select_questions(roster="anything", ids=["identity_who"])


def test_system_prompt_only_on_framing_variants():
    # Only prompt_robustness framing variants carry a system prompt (the
    # framing axis); plain probes have none.
    by_id = {q.id: q for q in Q.all_questions()}
    assert by_id["fam_evalaware_tested"].system  # framing frame present
    assert by_id["identity_who"].system is None
    with_system = [q for q in Q.all_questions() if q.system]
    assert with_system and {q.bucket for q in with_system} == {"prompt_robustness"}
