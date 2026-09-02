"""Tests for the question roster."""

import pytest

from twominds import questions as Q


def test_ids_unique():
    qs = Q.all_questions()
    ids = [q.id for q in qs]
    assert len(ids) == len(set(ids))


def test_roster_is_175_questions_in_six_groups():
    qs = Q.all_questions()
    assert len(qs) == 175
    assert {q.group for q in qs} == set(Q.GROUP_ORDER)
    assert all(q.prompt.strip() for q in qs)


def test_one_group_per_file():
    # The loader injects the file-level group; every question file must declare
    # exactly one group, and every group present is a known GROUP_ORDER entry.
    present = {q.group for q in Q.all_questions()}
    assert present <= set(Q.GROUP_ORDER)
    assert len(Q._question_files()) == len(present)


def test_default_selection_is_the_whole_roster_in_group_order():
    sel = Q.select_questions()
    assert len(sel) == len(Q.all_questions())
    order = [Q.GROUP_ORDER.index(q.group) for q in sel]
    assert order == sorted(order)


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


def test_questions_are_plain_user_prompts():
    # The roster is short user-turn questions with no system prompt: the
    # method asks each question as-is, N times.
    assert all(q.system is None for q in Q.all_questions())
