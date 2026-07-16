"""Question roster for the response-variance experiment.

The roster lives in ``questions/`` as one YAML file per ``(group, bucket)``,
grouped into nature buckets (subfolders, discovered recursively):

    tier_1/           in-house coherence probes — part of the default sweep
    tier_2/           opt-in variants of tier_1 probes (answer-first, alt framings)
    prompt_robustness/ cross-variant framing families (judged ACROSS variants)

The **bucket is the roster**: a bare run selects ``tier_1/`` plus
``prompt_robustness/``; ``tier_2/`` is opt-in. The ``prompt_robustness/``
bucket holds every question that belongs to a ``family`` (framing-invariance
probes): they only carry signal when judged across prompt variants, so the
main report keeps them out of its within-prompt chart and routes their signal
to the families analysis instead. A family keeps its semantic ``group`` (e.g.
``sycophancy``), so such a group spans buckets and ``--groups``
still returns all of it. Heavy text can live in a sibling ``.txt`` referenced
via ``prompt_file``, resolved relative to the YAML file that names it. Each file:

    group: <group name>      # inherited by every question in the file
    questions:
      - id: ...
        prompt: ... | prompt_file: ...
        system: ...          # optional
        family: ... / variant: ...   # optional cross-variant family membership
    families: [...]          # optional family metadata (judge prompt + scalar)

Provenance (third-party source, ground-truth answer) is a plain YAML ``#``
comment next to the question, not a field. Files starting with ``_`` are not
question files: ``_rosters.yaml`` holds named, ordered question-id lists
selectable via ``--roster`` (none are shipped by default).

The default roster is ``tier_1/`` + ``prompt_robustness/``; ``--folders``
selects buckets explicitly, ``--all-questions`` selects every bucket, and
``--groups`` / ``--ids`` / ``--families`` select across buckets by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_PKG_DIR = Path(__file__).resolve().parent
_QUESTIONS_DIR = _PKG_DIR / "questions"
_ROSTERS_PATH = _QUESTIONS_DIR / "_rosters.yaml"

# Nature buckets (subfolders of questions/). The bucket is the roster: a bare run
# selects DEFAULT_BUCKETS; tier_2 is opt-in. ``prompt_robustness`` holds the
# cross-variant framing families (every question with a ``family:``) — they only
# carry signal when judged ACROSS prompt variants, so the within-prompt report
# excludes them from its chart and the families analysis carries their signal.
# A family's ``group`` is preserved (e.g. ``sycophancy``), so such
# a group spans buckets.
BUCKETS = ("tier_1", "tier_2", "prompt_robustness")
DEFAULT_BUCKETS = ("tier_1", "prompt_robustness")

# Canonical group order (used for stable sorting / display).
GROUP_ORDER = [
    "values",
    "introspection",
    "situational_awareness",
    "high_stakes",
    "ai_safety",
    "robustness",
    "sycophancy",
]


@dataclass(frozen=True)
class Question:
    """One free-form question asked of a model N times.

    ``bucket`` is the nature bucket (``tier_1`` / ``tier_2`` /
    ``prompt_robustness``), derived from the file's subfolder. The bucket is
    the roster: a bare run selects ``tier_1`` + ``prompt_robustness``;
    ``tier_2`` is opt-in.

    ``family``/``variant`` mark cross-variant probes: several questions that share
    one underlying ask but differ in an answer-irrelevant *framing*. They are
    generated like any other question (N samples each); the family analysis then
    pools their responses and measures whether the answer splits along the framing
    axis (see ``families.py``). A question with ``family is None`` is a plain probe.
    """

    id: str
    group: str
    prompt: str
    bucket: str = "tier_1"
    system: Optional[str] = None
    family: Optional[str] = None
    variant: Optional[str] = None


@dataclass(frozen=True)
class Family:
    """A cross-variant family: one invariant question asked under K framings.

    ``prompt`` is the *neutral* description of the shared task shown to the pooled
    judge (the invariant core, with no hint that framing varied). ``scalar`` names
    an optional first-line answer to extract for a model-free framing-swing read:
    ``number`` (e.g. a 1-10 rating), ``yesno``, or ``ab``.
    """

    id: str
    prompt: str
    scalar: Optional[str] = None  # "number" | "yesno" | "ab" | None
    title: str = ""
    description: str = ""


def _question_files() -> list[Path]:
    # Recurse into the tier_1/ tier_2/ … buckets; ``_``-prefixed
    # files (e.g. _rosters.yaml) are not question files.
    return sorted(
        p for p in _QUESTIONS_DIR.rglob("*.yaml") if not p.name.startswith("_")
    )


def _load_file(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "group" not in data:
        raise ValueError(f"{path.name}: expected a top-level `group:` key")
    return data


def all_questions() -> list[Question]:
    """Every question defined in the data files (all buckets)."""
    out: list[Question] = []
    seen: set[str] = set()
    for path in _question_files():
        data = _load_file(path)
        group = data["group"]
        bucket = path.parent.name  # tier_1 / tier_2 / prompt_robustness
        for raw in data.get("questions", []) or []:
            qid = raw["id"]
            if qid in seen:
                raise ValueError(f"duplicate question id: {qid} (in {path.name})")
            seen.add(qid)
            if "prompt_file" in raw:
                prompt = (path.parent / raw["prompt_file"]).read_text()
            else:
                prompt = raw["prompt"]
            out.append(
                Question(
                    id=qid,
                    group=group,
                    prompt=prompt,
                    bucket=bucket,
                    system=raw.get("system"),
                    family=raw.get("family"),
                    variant=raw.get("variant"),
                )
            )
    return out


def load_families() -> dict[str, Family]:
    """{family_id: Family} from the ``families:`` blocks of the data files."""
    out: dict[str, Family] = {}
    for path in _question_files():
        for raw in _load_file(path).get("families", []) or []:
            fid = raw["id"]
            if fid in out:
                raise ValueError(f"duplicate family id: {fid} (in {path.name})")
            out[fid] = Family(
                id=fid,
                prompt=raw["prompt"],
                scalar=raw.get("scalar"),
                title=raw.get("title", ""),
                description=raw.get("description", ""),
            )
    return out


def load_rosters() -> dict[str, list[str]]:
    """Named, ordered question-id lists from ``_rosters.yaml``."""
    if not _ROSTERS_PATH.exists():
        return {}
    data = yaml.safe_load(_ROSTERS_PATH.read_text()) or {}
    return dict(data.get("rosters", {}) or {})


def _group_sort_key(q: Question) -> tuple[int, str]:
    idx = GROUP_ORDER.index(q.group) if q.group in GROUP_ORDER else len(GROUP_ORDER)
    return (idx, q.id)


def select_questions(
    groups: Optional[list[str]] = None,
    *,
    buckets: Optional[list[str]] = None,
    ids: Optional[list[str]] = None,
    families: Optional[list[str]] = None,
    roster: Optional[str] = None,
) -> list[Question]:
    """Select a roster.

    - ``ids``: exact ids (any bucket/family state), in the given order.
    - ``roster``: a named id-list from ``_rosters.yaml``, in its frozen order.
    - ``families``: every variant question of the named families (any bucket),
      sorted by (family, variant, id). Takes precedence over groups.
    - ``groups``: every question in the named groups (across all buckets).
    - else by bucket: ``buckets`` (defaults to ``DEFAULT_BUCKETS`` — i.e. the
      ``tier_1`` bucket — when omitted). Pass ``buckets=BUCKETS`` for everything.
    """
    qs = all_questions()
    if ids is not None and roster is not None:
        raise ValueError("pass either ids or roster, not both")

    if roster is not None:
        rosters = load_rosters()
        if roster not in rosters:
            raise KeyError(f"unknown roster: {roster!r} (known: {sorted(rosters)})")
        ids = rosters[roster]

    if ids is not None:
        by_id = {q.id: q for q in qs}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"unknown question id(s): {missing}")
        return [by_id[i] for i in ids]

    if families:
        known_fams = {q.family for q in qs if q.family}
        bad = [f for f in families if f not in known_fams]
        if bad:
            raise KeyError(f"unknown family(ies): {bad} (known: {sorted(known_fams)})")
        sel = [q for q in qs if q.family in families]
        return sorted(sel, key=lambda q: (q.family or "", q.variant or "", q.id))

    if groups:
        known = set(GROUP_ORDER) | {q.group for q in qs}
        bad = [g for g in groups if g not in known]
        if bad:
            raise KeyError(f"unknown group(s): {bad} (known: {sorted(known)})")
        return sorted((q for q in qs if q.group in groups), key=_group_sort_key)

    bks = tuple(buckets) if buckets else DEFAULT_BUCKETS
    bad = [b for b in bks if b not in BUCKETS]
    if bad:
        raise KeyError(f"unknown bucket(s): {bad} (known: {list(BUCKETS)})")
    qs = [q for q in qs if q.bucket in bks]
    return sorted(qs, key=_group_sort_key)
