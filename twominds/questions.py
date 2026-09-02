"""Question roster for the coherence experiment.

The roster lives in ``questions/`` as one YAML file per question group:

    group: <group name>      # inherited by every question in the file
    questions:
      - id: ...
        prompt: ... | prompt_file: ...
        system: ...          # optional

Heavy text can live in a sibling ``.txt`` referenced via ``prompt_file``,
resolved relative to the YAML file that names it. Provenance (third-party
source, ground-truth answer) is a plain YAML ``#`` comment next to the
question, not a field. A bare run selects every question; ``--groups`` and
``--ids`` narrow the selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_PKG_DIR = Path(__file__).resolve().parent
_QUESTIONS_DIR = _PKG_DIR / "questions"

# Canonical group order (used for stable sorting / display).
GROUP_ORDER = [
    "values",
    "introspection",
    "situational_awareness",
    "high_stakes",
    "ai_safety",
    "sycophancy",
]


@dataclass(frozen=True)
class Question:
    """One free-form question asked of a model N times."""

    id: str
    group: str
    prompt: str
    system: Optional[str] = None


def _question_files() -> list[Path]:
    return sorted(_QUESTIONS_DIR.glob("*.yaml"))


def _load_file(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "group" not in data:
        raise ValueError(f"{path.name}: expected a top-level `group:` key")
    return data


def all_questions() -> list[Question]:
    """Every question defined in the data files."""
    out: list[Question] = []
    seen: set[str] = set()
    for path in _question_files():
        data = _load_file(path)
        group = data["group"]
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
                Question(id=qid, group=group, prompt=prompt, system=raw.get("system"))
            )
    return out


def _group_sort_key(q: Question) -> tuple[int, str]:
    idx = GROUP_ORDER.index(q.group) if q.group in GROUP_ORDER else len(GROUP_ORDER)
    return (idx, q.id)


def select_questions(
    groups: Optional[list[str]] = None,
    *,
    ids: Optional[list[str]] = None,
) -> list[Question]:
    """Select a roster.

    - ``ids``: exact ids, in the given order (overrides groups).
    - ``groups``: every question in the named groups.
    - else: the whole roster.
    """
    qs = all_questions()

    if ids is not None:
        by_id = {q.id: q for q in qs}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"unknown question id(s): {missing}")
        return [by_id[i] for i in ids]

    if groups:
        known = set(GROUP_ORDER) | {q.group for q in qs}
        bad = [g for g in groups if g not in known]
        if bad:
            raise KeyError(f"unknown group(s): {bad} (known: {sorted(known)})")
        return sorted((q for q in qs if q.group in groups), key=_group_sort_key)

    return sorted(qs, key=_group_sort_key)
