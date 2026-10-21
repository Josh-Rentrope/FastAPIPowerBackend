"""Conversational eligibility screening against an encoded criteria set.

## What this is and is not

This is a **screening aid**. It takes answers a person gives in conversation,
checks them against rules transcribed from a protocol's inclusion/exclusion
criteria, and reports which rules pass, which fail, and which cannot be answered
by interview at all.

It never returns "you are eligible". Several criteria in any real protocol —
PD-L1 CPS, RECIST measurability, organ function — can only be established by the
study team with tests. Those are marked `requires_site_assessment` and are
reported as unknown rather than assumed to pass. Presenting this as a
determination would be both wrong and, in a clinical context, harmful.

## The private seam

Which rules exist is public. How much weight they carry, how ambiguity is
resolved, and what eventually replaces this reference implementation is not.
`matchers.load_matcher()` reads the entry-point group `clinical_trials.matchers`,
the same shape as `bone_viewer.solvers` and `bone_viewer.backends`: the public
package ships a transparent reference matcher, and a private package can register
a better one without changing any call site.

## Answers

`answers` is a flat mapping of field key -> value. Values are coerced to the
field's declared type, so a conversational "I'm 63" and a form's `"63"` reach the
comparison identically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
CRITERIA_DIR = os.path.join(HERE, "criteria")

# Criteria that exist but cannot be established by asking the patient;
# the study team measures them. Marked so a "pass" is never inferred.
REQUIRES_SITE_ASSESSMENT = {"measurable_disease"}

PASS, FAIL, UNKNOWN, NOT_APPLICABLE = "pass", "fail", "unknown", "not_applicable"


@dataclass
class RuleResult:
    id: str
    kind: str
    text: str
    status: str
    field: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        d = {"id": self.id, "kind": self.kind, "text": self.text, "status": self.status}
        if self.field:
            d["field"] = self.field
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class Evaluation:
    criteria_id: str
    verdict: str                  # likely_eligible | ineligible | needs_info
    summary: str
    results: list[RuleResult] = field(default_factory=list)
    blocking: list[RuleResult] = field(default_factory=list)
    next_questions: list[dict] = field(default_factory=list)
    unanswered_required: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "criteria_id": self.criteria_id,
            "verdict": self.verdict,
            "summary": self.summary,
            "disclaimer": (
                "Screening aid only. Not an eligibility determination. Final "
                "eligibility is decided by the study team after tests this "
                "questionnaire cannot perform."
            ),
            "blocking": [r.as_dict() for r in self.blocking],
            "results": [r.as_dict() for r in self.results],
            "next_questions": self.next_questions,
            "unanswered_required": self.unanswered_required,
        }


# ── loading ─────────────────────────────────────────────────────────────────


def list_criteria() -> list[dict]:
    out = []
    if not os.path.isdir(CRITERIA_DIR):
        return out
    for name in sorted(os.listdir(CRITERIA_DIR)):
        if not name.endswith(".json"):
            continue
        try:
            c = json.load(open(os.path.join(CRITERIA_DIR, name), encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "id": c.get("id", name[:-5]),
            "label": c.get("label"),
            "condition": c.get("condition"),
            "field_count": len(c.get("fields", [])),
            "rule_count": len(c.get("rules", [])),
        })
    return out


def get_criteria(criteria_id: str) -> dict | None:
    path = os.path.join(CRITERIA_DIR, f"{criteria_id}.json")
    if not os.path.exists(path):
        # Allow a mismatch between file name and declared id.
        for entry in list_criteria():
            if entry["id"] == criteria_id:
                return get_criteria(os.path.splitext(
                    next(f for f in os.listdir(CRITERIA_DIR)
                         if f.startswith(criteria_id) or entry["id"] == criteria_id))[0])
        return None
    return json.load(open(path, encoding="utf-8"))


# ── coercion ────────────────────────────────────────────────────────────────

_TRUE = {"true", "yes", "y", "1", "sure", "correct", "yep", "yeah"}
_FALSE = {"false", "no", "n", "0", "nope", "not"}
_UNKNOWN = {"unknown", "not sure", "unsure", "don't know", "dont know", "n/a", "na", ""}


def coerce(value: Any, ftype: str) -> Any:
    """Force a value to the field's type, or return None if it cannot be read.

    Conversational answers arrive as free text ("about 63", "no, never"), so this
    is deliberately forgiving on the way in and strict on the way out: anything it
    cannot interpret becomes None, which the evaluator treats as unknown rather
    than guessing.
    """
    if value is None:
        return None
    if isinstance(value, bool) and ftype == "boolean":
        return value
    s = str(value).strip().lower()

    if s in _UNKNOWN:
        return None

    if ftype == "boolean":
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        return None

    if ftype == "integer":
        digits = "".join(ch for ch in s if ch.isdigit() or ch == "-")
        try:
            return int(digits)
        except (TypeError, ValueError):
            return None

    if ftype == "enum":
        return s or None

    return s or None


# ── evaluation ──────────────────────────────────────────────────────────────


def _cmp(op: str, actual: Any, expected: Any) -> bool | None:
    if actual is None:
        return None
    try:
        if op == ">=":
            return actual >= expected
        if op == "<=":
            return actual <= expected
        if op == ">":
            return actual > expected
        if op == "<":
            return actual < expected
        if op == "==":
            return actual == expected
        if op == "!=":
            return actual != expected
        if op == "in":
            return str(actual).lower() in [str(v).lower() for v in expected]
        if op == "not_in":
            return str(actual).lower() not in [str(v).lower() for v in expected]
        if op == "is_true":
            return actual is True
        if op == "is_false":
            return actual is False
    except TypeError:
        return None
    return None


def evaluate(criteria: dict, answers: dict, max_questions: int = 3) -> Evaluation:
    fields = {f["key"]: f for f in criteria.get("fields", [])}
    typed = {k: coerce(v, fields[k]["type"]) if k in fields else v
             for k, v in (answers or {}).items()}

    results: list[RuleResult] = []
    unknown_required: list[str] = []

    for rule in criteria.get("rules", []):
        fkey = rule.get("field")
        op = rule.get("op")

        # "conditional" only applies when its guard holds -- e.g. the HPV rule
        # matters only for an oropharyngeal primary.
        if op == "conditional":
            when = rule.get("when") or {}
            guard = when.get("field")
            if guard:
                gval = typed.get(guard)
                if gval is None:
                    results.append(RuleResult(rule["id"], rule["kind"], rule["text"], UNKNOWN,
                                              field=guard, detail="depends on an unanswered question"))
                    continue
                if str(gval).lower() != str(when.get("equals", "")).lower():
                    results.append(RuleResult(rule["id"], rule["kind"], rule["text"], NOT_APPLICABLE))
                    continue
            # Falls through to an equality test on the rule's own field.
            actual = typed.get(fkey)
            if actual is None:
                results.append(RuleResult(rule["id"], rule["kind"], rule["text"], UNKNOWN, field=fkey))
                continue
            holds = str(actual).lower() == str(rule.get("value", "")).lower()
        else:
            actual = typed.get(fkey)
            holds = _cmp(op, actual, rule.get("value"))
            site_assessed = fkey in REQUIRES_SITE_ASSESSMENT
            if holds is None:
                # An unanswered site-assessed criterion is unknown, never assumed
                # to pass -- but an *answered* one is used, with a note that the
                # study team still has to confirm it. Discarding a patient's
                # answer and asking again would be both wrong and irritating.
                results.append(RuleResult(
                    rule["id"], rule["kind"], rule["text"], UNKNOWN, field=fkey,
                    detail=("measured by the study team, not by interview"
                            if site_assessed else None)))
                if rule.get("required"):
                    unknown_required.append(fkey)
                continue

        # An inclusion passes when it holds; an exclusion passes when it does not.
        if rule["kind"] == "inclusion":
            status = PASS if holds else FAIL
        else:
            status = PASS if not holds else FAIL

        detail = ("reported by the patient; the study team must confirm"
                  if fkey in REQUIRES_SITE_ASSESSMENT else None)
        results.append(RuleResult(rule["id"], rule["kind"], rule["text"], status,
                                  field=fkey, detail=detail))

    blocking = [r for r in results if r.status == FAIL]

    # Next questions: required rules still unknown, in declaration order, de-duped.
    next_questions: list[dict] = []
    seen: set[str] = set()
    for r in results:
        if r.status != UNKNOWN or not r.field or r.field in seen:
            continue
        rule = next((x for x in criteria.get("rules", []) if x["id"] == r.id), {})
        # Only genuinely required criteria are worth interrupting someone for.
        # Optional ones (e.g. "months since checkpoint inhibitor" for someone who
        # never had one) are still evaluated if volunteered, but never asked.
        if rule.get("required") is not True:
            continue
        seen.add(r.field)
        f = fields.get(r.field, {})
        next_questions.append({
            "key": r.field,
            "question": f.get("question", f"Can you tell me about {r.field.replace('_', ' ')}?"),
            "type": f.get("type", "text"),
            "options": f.get("options"),
            "help": f.get("help"),
            "because": r.text,
        })
        if len(next_questions) >= max_questions:
            break

    required_ids = {r["id"] for r in criteria.get("rules", []) if r.get("required")}
    unanswered_required = sorted({r.field for r in results
                                  if r.status == UNKNOWN and r.id in required_ids and r.field})

    if blocking:
        verdict = "ineligible"
        summary = (f"{len(blocking)} criterion(s) in this protocol's criteria appear "
                   f"not to be met. That is a screening signal, not a decision — "
                   f"please discuss it with the study team.")
    elif unanswered_required:
        verdict = "needs_info"
        summary = (f"{len(unanswered_required)} required criterion(s) still need an "
                   f"answer before this can be screened.")
    else:
        verdict = "likely_eligible"
        summary = ("No exclusion identified and the included criteria answered so "
                   "far are met. Site-measured requirements still have to be "
                   "confirmed by the study team.")

    return Evaluation(
        criteria_id=criteria.get("id", ""),
        verdict=verdict,
        summary=summary,
        results=results,
        blocking=blocking,
        next_questions=next_questions,
        unanswered_required=unanswered_required,
    )
