"""Eligibility screening endpoints.

These expose the same evaluation the conversational assistant uses, so a form-based
UI and the chat loop cannot drift apart. Both call `qualification.load_matcher()`.

The response deliberately never says "eligible". See qualification/engine.py.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

import qualification
from qualification import segmenter

router = APIRouter(prefix="/api/qualify", tags=["Qualification"])


class ScreeningRequest(BaseModel):
    answers: dict[str, Any] = {}


class SegmentRequest(BaseModel):
    condition: str | None = None
    status: str | None = None
    answers: dict[str, Any] = {}


# ── segment: which question to ask next ─────────────────────────────────────
#
# Declared BEFORE the `/{criteria_id}` catch-all below. FastAPI matches in
# declaration order, so "segment" would otherwise be read as a criteria id and
# 404 at the lookup instead of reaching this handler.

def _segment(request: Request, condition, status, answers):
    if not getattr(request.app.state, "engine", None):
        raise HTTPException(503, "database not configured")
    conn = request.app.state.engine.raw_connection()
    try:
        cur = conn.cursor()
        return segmenter.next_question(cur, condition, status, answers or {})
    finally:
        conn.close()


@router.get("/segment")
def segment_get(request: Request,
                condition: str | None = Query(None),
                status: str | None = Query(None)):
    """Next question for a plain filter-only pool (no answers yet)."""
    return _segment(request, condition, status, {})


@router.post("/segment")
def segment_post(body: SegmentRequest, request: Request):
    """Next question given everything answered so far.

    `answers` carries `age`, `sex`, `healthy_volunteer` (the form fields) and
    `facts`, keyed `"prior_therapy:chemotherapy": true`. Stateless, like the
    screening endpoint, so the caller owns the conversation.
    """
    return _segment(request, body.condition, body.status, body.answers)


@router.get("/criteria")
def list_all():
    """Every criteria set available, and which matcher is active.

    `matcher.private` is the tell for whether a registered implementation is in
    use or the transparent reference one is.
    """
    return {
        "criteria": qualification.list_criteria(),
        "matcher": qualification.describe_matcher(),
    }


@router.get("/criteria/{criteria_id}")
def get_one(criteria_id: str):
    """The full definition: every question, and every rule it feeds."""
    c = qualification.get_criteria(criteria_id)
    if not c:
        raise HTTPException(404, f"unknown criteria set: {criteria_id}")
    return c


@router.post("/{criteria_id}")
def screen(criteria_id: str, body: ScreeningRequest):
    """Evaluate answers and return a verdict plus the next questions to ask.

    Stateless on purpose: the caller keeps the answers. That keeps this endpoint
    usable from a form, a chat loop, and a test without a session.
    """
    c = qualification.get_criteria(criteria_id)
    if not c:
        raise HTTPException(404, f"unknown criteria set: {criteria_id}")
    matcher = qualification.load_matcher()
    return matcher.evaluate(c, body.answers).as_dict()
