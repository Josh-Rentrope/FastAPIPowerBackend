"""Eligibility screening endpoints.

These expose the same evaluation the conversational assistant uses, so a form-based
UI and the chat loop cannot drift apart. Both call `qualification.load_matcher()`.

The response deliberately never says "eligible". See qualification/engine.py.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import qualification

router = APIRouter(prefix="/api/qualify", tags=["Qualification"])


class ScreeningRequest(BaseModel):
    answers: dict[str, Any] = {}


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
