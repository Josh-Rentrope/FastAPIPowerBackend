"""Server-issued sessions for the chat feature.

## What this replaces

The original chat identified callers by an `X-User-Email` request header. Its own
docstring called it "INSECURE ... for demo purposes only", and it was worse than
that in practice: the shipped front end hardcoded one UUID into that header and
ignored the value it was passed, so **every visitor shared a single identity** and
could read every other visitor's conversations by listing them.

## What this does instead

The server mints an opaque 256-bit token, stores only its SHA-256, and returns it
as an HttpOnly cookie. Identity is therefore asserted by the server and cannot be
chosen by the caller. A dump of the `sessions` table yields nothing usable,
because the raw token exists only in the response that set the cookie.

## Why SameSite=Lax is correct here

The front end is at `trials.codingshinobi.org` and the API at
`api.trials.codingshinobi.org`. Those are different origins but the same
registrable domain, so requests between them are *same-site* and a Lax cookie is
sent normally. `SameSite=None` would be needed only for genuinely cross-site
calls, and would require `Secure` plus a browser that trusts the third-party
context. Keeping Lax means the cookie is not attached to requests originating
from other sites at all, which is the CSRF protection for the mutating routes.

Note that "same-site" is not "same-origin": the browser still treats this as a
CORS request, so `allow_credentials=True` in the app, an exact origin in
`allow_origins`, and `credentials: 'include'` on the client are all required.
Every one of those three fails silently if omitted, which is a lesson Bone Viewer
already paid for once.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from typing import Optional

from fastapi import Cookie, Depends, HTTPException, Request, Response, status

COOKIE_NAME = os.environ.get("CT_SESSION_COOKIE", "ct_session")
COOKIE_MAX_AGE = int(os.environ.get("CT_SESSION_MAX_AGE", str(60 * 60 * 24 * 30)))
COOKIE_SECURE = os.environ.get("CT_SESSION_COOKIE_SECURE", "1") == "1"
COOKIE_SAMESITE = os.environ.get("CT_SESSION_SAMESITE", "lax")

# Fixed identity for the assistant's own messages, so the client can tell them
# apart from its own.
AI_SENDER_ID = "00000000-0000-0000-0000-000000000000"

_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_hash   TEXT NOT NULL UNIQUE,
    display_name TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue(conn, display_name: Optional[str] = None) -> tuple[str, str]:
    """Create a session. Returns (session_id, raw_token).

    The raw token is returned exactly once and never stored.
    """
    token = secrets.token_urlsafe(32)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (token_hash, display_name) VALUES (%s, %s) RETURNING id",
        (_hash(token), display_name or "guest"),
    )
    session_id = cur.fetchone()[0]
    cur.close()
    return str(session_id), token


def resolve(conn, raw_token: Optional[str]) -> Optional[dict]:
    """Look up a session by its raw cookie value, or None."""
    if not raw_token:
        return None
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE sessions SET last_seen_at = now()
        WHERE token_hash = %s
        RETURNING id, display_name
        """,
        (_hash(raw_token),),
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        return None
    return {"id": str(row[0]), "display_name": row[1]}


def set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )


class SessionDep:
    """FastAPI dependency: resolve the caller's session from the cookie.

    Raises 401 rather than inventing an identity. An endpoint that wants to work
    without a session should not depend on this.
    """

    def __call__(self, request: Request,
                 ct_session: Optional[str] = Cookie(default=None)) -> dict:
        # Cookie(...) with a name of ct_session picks up COOKIE_NAME's value via
        # the parameter name, so the env-configured name only works if it matches.
        token = ct_session or request.cookies.get(COOKIE_NAME)
        session = resolve(request.app.state.get_conn(), token)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No valid session. POST /api/session first.",
            )
        return session


require_session = SessionDep()
Session = Depends(require_session)
