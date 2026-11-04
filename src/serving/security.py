"""Authentication, authorization, rate limiting and request-size limits.

Design (see docs/threat_model.md for the threats each control answers):

* **API keys with roles.** ``sensor`` keys may submit flows for scoring;
  ``analyst`` keys may also submit ground-truth labels (``/feedback``).
  Only SHA-256 digests of the keys are configured on the server, keys are
  compared in constant time, and the service *fails closed*: with no keys
  configured every protected endpoint answers 401.
* **Labels are an analyst privilege.** The old API accepted ``Label`` in
  the sensor payload and fed it straight into retraining - anyone who
  could reach /predict could poison the model. Labels now only enter via
  ``/feedback``, tied to a logged prediction id and the analyst's key id.
* **Token-bucket rate limiting** per key (OWASP API4: unrestricted
  resource consumption).
* **Body-size cap** enforced while the body streams in, so a missing or
  lying Content-Length cannot get around it.

Configuration (environment):
    IDS_API_KEYS       comma-separated ``name:role:sha256hex`` entries
    IDS_RATE_PER_MIN   sustained requests/minute per key (default 600)
    IDS_RATE_BURST     bucket size (default 60)
    IDS_MAX_BODY_BYTES request body cap in bytes (default 262144)

Generate a key:
    python -m src.serving.security new-key alice analyst
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

ROLES = ("sensor", "analyst")
# analyst can do everything a sensor can
ROLE_GRANTS = {"sensor": {"sensor"}, "analyst": {"sensor", "analyst"}}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class Principal:
    name: str
    role: str


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_keys(spec: str | None) -> list[tuple[str, str, str]]:
    """``name:role:sha256hex,...`` -> [(name, role, digest)]; strict."""
    entries = []
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError("IDS_API_KEYS entries must be name:role:sha256hex")
        name, role, digest = parts
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r} (expected one of {ROLES})")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"key digest for {name!r} is not a sha256 hex digest")
        entries.append((name, role, digest))
    return entries


def authenticate(raw_key: str | None, entries: list[tuple[str, str, str]]
                 ) -> Principal | None:
    """Constant-time lookup; checks every entry so timing leaks nothing."""
    if not raw_key:
        return None
    digest = hash_key(raw_key)
    found = None
    for name, role, expected in entries:
        if hmac.compare_digest(digest, expected):
            found = Principal(name, role)
    return found


class TokenBucket:
    """Per-key token bucket; thread-safe (sync endpoints run in a pool)."""

    def __init__(self, rate_per_min: float, burst: int,
                 clock=time.monotonic) -> None:
        self.rate = rate_per_min / 60.0
        self.burst = burst
        self.clock = clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        """(allowed, seconds until a token is available)."""
        now = self.clock()
        with self._lock:
            tokens, last = self._state.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self._state[key] = (tokens - 1, now)
                return True, 0.0
            self._state[key] = (tokens, now)
            return False, (1 - tokens) / self.rate if self.rate else 60.0


class SecurityConfig:
    """Reads the environment once per app start (tests rebuild it)."""

    def __init__(self) -> None:
        self.entries = parse_keys(os.environ.get("IDS_API_KEYS"))
        self.limiter = TokenBucket(
            float(os.environ.get("IDS_RATE_PER_MIN", 600)),
            int(os.environ.get("IDS_RATE_BURST", 60)))
        self.max_body = int(os.environ.get("IDS_MAX_BODY_BYTES", 256 * 1024))


CONFIG = SecurityConfig()


def reload_config() -> SecurityConfig:
    global CONFIG
    CONFIG = SecurityConfig()
    return CONFIG


def require(role: str):
    """FastAPI dependency: authenticated principal holding ``role``."""

    def dependency(raw_key: str | None = Depends(api_key_header)) -> Principal:
        principal = authenticate(raw_key, CONFIG.entries)
        if principal is None:
            raise HTTPException(status_code=401, detail="missing or invalid API key",
                                headers={"WWW-Authenticate": "APIKey"})
        if role not in ROLE_GRANTS[principal.role]:
            raise HTTPException(status_code=403, detail="insufficient role")
        allowed, retry = CONFIG.limiter.allow(principal.name)
        if not allowed:
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                headers={"Retry-After": str(max(1, round(retry)))})
        return principal

    return dependency


class BodySizeLimitMiddleware:
    """Pure ASGI middleware: 413 once the streamed body exceeds the cap."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = CONFIG.max_body
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            return await _reject(send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _reject(send)


class _BodyTooLarge(Exception):
    pass


async def _reject(send) -> None:
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body",
                "body": b'{"detail":"request body too large"}'})


async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def _cli() -> None:
    if len(sys.argv) != 4 or sys.argv[1] != "new-key" or sys.argv[3] not in ROLES:
        sys.exit("usage: python -m src.serving.security new-key <name> <sensor|analyst>")
    raw = secrets.token_urlsafe(32)
    print(f"API key (give to the client, store nowhere else): {raw}")
    print(f"IDS_API_KEYS entry: {sys.argv[2]}:{sys.argv[3]}:{hash_key(raw)}")


if __name__ == "__main__":
    _cli()
