"""
The brief URL scheme, as pure functions.

Kept separate from `server.py` so that both sides of the contract — the
publisher that writes a URL into Slack and the handler that later resolves it —
are built from the same code, and so the scheme can be tested without binding
a socket.

Scheme:

    /briefs/<lead_id>              current version, HTML
    /briefs/<lead_id>/v<N>         a specific version, HTML
    /briefs/<lead_id>/history      every version of this lead
    /briefs/<lead_id>.json         current version, structured record
    /briefs/<lead_id>/v<N>.json    a specific version, structured record
    /healthz                       liveness probe

The bare `/briefs/<lead_id>` URL is the one that goes into Slack: it always
resolves to the newest brief, so a re-analysed lead does not leave a stale
link behind. `/history` is how you get back to what an earlier run concluded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from leads_agent.briefs.identity import is_valid_lead_id

ROUTE_PREFIX = "/briefs"
HEALTH_PATH = "/healthz"

_VERSION_SEGMENT = re.compile(r"^v(\d{1,6})$")


class RouteKind(str, Enum):
    """What a resolved request is asking for."""

    health = "health"
    current_html = "current_html"
    version_html = "version_html"
    history = "history"
    current_json = "current_json"
    version_json = "version_json"


@dataclass(frozen=True)
class Route:
    """A parsed request path."""

    kind: RouteKind
    lead_id: str | None = None
    version: int | None = None


def brief_path(lead_id: str, version: int | None = None) -> str:
    """Path for a brief — the current one when `version` is None."""
    if version is None:
        return f"{ROUTE_PREFIX}/{lead_id}"
    return f"{ROUTE_PREFIX}/{lead_id}/v{version}"


def history_path(lead_id: str) -> str:
    """Path for a lead's version-history page."""
    return f"{ROUTE_PREFIX}/{lead_id}/history"


def absolute_url(base_url: str, path: str) -> str:
    """Join a configured base URL with a brief path, tolerating trailing slashes."""
    return f"{base_url.rstrip('/')}{path}"


def resolve_route(path: str) -> Route | None:
    """
    Parse a request path into a `Route`, or None if it matches nothing.

    Lead ids are validated against `identity.LEAD_ID_PATTERN` before they are
    returned, so a caller can interpolate `route.lead_id` straight into an S3
    key without a traversal check of its own.
    """
    path = path.split("?", 1)[0].split("#", 1)[0]

    if path in (HEALTH_PATH, "/health"):
        return Route(RouteKind.health)

    if not path.startswith(ROUTE_PREFIX + "/"):
        return None

    remainder = path[len(ROUTE_PREFIX) + 1 :].strip("/")
    if not remainder:
        return None

    segments = remainder.split("/")
    if len(segments) > 2:
        return None

    head = segments[0]
    if head.endswith(".json"):
        lead_id, kind, version = head[: -len(".json")], RouteKind.current_json, None
        if len(segments) != 1:
            return None
        return Route(kind, lead_id, version) if is_valid_lead_id(lead_id) else None

    lead_id = head
    if not is_valid_lead_id(lead_id):
        return None

    if len(segments) == 1:
        return Route(RouteKind.current_html, lead_id)

    tail = segments[1]
    if tail == "history":
        return Route(RouteKind.history, lead_id)

    as_json = tail.endswith(".json")
    if as_json:
        tail = tail[: -len(".json")]

    match = _VERSION_SEGMENT.match(tail)
    if not match:
        return None
    version = int(match.group(1))
    if version < 1:
        return None

    kind = RouteKind.version_json if as_json else RouteKind.version_html
    return Route(kind, lead_id, version)
