"""
A minimal HTTP listener for briefs, alongside Socket Mode.

Why the standard library and not a framework: the whole surface is four
read-only GET routes serving bytes that already exist in S3. `slack-bolt`
brings no server of its own in Socket Mode, `httpx` is a client, and nothing
else in the tree can listen on a socket — so a framework would mean a new
dependency and a second runtime model (ASGI) for four routes. `http.server`
with a `ThreadingHTTPServer` is boring, has no version to keep current, and
fits the traffic: a handful of requests a day from people clicking a Slack
link.

How it coexists with Socket Mode: `SocketModeHandler.start()` blocks the main
thread forever, so the listener runs in its own daemon thread. Daemon matters —
Ctrl+C still stops the process, and the listener can never be the reason the
bot fails to exit. Nothing is shared between the two except the `BriefStore`,
which holds no mutable state of its own.

This is read-only and unauthenticated by design. Exposure is a *network*
decision, made by which address docker-compose publishes the port on (see
docs/DEPLOYMENT.md) — not something this module tries to solve with a token.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from leads_agent.briefs import brief_store
from leads_agent.briefs.render import render_history_html
from leads_agent.briefs.routes import (
    RouteKind,
    brief_path,
    resolve_route,
)
from leads_agent.briefs.storage import BriefStore

if TYPE_CHECKING:
    from leads_agent.config import Settings

logger = logging.getLogger(__name__)

HTML = "text/html; charset=utf-8"
JSON = "application/json; charset=utf-8"
TEXT = "text/plain; charset=utf-8"

_NOT_FOUND_HTML = (
    "<!doctype html><html lang=en><head><meta charset=utf-8>"
    "<title>Not found</title></head><body style='font:16px system-ui;padding:40px'>"
    "<h1>Not found</h1><p>No brief at this address.</p></body></html>"
)


@dataclass(frozen=True)
class Response:
    """A rendered HTTP response, independent of the server that sends it."""

    status: int
    content_type: str
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


def _not_found() -> Response:
    return Response(404, HTML, _NOT_FOUND_HTML.encode("utf-8"))


def handle_route(store: BriefStore, path: str) -> Response:
    """
    Resolve a request path against the store.

    Split out from the request handler so the routing and lookup behaviour can
    be tested without binding a socket.
    """
    route = resolve_route(path)
    if route is None:
        return _not_found()

    if route.kind == RouteKind.health:
        return Response(200, TEXT, b"ok\n")

    lead_id = route.lead_id
    if lead_id is None:
        return _not_found()

    index = store.read_index(lead_id)

    if route.kind == RouteKind.history:
        if index is None:
            return _not_found()
        return Response(
            200,
            HTML,
            render_history_html(
                lead_id,
                index.versions,
                current_version=index.current_version,
                brief_url_for=lambda v: brief_path(lead_id, v),
            ).encode("utf-8"),
        )

    # `current_*` routes need the index to know which version "current" is;
    # `version_*` routes go straight to the object and do not.
    if route.kind in (RouteKind.current_html, RouteKind.current_json):
        if index is None or index.current_version < 1:
            return _not_found()
        version = index.current_version
    else:
        version = route.version or 0

    if route.kind in (RouteKind.current_json, RouteKind.version_json):
        record = store.read_record(lead_id, version)
        return Response(200, JSON, record) if record is not None else _not_found()

    html = store.read_html(lead_id, version)
    return Response(200, HTML, html.encode("utf-8")) if html is not None else _not_found()


def make_handler(store: BriefStore) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to one store."""

    class BriefRequestHandler(BaseHTTPRequestHandler):
        # Keep-alive is worth having: a brief is one HTML document and no
        # sub-resources, but Slack's unfurler and browsers both send a HEAD
        # before the GET.
        protocol_version = "HTTP/1.1"
        server_version = "leads-agent-briefs"
        sys_version = ""

        def _respond(self, response: Response, *, include_body: bool = True) -> None:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            # Briefs name individuals and judge them; they must not be cached
            # by intermediaries or indexed anywhere.
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Robots-Tag", "noindex, nofollow")
            self.send_header("Referrer-Policy", "no-referrer")
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.end_headers()
            if include_body:
                self.wfile.write(response.body)

        def _serve(self, *, include_body: bool) -> None:
            try:
                response = handle_route(store, self.path)
            except Exception:
                logger.exception("Brief request failed: %s", self.path)
                response = Response(500, TEXT, b"internal error\n")
            self._respond(response, include_body=include_body)

        def do_GET(self) -> None:  # Name fixed by BaseHTTPRequestHandler.
            self._serve(include_body=True)

        def do_HEAD(self) -> None:  # Name fixed by BaseHTTPRequestHandler.
            self._serve(include_body=False)

        def log_message(self, format: str, *args) -> None:
            """Send access logs through logging instead of straight to stderr."""
            logger.info("brief http %s - %s", self.address_string(), format % args)

    return BriefRequestHandler


def start_brief_server(
    settings: Settings,
    client: object | None = None,
) -> ThreadingHTTPServer | None:
    """
    Start the brief HTTP listener in a daemon thread.

    Returns the server (so tests and callers can shut it down) or None when
    serving is disabled, unconfigured, or the port could not be bound. A bind
    failure is logged and swallowed: the bot's job is classifying leads, and it
    must keep doing that even if nobody can read the briefs.
    """
    if not settings.briefs_http_enabled:
        return None

    store = brief_store(settings, client)
    if store is None:
        logger.warning("Brief HTTP server not started: briefs are disabled or unconfigured")
        return None

    address = (settings.briefs_http_host, settings.briefs_http_port)
    try:
        server = ThreadingHTTPServer(address, make_handler(store))
    except OSError:
        logger.exception("Could not bind brief HTTP server on %s:%s", *address)
        return None

    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="brief-http", daemon=True)
    thread.start()

    logger.info(
        "Serving briefs on http://%s:%s%s (bucket: %s)",
        settings.briefs_http_host,
        settings.briefs_http_port,
        brief_path("<lead_id>"),
        store.bucket,
    )
    return server
