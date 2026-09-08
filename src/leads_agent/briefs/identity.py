"""
Stable identity for a lead.

A lead has no natural primary key: HubSpot's payload reaches us as Slack text,
and the same person can be re-analysed weeks later from a fresh message. The
brief history only works if that second analysis lands under the *same* id, so
the id is derived from the lead's own durable identity (email first, then
company + name) rather than from anything about the run.

The id is `<slug>-<digest>`: the slug makes an object listing readable in the
S3 console, the digest makes it stable and collision-resistant. The email
itself is deliberately not in the id — these URLs get pasted into Slack, and a
brief already contains enough about a named individual without their address
being in the link.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# The full set of characters a lead id may contain. Anything arriving from a
# URL is matched against this before it is interpolated into an S3 key, which
# is what keeps `../` out of the object store.
LEAD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")

_NON_SLUG = re.compile(r"[^a-z0-9]+")

_DIGEST_LENGTH = 10
_SLUG_MAX_LENGTH = 48


def _slugify(text: str, *, max_length: int = _SLUG_MAX_LENGTH) -> str:
    """Lowercase, hyphenated, ASCII-only fragment safe for a URL path segment."""
    slug = _NON_SLUG.sub("-", text.lower()).strip("-")
    return slug[:max_length].strip("-")


def _identity(lead: Any) -> tuple[str, str]:
    """
    The identity of a lead, as `(digest_source, slug_source)`.

    Email wins because it is the one field a person keeps across enquiries.
    Company + name is the fallback; raw text is the last resort and is only
    stable for byte-identical messages, which is the best that can be done
    when the lead gave us nothing else.

    Both halves come from the *same* branch on purpose. Taking the slug from
    the company name while keying the digest on email would mean a lead whose
    company is typed differently the second time ("Northgate Mfg Inc" vs
    "Northgate Manufacturing") gets a different id and starts a fresh, empty
    history — which is exactly the failure this id exists to prevent. With an
    email in hand the slug is therefore its domain, which moves only when the
    digest does.
    """
    email = (getattr(lead, "email", None) or "").strip().lower()
    if email:
        domain = email.rpartition("@")[2] or email
        return f"email:{email}", domain

    company = (getattr(lead, "company", None) or "").strip().lower()
    first = (getattr(lead, "first_name", None) or "").strip().lower()
    last = (getattr(lead, "last_name", None) or "").strip().lower()
    if company or first or last:
        name = f"{first} {last}".strip()
        return f"name:{company}|{name}", (company or name)

    return f"raw:{(getattr(lead, 'raw_text', None) or '').strip()}", ""


def lead_id_for(lead: Any) -> str:
    """
    Derive the stable, URL-safe id used for this lead's brief history.

    Re-analysing the same lead returns the same id, so successive briefs
    accumulate as versions instead of scattering across unrelated paths.
    """
    digest_source, slug_source = _identity(lead)
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]

    slug = _slugify(slug_source) if slug_source else ""
    return f"{slug}-{digest}" if slug else f"lead-{digest}"


def is_valid_lead_id(lead_id: str) -> bool:
    """Whether a lead id from an untrusted source is safe to use in a key."""
    return bool(LEAD_ID_PATTERN.match(lead_id))
