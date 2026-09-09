"""
S3 persistence for lead briefs, with explicit version history.

Versions are *object paths*, not S3 bucket versions:

    <prefix>/<lead_id>/index.json     pointer + version log
    <prefix>/<lead_id>/v0001.html     the brief, as served
    <prefix>/<lead_id>/v0001.json     the classification, as data

Bucket versioning would give history for free, but you cannot see it in a
listing, cannot link to a specific revision, and cannot diff two revisions
without reaching for version ids. A lead's verdict is *designed* to change as
evidence arrives (`needs_verification` -> `in_icp` once revenue is confirmed),
so the history is a feature to be browsed rather than a recovery mechanism —
which makes a visible, listable key per version the simpler thing to reason
about.

The HTML is for people; the JSON alongside it is what later answers questions
like "which ICP gate fails most often?" without re-parsing prose.

Version numbers are zero-padded to four digits so that a plain `aws s3 ls`
returns them in chronological order.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"

_VERSION_PAD = 4

# S3 error codes that mean "this object/prefix does not exist yet" rather than
# "something is wrong". A first-ever brief for a lead hits these every time,
# so they must not be logged as failures.
_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound", "NoSuchBucket"})


class BriefVersion(BaseModel):
    """One published version of a lead's brief, as recorded in the index."""

    version: int
    created_at: str
    verdict: str | None = None
    action: str | None = None
    company: str | None = None
    contact: str | None = None


class BriefIndex(BaseModel):
    """
    The pointer object for one lead.

    `current_version` is what a bare `/briefs/<lead_id>` URL resolves to. It is
    stored rather than inferred so that a future "pin this lead to v2" needs no
    schema change.
    """

    lead_id: str
    current_version: int = 0
    versions: list[BriefVersion] = Field(default_factory=list)

    def latest(self) -> BriefVersion | None:
        return max(self.versions, key=lambda v: v.version) if self.versions else None


def _error_code(exc: Exception) -> str | None:
    """Pull the S3 error code out of a botocore ClientError, if that is what this is."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return str(code) if code is not None else None


def _is_not_found(exc: Exception) -> bool:
    return _error_code(exc) in _NOT_FOUND_CODES


class BriefStore:
    """
    Read/write access to one bucket's worth of briefs.

    The S3 client is injected rather than constructed here so the store can be
    exercised against a stub with no credentials present.
    """

    def __init__(self, client: Any, bucket: str, prefix: str = "briefs") -> None:
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    # --- Key scheme -------------------------------------------------------

    def lead_prefix(self, lead_id: str) -> str:
        return f"{self.prefix}/{lead_id}" if self.prefix else lead_id

    def index_key(self, lead_id: str) -> str:
        return f"{self.lead_prefix(lead_id)}/{INDEX_FILENAME}"

    def html_key(self, lead_id: str, version: int) -> str:
        return f"{self.lead_prefix(lead_id)}/v{version:0{_VERSION_PAD}d}.html"

    def json_key(self, lead_id: str, version: int) -> str:
        return f"{self.lead_prefix(lead_id)}/v{version:0{_VERSION_PAD}d}.json"

    # --- Reads ------------------------------------------------------------

    def _get_bytes(self, key: str) -> bytes | None:
        """Fetch an object, returning None when it simply is not there."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # boto3 error classes vary by client, so inspect the code
            if _is_not_found(exc):
                return None
            raise
        body = response["Body"]
        return body.read() if hasattr(body, "read") else bytes(body)

    def _exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # boto3 error classes vary by client, so inspect the code
            if _is_not_found(exc):
                return False
            raise
        return True

    def read_index(self, lead_id: str) -> BriefIndex | None:
        """The lead's index, or None when nothing has ever been published for it."""
        raw = self._get_bytes(self.index_key(lead_id))
        if raw is None:
            return None
        try:
            return BriefIndex.model_validate_json(raw)
        except ValueError:
            # A corrupt index must not make the lead unpublishable: treat it as
            # absent and let `next_version` probe the object store instead.
            logger.warning("Unreadable brief index for %s; treating as empty", lead_id)
            return None

    def read_html(self, lead_id: str, version: int) -> str | None:
        raw = self._get_bytes(self.html_key(lead_id, version))
        return raw.decode("utf-8") if raw is not None else None

    def read_record(self, lead_id: str, version: int) -> bytes | None:
        return self._get_bytes(self.json_key(lead_id, version))

    # --- Writes -----------------------------------------------------------

    def next_version(self, lead_id: str, index: BriefIndex | None) -> int:
        """
        Allocate the next free version number.

        The index gives the candidate; the object store gets the final say. The
        probe costs one HEAD in the common case and is what lets a lost or
        corrupt index heal itself instead of overwriting a published brief.
        """
        candidate = 1
        if index is not None:
            latest = index.latest()
            highest = max(latest.version, index.current_version) if latest else index.current_version
            candidate = highest + 1

        # Bounded: a runaway loop here would be worse than a duplicate.
        for offset in range(64):
            version = candidate + offset
            if not self._exists(self.html_key(lead_id, version)):
                return version
        return candidate + 64

    def publish(
        self,
        lead_id: str,
        *,
        html: str,
        record: dict[str, Any],
        verdict: str | None = None,
        action: str | None = None,
        company: str | None = None,
        contact: str | None = None,
    ) -> tuple[int, str]:
        """
        Write a new version of a lead's brief.

        The HTML and the JSON go first, the index last: if the run dies
        mid-write the index still points at a version that exists, and the
        orphaned objects are found again by `next_version`'s probe.

        Returns:
            (version, html_key)
        """
        index = self.read_index(lead_id) or BriefIndex(lead_id=lead_id)
        version = self.next_version(lead_id, index)
        html_key = self.html_key(lead_id, version)
        created_at = datetime.now(UTC).isoformat(timespec="seconds")

        self.client.put_object(
            Bucket=self.bucket,
            Key=html_key,
            Body=html.encode("utf-8"),
            ContentType="text/html; charset=utf-8",
            CacheControl="no-store",
        )
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.json_key(lead_id, version),
            Body=json.dumps(record, indent=2, default=str, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json; charset=utf-8",
        )

        index.versions = [v for v in index.versions if v.version != version]
        index.versions.append(
            BriefVersion(
                version=version,
                created_at=created_at,
                verdict=verdict,
                action=action,
                company=company,
                contact=contact,
            )
        )
        index.versions.sort(key=lambda v: v.version)
        index.current_version = version

        self.client.put_object(
            Bucket=self.bucket,
            Key=self.index_key(lead_id),
            Body=index.model_dump_json(indent=2).encode("utf-8"),
            ContentType="application/json; charset=utf-8",
            CacheControl="no-store",
        )

        return version, html_key
