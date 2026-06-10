from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import time

import requests

from core.config import Settings
from core.utils import compact_join, normalize_whitespace, write_json

CROSSREF_API_URL = "https://api.crossref.org/works"
_JATS_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str
    comment: str


def _clean_abstract(raw: str) -> str:
    """Strip JATS/XML tags that Crossref wraps abstracts in."""
    without_tags = _JATS_TAG.sub(" ", raw or "")
    return normalize_whitespace(without_tags)


def _date_parts_to_iso(container: dict | None) -> str:
    """Convert a Crossref date container ({"date-parts": [[y, m, d]]}) to YYYY-MM-DD."""
    if not container:
        return ""
    parts = container.get("date-parts") or [[]]
    first = parts[0] if parts else []
    if not first or first[0] is None:
        return ""
    year = int(first[0])
    month = int(first[1]) if len(first) > 1 and first[1] else 1
    day = int(first[2]) if len(first) > 2 and first[2] else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _extract_authors(item: dict) -> list[str]:
    authors: list[str] = []
    for person in item.get("author", []) or []:
        name = compact_join([person.get("given", ""), person.get("family", "")], sep=" ")
        name = normalize_whitespace(name)
        if name:
            authors.append(name)
    return authors


def _extract_pdf_url(item: dict) -> str:
    for link in item.get("link", []) or []:
        if link.get("content-type") == "application/pdf" and link.get("URL"):
            return link["URL"]
    return ""


def _published_date(item: dict) -> str:
    for key in ("published", "published-online", "published-print", "issued"):
        iso = _date_parts_to_iso(item.get(key))
        if iso:
            return iso
    return ""


def parse_crossref_payload(payload: dict) -> list[PaperRecord]:
    """Parse a Crossref `/works` payload into a list of `PaperRecord`."""
    items = (payload.get("message") or {}).get("items") or []
    records: list[PaperRecord] = []
    seen: set[str] = set()

    for item in items:
        doi = normalize_whitespace(item.get("DOI", ""))
        title_list = item.get("title") or []
        title = normalize_whitespace(title_list[0]) if title_list else ""
        summary = _clean_abstract(item.get("abstract", ""))

        # Drop records that are not usable for a RAG corpus.
        if not doi or not title or not summary:
            continue
        if doi in seen:
            continue
        seen.add(doi)

        categories = [normalize_whitespace(s) for s in (item.get("subject") or []) if s]
        container = item.get("container-title") or []

        records.append(
            PaperRecord(
                paper_id=doi,
                title=title,
                summary=summary,
                authors=_extract_authors(item),
                categories=categories,
                primary_category=categories[0] if categories else "",
                published=_published_date(item),
                updated=_date_parts_to_iso(item.get("indexed") or item.get("deposited")),
                abs_url=item.get("URL", ""),
                pdf_url=_extract_pdf_url(item),
                comment=normalize_whitespace(container[0]) if container else item.get("type", ""),
            )
        )
    return records


def fetch_source_records(settings: Settings) -> list[PaperRecord]:
    """Call the Crossref API, persist the raw response, and parse it into records."""
    params = {
        "query": settings.source_query,
        "filter": settings.source_filter,
        "rows": settings.max_results,
        # Polite-pool identification recommended by Crossref.
        "mailto": "day10-lab@example.com",
    }
    headers = {"User-Agent": "Day10DataObservabilityLab/0.1 (mailto:day10-lab@example.com)"}

    payload: dict | None = None
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.get(CROSSREF_API_URL, params=params, headers=headers, timeout=30)
            if response.status_code in {429, 503}:
                time.sleep(2 * attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:  # network / transient errors
            last_error = exc
            time.sleep(2 * attempt)

    if payload is None:
        raise RuntimeError(f"Failed to fetch from Crossref after retries: {last_error}")

    write_json(settings.paths.raw_api_response, payload)
    records = parse_crossref_payload(payload)
    write_json(settings.paths.raw_records_json, [asdict(record) for record in records])
    return records


def load_raw_records(path: Path) -> list[PaperRecord]:
    """Read a saved JSON snapshot of parsed records back into `PaperRecord` objects."""
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [PaperRecord(**row) for row in raw]
