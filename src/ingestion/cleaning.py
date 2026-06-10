from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from core.utils import compact_join, normalize_whitespace
from ingestion.crossref import PaperRecord


def _parse_published(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _build_text_for_embedding(row: dict) -> str:
    return (
        f"Title: {row['title']}\n"
        f"Authors: {row['authors_joined']}\n"
        f"Categories: {row['categories_joined']}\n"
        f"Published: {row['published']}\n"
        f"Summary: {row['summary']}"
    )


def build_clean_dataframe(records: list[PaperRecord], run_date: datetime) -> pd.DataFrame:
    """Clean raw records into a dataframe ready to embed and evaluate."""
    run_day = run_date.date()
    rows: list[dict] = []

    for record in records:
        title = normalize_whitespace(record.title)
        summary = normalize_whitespace(record.summary)
        published_date = _parse_published(record.published)

        # Skip records that cannot support retrieval or freshness analysis.
        if not title or not summary or published_date is None:
            continue

        authors = [normalize_whitespace(a) for a in record.authors if a]
        categories = [normalize_whitespace(c) for c in record.categories if c]

        row = {
            "paper_id": record.paper_id,
            "title": title,
            "summary": summary,
            "authors": authors,
            "categories": categories,
            "primary_category": record.primary_category,
            "authors_joined": compact_join(authors),
            "categories_joined": compact_join(categories),
            "published": published_date.isoformat(),
            "updated": record.updated,
            "age_days": (run_day - published_date).days,
            "summary_chars": len(summary),
            "abs_url": record.abs_url,
            "pdf_url": record.pdf_url,
            "comment": record.comment,
        }
        row["text_for_embedding"] = _build_text_for_embedding(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset="paper_id", keep="first")
    df = df.sort_values("published", ascending=False).reset_index(drop=True)
    return df
