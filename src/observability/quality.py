from __future__ import annotations

from typing import Any

import pandas as pd

from core.config import Settings
from core.utils import write_json

MIN_ROWS = 5
MIN_SUMMARY_CHARS = 50


def run_data_quality_checks(df: pd.DataFrame, settings: Settings, report_name: str) -> dict[str, Any]:
    """Run a small battery of data quality checks and persist the result."""
    total_rows = int(len(df))
    threshold = settings.freshness_threshold_days

    checks: list[dict[str, Any]] = []

    def add(name: str, success: bool, observed: Any, detail: str) -> None:
        checks.append({"name": name, "success": bool(success), "observed": observed, "detail": detail})

    add(
        "row_count_minimum",
        total_rows >= MIN_ROWS,
        total_rows,
        f"Expected at least {MIN_ROWS} rows.",
    )

    if total_rows:
        null_ids = int(df["paper_id"].isna().sum() + (df["paper_id"] == "").sum())
        add("paper_id_not_null", null_ids == 0, null_ids, "paper_id must be present.")

        duplicate_ids = int(df["paper_id"].duplicated().sum())
        add("paper_id_unique", duplicate_ids == 0, duplicate_ids, "paper_id must be unique.")

        empty_titles = int((df["title"].fillna("").str.strip() == "").sum())
        add("title_not_null", empty_titles == 0, empty_titles, "title must be non-empty.")

        short_summaries = int((df["summary_chars"] < MIN_SUMMARY_CHARS).sum())
        add(
            "summary_min_length",
            short_summaries == 0,
            short_summaries,
            f"summary should have at least {MIN_SUMMARY_CHARS} characters.",
        )

        stale_rows = int((df["age_days"] > threshold).sum())
        add(
            "freshness_within_threshold",
            stale_rows == 0,
            stale_rows,
            f"rows should be at most {threshold} days old.",
        )

    passed = sum(1 for c in checks if c["success"])
    failed = len(checks) - passed
    result = {
        "report_name": report_name,
        "total_rows": total_rows,
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "success": failed == 0,
        "checks": checks,
    }

    write_json(settings.paths.quality_dir / f"{report_name}.json", result)
    return result


def build_freshness_report(df: pd.DataFrame, settings: Settings, report_path) -> dict[str, Any]:
    """Summarize how fresh the dataset is and persist a JSON report."""
    threshold = settings.freshness_threshold_days
    total_rows = int(len(df))

    if total_rows:
        stale_rows = int((df["age_days"] > threshold).sum())
        payload = {
            "total_rows": total_rows,
            "latest_published": str(df["published"].max()),
            "oldest_published": str(df["published"].min()),
            "min_age_days": int(df["age_days"].min()),
            "max_age_days": int(df["age_days"].max()),
            "stale_rows": stale_rows,
            "fresh_rows": total_rows - stale_rows,
            "freshness_threshold_days": threshold,
            "is_fresh": stale_rows == 0,
        }
    else:
        payload = {
            "total_rows": 0,
            "latest_published": None,
            "oldest_published": None,
            "stale_rows": 0,
            "fresh_rows": 0,
            "freshness_threshold_days": threshold,
            "is_fresh": False,
        }

    write_json(report_path, payload)
    return payload
