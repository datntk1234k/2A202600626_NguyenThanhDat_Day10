from __future__ import annotations

import pandas as pd

from core.utils import write_json

STALE_DATE = "2001-01-01"
STALE_AGE_DAYS = 9000
NOISE_TOKEN = " lorem ipsum dolor sit amet garbled-token-xyz"


def _rebuild_text(row: pd.Series) -> str:
    return (
        f"Title: {row['title']}\n"
        f"Authors: {row['authors_joined']}\n"
        f"Categories: {row['categories_joined']}\n"
        f"Published: {row['published']}\n"
        f"Summary: {row['summary']}"
    )


def corrupt_clean_dataframe(df: pd.DataFrame, output_log_path) -> pd.DataFrame:
    """Simulate several realistic data-quality failures and log every change.

    The input is assumed sorted by `published` descending (newest first), as
    produced by `ingestion.cleaning.build_clean_dataframe`.
    """
    work = df.copy().reset_index(drop=True)
    log: dict[str, object] = {}

    # 1. Drop the newest records (simulates a broken/late ingestion run).
    n_drop = min(3, len(work) - 1)
    dropped_ids = work.head(n_drop)["paper_id"].tolist()
    work = work.iloc[n_drop:].reset_index(drop=True)
    log["dropped_latest_paper_ids"] = dropped_ids

    # 2. Blank out summaries on every 3rd remaining row.
    blank_idx = list(range(0, len(work), 3))
    work.loc[blank_idx, "summary"] = ""
    work.loc[blank_idx, "summary_chars"] = 0
    log["blanked_summary_paper_ids"] = work.loc[blank_idx, "paper_id"].tolist()

    # 3. Inject noise into the summary of every 4th row.
    noise_idx = list(range(1, len(work), 4))
    work.loc[noise_idx, "summary"] = work.loc[noise_idx, "summary"].astype(str) + NOISE_TOKEN
    log["noised_paper_ids"] = work.loc[noise_idx, "paper_id"].tolist()

    # 4. Truncate titles on every 5th row.
    trunc_idx = list(range(2, len(work), 5))
    work.loc[trunc_idx, "title"] = work.loc[trunc_idx, "title"].astype(str).str.slice(0, 8)
    log["truncated_title_paper_ids"] = work.loc[trunc_idx, "paper_id"].tolist()

    # 5. Make publication dates stale on every 6th row.
    stale_idx = list(range(3, len(work), 6))
    work.loc[stale_idx, "published"] = STALE_DATE
    work.loc[stale_idx, "age_days"] = STALE_AGE_DAYS
    log["staled_paper_ids"] = work.loc[stale_idx, "paper_id"].tolist()

    # 6. Add duplicate rows (breaks the uniqueness invariant).
    dup_rows = work.head(2).copy()
    work = pd.concat([work, dup_rows], ignore_index=True)
    log["duplicated_paper_ids"] = dup_rows["paper_id"].tolist()

    # 7. Rebuild text_for_embedding so embeddings reflect the corruption.
    work["text_for_embedding"] = work.apply(_rebuild_text, axis=1)

    log["rows_before"] = int(len(df))
    log["rows_after"] = int(len(work))
    write_json(output_log_path, log)
    return work
