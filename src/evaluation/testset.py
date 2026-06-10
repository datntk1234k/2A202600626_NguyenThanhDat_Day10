from __future__ import annotations

from typing import Any

import pandas as pd

from core.utils import first_sentence, write_json

MIN_DOCUMENTS = 5
MAX_PAPERS = 6


def build_test_set(df: pd.DataFrame, output_path) -> list[dict[str, Any]]:
    """Build an evaluation set from the cleaned dataframe.

    Question phrasings intentionally match the keyword routing in
    `retrieval.qa._extract_answer` so the deterministic QA path can answer them,
    and each question embeds the title in single quotes for exact lookup.
    """
    if len(df) < MIN_DOCUMENTS:
        raise ValueError(
            f"Need at least {MIN_DOCUMENTS} documents to build a test set, got {len(df)}."
        )

    # Pick papers that carry the richest metadata for varied questions.
    candidates = df[(df["authors_joined"] != "") & (df["categories_joined"] != "")]
    if len(candidates) < MIN_DOCUMENTS:
        candidates = df
    selected = candidates.head(MAX_PAPERS)

    test_set: list[dict[str, Any]] = []
    counter = 1

    def add(question_type: str, question: str, ground_truth: str, paper_id: str) -> None:
        nonlocal counter
        if not ground_truth:
            return
        test_set.append(
            {
                "id": f"q{counter:03d}",
                "question_type": question_type,
                "question": question,
                "ground_truth": ground_truth,
                "ground_truth_doc_ids": [paper_id],
            }
        )
        counter += 1

    for row in selected.to_dict(orient="records"):
        title = row["title"]
        paper_id = row["paper_id"]

        add("summary", f"What is the paper titled '{title}' about?",
            first_sentence(row["summary"]), paper_id)
        add("authors", f"Who authored the paper titled '{title}'?",
            row["authors_joined"], paper_id)
        add("date", f"When was the paper titled '{title}' published?",
            row["published"], paper_id)
        add("categories", f"What categories does the paper titled '{title}' belong to?",
            row["categories_joined"], paper_id)

    write_json(output_path, test_set)
    return test_set
