# Phase 1 — Baseline Report

## Source

- **source_api**: Crossref REST API
- **source_query**: agentic retrieval augmented generation large language model
- **source_filter**: from-pub-date:2025-12-12,has-abstract:true
- **max_results**: 24
- **rows_after_clean**: 23
- **embedding_model**: sentence-transformers/all-MiniLM-L6-v2
- **collection**: papers-baseline

## Retrieval & Evaluation Metrics

| Metric | Value |
| --- | --- |
| Samples | 18 |
| Retrieval hit rate | 1.0000 |
| Mean token F1 | 1.0000 |
| Judge accuracy | 1.0000 |
| Mean judge score | 5 |

## Metrics by Question Type

| Question type | N | Hit rate | Mean F1 | Judge acc |
| --- | --- | --- | --- | --- |
| authors | 6 | 1.0000 | 1.0000 | 1.0000 |
| date | 6 | 1.0000 | 1.0000 | 1.0000 |
| summary | 6 | 1.0000 | 1.0000 | 1.0000 |

## Data Quality

- Result: **PASS** (6/6 checks passed)

| Check | Success | Observed |
| --- | --- | --- |
| row_count_minimum | ✅ | 23 |
| paper_id_not_null | ✅ | 0 |
| paper_id_unique | ✅ | 0 |
| title_not_null | ✅ | 0 |
| summary_min_length | ✅ | 0 |
| freshness_within_threshold | ✅ | 0 |

## Freshness

- **total_rows**: 23
- **latest_published**: 2026-06-02
- **oldest_published**: 2025-12-19
- **min_age_days**: 8
- **max_age_days**: 173
- **stale_rows**: 0
- **fresh_rows**: 23
- **freshness_threshold_days**: 180
- **is_fresh**: True
