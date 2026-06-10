# Corruption Flow — Comparison Report

## Metrics: Baseline vs Corrupted vs Repaired

| Metric | Baseline | Corrupted | Δ (corrupt) | Repaired | Δ (repair) |
| --- | --- | --- | --- | --- | --- |
| Retrieval hit rate | 1.0000 | 0.5000 | -0.5000 ⬇️ | 1.0000 | +0.0000 ➖ |
| Mean token F1 | 1.0000 | 0.4444 | -0.5556 ⬇️ | 1.0000 | +0.0000 ➖ |
| Judge accuracy | 1.0000 | 0.4444 | -0.5556 ⬇️ | 1.0000 | +0.0000 ➖ |
| Mean judge score | 5 | 2.7778 | -2.2222 ⬇️ | 5 | +0.0000 ➖ |

## Token F1 by Question Type

| Question type | Baseline | Corrupted | Repaired |
| --- | --- | --- | --- |
| authors | 1.0000 | 0.5000 | 1.0000 |
| date | 1.0000 | 0.5000 | 1.0000 |
| summary | 1.0000 | 0.3333 | 1.0000 |

## Corruption Scenario

Rows: **23 → 22**.

| Failure injected | Rows affected |
| --- | --- |
| Dropped newest records | 3 |
| Blanked summaries | 7 |
| Injected noise | 5 |
| Truncated titles | 4 |
| Staled publication dates | 3 |
| Duplicated rows | 2 |

## Data Quality

| Dataset | Result | Passed | Failed |
| --- | --- | --- | --- |
| Corrupted | FAIL | 3 | 3 |
| Repaired | PASS | 6 | 0 |

## Freshness

| Dataset | Is fresh | Stale rows | Latest published |
| --- | --- | --- | --- |
| Corrupted | False | 3 | 2026-05-06 |
| Repaired | True | 0 | 2026-06-02 |

## Analysis

- **Retrieval impact**: hit rate fell by **50.0%**. Dropping the 3 newest papers removed the ground-truth documents for their questions, so retrieval can no longer surface the correct source.
- **Answer impact**: token F1 fell by **55.6%**. Blanked/noised summaries break `summary` questions, while staled dates break `date` questions — the QA layer extracts the wrong field value.
- **Quality vs evaluation**: the data-quality checks flag the duplicates, short summaries and stale rows *before* any LLM call, demonstrating that cheap observability catches the same defects that later degrade the agent.
- **Repair**: rebuilding from the raw source fully restores metrics to the baseline level, confirming the regression was caused by the data and not the model.
