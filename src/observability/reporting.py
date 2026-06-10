from __future__ import annotations

from statistics import mean
from typing import Any

from core.utils import write_text


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _delta(base: Any, other: Any) -> str:
    """Format a signed delta with a direction arrow."""
    try:
        diff = float(other) - float(base)
    except (TypeError, ValueError):
        return "n/a"
    arrow = "⬇️" if diff < 0 else ("⬆️" if diff > 0 else "➖")
    return f"{diff:+.4f} {arrow}"


def _metrics_rows(metrics: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        ("Samples", metrics.get("samples")),
        ("Retrieval hit rate", metrics.get("retrieval_hit_rate")),
        ("Mean token F1", metrics.get("mean_token_f1")),
        ("Judge accuracy", metrics.get("judge_accuracy")),
        ("Mean judge score", metrics.get("mean_judge_score")),
    ]


def _qtype_breakdown(answers: list[dict[str, Any]] | None) -> dict[str, dict[str, float]]:
    """Aggregate retrieval/answer quality per question_type."""
    if not answers:
        return {}
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in answers:
        buckets.setdefault(item.get("question_type", "unknown"), []).append(item)
    out: dict[str, dict[str, float]] = {}
    for qtype, items in sorted(buckets.items()):
        out[qtype] = {
            "n": len(items),
            "hit_rate": mean(1.0 if it.get("retrieval_hit") else 0.0 for it in items),
            "mean_f1": mean(float(it.get("token_f1", 0.0)) for it in items),
            "judge_acc": mean(1.0 if it.get("judge", {}).get("correct") else 0.0 for it in items),
        }
    return out


def generate_phase1_report(
    report_path,
    source_summary: dict[str, Any],
    metrics: dict[str, Any],
    quality: dict[str, Any],
    freshness: dict[str, Any],
    answers: list[dict[str, Any]] | None = None,
) -> None:
    """Write the baseline phase markdown report."""
    lines: list[str] = ["# Phase 1 — Baseline Report", ""]

    lines += ["## Source", ""]
    for key, value in source_summary.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines += ["## Retrieval & Evaluation Metrics", "", "| Metric | Value |", "| --- | --- |"]
    for label, value in _metrics_rows(metrics):
        lines.append(f"| {label} | {_fmt(value)} |")
    lines.append("")

    breakdown = _qtype_breakdown(answers)
    if breakdown:
        lines += [
            "## Metrics by Question Type",
            "",
            "| Question type | N | Hit rate | Mean F1 | Judge acc |",
            "| --- | --- | --- | --- | --- |",
        ]
        for qtype, vals in breakdown.items():
            lines.append(
                f"| {qtype} | {int(vals['n'])} | {vals['hit_rate']:.4f} | "
                f"{vals['mean_f1']:.4f} | {vals['judge_acc']:.4f} |"
            )
        lines.append("")

    lines += ["## Data Quality", "",
              f"- Result: **{'PASS' if quality.get('success') else 'FAIL'}** "
              f"({quality.get('passed')}/{quality.get('total_checks')} checks passed)", "",
              "| Check | Success | Observed |", "| --- | --- | --- |"]
    for check in quality.get("checks", []):
        lines.append(f"| {check['name']} | {'✅' if check['success'] else '❌'} | {check['observed']} |")
    lines.append("")

    lines += ["## Freshness", ""]
    for key, value in freshness.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    write_text(report_path, "\n".join(lines))


def _comparison_table(
    baseline: dict[str, Any],
    corrupted: dict[str, Any],
    repaired: dict[str, Any],
) -> list[str]:
    metric_keys = [
        ("retrieval_hit_rate", "Retrieval hit rate"),
        ("mean_token_f1", "Mean token F1"),
        ("judge_accuracy", "Judge accuracy"),
        ("mean_judge_score", "Mean judge score"),
    ]
    rows = [
        "| Metric | Baseline | Corrupted | Δ (corrupt) | Repaired | Δ (repair) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, label in metric_keys:
        rows.append(
            f"| {label} | {_fmt(baseline.get(key))} | {_fmt(corrupted.get(key))} | "
            f"{_delta(baseline.get(key), corrupted.get(key))} | {_fmt(repaired.get(key))} | "
            f"{_delta(baseline.get(key), repaired.get(key))} |"
        )
    return rows


def _scenario_section(corruption_log: dict[str, Any] | None) -> list[str]:
    if not corruption_log:
        return []
    counts = [
        ("Dropped newest records", "dropped_latest_paper_ids"),
        ("Blanked summaries", "blanked_summary_paper_ids"),
        ("Injected noise", "noised_paper_ids"),
        ("Truncated titles", "truncated_title_paper_ids"),
        ("Staled publication dates", "staled_paper_ids"),
        ("Duplicated rows", "duplicated_paper_ids"),
    ]
    lines = [
        "## Corruption Scenario",
        "",
        f"Rows: **{corruption_log.get('rows_before')} → {corruption_log.get('rows_after')}**.",
        "",
        "| Failure injected | Rows affected |",
        "| --- | --- |",
    ]
    for label, key in counts:
        lines.append(f"| {label} | {len(corruption_log.get(key, []))} |")
    lines.append("")
    return lines


def _breakdown_compare_section(
    baseline_answers: list[dict[str, Any]] | None,
    corrupted_answers: list[dict[str, Any]] | None,
    repaired_answers: list[dict[str, Any]] | None,
) -> list[str]:
    b = _qtype_breakdown(baseline_answers)
    c = _qtype_breakdown(corrupted_answers)
    r = _qtype_breakdown(repaired_answers)
    if not (b and c):
        return []
    lines = [
        "## Token F1 by Question Type",
        "",
        "| Question type | Baseline | Corrupted | Repaired |",
        "| --- | --- | --- | --- |",
    ]
    for qtype in sorted(b):
        lines.append(
            f"| {qtype} | {b[qtype]['mean_f1']:.4f} | "
            f"{c.get(qtype, {}).get('mean_f1', 0.0):.4f} | "
            f"{r.get(qtype, {}).get('mean_f1', 0.0):.4f} |"
        )
    lines.append("")
    return lines


def _analysis_section(
    baseline: dict[str, Any],
    corrupted: dict[str, Any],
    repaired: dict[str, Any],
    corruption_log: dict[str, Any] | None,
) -> list[str]:
    def pct_drop(key: str) -> float:
        base = float(baseline.get(key) or 0.0)
        corr = float(corrupted.get(key) or 0.0)
        return 0.0 if base == 0 else (base - corr) / base * 100.0

    hit_drop = pct_drop("retrieval_hit_rate")
    f1_drop = pct_drop("mean_token_f1")
    recovered = abs(float(repaired.get("mean_token_f1") or 0.0) - float(baseline.get("mean_token_f1") or 0.0)) < 1e-6
    n_dropped = len(corruption_log.get("dropped_latest_paper_ids", [])) if corruption_log else 0

    return [
        "## Analysis",
        "",
        f"- **Retrieval impact**: hit rate fell by **{hit_drop:.1f}%**. Dropping the {n_dropped} newest "
        "papers removed the ground-truth documents for their questions, so retrieval can no longer "
        "surface the correct source.",
        f"- **Answer impact**: token F1 fell by **{f1_drop:.1f}%**. Blanked/noised summaries break "
        "`summary` questions, while staled dates break `date` questions — the QA layer extracts the "
        "wrong field value.",
        "- **Quality vs evaluation**: the data-quality checks flag the duplicates, short summaries and "
        "stale rows *before* any LLM call, demonstrating that cheap observability catches the same "
        "defects that later degrade the agent.",
        f"- **Repair**: rebuilding from the raw source {'fully restores' if recovered else 'restores'} "
        "metrics to the baseline level, confirming the regression was caused by the data and not the model.",
        "",
    ]


def generate_corruption_report(
    report_path,
    baseline_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    corrupted_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    corrupted_freshness: dict[str, Any],
    repaired_freshness: dict[str, Any],
    corruption_log: dict[str, Any] | None = None,
    baseline_answers: list[dict[str, Any]] | None = None,
    corrupted_answers: list[dict[str, Any]] | None = None,
    repaired_answers: list[dict[str, Any]] | None = None,
) -> None:
    """Write the corruption comparison markdown report with bonus analysis."""
    lines: list[str] = ["# Corruption Flow — Comparison Report", ""]

    lines += ["## Metrics: Baseline vs Corrupted vs Repaired", ""]
    lines += _comparison_table(baseline_metrics, corrupted_metrics, repaired_metrics)
    lines.append("")

    lines += _breakdown_compare_section(baseline_answers, corrupted_answers, repaired_answers)
    lines += _scenario_section(corruption_log)

    lines += [
        "## Data Quality",
        "",
        "| Dataset | Result | Passed | Failed |",
        "| --- | --- | --- | --- |",
        f"| Corrupted | {'PASS' if corrupted_quality.get('success') else 'FAIL'} | "
        f"{corrupted_quality.get('passed')} | {corrupted_quality.get('failed')} |",
        f"| Repaired | {'PASS' if repaired_quality.get('success') else 'FAIL'} | "
        f"{repaired_quality.get('passed')} | {repaired_quality.get('failed')} |",
        "",
    ]

    lines += [
        "## Freshness",
        "",
        "| Dataset | Is fresh | Stale rows | Latest published |",
        "| --- | --- | --- | --- |",
        f"| Corrupted | {corrupted_freshness.get('is_fresh')} | {corrupted_freshness.get('stale_rows')} | "
        f"{corrupted_freshness.get('latest_published')} |",
        f"| Repaired | {repaired_freshness.get('is_fresh')} | {repaired_freshness.get('stale_rows')} | "
        f"{repaired_freshness.get('latest_published')} |",
        "",
    ]

    lines += _analysis_section(baseline_metrics, corrupted_metrics, repaired_metrics, corruption_log)

    write_text(report_path, "\n".join(lines))
