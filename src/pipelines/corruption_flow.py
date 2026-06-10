from __future__ import annotations

import pandas as pd

from core.config import load_settings
from core.utils import now_utc, read_json, write_csv, write_json
from evaluation.metrics import evaluate_pipeline
from ingestion.cleaning import build_clean_dataframe
from ingestion.corruption import corrupt_clean_dataframe
from ingestion.crossref import load_raw_records
from observability.quality import build_freshness_report, run_data_quality_checks
from observability.reporting import generate_corruption_report
from retrieval.index import LocalEmbeddingIndex


def main() -> None:
    """Corrupt -> evaluate -> repair -> compare flow."""
    settings = load_settings()
    paths = settings.paths

    # 1. Load baseline metrics and the clean dataset.
    baseline_metrics = read_json(paths.baseline_metrics)
    clean_df = pd.DataFrame(read_json(paths.clean_json))
    print(f"Loaded baseline clean dataset: {len(clean_df)} rows")

    # 2-3. Create the corrupted dataset and persist artifacts.
    corrupted_df = corrupt_clean_dataframe(clean_df, paths.corruption_log)
    write_csv(corrupted_df, paths.corrupted_clean_csv)
    write_json(paths.corrupted_clean_json, corrupted_df.to_dict(orient="records"))
    print(f"Corrupted dataset: {len(corrupted_df)} rows")

    # 4. Rebuild index + evaluate on the SAME test set.
    corrupted_index = LocalEmbeddingIndex.build(corrupted_df, settings, paths.corrupted_embeddings_json)
    corrupted_bundle = evaluate_pipeline(
        settings=settings,
        index=corrupted_index,
        test_set_path=paths.eval_testset,
        metrics_output_path=paths.corrupted_metrics,
        answers_output_path=paths.corrupted_answers,
    )

    # 5. Observability on corrupted data.
    corrupted_quality = run_data_quality_checks(corrupted_df, settings, "corrupted_quality")
    corrupted_freshness = build_freshness_report(
        corrupted_df, settings, paths.quality_dir / "freshness_corrupted.json"
    )

    # 6. Repair from the raw source snapshot.
    records = load_raw_records(paths.raw_records_json)
    repaired_df = build_clean_dataframe(records, now_utc())
    write_csv(repaired_df, paths.repaired_clean_csv)
    write_json(paths.repaired_clean_json, repaired_df.to_dict(orient="records"))
    print(f"Repaired dataset: {len(repaired_df)} rows")

    # 7. Rebuild index + evaluate repaired data.
    repaired_index = LocalEmbeddingIndex.build(repaired_df, settings, paths.repaired_embeddings_json)
    repaired_bundle = evaluate_pipeline(
        settings=settings,
        index=repaired_index,
        test_set_path=paths.eval_testset,
        metrics_output_path=paths.repaired_metrics,
        answers_output_path=paths.repaired_answers,
    )
    repaired_quality = run_data_quality_checks(repaired_df, settings, "repaired_quality")
    repaired_freshness = build_freshness_report(
        repaired_df, settings, paths.quality_dir / "freshness_repaired.json"
    )

    # 8. Comparison report (with per-question-type breakdown + analysis).
    baseline_answers = read_json(paths.baseline_answers)
    generate_corruption_report(
        paths.comparison_report,
        baseline_metrics=baseline_metrics,
        corrupted_metrics=corrupted_bundle.summary,
        repaired_metrics=repaired_bundle.summary,
        corrupted_quality=corrupted_quality,
        repaired_quality=repaired_quality,
        corrupted_freshness=corrupted_freshness,
        repaired_freshness=repaired_freshness,
        corruption_log=read_json(paths.corruption_log),
        baseline_answers=baseline_answers,
        corrupted_answers=corrupted_bundle.answers,
        repaired_answers=repaired_bundle.answers,
    )

    def line(label: str, metrics: dict) -> str:
        return (
            f"  {label:<10} hit={metrics.get('retrieval_hit_rate')}  "
            f"f1={metrics.get('mean_token_f1')}  "
            f"judge_acc={metrics.get('judge_accuracy')}"
        )

    print("\nComparison:")
    print(line("baseline", baseline_metrics))
    print(line("corrupted", corrupted_bundle.summary))
    print(line("repaired", repaired_bundle.summary))
    print(f"\nReport: {paths.comparison_report}")


if __name__ == "__main__":
    main()
