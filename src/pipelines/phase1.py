from __future__ import annotations

from core.config import load_settings
from core.utils import now_utc, read_json, write_csv, write_json
from evaluation.metrics import evaluate_pipeline
from evaluation.testset import build_test_set
from ingestion.cleaning import build_clean_dataframe
from ingestion.crossref import fetch_source_records, load_raw_records
from observability.quality import build_freshness_report, run_data_quality_checks
from observability.reporting import generate_phase1_report
from retrieval.index import LocalEmbeddingIndex


def main() -> None:
    """Build the baseline RAG pipeline end-to-end."""
    settings = load_settings()
    paths = settings.paths

    # 1. Load or fetch raw records.
    if settings.refresh_source or not paths.raw_records_json.exists():
        print("Fetching raw records from source ...")
        records = fetch_source_records(settings)
    else:
        print("Loading cached raw records ...")
        records = load_raw_records(paths.raw_records_json)
    print(f"  -> {len(records)} raw records")

    # 2. Clean and persist the dataset.
    df = build_clean_dataframe(records, now_utc())
    if df.empty:
        raise RuntimeError("Cleaning produced an empty dataframe; check the source query/filter.")
    write_csv(df, paths.clean_csv)
    write_json(paths.clean_json, df.to_dict(orient="records"))
    print(f"  -> {len(df)} clean rows")

    # 3. Build the embedding index (baseline collection).
    index = LocalEmbeddingIndex.build(df, settings)
    print("  -> embedding index built")

    # 4. Create or load the evaluation set.
    if settings.refresh_test_set or not paths.eval_testset.exists():
        test_set = build_test_set(df, paths.eval_testset)
    else:
        test_set = read_json(paths.eval_testset)
    print(f"  -> {len(test_set)} eval questions")

    # 5. Evaluate.
    bundle = evaluate_pipeline(
        settings=settings,
        index=index,
        test_set_path=paths.eval_testset,
        metrics_output_path=paths.baseline_metrics,
        answers_output_path=paths.baseline_answers,
    )
    print("  -> evaluation complete")

    # 6. Observability: quality checks + freshness.
    quality = run_data_quality_checks(df, settings, "phase1_quality")
    freshness = build_freshness_report(df, settings, paths.freshness_report)

    # 7. Markdown report.
    source_summary = {
        "source_api": settings.source_api,
        "source_query": settings.source_query,
        "source_filter": settings.source_filter,
        "max_results": settings.max_results,
        "rows_after_clean": len(df),
        "embedding_model": settings.embedding_model,
        "collection": settings.baseline_collection_name,
    }
    generate_phase1_report(
        paths.baseline_report,
        source_summary=source_summary,
        metrics=bundle.summary,
        quality=quality,
        freshness=freshness,
        answers=bundle.answers,
    )

    print("\nBaseline metrics:")
    for key in ("samples", "retrieval_hit_rate", "mean_token_f1", "judge_accuracy", "mean_judge_score"):
        print(f"  {key}: {bundle.summary.get(key)}")
    print(f"\nReport: {paths.baseline_report}")


if __name__ == "__main__":
    main()
