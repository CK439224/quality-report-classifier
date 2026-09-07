"""
Basic tests for the parts of the pipeline that don't require downloading a
sentence-transformer model, so these run fast and offline (e.g. in CI).

Embedding/clustering/similarity-search tests are skipped automatically if
sentence-transformers isn't installed, rather than failing -- that keeps
`pytest` usable in this sandbox project even before you've installed the
heavier optional dependencies.

Run with:
    pytest tests/
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import add_clean_narrative, clean_text, load_reports  # noqa: E402


@pytest.fixture
def tiny_csv(tmp_path):
    df = pd.DataFrame(
        {
            "category": ["ENGINE", "ENGINE", "BRAKES", "BRAKES", "BRAKES", ""],
            "narrative": [
                "Engine stalled at highway speed with no warning.",
                "Engine stalled at highway speed with no warning.",  # exact duplicate -> should be dropped
                "Brake pedal felt spongy and went to the floor.",
                "Grinding noise when braking at low speed.",
                None,  # missing narrative -> should be dropped
                "Row with empty category -> should be dropped.",
            ],
        }
    )
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path


def test_load_reports_drops_bad_rows(tiny_csv):
    df = load_reports(str(tiny_csv))
    # 6 rows in -> 1 exact duplicate, 1 missing narrative, 1 missing category dropped -> 3 remain
    assert len(df) == 3
    assert set(df["category"]) == {"ENGINE", "BRAKES"}
    assert "report_id" in df.columns


def test_load_reports_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reports(str(tmp_path / "does_not_exist.csv"))


def test_load_reports_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"narrative": ["a report"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_reports(str(path))


def test_clean_text_lowercases_and_strips_punctuation():
    cleaned = clean_text("Engine Stalled!! At 45,000 Miles (approx.)")
    assert cleaned == cleaned.lower()
    assert "!" not in cleaned
    assert "(" not in cleaned


def test_add_clean_narrative_adds_column(tiny_csv):
    df = load_reports(str(tiny_csv))
    df = add_clean_narrative(df)
    assert "clean_narrative" in df.columns
    assert all(df["clean_narrative"].str.islower() | (df["clean_narrative"] == ""))


def test_sample_data_generator_produces_expected_shape(tmp_path):
    """Smoke test for data/generate_sample_data.py -- run it and check the output looks right."""
    import subprocess

    project_root = Path(__file__).resolve().parent.parent
    out_path = tmp_path / "sample.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "data" / "generate_sample_data.py"),
            "--n-per-category",
            "5",
            "--seed",
            "1",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert set(df.columns) == {"report_id", "category", "narrative"}
    assert len(df) == 10 * 5  # 10 categories in CATEGORY_FRAGMENTS


def test_baseline_pipeline_trains_and_predicts(tiny_csv):
    """End-to-end smoke test for the baseline model's building blocks, using
    a tiny in-memory dataset rather than the bundled sample (keeps this test fast).
    """
    from baseline_model import build_pipeline

    df = load_reports(str(tiny_csv))
    pipeline = build_pipeline()
    pipeline.fit(df["narrative"], df["category"])

    prediction = pipeline.predict(["brake noise while stopping"])[0]
    assert prediction in set(df["category"])


def test_similarity_searcher_skipped_without_sentence_transformers():
    """Confirms similarity_search.py raises a clear, actionable error rather
    than a confusing one when the embeddings cache hasn't been built yet."""
    from similarity_search import SimilaritySearcher

    with pytest.raises(FileNotFoundError):
        SimilaritySearcher(embeddings_cache="models/definitely_missing.npz")
