"""
Shared text-cleaning and data-loading utilities used by every model in this
project, so preprocessing stays identical across the baseline, embedding,
and clustering approaches -- that consistency matters if you want a fair
comparison between them.
"""
import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"category", "narrative"}


def load_reports(csv_path: str) -> pd.DataFrame:
    """Load a reports CSV and do basic validation / cleanup.

    Expects at least `category` and `narrative` columns (report_id is
    optional and added if missing). Drops rows with empty narratives or
    categories, and drops exact-duplicate narratives.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No data file at {csv_path}. Run `python data/generate_sample_data.py` "
            f"for a synthetic sample, or `python src/fetch_data.py` for real NHTSA data."
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {missing}")

    if "report_id" not in df.columns:
        df.insert(0, "report_id", range(1, len(df) + 1))

    before = len(df)
    df = df.dropna(subset=["category", "narrative"])
    df = df[df["narrative"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["narrative"])
    after = len(df)
    if after < before:
        print(f"[preprocess] dropped {before - after} row(s) with missing/empty/duplicate narratives")

        df = df.reset_index(drop=True)
        df["category"] = df["category"].astype(object)
        df["narrative"] = df["narrative"].astype(object)
    return df


def clean_text(text: str) -> str:
    """Light-touch cleaning: lowercase, collapse whitespace, strip stray
    punctuation. Deliberately not too aggressive -- TF-IDF and sentence
    embeddings both do fine with mild noise, and over-cleaning (e.g.
    stripping all numbers) can throw away real signal like mileage.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.,%\- ]", "", text)
    return text.strip()


def add_clean_narrative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_narrative"] = df["narrative"].apply(clean_text).astype(object)
    return df


def summarize_categories(df: pd.DataFrame) -> None:
    """Print a quick class-balance summary -- worth looking at before
    trusting any accuracy number, since these datasets are rarely balanced.
    """
    counts = df["category"].value_counts()
    print(f"{len(df)} reports across {len(counts)} categories:")
    for category, count in counts.items():
        pct = 100 * count / len(df)
        print(f"  {category:<25} {count:>5}  ({pct:.1f}%)")
