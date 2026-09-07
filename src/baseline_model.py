"""
Baseline classifier: TF-IDF features + Logistic Regression.

This is deliberately simple. It's the model to reach for first on any text
classification problem, it trains in seconds, and its coefficients are
directly readable per class -- which makes it a good one to be able to
explain end-to-end (what TF-IDF actually weights, why a linear model here
is a reasonable starting point) rather than just a thing you ran.

Usage:
    python src/baseline_model.py
    python src/baseline_model.py --data data/nhtsa_real.csv --model-out models/baseline.joblib
"""
import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import add_clean_narrative, load_reports, summarize_categories


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # categories are rarely balanced in the real world
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/sample_reports.csv")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="models/baseline.joblib")
    args = parser.parse_args()

    df = load_reports(args.data)
    summarize_categories(df)
    df = add_clean_narrative(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_narrative"],
        df["category"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["category"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\n=== Classification report (test set) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== Confusion matrix ===")
    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    header = "".join(f"{l[:10]:>12}" for l in labels)
    print(f"{'':>22}{header}")
    for label, row in zip(labels, cm):
        row_str = "".join(f"{v:>12}" for v in row)
        print(f"{label:>22}{row_str}")

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"\nSaved trained pipeline to {out_path}")


if __name__ == "__main__":
    main()
