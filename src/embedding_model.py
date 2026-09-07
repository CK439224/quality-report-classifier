"""
Embedding-based classifier: sentence-transformer embeddings + a linear
classifier on top, instead of TF-IDF.

The point of building this alongside baseline_model.py is to compare them
honestly. Dense embeddings should generalize better to paraphrasing and
vocabulary variation than bag-of-words TF-IDF, but on a small or
narrow-vocabulary dataset that advantage can be small or nonexistent --
knowing when the simple model is "good enough" is a real, useful skill, not
a consolation prize. Run both and look at the actual numbers before
assuming the fancier one wins.

Uses `all-MiniLM-L6-v2`: small (~80MB), fast on CPU, good general-purpose
default for a portfolio project. Swap MODEL_NAME for a larger model if you
want to push accuracy further and don't mind the extra download/compute.

Usage:
    python src/embedding_model.py
    python src/embedding_model.py --data data/nhtsa_real.csv --model-out models/embedding_clf.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from preprocess import load_reports, summarize_categories

MODEL_NAME = "all-MiniLM-L6-v2"


def embed_narratives(narratives, encoder: SentenceTransformer) -> np.ndarray:
    return encoder.encode(list(narratives), show_progress_bar=True, normalize_embeddings=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/sample_reports.csv")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="models/embedding_clf.joblib")
    parser.add_argument("--embeddings-out", type=str, default="models/embeddings.npz", help="Cache narrative embeddings for reuse by similarity_search.py")
    args = parser.parse_args()

    df = load_reports(args.data)
    summarize_categories(df)

    print(f"\nLoading sentence-transformer '{MODEL_NAME}' (downloads on first run)...")
    encoder = SentenceTransformer(MODEL_NAME)

    print("Embedding narratives...")
    X = embed_narratives(df["narrative"], encoder)
    y = df["category"].to_numpy(dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== Classification report (test set) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== Confusion matrix ===")
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    header = "".join(f"{l[:10]:>12}" for l in labels)
    print(f"{'':>22}{header}")
    for label, row in zip(labels, cm):
        row_str = "".join(f"{v:>12}" for v in row)
        print(f"{label:>22}{row_str}")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "encoder_name": MODEL_NAME}, model_out)
    print(f"\nSaved trained classifier to {model_out}")

    # Cache full-dataset embeddings for similarity_search.py / app.py so they
    # don't need to re-embed everything on every run.
    embeddings_out = Path(args.embeddings_out)
    embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    full_embeddings = embed_narratives(df["narrative"], encoder)
    np.savez(
        embeddings_out,
        embeddings=full_embeddings,
        report_id=df["report_id"].to_numpy(),
        category=df["category"].to_numpy(dtype=object),
        narrative=df["narrative"].to_numpy(dtype=object),
    )
    print(f"Cached {len(df)} narrative embeddings to {embeddings_out}")


if __name__ == "__main__":
    main()
