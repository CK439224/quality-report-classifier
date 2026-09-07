"""
Unsupervised pattern discovery: cluster report embeddings and surface the
top distinguishing terms per cluster.

This is the part of the project that maps most directly to real value for a
quality team -- finding failure patterns that don't line up neatly with the
existing category taxonomy, rather than just re-predicting labels that
already exist. On this project's data the clusters will mostly recover the
existing categories (expected, since the synthetic/NHTSA categories are
already fairly clean); the interesting real-world use is running this on
UNlabeled or coarsely-labeled data to find structure nobody has named yet.

Uses embeddings cached by embedding_model.py if available (models/embeddings.npz),
otherwise embeds narratives itself.

Usage:
    python src/embedding_model.py          # first, to cache embeddings
    python src/clustering.py
    python src/clustering.py --k 8         # override automatic k selection
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from preprocess import load_reports


def get_embeddings(df: pd.DataFrame, cache_path: str):
    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached embeddings from {cache}")
        data = np.load(cache, allow_pickle=True)
        cached_ids = list(data["report_id"])
        if list(df["report_id"]) == cached_ids:
            return data["embeddings"]
        print("  cached embeddings don't match current data/report_ids -- recomputing")

    print("No usable cache found -- embedding narratives now (this will download the model on first run)...")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return encoder.encode(list(df["narrative"]), show_progress_bar=True, normalize_embeddings=True)


def choose_k(embeddings: np.ndarray, k_range=range(4, 15)) -> int:
    best_k, best_score = None, -1
    for k in k_range:
        if k >= len(embeddings):
            break
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"  k={k:>2}  silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def top_terms_per_cluster(narratives: pd.Series, cluster_labels: np.ndarray, top_n: int = 8):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, stop_words="english")
    tfidf = vectorizer.fit_transform(narratives)
    terms = np.array(vectorizer.get_feature_names_out())

    results = {}
    for cluster_id in sorted(set(cluster_labels)):
        mask = cluster_labels == cluster_id
        cluster_mean = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_idx = cluster_mean.argsort()[::-1][:top_n]
        results[cluster_id] = list(terms[top_idx])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/sample_reports.csv")
    parser.add_argument("--embeddings-cache", type=str, default="models/embeddings.npz")
    parser.add_argument("--k", type=int, default=None, help="Number of clusters; auto-selected via silhouette score if omitted")
    args = parser.parse_args()

    df = load_reports(args.data)
    embeddings = get_embeddings(df, args.embeddings_cache)

    if args.k is None:
        print("\nSearching for a good number of clusters (silhouette score, higher is better)...")
        k = choose_k(embeddings)
        print(f"Selected k={k}")
    else:
        k = args.k

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    df = df.assign(cluster=cluster_labels)

    top_terms = top_terms_per_cluster(df["narrative"], cluster_labels)

    print(f"\n=== {k} clusters discovered ===")
    for cluster_id in sorted(top_terms):
        subset = df[df["cluster"] == cluster_id]
        dominant_category = subset["category"].value_counts().idxmax()
        purity = subset["category"].value_counts().iloc[0] / len(subset)
        print(f"\nCluster {cluster_id}  (n={len(subset)}, dominant label='{dominant_category}' at {purity:.0%} purity)")
        print(f"  top terms: {', '.join(top_terms[cluster_id])}")

    print("\n=== Cluster vs. existing category crosstab ===")
    print(pd.crosstab(df["cluster"], df["category"]))
    print(
        "\nHigh purity per cluster mostly means the clusters are rediscovering the "
        "existing labels -- expected on cleanly-labeled data. Look for clusters that "
        "split a single category into two distinct sub-patterns, or that mix categories "
        "in a way that suggests the original label wasn't quite right; those are the "
        "interesting findings to write up."
    )


if __name__ == "__main__":
    main()
