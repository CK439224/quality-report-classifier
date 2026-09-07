"""
Nearest-neighbor retrieval over cached report embeddings: given a new report
narrative, find the most similar historical reports.

This mirrors the actual instinct behind a lot of root-cause investigation --
"has this happened before, and what did we find out last time?" -- more
directly than a category prediction alone does. It's also the piece app.py
uses for the interactive demo.

Requires models/embeddings.npz, created by running:
    python src/embedding_model.py

Usage (CLI):
    python src/similarity_search.py --query "engine stalled at highway speed with no warning lights"
"""
import argparse
from pathlib import Path

import numpy as np


class SimilaritySearcher:
    def __init__(self, embeddings_cache: str = "models/embeddings.npz", model_name: str = "all-MiniLM-L6-v2"):
        cache = Path(embeddings_cache)
        if not cache.exists():
            raise FileNotFoundError(
                f"No embeddings cache at {embeddings_cache}. Run `python src/embedding_model.py` first "
                f"to generate it."
            )
        data = np.load(cache, allow_pickle=True)
        self.embeddings = data["embeddings"]  # already L2-normalized by embedding_model.py
        self.report_id = data["report_id"]
        self.category = data["category"]
        self.narrative = data["narrative"]
        self._model_name = model_name
        self._encoder = None  # lazy-loaded, since importing sentence_transformers is slow

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def search(self, query: str, top_k: int = 5):
        query_vec = self.encoder.encode([query], normalize_embeddings=True)[0]
        # Embeddings are normalized, so dot product == cosine similarity.
        scores = self.embeddings @ query_vec
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "report_id": int(self.report_id[i]),
                "category": str(self.category[i]),
                "narrative": str(self.narrative[i]),
                "similarity": float(scores[i]),
            }
            for i in top_idx
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=str, required=True, help="A new report narrative to find similar past reports for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embeddings-cache", type=str, default="models/embeddings.npz")
    args = parser.parse_args()

    searcher = SimilaritySearcher(args.embeddings_cache)
    results = searcher.search(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}\n")
    print(f"Top {len(results)} similar historical reports:\n")
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. [{r['category']}] similarity={r['similarity']:.3f}  (report #{r['report_id']})")
        print(f"   {r['narrative']}\n")


if __name__ == "__main__":
    main()
