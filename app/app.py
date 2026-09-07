"""
Interactive demo: paste a new quality/failure report, get a predicted
category plus the most similar historical reports.

Run with:
    streamlit run app/app.py

Requires the models to be trained first:
    python src/baseline_model.py
    python src/embedding_model.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable regardless of the working directory this is launched from.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import joblib  # noqa: E402
from preprocess import clean_text  # noqa: E402
from similarity_search import SimilaritySearcher  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline.joblib"
EMBEDDINGS_CACHE_PATH = PROJECT_ROOT / "models" / "embeddings.npz"

st.set_page_config(page_title="Quality Report Triage", page_icon="🔧", layout="centered")


@st.cache_resource
def load_baseline_model():
    if not BASELINE_MODEL_PATH.exists():
        return None
    return joblib.load(BASELINE_MODEL_PATH)


@st.cache_resource
def load_searcher():
    if not EMBEDDINGS_CACHE_PATH.exists():
        return None
    return SimilaritySearcher(str(EMBEDDINGS_CACHE_PATH))


def main():
    st.title("🔧 Quality Report Triage")
    st.caption(
        "Paste a new quality / failure report narrative to get a predicted category and the "
        "most similar historical reports — the way a quality engineer would triage an incoming NCR."
    )

    pipeline = load_baseline_model()
    searcher = load_searcher()

    if pipeline is None or searcher is None:
        st.warning(
            "Models aren't trained yet. From the project root, run:\n\n"
            "```\npython src/baseline_model.py\npython src/embedding_model.py\n```\n\n"
            "then restart this app."
        )
        return

    example = (
        "Engine stalled without warning at highway speed. Check engine light came on right "
        "before it happened. This is the second time this has occurred in the last month."
    )
    query = st.text_area("Report narrative", value="", placeholder=example, height=140)

    top_k = st.slider("How many similar historical reports to show", min_value=3, max_value=10, value=5)

    if st.button("Triage this report", type="primary") and query.strip():
        with st.spinner("Classifying..."):
            cleaned = clean_text(query)
            predicted_category = pipeline.predict([cleaned])[0]

            probs = pipeline.predict_proba([cleaned])[0]
            classes = pipeline.classes_
            prob_df = (
                pd.DataFrame({"category": classes, "confidence": probs})
                .sort_values("confidence", ascending=False)
                .reset_index(drop=True)
            )

        st.subheader(f"Predicted category: `{predicted_category}`")
        st.caption("From the TF-IDF + Logistic Regression baseline model.")
        st.bar_chart(prob_df.set_index("category")["confidence"])

        with st.spinner("Finding similar historical reports..."):
            similar = searcher.search(query, top_k=top_k)

        st.subheader("Most similar historical reports")
        st.caption("From the sentence-embedding nearest-neighbor search — this is the 'has this happened before?' view.")
        for r in similar:
            with st.container(border=True):
                st.markdown(f"**[{r['category']}]** &nbsp; similarity: `{r['similarity']:.3f}` &nbsp; (report #{r['report_id']})")
                st.write(r["narrative"])

    st.divider()
    with st.expander("About this demo"):
        st.markdown(
            "- The category prediction comes from `src/baseline_model.py` (TF-IDF + Logistic Regression).\n"
            "- The similar-reports list comes from `src/similarity_search.py`, which does cosine "
            "similarity search over sentence embeddings cached by `src/embedding_model.py`.\n"
            "- The underlying data is either the bundled synthetic sample (`data/sample_reports.csv`) "
            "or real NHTSA complaint data pulled with `src/fetch_data.py` — check which one is currently "
            "trained by looking at what you passed to `--data` when you last ran the training scripts."
        )


if __name__ == "__main__":
    main()
