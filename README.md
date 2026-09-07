# Quality Report Root-Cause Classifier

Automatically triage free-text quality/failure reports into root-cause categories, and surface similar historical reports for a given new one — the way an experienced quality engineer would when doing initial intake on a nonconformance report (NCR) or complaint.

## Why this project

In manufacturing and product quality work, a huge amount of signal lives in unstructured text: NCRs, CAPA descriptions, customer complaints, field failure reports. New reports usually get triaged manually — someone reads the narrative and assigns it to a category (electrical, mechanical, process, supplier, etc.) so it routes to the right team and rolls up into the right metrics. That triage step is slow, inconsistent between reviewers, and doesn't scale as report volume grows.

This project treats that triage step as a text classification problem, and goes one step further: alongside predicting a category, it retrieves the most similar *past* reports for a new one, which is closer to what a quality engineer actually wants when investigating a new failure ("has this happened before, and what did we find out last time?").

Real proprietary NCR/CAPA data isn't public, so this project uses the [NHTSA vehicle complaints database](https://www.nhtsa.gov/nhtsa-datasets-and-apis) as a stand-in — it's free, real-world, and structurally very close to a quality report: a free-text narrative (`Summary`) paired with a category label (`Component`, e.g. `ENGINE`, `ELECTRICAL SYSTEM`, `BRAKES`). The modeling approach transfers directly to real quality/CAPA data if you have access to it later — you'd just point `fetch_data.py`'s output at your own export instead.

## Approach

The project is built in three stages, each a step up in sophistication, so the progression itself is something you can talk through in an interview:

1. **Baseline — TF-IDF + Logistic Regression** (`src/baseline_model.py`). Simple, fast, interpretable. This is the model to reach for first on any text classification problem, and it's a good one to be able to explain from scratch (what TF-IDF actually weights, why logistic regression's coefficients are readable per class).
2. **Embeddings — Sentence-Transformer + classifier** (`src/embedding_model.py`). Replaces bag-of-words features with dense semantic embeddings, which should handle paraphrasing and vocabulary variation better than TF-IDF. The point of building both is to *compare them honestly* — embeddings aren't automatically better on small/narrow-vocabulary datasets, and knowing when the simple model is good enough is a real skill.
3. **Unsupervised pattern discovery — clustering** (`src/clustering.py`). Clusters report embeddings (KMeans) and surfaces the top distinguishing terms per cluster. This is the part that's actually novel value for a quality team: finding emerging failure patterns that don't map cleanly onto the existing category labels.

On top of all three, `src/similarity_search.py` does nearest-neighbor lookup over embeddings so a new report can be matched against the most similar historical ones — and `app/app.py` wraps that in a small interactive demo.

## Project structure

```
quality-report-classifier/
├── data/
│   ├── sample_reports.csv     # bundled synthetic sample so the pipeline runs immediately
│   └── README.md              # how to pull real NHTSA data instead
├── src/
│   ├── fetch_data.py          # pulls real complaint narratives from the NHTSA API
│   ├── preprocess.py          # text cleaning shared by all models
│   ├── baseline_model.py      # TF-IDF + Logistic Regression
│   ├── embedding_model.py     # sentence-embeddings + classifier
│   ├── clustering.py          # unsupervised pattern discovery
│   └── similarity_search.py   # nearest-neighbor retrieval over embeddings
├── app/
│   └── app.py                 # Streamlit demo
├── tests/
│   └── test_pipeline.py
├── models/                    # trained models get saved here (gitignored)
└── requirements.txt
```

## Getting started

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the baseline classifier on the bundled sample data (works offline, no setup needed):

```bash
python src/baseline_model.py
```

Run the embedding-based classifier (downloads a small pretrained model on first run):

```bash
python src/embedding_model.py
```

Discover clusters/topics in the reports:

```bash
python src/clustering.py
```

Launch the interactive demo:

```bash
streamlit run app/app.py
```

## Using real data instead of the bundled sample

`data/sample_reports.csv` is a **synthetic** dataset generated from templates (see `data/README.md` for exactly how) — it exists so you can run the whole pipeline immediately without any network calls or long downloads. It's good enough to prove the code works, but the writing style is repetitive in a way real complaint narratives aren't, so metrics on it will look better than they will on real data. That's expected, and worth saying out loud in an interview rather than hiding it.

To pull real complaint narratives from NHTSA (free, no API key required):

```bash
python src/fetch_data.py --make honda --model accord --model-year 2015 2016 2017 --out data/nhtsa_real.csv
```

See `data/README.md` for more on the API, how to pull a broader/more diverse set of vehicles, and known quirks in the field names. **Note:** this sandbox's network access is restricted and can't reach `api.nhtsa.gov` directly, so `fetch_data.py` is written and documented but hasn't been run end-to-end from here — run it from your own machine, and sanity-check the first few rows of output against the raw API response before trusting it for training.

## Evaluation

Both `baseline_model.py` and `embedding_model.py` print precision/recall/F1 per class and a confusion matrix, not just accuracy. That matters here because the categories are imbalanced (some failure types are just rarer), and accuracy alone would hide a model that's great on the common categories and useless on the rare ones — which is exactly backwards from what a quality team needs, since rare/novel failure modes are often the ones worth catching.

## What's actually been verified vs. what hasn't

This scaffold was built in a sandboxed environment with restricted network access, so it's worth being precise about what was actually run versus just written carefully:

- **Verified end-to-end:** `data/generate_sample_data.py`, `src/preprocess.py`, and `src/baseline_model.py` were all run successfully against the bundled sample data (see `tests/test_pipeline.py`, 8/8 passing). The baseline classifier trains and evaluates correctly.
- **Written and syntax-checked, but not run end-to-end here:** `src/embedding_model.py`, `src/clustering.py`, `src/similarity_search.py`, and `app/app.py` all depend on `sentence-transformers` (and its `torch` dependency), which this sandbox couldn't install due to restricted egress to `download.pytorch.org`. `src/fetch_data.py` similarly couldn't be run here because `api.nhtsa.gov` isn't reachable from this sandbox. All of these compile cleanly and were written against documented library APIs, but **run them yourself and fix anything that comes up** before treating them as done — that's normal, expected work for a real project, not a sign something is wrong with the scaffold.

That baseline result is also worth reading carefully: it came back at 100% accuracy on the held-out test set, which is a red flag, not a win — it means the synthetic template data is too easy (too little vocabulary variation between reports in the same category) for this to be a meaningful benchmark. Don't put that number in a portfolio; it's exactly the kind of "the model is perfect" result that should make you suspicious of the data rather than proud of the model. Regenerate with real NHTSA data (`src/fetch_data.py`) before evaluating anything for real.

## Honest limitations

- The bundled sample data is synthetic and will overstate how well the approach works on real reports.
- NHTSA's `Component` categories are automotive-specific and coarser than a real internal CAPA taxonomy would be; treat this as a proof of concept for the *method*, not a drop-in tool.
- No hyperparameter tuning has been done — both models use reasonable defaults, deliberately, so the comparison between them isn't confounded by one being more tuned than the other.
- Clustering quality (`clustering.py`) hasn't been validated against ground truth beyond eyeballing the top terms per cluster; silhouette score is printed but should be read as a rough signal, not proof.

## Possible next steps

- Swap in real CAPA/NCR export data (see `data/README.md`) and re-run all three stages.
- Add a Pareto-style dashboard of predicted categories over time — the classic quality-engineering view of "what's driving defects this month."
- Try a small fine-tuned transformer classifier and compare against the embedding + classifier approach.
- Active-learning loop: route low-confidence predictions to a human reviewer, and use their corrections to retrain.
