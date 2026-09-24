# Zepto end-to-end AI/ML capstone

This repository contains three connected modules:

- `data_pipeline`: raw scrape -> clean -> convert -> store -> query
- `analytics`: Titanic EDA + modeling + saved end-to-end pipeline
- `support_assistant`: grounded Zepto policy assistant with local embeddings and a FastAPI API

## Submission repository

This project is submitted as one public repository:

https://github.com/vishnu0414/end-to-end-capstone

The repository contains the three module folders at its root and uses one consolidated root `requirements.txt`.

## Project setup

A single root `requirements.txt` is used for the whole repository. Install everything with:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Module 1: Data Pipeline

The data pipeline is implemented in `data_pipeline/` and follows the required fixed-rate conversion:

- `1 GBP = 105.50 INR`
- price conversion is performed in the `cleaner.py` logic and stored into `price_inr`

Run it with:

```bash
python data_pipeline/main.py
```

This pipeline scrapes the first five catalog pages from the public Books To Scrape site, cleans the records, stores them in a normalized SQLite database, and executes multiple SQL queries plus pandas validation.

## Module 2: Analytics Pipeline

The analytics module lives in `analytics/` and includes:

- a one-time dataset load from `sns.load_dataset('titanic')`
- offline fallback saved to `analytics/titanic.csv`
- missing-value handling and profiling
- EDA charts and written interpretation
- stratified modeling with preprocessing, evaluation, tuning, and imbalance comparison
- a saved end-to-end sklearn pipeline in `analytics/titanic_full_pipeline.joblib`

Run it with:

```bash
python analytics/titanic_pipeline.py
```

## Module 3: Support Assistant

The support assistant is a lightweight grounded assistant that answers Zepto policy questions using the eight local policy documents.

Run the browser-first chatbot from the repository root with:

```bash
python -m uvicorn support_assistant.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/` in a browser. The visible product is the chat interface; its internal `POST /ask` route is retained for the FastAPI contract and automated grading. Swagger and ReDoc are disabled.

The default `MOCK_LLM=1` path is fully offline and requires no API key. The chat routes policy questions through retrieval and returns a deterministic answer grounded in the top ChromaDB result; unrelated questions receive the fixed general-question response.

```bash
# Optional API-level verification of the same chat flow:
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the delivery time?\"}"
```

The assistant ingests the eight files in `support_assistant/docs/`, embeds one chunk per document with `all-MiniLM-L6-v2`, stores vectors in the `zepto_policies` ChromaDB collection, and routes requests through the three-node LangGraph. Only intent classification and answer generation branch to a real LLM when `MOCK_LLM=0`; retrieval always uses local embeddings and ChromaDB. The required default performs no LLM network call.

## Design decisions

- The Module 1 scraper keeps the public practice dataset scope stable and uses a normalized two-table SQLite model.
- The Module 2 pipeline keeps a single source-of-truth dataset and treats downstream modeling as a continuation of that same cleaned data.
- The Module 2 final saved artifact includes the preprocessor and estimator together so raw input can be passed directly to the pipeline.
- The Module 3 mock mode is deterministic and schema validated, while the optional real-LLM path uses the same grounded prompt and retries invalid JSON responses.
- The Module 3 user experience is a browser chat at `/`; the rubric-required `POST /ask` route is an internal JSON boundary rather than the visible interface.

## Submission checklist

- All required source code and interpretations are stored as `.py`, `.ipynb`, or Markdown files inside this repository.
- Generated chart PNGs are supporting artifacts only; written interpretations remain in `analytics/README.md` and the notebooks.
- Git history includes the `feature/capstone-merge` branch with two commits merged into `main`.
- The required baseline uses no paid services: scraping, Titanic fallback data, local embeddings, ChromaDB, and mock LLM responses run locally or through the stated free public data source.
