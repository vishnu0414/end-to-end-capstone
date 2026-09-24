# Data Pipeline Project

This project scrapes the first five pages from the `books.toscrape.com` catalogue, cleans the book records, stores them in a normalized SQLite database, and validates the final results with SQL and pandas.

## Pipeline flow

1. Scrape catalogue pages for books.
2. Clean fields: title, price, rating, availability, category.
3. Convert GBP prices to INR at a fixed rate of 105.50.
4. Insert records into a SQLite schema with `categories` and `books` tables.
5. Run multiple SQL queries including a JOIN.
6. Validate results using `pd.read_sql()` and `pd.merge()`.

## Folder structure

- `scraper.py` fetches catalogue data and category detail pages.
- `cleaner.py` normalizes and filters raw values.
- `database.py` creates the SQLite schema and inserts cleaned rows.
- `queries.py` runs six SQL queries and writes outputs.
- `main.py` runs the full pipeline.
- `data/` stores the cleaned CSV output.
- `database/` stores the SQLite database.
- `outputs/` stores query results.

## Notes

- The scraper intentionally uses the first five catalogue pages to guarantee the required minimum number of books while keeping the pipeline reliable and simple.
- Rows with malformed values are dropped rather than guessed, which preserves data integrity.
- The database uses a normalized two-table design: `categories` and `books`.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python data_pipeline/main.py
```
