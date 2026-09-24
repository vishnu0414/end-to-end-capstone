from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "books.db"
OUTPUT_PATH = BASE_DIR / "outputs" / "sql_results.txt"

QUERIES = {
    "select_where": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4;
    """,
    "order_by": """
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC;
    """,
    "limit": """
        SELECT title, price_gbp, rating
        FROM books
        ORDER BY rating DESC
        LIMIT 10;
    """,
    "distinct": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "between": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp;
    """,
    "join": """
        SELECT
            c.category_name,
            b.title,
            b.rating,
            b.price_gbp,
            b.price_inr,
            b.in_stock
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_inr DESC
        LIMIT 10;
    """,
}


def run_queries():
    """Execute all SQL queries and save a text transcript."""
    connection = sqlite3.connect(DB_PATH)
    results = {}

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for name, query in QUERIES.items():
            print(f"\n{'=' * 60}")
            print(name.upper())
            print('=' * 60)
            print(query.strip())

            df = pd.read_sql(query, connection)
            print(df.to_string(index=False))

            output.write(f"\n{'=' * 60}\n")
            output.write(f"{name.upper()}\n")
            output.write(f"{'=' * 60}\n")
            output.write(query.strip() + "\n\n")
            output.write(df.to_string(index=False))
            output.write("\n")

            results[name] = df

    connection.close()
    return results
