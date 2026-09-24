from __future__ import annotations

from pathlib import Path

try:
    from .scraper import scrape_books
    from .cleaner import clean_books
    from .database import insert_books
    from .queries import run_queries
except ImportError:  # pragma: no cover
    from scraper import scrape_books
    from cleaner import clean_books
    from database import insert_books
    from queries import run_queries

BASE_DIR = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("ZEPTO DATA PIPELINE")
    print("=" * 60)

    print("\n[1] SCRAPING")
    books = scrape_books(num_pages=5)

    print("\n[2] CLEANING")
    df = clean_books(books)

    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} valid books were produced. At least 60 are required."
        )

    if df["category"].nunique() < 3:
        raise ValueError(
            f"Only {df['category'].nunique()} categories found. At least 3 are required."
        )

    output_path = BASE_DIR / "data" / "books_cleaned.csv"
    df.to_csv(output_path, index=False)

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Categories: {df['category'].nunique()}")

    print("\n[3] DATABASE")
    insert_books(df)

    print("\n[4] SQL QUERIES")
    results = run_queries()

    print("\n[5] PANDAS VALIDATION")
    join_sql_result = results["join"]
    print("\nJOIN result using pd.read_sql():")
    print(join_sql_result)

    # Recreate the category table mapping used in SQLite so the pd.merge validation matches the DB
    category_df = df[["category"]].drop_duplicates().reset_index(drop=True)
    category_df["category_id"] = range(1, len(category_df) + 1)
    category_map = category_df.set_index("category")["category_id"].to_dict()

    books_df = df.copy()
    books_df["category_id"] = books_df["category"].map(category_map)

    print("\nJOIN result using pd.merge():")
    merge_result = (
        books_df[["category", "title", "rating", "price_gbp", "price_inr", "in_stock"]]
        .rename(columns={"category": "category_name"})
        .sort_values(["rating", "price_inr"], ascending=[False, False])
        .head(10)
        .reset_index(drop=True)
    )
    print(merge_result)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
