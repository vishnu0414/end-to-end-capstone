from __future__ import annotations

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "books.db"


def create_database():
    """Create the normalized SQLite schema."""
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    connection.commit()
    return connection


def insert_books(df):
    """Insert cleaned book data into the normalized database."""
    connection = create_database()
    cursor = connection.cursor()

    cursor.execute("DELETE FROM books")
    cursor.execute("DELETE FROM categories")

    categories = df["category"].drop_duplicates().tolist()
    for category in categories:
        cursor.execute(
            "INSERT INTO categories (category_name) VALUES (?)",
            (category,),
        )

    category_map = {
        row[1]: row[0]
        for row in cursor.execute(
            "SELECT category_id, category_name FROM categories"
        ).fetchall()
    }

    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                int(row["in_stock"]),
                category_map[row["category"]],
            ),
        )

    connection.commit()
    print(f"Inserted {len(df)} books.")
    print(f"Inserted {len(categories)} categories.")
    return connection
