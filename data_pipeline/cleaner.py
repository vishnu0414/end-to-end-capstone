from __future__ import annotations

import pandas as pd

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

GBP_TO_INR = 105.50


def clean_price(value):
    """Convert a price such as £51.77 into float 51.77."""
    try:
        cleaned = str(value)
        cleaned = cleaned.replace("Â£", "").replace("\xa3", "").replace("£", "")
        return float(cleaned.strip())
    except (TypeError, ValueError):
        return None


def clean_rating(value):
    """Convert textual rating into integer 1-5."""
    return RATING_MAP.get(str(value).strip())


def clean_stock(value):
    """Convert availability text into boolean."""
    if pd.isna(value):
        return None

    text = str(value).lower()

    if "in stock" in text:
        return True

    if "out of stock" in text:
        return False

    return None


def clean_books(books):
    """Clean and transform scraped book records."""
    df = pd.DataFrame(books)

    if df.empty:
        return df

    df["price_gbp"] = df["price"].apply(clean_price)
    df["rating"] = df["star_rating"].apply(clean_rating)
    df["in_stock"] = df["availability"].apply(clean_stock)
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    before = len(df)
    df = df.dropna(subset=["title", "price_gbp", "rating", "in_stock", "category"]).copy()
    dropped = before - len(df)

    print(f"Rows before cleaning: {before}")
    print(f"Rows dropped: {dropped}")
    print(f"Rows after cleaning: {len(df)}")

    df["price_gbp"] = df["price_gbp"].astype(float)
    df["price_inr"] = df["price_inr"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    df = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].copy()
    return df
