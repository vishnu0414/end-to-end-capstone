from __future__ import annotations

from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = "https://books.toscrape.com/catalogue/page-{}.html"


def build_detail_url(href: str) -> str:
    """Build the full URL for a book detail page from a catalogue-relative link."""
    return urljoin(urljoin(BASE_URL, "catalogue/"), href)


def fetch_page(url: str):
    """Fetch and parse an HTML page."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def scrape_book_category(url: str) -> str:
    """Open a book detail page and extract the category from breadcrumbs."""
    soup = fetch_page(url)
    breadcrumb_links = soup.select(".breadcrumb li a")

    if len(breadcrumb_links) >= 3:
        return breadcrumb_links[2].get_text(strip=True)

    breadcrumb_items = [link.get_text(strip=True) for link in breadcrumb_links]
    if len(breadcrumb_items) >= 2:
        return breadcrumb_items[-1]

    return "Unknown"


def scrape_book_page(url: str):
    """Scrape all books from one catalogue page."""
    soup = fetch_page(url)
    books = []

    for article in soup.select("article.product_pod"):
        title_tag = article.select_one("h3 a")
        price_tag = article.select_one(".price_color")
        availability_tag = article.select_one(".availability")
        rating_tag = article.select_one(".star-rating")

        if not all([title_tag, price_tag, availability_tag, rating_tag]):
            continue

        rating_classes = rating_tag.get("class", [])
        rating_text = next(
            (
                item
                for item in rating_classes
                if item in ["One", "Two", "Three", "Four", "Five"]
            ),
            None,
        )

        detail_url = build_detail_url(title_tag.get("href"))
        category = scrape_book_category(detail_url)

        books.append(
            {
                "title": title_tag.get("title") or title_tag.get_text(strip=True),
                "price": price_tag.get_text(strip=True),
                "star_rating": rating_text,
                "availability": availability_tag.get_text(" ", strip=True),
                "category": category,
            }
        )

    return books


def scrape_books(num_pages: int = 5):
    """Scrape books from the first N catalogue pages."""
    all_books = []

    for page_number in range(1, num_pages + 1):
        url = CATALOGUE_URL.format(page_number)
        print(f"Scraping page {page_number}: {url}")

        page_books = scrape_book_page(url)
        all_books.extend(page_books)
        print(f"  Books collected: {len(page_books)}")

    print(f"\nTotal books scraped: {len(all_books)}")
    return all_books


if __name__ == "__main__":
    books = scrape_books(num_pages=5)
    print(f"Sample: {books[:2]}")
