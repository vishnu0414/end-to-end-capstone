from data_pipeline.scraper import scrape_books

books = scrape_books(num_pages=5)
print('COUNT', len(books))
print('CATEGORY_COUNT', len({book['category'] for book in books}))
print('FIRST', books[0])
