import requests

urls = [
    'https://books.toscrape.com/',
    'https://books.toscrape.com/index.html',
    'https://books.toscrape.com/catalogue/',
    'https://books.toscrape.com/catalogue/page-1.html',
    'https://books.toscrape.com/catalogue/page-1/index.html',
    'https://books.toscrape.com/catalogue/page-1',
]

for url in urls:
    try:
        r = requests.get(url, timeout=15)
        print('URL:', url)
        print('STATUS:', r.status_code)
        print('FINAL:', r.url)
        print(r.text[:220].replace('\n', ' '))
        print('---')
    except Exception as e:
        print('URL:', url)
        print('ERROR:', type(e).__name__, e)
        print('---')
