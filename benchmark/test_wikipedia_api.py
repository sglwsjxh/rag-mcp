"""Test Wikipedia API with real browser User-Agent."""
import httpx

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 QuarkPC/6.8.7.860",
    "Accept": "application/json,text/html",
}
params = {
    "action": "query",
    "prop": "extracts",
    "explaintext": True,
    "titles": "Python (programming language)",
    "format": "json",
}

r = httpx.get("https://en.wikipedia.org/w/api.php", params=params, headers=headers, timeout=15)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for pid, p in pages.items():
        if pid != "-1":
            text = p.get("extract", "")
            print(f"Got {len(text)} chars")
            print(f"First 150: {text[:150]}")
else:
    print(f"Body: {r.text[:300]}")
