"""Load google/frames-benchmark, fetch articles via wikipedia library,
create knowledge bases by reasoning type category.

Usage:
    pip install -e ".[benchmark]"
    python benchmark/load_frames.py
"""
import ast
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote
from collections import defaultdict

import wikipedia
from datasets import load_dataset

from rag_mcp.knowledge_manager import KnowledgeManager


MAIN_CATEGORIES = {
    "Numerical reasoning": "numerical",
    "Temporal reasoning": "temporal",
    "Multiple constraints": "multi_constraint",
    "Tabular reasoning": "tabular",
    "Post processing": "postprocessing",
}


def classify_reasoning(reasoning_str: str) -> str:
    if not reasoning_str:
        return "other"
    for sub_type in reasoning_str.split(" | "):
        sub_type = sub_type.strip()
        if sub_type in MAIN_CATEGORIES:
            return MAIN_CATEGORIES[sub_type]
    return "other"


def parse_wiki_links(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return []


def url_to_title(url: str) -> str | None:
    m = re.search(r"/wiki/(.+)", url)
    if not m:
        return None
    title = m.group(1)
    title = re.sub(r"#.*", "", title)
    title = title.replace("_", " ")
    title = unquote(title)
    return title


def fetch_wikipedia_text(title: str) -> str:
    wikipedia.set_lang("en")
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.content
    except (wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError):
        return ""
    except Exception:
        return ""


def main():
    print("Loading google/frames-benchmark...")
    ds = load_dataset("google/frames-benchmark")
    rows = ds["test"]

    print("Analyzing queries...")
    type_urls: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        category = classify_reasoning(row["reasoning_types"])
        urls = parse_wiki_links(row.get("wiki_links"))
        for url in urls:
            if isinstance(url, str) and url.startswith("http"):
                type_urls[category].add(url)

    type_urls = {k: v for k, v in type_urls.items() if k != "other"}
    total = sum(len(v) for v in type_urls.values())
    print(f"  {total} unique URLs across {len(type_urls)} categories")
    for cat, urls in sorted(type_urls.items()):
        print(f"    {cat}: {len(urls)} URLs")

    MAX_PER_TYPE = 20
    km = KnowledgeManager()

    for category, urls in sorted(type_urls.items()):
        urls_list = sorted(urls)[:MAX_PER_TYPE]
        print(f"\nCreating KB: {category} ({len(urls_list)} articles)")
        km.add_database(category, f"Frames benchmark - {category}")

        for idx, url in enumerate(urls_list, 1):
            title = url.split("/")[-1][:40]
            print(f"  [{idx}/{len(urls_list)}] {title:40s}", end=" ")

            page_title = url_to_title(url)
            if not page_title:
                print("skip (bad url)")
                continue

            text = fetch_wikipedia_text(page_title)
            if not text or len(text.strip()) < 50:
                print("skip (empty)")
                continue

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(text)
                tmp_path = f.name

            try:
                n = km.add_file(category, tmp_path)
                print(f"{n} chunks")
            except Exception as e:
                print(f"FAIL: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    print("\nDone! Knowledge bases:")
    for kb in km.list():
        desc = kb.get("desc", "")
        if "benchmark" in desc:
            print(f"  {kb['name']}: {desc}")


if __name__ == "__main__":
    main()
