"""Debug: check wiki_links format."""
from datasets import load_dataset
from collections import defaultdict

ds = load_dataset("google/frames-benchmark")
rows = ds["test"]

for i in range(3):
    r = rows[i]
    wl = r.get("wiki_links")
    print(f"Row {i}:")
    print(f"  reasoning_types = {repr(r['reasoning_types'])}")
    print(f"  wiki_links type = {type(wl).__name__}")
    print(f"  wiki_links = {repr(wl)[:250]}")
    print(f"  link_1 = {r['wikipedia_link_1']}")
    print()

type_counts = defaultdict(int)
type_url_examples = defaultdict(list)
for row in rows:
    rt = row["reasoning_types"]
    type_counts[rt] += 1
    if len(type_url_examples[rt]) < 2:
        wl = row.get("wiki_links")
        type_url_examples[rt].append(repr(wl)[:100])

print("Reasoning type distribution:")
for rt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:3d} x {rt}")
    for ex in type_url_examples[rt]:
        print(f"        URL sample: {ex}")
