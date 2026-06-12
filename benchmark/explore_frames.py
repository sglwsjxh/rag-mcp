"""Explore the google/frames-benchmark dataset structure."""
from datasets import load_dataset, get_dataset_config_names

# Check available configs
print("Configs:", get_dataset_config_names("google/frames-benchmark"))
print()

# Load default
ds = load_dataset("google/frames-benchmark")
print("Splits:", list(ds.keys()))
for split in ds:
    row = ds[split][0]
    print(f"\n{split}: {len(ds[split])} rows")
    print(f"  Columns ({len(list(ds[split].features.keys()))}):")
    for col in ds[split].features.keys():
        val = row.get(col, "N/A")
        if isinstance(val, str) and len(val) > 80:
            val = val[:80] + "..."
        print(f"    - {col}: {type(val).__name__} = {val}")
    
    if "reasoning_types" in ds[split].features:
        types = set()
        for i in range(min(len(ds[split]), 100)):
            t = ds[split][i]["reasoning_types"]
            if isinstance(t, list):
                types.update(t)
            else:
                types.add(t)
        print(f"\n  Reasoning types (sample 100): {types}")
