from datasets import load_dataset
for split in ("train", "validation"):
    dataset = load_dataset("roneneldan/TinyStories", split=split, cache_dir="data/hf_cache")
    print(f"{split}: {len(dataset)} examples cached")
