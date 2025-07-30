import json
import random
from datasets import load_dataset

def is_valid_pair(example):
    en = example.get("translation", {}).get("en", "").strip()
    de = example.get("translation", {}).get("de", "").strip()
    if not en or not de:
        return False
    word_count = len(en.split())
    return 5 <= word_count <= 20

def extract_translation_samples(n=100, seed=42):
    dataset = load_dataset("wmt16", "de-en", split="test")
    valid_samples = [ex for ex in dataset if is_valid_pair(ex)]
    random.seed(seed)
    sampled = random.sample(valid_samples, n)
    results = []
    for example in sampled:
        en = example["translation"]["en"].strip()
        de = example["translation"]["de"].strip()
        results.append({
            "english": en,
            "german": de,
            "source": "WMT16"
        })
    return results

def save_to_jsonl(data, output_path="data/translation_100.jsonl"):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    samples = extract_translation_samples(n=100)
    save_to_jsonl(samples)
