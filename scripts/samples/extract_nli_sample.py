import json
import random
from datasets import load_dataset

def label_index_to_text(index):
    label_map = {0: "entailment", 1: "contradiction"}
    return label_map.get(index)

def is_valid_sample(sample):
    if not sample.get("premise") or not sample.get("hypothesis"):
        return False
    if sample.get("label") not in [0, 1]:
        return False
    return True

def extract_rte_samples(n=100, seed=42):
    dataset = load_dataset("super_glue", "rte", split="train")
    valid_samples = [ex for ex in dataset if is_valid_sample(ex)]
    print(f"Valid RTE samples found: {len(valid_samples)}")
    if len(valid_samples) < n:
        print(f"Only {len(valid_samples)} samples available. Sampling all.")
        n = len(valid_samples)
    random.seed(seed)
    sampled = random.sample(valid_samples, n)
    results = []
    for sample in sampled:
        results.append({
            "premise": sample["premise"].strip(),
            "hypothesis": sample["hypothesis"].strip(),
            "label": label_index_to_text(sample["label"]),
        })
    return results

def save_to_jsonl(data, output_path="data/nli_100.jsonl"):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    samples = extract_rte_samples(n=100)
    save_to_jsonl(samples)
