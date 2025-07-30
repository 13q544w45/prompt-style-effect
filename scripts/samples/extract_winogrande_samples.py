import os
import json
import random
from datasets import load_dataset

def extract_winogrande_samples(n=100, seed=42):
    dataset = load_dataset("winogrande", "winogrande_xl", split="train")
    random.seed(seed)
    sampled = random.sample(list(dataset), n)
    results = []
    for sample in sampled:
        results.append({
            "sentence": sample["sentence"],
            "option1": sample["option1"],
            "option2": sample["option2"],
            "answer": sample["answer"],
            "source": "winogrande"
        })
    return results

def save_to_jsonl(data, output_path="data/winogrande_100.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(data)} samples to {output_path}")

if __name__ == "__main__":
    samples = extract_winogrande_samples(n=100)
    save_to_jsonl(samples)

