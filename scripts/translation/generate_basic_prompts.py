import os
import json

def load_few_shot_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def generate_prompts():
    few_shot = load_few_shot_examples("prompts/translation_basic.txt")
    samples = load_samples("data/sample_dataset/translation_100.jsonl")
    results = []
    for sample in samples:
        query = sample["english"]
        answer = sample["german"]
        prompt = f"{few_shot}\n\nenglish: {query}\ngerman:"
        results.append({
            "prompt": prompt,
            "answer": answer
        })
    os.makedirs("data/generated_prompts", exist_ok=True)
    output_path = "data/generated_prompts/translation_basic_prompts.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(results)} prompts to {output_path}")

if __name__ == "__main__":
    generate_prompts()
