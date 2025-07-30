import os
import json

def load_cot_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_query_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_to_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(data)} prompts to {path}")

def generate_prompts():
    cot_examples_path = "prompts/translation_cot.txt"
    query_file = "data/sample_dataset/translation_100.jsonl"
    output_path = "data/generated_prompts/translation_cot_prompts.jsonl"
    prefix = load_cot_examples(cot_examples_path)
    queries = load_query_samples(query_file)
    prompts = []
    for sample in queries:
        query = sample["english"]
        answer = sample["german"]
        full_prompt = f"{prefix}\n\nenglish: {query}\ngerman:"
        prompts.append({
            "prompt": full_prompt,
            "answer": answer
        })
    save_to_jsonl(prompts, output_path)

if __name__ == "__main__":
    generate_prompts()
