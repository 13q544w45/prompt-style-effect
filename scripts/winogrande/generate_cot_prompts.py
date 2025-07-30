import os
import json

def load_few_shot_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def generate_prompts():
    few_shot_prefix = load_few_shot_examples("prompts/winogrande_cot.txt")
    samples = load_samples("data/sample_dataset/winogrande_100.jsonl")
    results = []
    for sample in samples:
        prompt = (
            f"{few_shot_prefix}\n\n"
            f"instruction: Choose which option the pronoun refers to.\n"
            f"sentence: {sample['sentence']}\n"
            f"option1: {sample['option1']}\n"
            f"option2: {sample['option2']}\n"
            f"Let's think step by step.\n"
            f"answer:"
        )
        results.append({
            "prompt": prompt,
            "answer": sample["answer"]
        })
    os.makedirs("data/generated_prompts", exist_ok=True)
    output_path = "data/generated_prompts/winogrande_cot_prompts.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(results)} prompts to {output_path}")

if __name__ == "__main__":
    generate_prompts()
