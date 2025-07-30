import os
import json

def load_few_shot_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def format_prompt(few_shot, premise, hypothesis):
    return (
        f"{few_shot}\n\n"
        f"instruction: Determine the logical relationship between a premise and a hypothesis. "
        f"Choose from: entailment, contradiction, or neutral.\n"
        f"premise: {premise}\n"
        f"hypothesis: {hypothesis}\n"
        f"Let's think step by step.\n"
        f"label:"
    )

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(data)} prompts to {path}")

def generate_prompts():
    few_shot_path = "prompts/nli_cot.txt"
    data_path = "data/sample_dataset/nli_100.jsonl"
    output_path = "data/generated_prompts/nli_cot_prompts.jsonl"
    few_shot = load_few_shot_examples(few_shot_path)
    samples = load_samples(data_path)
    results = []
    for sample in samples:
        prompt = format_prompt(few_shot, sample["premise"], sample["hypothesis"])
        results.append({
            "prompt": prompt,
            "answer": sample["label"]
        })
    save_jsonl(results, output_path)

if __name__ == "__main__":
    generate_prompts()
