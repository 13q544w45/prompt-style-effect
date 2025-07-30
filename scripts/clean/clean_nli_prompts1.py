import os
import json

INPUT_PATH = "data/generated_prompts/nli_instruction_prompts.jsonl"
OUTPUT_PATH = "data/model_inputs/nli_instruction_cleaned.jsonl"

def process_file():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    cleaned_data = []
    for item in data:
        cleaned_data.append({
            "prompt": item.get("prompt", ""),
            "label": ""
        })
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in cleaned_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(cleaned_data)} cleaned prompts to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_file()
