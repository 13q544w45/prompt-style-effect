import os
import json

def load_examples(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    example_blocks = []
    current_block = []
    for line in lines:
        if line.strip() == "":
            continue
        current_block.append(line)
        if len(current_block) == 2:
            example_blocks.append("\n".join(current_block))
            current_block = []
    return "\n\n".join(example_blocks)

def load_samples(input_file):
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
    return samples

def save_to_jsonl(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(data)} prompts to {output_path}")

def generate_prompts():
    INPUT_FILE = "data/sample_dataset/translation_100.jsonl"
    PROMPT_FILE = "prompts/translation_instruction.txt"
    OUTPUT_FILE = "data/generated_prompts/translation_instruction_prompts.jsonl"
    prefix = load_examples(PROMPT_FILE)
    queries = load_samples(INPUT_FILE)
    prompts = []
    for sample in queries:
        query_text = sample["english"]
        full_prompt = f"{prefix}\n\nenglish: {query_text}\ngerman:"
        prompts.append({
            "prompt": full_prompt,
            "answer": sample["german"],
        })
    save_to_jsonl(prompts, OUTPUT_FILE)

if __name__ == "__main__":
    generate_prompts()
