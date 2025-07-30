import os
import json

def load_few_shot_prompt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_test_samples(filepath):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
    return samples

def build_prompt(few_shot_prefix, question):
    return f"{few_shot_prefix}\nInstruction: Answer the question based only on your general knowledge.\nQ: {question}\nA:"

def save_generated_prompts(prompts, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in prompts:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(prompts)} prompts to {output_path}")

def generate_instruction_prompts():
    few_shot_path = "prompts/closedbookqa_instruction.txt"
    test_data_path = "data/sample_dataset/qa_100.jsonl"
    output_path = "data/generated_prompts/closedbookqa_instruction_prompts.jsonl"
    few_shot_prefix = load_few_shot_prompt(few_shot_path)
    test_samples = load_test_samples(test_data_path)
    results = []
    for i, sample in enumerate(test_samples):
        question = sample.get("question", "").strip()
        answer = sample.get("answer", "").strip()
        full_prompt = build_prompt(few_shot_prefix, question)
        results.append({
            "id": i,
            "prompt": full_prompt,
            "answer": answer
        })
    save_generated_prompts(results, output_path)

if __name__ == "__main__":
    generate_instruction_prompts()
