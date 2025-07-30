import os
import json
import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_FILE = "data/model_inputs/closedbookqa_instruction_cleaned.jsonl"
OUTPUT_FILE = "data/model_outputs/closedbookqa_instruction_gpt3.jsonl"

def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_outputs(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(data)} completions to {path}")

def call_gpt35(prompt, max_tokens=128):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "ERROR"

def run_inference():
    prompts = load_prompts(INPUT_FILE)
    outputs = []
    for i, item in enumerate(tqdm(prompts, desc="Running GPT-3.5 Inference")):
        prompt = item["prompt"]
        prediction = call_gpt35(prompt)
        outputs.append({
            "id": i,
            "prompt": prompt,
            "gpt3_answer": prediction
        })
    save_outputs(outputs, OUTPUT_FILE)

if __name__ == "__main__":
    run_inference()
    
