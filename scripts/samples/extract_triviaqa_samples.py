import json
import random
from datasets import load_dataset

def is_valid_sample(sample):
    answer = sample.get("answer", {}).get("value", "")
    question = sample.get("question", "")
    if not question or not answer:
        return False
    if isinstance(answer, list):
        answer = answer[0]
    if isinstance(answer, str):
        if len(answer.strip().split()) > 5:
            return False
        if len(answer.strip()) == 0:
            return False
        return True
    return False

def extract_samples(n=100, seed=42):
    dataset = load_dataset("trivia_qa", "unfiltered")["validation"]
    valid_samples = [s for s in dataset if is_valid_sample(s)]
    random.seed(seed)
    sampled = random.sample(valid_samples, n)
    results = []
    for sample in sampled:
        question = sample["question"].strip()
        answer = sample["answer"]["value"]
        if isinstance(answer, list):
            answer = answer[0]
        results.append({
            "question": question,
            "answer": answer.strip(),
            "source": "TriviaQA"
        })
    return results

def save_jsonl(data, output_path="data/qa_100.jsonl"):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    samples = extract_samples(n=100)
    save_jsonl(samples)
    