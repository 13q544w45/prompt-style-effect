import os
import json

def extract_final_answer(text):
    if not isinstance(text, str):
        return ""
    lowered = text.lower()
    key_phrase = "the answer is"
    if key_phrase in lowered:
        idx = lowered.rfind(key_phrase)
        final_answer = text[idx + len(key_phrase):].strip()
        return final_answer.strip(" .")
    return text.strip(" .")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path, accuracy=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
        if accuracy is not None:
            json.dump({"accuracy": round(accuracy, 4)}, f, ensure_ascii=False)
            f.write("\n")
    print(f"\nSaved evaluation results to {path}")

def normalize(text):
    return text.strip().lower()

def evaluate_predictions(ground_truth_path, prediction_path, output_path):
    gt_data = load_jsonl(ground_truth_path)
    pred_data = load_jsonl(prediction_path)
    assert len(gt_data) == len(pred_data), "Mismatch in number of samples."
    total = len(gt_data)
    correct = 0
    results = []
    for i in range(total):
        gt_item = gt_data[i]
        pred_item = pred_data[i]
        gt_answer = normalize(gt_item["answer"])
        raw_output = pred_data[i]["gpt3_answer"]
        gpt_answer = normalize(extract_final_answer(raw_output))
        is_correct = gt_answer == gpt_answer
        if is_correct:
            correct += 1
        results.append({
            "sentence": gt_item["sentence"],
            "option1": gt_item["option1"],
            "option2": gt_item["option2"],
            "answer": gt_answer,
            "gpt3 answer": gpt_answer,
            "correct": is_correct
        })
    accuracy = correct / total
    save_jsonl(results, output_path, accuracy)

def main():
    truth_path = "data/sample_dataset/winogrande_100.jsonl"
    prediction_path = "data/model_outputs/winogrande_cot_gpt3.jsonl"
    output_path = "evaluation/winogrande_cot.jsonl"
    evaluate_predictions(truth_path, prediction_path, output_path)

if __name__ == "__main__":
    main()
