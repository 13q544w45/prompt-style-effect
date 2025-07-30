import os
import json

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
        gt_answer = normalize(gt_item["label"])
        gpt_answer = normalize(pred_item["gpt3_answer"])
        is_correct = gt_answer == gpt_answer
        if is_correct:
            correct += 1
        results.append({
            "premise": gt_item["premise"],
            "hypothesis": gt_item["hypothesis"],
            "truth": gt_answer,
            "gpt3_answer": gpt_answer,
            "correct": is_correct
        })
    accuracy = correct / total
    save_jsonl(results, output_path, accuracy)

def main():
    truth_path = "data/sample_dataset/nli_100.jsonl"
    prediction_path = "data/model_outputs/nli_basic_gpt3.jsonl"
    output_path = "evaluation/nli_basic.jsonl"
    evaluate_predictions(truth_path, prediction_path, output_path)

if __name__ == "__main__":
    main()
