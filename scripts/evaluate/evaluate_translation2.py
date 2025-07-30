import os
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')

def extract_final_answer(text):
    if "The answer is:" in text:
        return text.split("The answer is:")[-1].strip()
    return text.strip()

def compute_bleu(reference, hypothesis):
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path, average_bleu=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
        if average_bleu is not None:
            json.dump({"average_bleu": round(average_bleu, 4)}, f, ensure_ascii=False)
            f.write("\n")

def normalize(text):
    return text.strip().lower()

def evaluate_predictions(truth_path, prediction_path, output_path):
    gt_data = load_jsonl(truth_path)
    pred_data = load_jsonl(prediction_path)
    assert len(gt_data) == len(pred_data), "Mismatch in number of samples."
    results = []
    bleu_scores = []
    for gt_item, pred_item in zip(gt_data, pred_data):
        source_en = gt_item["english"]
        reference_de = normalize(gt_item["german"])
        full_pred = pred_item["gpt3_answer"]
        final_answer = normalize(extract_final_answer(full_pred))
        bleu = compute_bleu(reference_de, final_answer)
        bleu_scores.append(bleu)
        results.append({
            "english_input": source_en,
            "reference": reference_de,
            "gpt3_answer": final_answer,
            "bleu_score": round(bleu, 4)
        })
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    save_jsonl(results, output_path, average_bleu=average_bleu)

def main():
    truth_path = "data/sample_dataset/translation_100.jsonl"
    prediction_path = "data/model_outputs/translation_cot_gpt3.jsonl"
    output_path = "evaluation/translation_cot.jsonl"
    evaluate_predictions(truth_path, prediction_path, output_path)

if __name__ == "__main__":
    main()
