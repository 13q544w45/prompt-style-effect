import os
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')

def compute_bleu(reference, hypothesis):
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

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
            json.dump({"average_bleu": round(accuracy, 4)}, f, ensure_ascii=False)
            f.write("\n")

def normalize(text):
    return text.strip().lower()

def evaluate_predictions(truth_path, prediction_path, output_path):
    gt_data = load_jsonl(truth_path)
    pred_data = load_jsonl(prediction_path)
    assert len(gt_data) == len(pred_data), "Mismatch in number of samples."
    results = []
    bleu_scores = []
    for i in range(len(gt_data)):
        source_en = gt_data[i]["english"]        
        reference_de = normalize(gt_data[i]["german"])  
        predicted_de = normalize(pred_data[i]["gpt3_answer"]) 
        bleu = compute_bleu(reference_de, predicted_de)
        bleu_scores.append(bleu)
        results.append({
            "english_input": source_en,
            "reference": reference_de,
            "gpt3_answer": predicted_de,
            "bleu_score": round(bleu, 4)
        })
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    save_jsonl(results, output_path, accuracy=average_bleu)

def main():
    truth_path = "data/sample_dataset/translation_100.jsonl"
    prediction_path = "data/model_outputs/translation_instruction_gpt3.jsonl"
    output_path = "evaluation/translation_instruction.jsonl"
    evaluate_predictions(truth_path, prediction_path, output_path)

if __name__ == "__main__":
    main()
