import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli")

def check_entailment(textA, textB):
    inputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=[(textA, textB)],
        add_special_tokens=True, return_tensors="pt",
    )
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    predicted_class_id = torch.argmax(probs).item()
    return predicted_class_id == 0  # 0 indicates entailment

def calculate_accuracy(r, question):
    ground_truth_answer = r['full_answer']
    responses = r['responses']
    
    entailment_count = 0
    response_not_counted = 0
    for response in responses:
        try:
            model_answer = response['answer']
            if check_entailment(f'{question} {ground_truth_answer}', f'{question} {model_answer}'):
                entailment_count += 1
        except Exception as e:
            print(' Error {} processing {}'.format(e, response))
            response_not_counted = response_not_counted + 1
            

    accuracy = entailment_count / (len(responses) - response_not_counted)
    return accuracy

# Example usage:
path = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_temp1_topp1_dp_100/results'
sample = 'result_0293788.pkl'

with open(f'{path}/{sample}', 'rb') as f:
    r = pickle.load(f)

# Load the GQA dataset files

questions_file_path = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/questions1.2/train_all_questions/train_all_questions_0.json'
with open(questions_file_path, 'r') as f:
    questions_data = json.load(f)
        
question_id = sample.split('_')[1].split('.')[0]

question = questions_data[question_id]['question']
accuracy = calculate_accuracy(r, question)
print(f"Accuracy: {accuracy * 100:.2f}%")