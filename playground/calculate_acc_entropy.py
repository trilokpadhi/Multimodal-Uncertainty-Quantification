import json
import torch
import torch.nn.functional as F
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm 

# Load the tokenizer and model for entailment checking
tokenizer = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli")

# Define functions
def compute_predictive_entropy(raw_logits, token_ids):
    """
    Compute predictive entropy given raw logits and sequence of token IDs.
    """
    try:
        log_probs = []
        for logits in raw_logits:
            log_probs.append(F.log_softmax(logits, dim=-1))

        observed_log_probs = []
        for log_prob, token_id in zip(log_probs, token_ids):
            observed_log_prob = log_prob[0, torch.arange(log_prob.shape[1]), token_id]
            observed_log_probs.append(torch.mean(observed_log_prob, dim=1))

        entropy = -torch.stack(observed_log_probs).mean(dim=0).squeeze()
        return entropy.item()
    except Exception as e:
        print(f"Error calculating entropy: {e}")
        return None

def check_entailment(textA, textB):
    """
    Check whether textB entails textA using a pre-trained model.
    """
    try:
        inputs = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(textA, textB)],
            add_special_tokens=True, return_tensors="pt",
        )
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_class_id = torch.argmax(probs).item()
        return predicted_class_id == 0  # 0 indicates entailment
    except Exception as e:
        print(f"Error checking entailment: {e}")
        return False

def calculate_accuracy(r, question):
    """
    Calculate accuracy based on entailment with ground truth.
    """
    try:
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
                print(f'Error {e} processing response: {response}')
                response_not_counted += 1

        accuracy = entailment_count / (len(responses) - response_not_counted)
        return accuracy
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return None

# Load GQA questions
questions_file_path = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/questions1.2/train_all_questions/train_all_questions_0.json'
try:
    with open(questions_file_path, 'r') as f:
        questions_data = json.load(f)
except Exception as e:
    print(f"Error loading questions data: {e}")
    questions_data = {}

# Path to the results
path = "/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_temp1_topp1_dp_100/results"

# Dictionary to store the results
results = {}

# Loop through each .pkl file in the directory and compute entropy and accuracy
for filename in tqdm(os.listdir(path)):
    if filename.endswith('.pkl'):
        filepath = os.path.join(path, filename)
        try:
            with open(filepath, 'rb') as f:
                r = pickle.load(f)

            question_id = filename.split('_')[1].split('.')[0]
            raw_logits = r['scores']
            token_ids = r['tokens']

            # Calculate entropy
            entropy = compute_predictive_entropy(raw_logits, token_ids)

            # Get the corresponding question
            question = questions_data.get(question_id, {}).get('question', 'Question not found')

            if question != 'Question not found':
                # Calculate accuracy
                accuracy = calculate_accuracy(r, question)

                # Store the results in the dictionary
                results[question_id] = {
                    "accuracy": accuracy,
                    "entropy": entropy
                }
            else:
                print(f'Question ID {question_id} not found in the questions data.')
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Print the results or save them to a file
print(results)

# Optionally, save the results to a JSON file
try:
    with open('/home/ubuntu/Multimodal-Uncertainty-Quantification/results.json', 'w') as f:
        json.dump(results, f, indent=4)
except Exception as e:
    print(f"Error saving results to JSON file: {e}")