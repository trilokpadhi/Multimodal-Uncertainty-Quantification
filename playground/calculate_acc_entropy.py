import json
import torch
import torch.nn.functional as F
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm 
from llava.mm_utils import get_model_name_from_path # type: ignore
from llava.model.builder import load_pretrained_model # type: ignore

# # Load the tokenizer and model for entailment checking
# tokenizer = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
# model = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli")

# Load the tokenizer and model for entailment checking
tokenizer_deberta = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
model_deberta = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli")


## Llava tokenizer 
model_path = "liuhaotian/llava-v1.6-vicuna-7b"
model_base = None
model_name = get_model_name_from_path(model_path)
tokenizer_llava, model_llava, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name
)

# Define functions
# def compute_predictive_entropy(raw_logits, token_ids_list):
#     """
#     Compute predictive entropy given raw logits and sequence of token IDs.
#     """
#     try:
#         log_probs = []
#         for logits, token_ids in zip(raw_logits, token_ids_list):
#             try:
#                 # log_probs.append(F.log_softmax(logits, dim=-1))
#                 log_softmax_for_all_tokens = F.log_softmax(logits, dim=-1)
#                 """
#                 for selected tokens
#                 """
#                 start_index = token_ids[0].tolist().index(362)
#                 end_index = token_ids[0].tolist().index(5527)
#                 log_softmax_for_selected_tokens = log_softmax_for_all_tokens[:, start_index:end_index, :]
#                 log_prob = log_softmax_for_selected_tokens[0, torch.arange(log_softmax_for_selected_tokens.shape[1]), token_ids[:, start_index: end_index]]
#                 log_probs.append(torch.mean(log_prob, dim=1))
#                 """
#                 Method 2: where i only take tokens between tokens with IDs 362 (“ation”) and 5527 (“conf”) within the sequence.
#                 """                   
#                 # log_prob = log_softmax_for_all_tokens[0, torch.arange(log_softmax_for_all_tokens.shape[1]), token_ids]
#                 # log_probs.append(torch.mean(log_prob, dim=1))
#             except Exception as e:
#                 print(f'Error calculating log probs {e}')

#         entropy = -torch.stack(log_probs).mean(dim=0).squeeze()
#         return entropy.item()
#     except Exception as e:
#         print(f"Error calculating entropy: {e}")
#         return None

def compute_predictive_entropy(raw_logits, token_ids_list):
    """
    Compute predictive entropy given raw logits and sequence of token IDs.
    Returns:
        entropy: Entropy of 20 responses
        log_likelihoods: Log Likelihood of 20 responses
        log_probs: Log probs of 20 responses token-wise, list of [1, token_length, log probs]
        tokens_list: tokens_list of 20 responses
        decoded_tokens_list: Decoded tokens of 20 responses
    """
    log_likelihoods = []
    log_probs = []
    tokens_list = []
    decoded_tokens_list = []
    
    try:
        for logits, token_ids in zip(raw_logits, token_ids_list):
            try:
                """
                For selected tokens
                """
                # Compute log softmax for all tokens
                log_softmax_for_all_tokens = F.log_softmax(logits, dim=-1)
                
                # Get the start and end index for the specific tokens (IDs 362 and 5527)
                # start index + 1 to not include token 'ation'
                start_index = token_ids[0].tolist().index(362) + 1
                end_index = token_ids[0].tolist().index(5527) 
                
                # Select log probabilities for the given token IDs
                
                log_softmax_for_selected_tokens = log_softmax_for_all_tokens[:, start_index :end_index, :]
                log_prob = log_softmax_for_selected_tokens[0, torch.arange(log_softmax_for_selected_tokens.shape[1]), token_ids[:, start_index:end_index]]
                
                # Store log probabilities and log likelihoods
                log_probs.append(log_prob.squeeze())  # Mean log probability
                log_likelihoods.append(torch.mean(log_prob))  # Log likelihood for the selected range
                
                # Store token IDs and decoded tokens (you can replace 'convert_ids_to_tokens' with your tokenizer method)
                tokens_list.append(token_ids[:, start_index:end_index].squeeze().tolist())
                decoded_tokens_list.append(tokenizer_llava.convert_ids_to_tokens(token_ids[:, start_index:end_index].squeeze()))
                """
                For All tokens
                """
                # # Compute log softmax for all tokens
                # log_softmax_for_all_tokens = F.log_softmax(logits, dim=-1)
                # log_prob = log_softmax_for_all_tokens[0, torch.arange(log_softmax_for_all_tokens.shape[1]), token_ids]
                
                # # Store log probabilities and log likelihoods
                # log_probs.append(log_prob.squeeze())  # Mean log probability
                # log_likelihoods.append(torch.mean(log_prob))  # Log likelihood for the selected range
                
                # # Store token IDs and decoded tokens (you can replace 'convert_ids_to_tokens' with your tokenizer method)
                # tokens_list.append(token_ids.squeeze().tolist())
                # decoded_tokens_list.append(tokenizer_llava.convert_ids_to_tokens(token_ids.squeeze()))
                
            
            except Exception as e:
                print(f'Error calculating log probs for one set: {e}')
                log_probs.append(None)
                log_likelihoods.append(None)
                tokens_list.append(None)
                decoded_tokens_list.append(None)
        
        # Filter out None values (in case some responses failed)
        log_likelihoods_filtered = [lp for lp in log_likelihoods if lp is not None]
        
        # Calculate entropy
        entropy = -torch.stack(log_likelihoods_filtered).mean(dim=0).squeeze()

        # Return the computed values
        return entropy.item(), log_likelihoods, log_probs, tokens_list, decoded_tokens_list
    
    except Exception as e:
        print(f"Error calculating entropy: {e}")
        return None, None, None, None, None
    

def check_entailment(textA, textB):
    """
    Check whether textB entails textA using a pre-trained model.
    """
    try:
        inputs = tokenizer_deberta.batch_encode_plus(
            batch_text_or_text_pairs=[(textA, textB)],
            add_special_tokens=True, return_tensors="pt",
        )
        logits = model_deberta(**inputs).logits
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
            entropy, _, _, _, _  = compute_predictive_entropy(raw_logits, token_ids)

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
    with open('/home/ubuntu/Multimodal-Uncertainty-Quantification/results_with_selected_tokens_new.json', 'w') as f:
        json.dump(results, f, indent=4)
except Exception as e:
    print(f"Error saving results to JSON file: {e}")