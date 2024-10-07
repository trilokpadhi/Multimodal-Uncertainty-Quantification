import torch
import pickle
import os
import json 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Entailment model
tokenizer_deberta = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
model_deberta = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli")

def extract_json_from_text(text):
    """Extracts JSON from the given text."""

    start_index = text.find("{")
    end_index = text.rfind("}") + 1
    json_string = text[start_index:end_index]
    
    return json.loads(json_string)

def get_log_likelihood(scores, tokens, tokens_contain_input_ids = True):
    """
    Get log likelihoods for each explanation from the raw scores
    """

    scores = torch.stack(scores).squeeze()
    log_softmax = torch.log_softmax(scores, dim=-1)
    
    if tokens_contain_input_ids:
        len_gen_tokens = log_softmax.shape[0]
        generated_tokens = tokens[-len_gen_tokens:]
        
    log_softmax_gen_tokens = log_softmax[range(len_gen_tokens), generated_tokens]
    log_likelihood = log_softmax_gen_tokens.sum()
    
    return log_likelihood


def calculate_entropy(loglikelihoods):
    """
    Calculate entropy for each explanation from the list of log likelihoods
    """
    loglikelihoods_tensor = torch.tensor(loglikelihoods)
    probs = torch.exp(loglikelihoods_tensor)
    # entropy = -torch.sum(probs * torch.log(probs))
    entropy = -torch.sum(torch.nan_to_num(probs * torch.log(probs), nan=0.0))
    
    return entropy.item()


def check_entailment(textA, textB):
    """
    Check whether textB entails textA using a pre-trained model.
    textA: premise : It is the ground truth full answer
    textB: hypothesis : It is the generated model explanation
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

# Load the files 
explanation_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/explanations'
grounding_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/grounding'
uncertainty_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_100/uncertainty'

# check if the directories exist
if not os.path.exists(explanation_dir):
    print('explanation directory does not exist')
    exit()
    
if not os.path.exists(grounding_dir):
    print('grounding directory does not exist')
    exit()
    
if not os.path.exists(uncertainty_dir):
    os.makedirs(uncertainty_dir)
    
explanation_files = os.listdir(explanation_dir)
grounding_files = os.listdir(grounding_dir)

# Load the files
uncertainty_from_grounding_entropy = {}

# get entropy from explanations
for file in tqdm(explanation_files, desc = 'Calculating entropy from explanations', total = len(explanation_files)):
    with open(explanation_dir + '/' + file, 'rb') as f:
        explanation_meta_data = pickle.load(f)
        question_id = file.split('_')[-1].replace('.pkl', '')
        
        log_likelihoods = []
        for key in explanation_meta_data.keys():
            if 'response' in key:
                response_meta_data = explanation_meta_data[key]
                log_likelihood = get_log_likelihood(response_meta_data['outputs']['scores'], response_meta_data['outputs']['sequences'].squeeze(), tokens_contain_input_ids = True)
                explanation_meta_data[key]['log_likelihood'] = log_likelihood
                log_likelihoods.append(log_likelihood.item())
                
            else:
                continue
        entropy = calculate_entropy(log_likelihoods)
        # explanation_meta_data['entropy'] = entropy
        uncertainty_from_grounding_entropy[question_id] = {}
        uncertainty_from_grounding_entropy[question_id]['uncertainty_from_entropy'] = entropy
            
# get grounding of the explanations
for file in tqdm(grounding_files, desc = 'Calculating grounding from explanations', total = len(grounding_files)):
    with open(grounding_dir + '/' + file, 'rb') as f:
        grounding_meta_data = pickle.load(f)
        question_id = file.split('_')[-1].replace('.pkl', '')
        
        uncertainty_from_grounding = []
        for key in grounding_meta_data.keys():
            if 'response' in key:
                response_meta_data = grounding_meta_data[key]
                try:
                    grounding = response_meta_data['grounding_score'] 
                except:
                    continue
                uncertainty_from_grounding.append(1 - grounding)
                
            else:
                continue
        uncertainty_from_grounding_ = sum(uncertainty_from_grounding) / len(uncertainty_from_grounding)
        uncertainty_from_grounding_entropy[question_id]['uncertainty_from_grounding'] = uncertainty_from_grounding_
            
                
# get accuracy from groundings
for file in tqdm(grounding_files, desc = 'Calculating accuracy from groundings', total = len(grounding_files)):
    with open(grounding_dir + '/' + file, 'rb') as f:
        grounding_meta_data = pickle.load(f)
        question_id = file.split('_')[-1].replace('.pkl', '')
        
        entailment = []
        for key in grounding_meta_data.keys():
            if 'response' in key:
                response_meta_data = grounding_meta_data[key]
                try:
                    model_explanation = extract_json_from_text(response_meta_data['decoded_outputs'])['explanation'] 
                except:
                    continue
                ground_truth = grounding_meta_data['full_answers'][0]
                entailment.append(check_entailment(ground_truth, model_explanation))
                
            else:
                continue
        entailed_responses = entailment.count(True)
        accuracy = entailed_responses / len(entailment)
        uncertainty_from_grounding_entropy[question_id]['accuracy'] = accuracy
        
# # get model uncertainty from groundings
# for file in tqdm(grounding_files, desc = 'Calculating model uncertainty from groundings', total = len(grounding_files)):
#     with open(grounding_dir + '/' + file, 'rb') as f:
#         grounding_meta_data = pickle.load(f)
#         question_id = file.split('_')[-1].replace('.pkl', '')
        
#         model_uncertainty = []
#         for key in grounding_meta_data.keys():
#             if 'response' in key:
#                 response_meta_data = grounding_meta_data[key]
#                 try:
#                     model_uncertainty.append(response_meta_data['model_uncertainty'])
#                 except:
#                     continue
#             else:
#                 continue
#         model_uncertainty_ = sum(model_uncertainty) / len(model_uncertainty)
#         uncertainty_from_grounding_entropy[question_id]['model_uncertainty'] = model_uncertainty_
        

# save the uncertainty
with open(uncertainty_dir + '/uncertainty_random_100.pkl', 'wb') as f:
    pickle.dump(uncertainty_from_grounding_entropy, f)
        
            
