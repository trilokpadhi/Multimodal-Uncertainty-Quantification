import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
import numpy as np
from collections import Counter
from math import log2
import os
import pickle
import json

from rouge_score import rouge_scorer
import itertools
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Initialize NLTK resources
nltk.download('punkt')

def initialize_rouge():
    """Initialize the ROUGE scorer with ROUGE-L."""
    return rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def initialize_deberta(deberta_model_name='microsoft/deberta-base-mnli', device='cpu'):
    """
    Initialize the DeBERTa tokenizer and model.

    Args:
        deberta_model_name (str): Model name or path.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        tokenizer: DeBERTa tokenizer.
        model: DeBERTa model for sequence classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return tokenizer, model

def get_json_from_response(response):
    """
    Extract JSON data from a response string containing extraneous characters.

    Args:
        response (str): Response string containing JSON data.

    Returns:
        dict: Parsed JSON data.
    """
    start_token = response.find('{')
    end_token = response.rfind('}')
    response_json_str = response[start_token:end_token+1]
    response_json = json.loads(response_json_str)
    return response_json

def calculated_predictive_entropy_from_log_probs(log_probs):
    """
    Calculate predictive entropy from log probabilities.

    Args:
        log_probs (list): List of log probabilities of the sentences.

    Returns:
        float: Total predictive entropy.
    """
    probabilities = [np.exp(lp) for lp in log_probs]
    probabilities = np.array(probabilities)
    log_probs = np.array(log_probs)
    total_predicitive_entropy = np.sum(probabilities * log_probs)
    return total_predicitive_entropy

def get_lexical_similarity(responses, scorer):
    """
    Compute the average ROUGE-L F1 score across all unique pairs of responses.

    Args:
        responses (list): List of response strings.
        scorer: Initialized ROUGE scorer.

    Returns:
        float: Average ROUGE-L F1 score.
    """
    rouge_l_scores = []
    for resp1, resp2 in itertools.combinations(responses, 2):
        score = scorer.score(resp1, resp2)
        rouge_l_f1 = score['rougeL'].fmeasure
        rouge_l_scores.append(rouge_l_f1)
    average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    return average_rouge_l

def get_accuracy(full_answers, question, responses, tokenizer, model, device):
    """
    Calculate accuracy by checking entailment of responses with full answers.

    Args:
        full_answers (str): The correct full answer.
        question (str): The associated question.
        responses (list): List of response strings.
        tokenizer: DeBERTa tokenizer.
        model: DeBERTa model.
        device (str): Device for computation.

    Returns:
        float: Accuracy score.
    """
    correct = 0
    for response in responses:
        premise = f"{question} {full_answers}"
        hypothesis = f"{question} {response}"
        
        inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors='pt', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
        
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
            if pred == 2:
                correct += 1
    accuracy = correct / len(responses) if responses else 0
    return accuracy

def get_semantic_entropy(question, explanation_with_log_probs, tokenizer, model, device):
    """
    Calculate semantic entropy by clustering explanations based on bidirectional entailment.

    Args:
        question (str): The question for which the explanations are generated.
        explanation_with_log_probs (dict): Dictionary mapping explanations to their log probabilities.
        tokenizer: DeBERTa tokenizer.
        model: DeBERTa model.
        device (str): Device for computation.

    Returns:
        tuple: (semantic entropy, number of clusters)
    """
    explanations = list(explanation_with_log_probs.keys())
    log_probs = list(explanation_with_log_probs.values())

    # Convert log probabilities to probabilities
    probs = [math.exp(lp) for lp in log_probs]
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    clusters = []

    for explanation in explanations:
        assigned = False
        for cluster in clusters:
            representative = cluster[0]
            premise1 = f"{question} {explanation}"
            premise2 = f"{question} {representative}"

            inputs1 = tokenizer.encode_plus(premise1, premise2, return_tensors='pt', truncation=True)
            inputs2 = tokenizer.encode_plus(premise2, premise1, return_tensors='pt', truncation=True)

            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}

            with torch.no_grad():
                logits1 = model(**inputs1).logits
                logits2 = model(**inputs2).logits

            pred1 = torch.argmax(logits1, dim=1).item()
            pred2 = torch.argmax(logits2, dim=1).item()

            if pred1 == 2 and pred2 == 2:
                cluster.append(explanation)
                assigned = True
                break

        if not assigned:
            clusters.append([explanation])

    # Compute the probability of each cluster by averaging probabilities of its explanations
    cluster_probs = []
    for cluster in clusters:
        cluster_prob = np.mean([probs[explanations.index(exp)] for exp in cluster])
        cluster_probs.append(cluster_prob)

    # Compute semantic entropy: SE(x) = -sum(p(c) * log(p(c)))
    entropy = -sum(p * math.log(p) for p in cluster_probs if p > 0)
    num_clusters = len(clusters)

    return entropy, num_clusters

def process_file(file_path, explanation_dir, deberta_model_name, gpu_id):
    """
    Worker function to process a single file and compute uncertainty metrics.

    Args:
        file_path (str): Filename to process.
        explanation_dir (str): Directory containing explanation files.
        deberta_model_name (str): Name/path of the DeBERTa model.
        gpu_id (int): GPU ID to use for this worker.

    Returns:
        tuple: (question_id, uncertainty_scores_baseline)
    """
    try:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        tokenizer, model = initialize_deberta(deberta_model_name, device)
        scorer = initialize_rouge()

        with open(os.path.join(explanation_dir, file_path), 'rb') as f:
            question_id = file_path.split('_')[1].split('.')[0]  # Extract question_id
            explanation_metadata = pickle.load(f)
            log_probs = []
            explanations = []
            explanation_with_log_probs = {}
            question = explanation_metadata['questions'][0]

            for key, value in explanation_metadata.items():
                if 'response' in key:
                    metadata = explanation_metadata[key]
                    # Calculate log probability
                    log_prob_sentence = np.sum(metadata.get('transition_scores', []))
                    log_probs.append(log_prob_sentence)

                    # Extract explanation from decoded_outputs
                    try:
                        response_json = get_json_from_response(metadata['decoded_outputs'])
                        explanation = response_json['explanation']
                        explanations.append(explanation)
                        explanation_with_log_probs[explanation] = log_prob_sentence
                    except Exception as e:
                        print(f"Error parsing JSON in file {file_path}: {e}")
                        continue
                else:
                    continue

            # Compute uncertainty metrics
            predictive_entropy = calculated_predictive_entropy_from_log_probs(log_probs)
            rogue_l_score = get_lexical_similarity(explanations, scorer)
            semantic_entropy, num_clusters = get_semantic_entropy(question, explanation_with_log_probs, tokenizer, model, device)
            explanation_list = list(explanation_with_log_probs.keys())
            full_answers = explanation_metadata.get('full_answers', [''])[0]
            accuracy = get_accuracy(full_answers, question, explanation_list, tokenizer, model, device)

            # Compile uncertainty scores
            uncertainty_scores_baseline = {
                'predictive_entropy': predictive_entropy,
                'lexical_similarity': rogue_l_score,
                'semantic_entropy': semantic_entropy,
                'num_clusters': num_clusters,
                'accuracy': accuracy
            }

        return question_id, uncertainty_scores_baseline

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

def main():
    # Define directories
    explanation_dir = '/mnt/myebsvolume/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_yes_gsam_grounding_random_10000/explanations'
    uncertainty_dir = '/home/ec2-user/Multimodal-Uncertainty-Quantification/runs/uncertainty'
    deberta_model_name = 'microsoft/deberta-base-mnli'

    # List all files to process
    files = os.listdir(explanation_dir)

    # Dictionary to store results
    responses = {}

    # Define number of worker processes and assign GPUs
    num_cpus = 6
    num_gpus = 8
    workers = min(num_cpus, num_gpus)  # Typically, align workers with available CPUs and GPUs

    # Assign GPUs to workers (cycle through available GPUs if workers < GPUs)
    gpu_ids = list(range(num_gpus))
    assigned_gpus = [gpu_ids[i % num_gpus] for i in range(workers)]

    # Initialize ProcessPoolExecutor with specified number of workers
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Prepare arguments for each worker
        futures = []
        for idx, file in enumerate(files):
            gpu_id = assigned_gpus[idx % workers]  # Assign GPU in a round-robin fashion
            futures.append(executor.submit(process_file, file, explanation_dir, deberta_model_name, gpu_id))

        # Initialize tqdm progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing files in parallel'):
            try:
                question_id, uncertainty_scores_baseline = future.result()
                if question_id is not None:
                    responses[question_id] = uncertainty_scores_baseline
            except Exception as e:
                print(f"Error retrieving result: {e}")

    # Make the uncertainty directory if it does not exist
    if not os.path.exists(uncertainty_dir):
        os.makedirs(uncertainty_dir)

    # Save the responses in a pickle file
    output_path = os.path.join(uncertainty_dir, 'uncertainty_scores_baseline.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(responses, f)
    print(f"Uncertainty scores saved to {output_path}")

if __name__ == '__main__':
    main()