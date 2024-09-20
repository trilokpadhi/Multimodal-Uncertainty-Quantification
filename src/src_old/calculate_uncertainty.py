import torch
import torch.multiprocessing as mp
from tqdm import tqdm 
from graph import extract_entities_and_relationships_llama3, extract_triples_from_llama3_json, extract_triples_scene_graph
from utils import extract_json_from_text, ModelArgs
from sentence_transformers import util
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import logging
import pickle 
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_results_if_not_empty(results, save_path):
    """Save results to the specified path if they are not empty."""
    if results:
        save_intermediate_results(results, save_path)
    else:
        logging.info("No new results to save.")

def save_intermediate_results(results, save_path):
    """Save results to the specified path."""
    try:
        if os.path.exists(save_path):
            # with open(save_path, 'r') as f:
            #     existing_results = json.load(f)
            # Load existing results using pickle
            with open(save_path, 'rb') as f:
                existing_results = pickle.load(f)
        else:
            existing_results = {}

        if results:
            existing_results.update(results)
            temp_save_path = f"{save_path}.tmp"
            # with open(temp_save_path, 'w') as f:
            #     json.dump(existing_results, f)
            with open(temp_save_path, 'wb') as f:
                pickle.dump(existing_results, f)
            os.rename(temp_save_path, save_path)  # Atomic move to avoid partial writes
            logging.info(f"Intermediate results saved to {save_path}.")
        else:
            logging.info("No new results to save.")
    
    except Exception as e:
        logging.error(f"Error saving intermediate results: {e}")


def calculate_uncertainty_by_self_consistency():
    # Placeholder function; implement as needed
    pass

def preprocess_triple(triple):
    """Normalize the triple by converting to lowercase and joining the elements into a single string."""
    try:
        return ' '.join(word.lower() for word in triple)
    except Exception as e:
        logging.error(f"Error in preprocess_triple: {e}")
        return None

def calculate_confidence_metric(model_args, ground_truth_set, output_set, k=3):
    """Calculate the confidence metric based on the top-K similarities between two sets of triples."""
    
    try:
        ground_truth_list = [preprocess_triple(triple) for triple in ground_truth_set if triple]
        output_list = [preprocess_triple(triple) for triple in output_set if triple]
        
        if not ground_truth_list or not output_list:
            logging.error("One of the input lists is empty or all triples failed to preprocess.")
            return None, None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings_gt = model_args.sentence_transformer.encode(ground_truth_list, convert_to_tensor=True).to(device)
        embeddings_out = model_args.sentence_transformer.encode(output_list, convert_to_tensor=True).to(device)
        
        if embeddings_gt.shape[0] == 0 or embeddings_out.shape[0] == 0:
            logging.error("One of the input tensors is empty.")
            return None, None
        
        similarity_matrix = util.pytorch_cos_sim(embeddings_gt, embeddings_out).cpu().numpy()
        sorted_similarities = np.sort(similarity_matrix.flatten())[::-1]
        top_k_similarities = sorted_similarities[:k]
        confidence_metric = np.mean(top_k_similarities)
        
        return confidence_metric.item(), similarity_matrix
    
    except ValueError as ve:
        logging.error(f"ValueError in calculate_confidence_metric: {ve}")
        return None, None
    
    except TypeError as te:
        logging.error(f"TypeError in calculate_confidence_metric: {te}")
        return None, None
    
    except Exception as e:
        logging.error(f"An unexpected error occurred in calculate_confidence_metric: {e}")
        return None, None

def save_results_image_wise(results, save_path, image_id):
    """Save results for a specific image ID to the specified path."""
    try:
        filename = os.path.join(save_path, f"{image_id}.pkl")
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                existing_results = pickle.load(f)
        else:
            existing_results = {}

        if results:
            existing_results.update(results)
            with open(filename, 'wb') as f:
                pickle.dump(existing_results, f)
            logging.info(f"Results for image_id {image_id} saved to {filename}.")
        else:
            logging.info(f"No new results to save for image_id {image_id}.")
    
    except Exception as e:
        logging.error(f"Error saving results for image_id {image_id}: {e}")

def calculate_uncertainty_by_grounding_worker(args, keys, output_dir):
    try:
        args.llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        args.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model_id)
        args.llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model_args = ModelArgs(args)

        for idx in tqdm(keys, desc=f"Process {mp.current_process().pid}"):
            responses = args.responses.get(idx, {}).get('responses', [])
            image_id = args.gqa_data.get(idx, {}).get('imageId')
            results_for_image = {}  # Store results specifically for this image_id
            
            for j, response in enumerate(responses):
                try:
                    explanation = response.get('explanation')
                    if not explanation:
                        logging.warning(f"No explanation found for response {j}. Skipping...")
                        continue
                    
                    llama3_response_graph = extract_entities_and_relationships_llama3(model_args, explanation)
                    llama3_response_graph_jsonified = extract_json_from_text(llama3_response_graph)
                    response_triples = extract_triples_from_llama3_json(llama3_response_graph_jsonified)
                
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error for response {j}: {e}")
                except Exception as e:
                    logging.error(f"Error processing response {j}: {e}")

        
    except Exception as e:
        logging.error(f"An unexpected error occurred in process {mp.current_process().pid}: {e}")

def calculate_uncertainty_by_grounding(args):
    if not args.debug:
        try:
            keys = list(args.responses.keys())[:4]
            num_processes = min(torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(), 6)
            chunk_size = max(1, len(keys) // num_processes)
            chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            logging.info(f"Divided into {len(chunks)} chunks")

            output_dir = os.path.join(args.uncertainty_path, "image_results")
            os.makedirs(output_dir, exist_ok=True)

            processes = []

            for i in range(num_processes):
                p = mp.Process(target=calculate_uncertainty_by_grounding_worker, args=(args, chunks[i], output_dir))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            logging.info("All processes have finished")

        except Exception as e:
            logging.error(f"An unexpected error occurred in calculate_uncertainty_by_grounding: {e}")
    else:
        # Non-parallel execution
        try:
            keys = list(args.responses.keys())[:args.samples]
            output_dir = os.path.join(args.uncertainty_path, "image_results")
            os.makedirs(output_dir, exist_ok=True)

            # Directly process all keys using the worker function
            calculate_uncertainty_by_grounding_worker(args, keys, output_dir)

            logging.info("Processing completed")

        except Exception as e:
            logging.error(f"An unexpected error occurred in calculate_uncertainty_by_grounding: {e}")