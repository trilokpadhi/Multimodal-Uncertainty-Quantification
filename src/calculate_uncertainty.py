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

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def calculate_uncertainty_by_self_consistency():
#     # Placeholder function; implement as needed
#     pass

# def preprocess_triple(triple):
#     """Normalize the triple by converting to lowercase and joining the elements into a single string."""
#     try:
#         return ' '.join(word.lower() for word in triple)
#     except Exception as e:
#         logging.error(f"Error in preprocess_triple: {e}")
#         return None

# def calculate_confidence_metric(model_args, ground_truth_set, output_set, k=3):
#     """Calculate the confidence metric based on the top-K similarities between two sets of triples."""
    
#     try:
#         ground_truth_list = [preprocess_triple(triple) for triple in ground_truth_set if triple]
#         output_list = [preprocess_triple(triple) for triple in output_set if triple]
        
#         if not ground_truth_list or not output_list:
#             logging.error("One of the input lists is empty or all triples failed to preprocess.")
#             return None
        
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         embeddings_gt = model_args.sentence_transformer.encode(ground_truth_list, convert_to_tensor=True).to(device)
#         embeddings_out = model_args.sentence_transformer.encode(output_list, convert_to_tensor=True).to(device)
        
#         if embeddings_gt.shape[0] == 0 or embeddings_out.shape[0] == 0:
#             logging.error("One of the input tensors is empty.")
#             return None
        
#         similarity_matrix = util.pytorch_cos_sim(embeddings_gt, embeddings_out).cpu().numpy()
#         sorted_similarities = np.sort(similarity_matrix.flatten())[::-1]
#         top_k_similarities = sorted_similarities[:k]
#         confidence_metric = np.mean(top_k_similarities)
        
#         return confidence_metric.item()
    
#     except ValueError as ve:
#         logging.error(f"ValueError in calculate_confidence_metric: {ve}")
#         return None
    
#     except TypeError as te:
#         logging.error(f"TypeError in calculate_confidence_metric: {te}")
#         return None
    
#     except Exception as e:
#         logging.error(f"An unexpected error occurred in calculate_confidence_metric: {e}")
#         return None

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

# def calculate_uncertainty_by_grounding_worker(args, keys, results_queue):
#     try:
#         args.llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#         args.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model_id)
#         args.llama_model = AutoModelForCausalLM.from_pretrained(
#             args.llama_model_id,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#         )
#         model_args = ModelArgs(args)
#         results = {}
#         save_interval = 10

#         for i, idx in enumerate(tqdm(keys, desc=f"Process {mp.current_process().pid}")):
#             responses = args.responses.get(idx, {}).get('responses', [])
#             for j, response in enumerate(responses):
#                 try:
#                     explanation = response.get('explanation')
#                     if not explanation:
#                         logging.warning(f"No explanation found for response {j}. Skipping...")
#                         continue
                    
#                     llama3_response_graph = extract_entities_and_relationships_llama3(model_args, explanation)
#                     llama3_response_graph_jsonified = extract_json_from_text(llama3_response_graph)
#                     response_triples = extract_triples_from_llama3_json(llama3_response_graph_jsonified)
                    
#                     image_id = args.gqa_data.get(idx, {}).get('imageId')
#                     scene_graph = args.scene_graphs_data.get(image_id)
                    
#                     if not scene_graph:
#                         logging.warning(f"No scene graph found for image ID {image_id}. Skipping...")
#                         continue

#                     scene_graph_triples = extract_triples_scene_graph(scene_graph)
#                     confidence_metric = calculate_confidence_metric(model_args, scene_graph_triples, response_triples, len(response_triples))
                    
#                     if confidence_metric is not None:
#                         response['confidence_metric'] = confidence_metric
                
#                 except json.JSONDecodeError as e:
#                     logging.error(f"JSON parsing error for response {j}: {e}")
#                 except Exception as e:
#                     logging.error(f"Error processing response {j}: {e}")

#             if idx not in results:
#                 results[idx] = []
#             results[idx].extend(responses)

#             # Save results at every interval
#             if (i + 1) % save_interval == 0:
#                 save_results_if_not_empty(results, args.uncertainty_path)
#                 results.clear()

#         # Save any remaining results after the loop
#         save_results_if_not_empty(results, args.uncertainty_path)
#         results_queue.put(results)
    
#     except Exception as e:
#         logging.error(f"An unexpected error occurred in process {mp.current_process().pid}: {e}")

# def calculate_uncertainty_by_grounding(args):

#     if not args.debug:
#         try:
#             keys = list(args.responses.keys())[:4]
#             num_processes = min(torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(), 6)
#             chunk_size = max(1, len(keys) // num_processes)
#             chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
#             logging.info(f"Divided into {len(chunks)} chunks")
            
#             results_queue = mp.Queue()
#             processes = []

#             for i in range(num_processes):
#                 p = mp.Process(target=calculate_uncertainty_by_grounding_worker, args=(args, chunks[i], results_queue))
#                 p.start()
#                 processes.append(p)

#             results = {}
#             for p in processes:
#                 p.join()

#             while not results_queue.empty():
#                 result = results_queue.get()
#                 for idx, responses in result.items():
#                     if idx not in results:
#                         results[idx] = []
#                     results[idx].extend(responses)

#             # Final save of all results
#             save_results_if_not_empty(results, args.uncertainty_path)
        
#         except Exception as e:
#             logging.error(f"An unexpected error occurred in calculate_uncertainty_by_grounding: {e}")
#     else:
#         # Non-parallel execution
#         try:
#             keys = list(args.responses.keys())[:args.samples]
#             results = {}
#             results_queue = []

#             # Directly process all keys using the worker function
#             result = calculate_uncertainty_by_grounding_worker(args, keys, results_queue)
            
#             for idx, responses in result.items():
#                 if idx not in results:
#                     results[idx] = []
#                 results[idx].extend(responses)

#             # Final save of all results
#             save_results_if_not_empty(results, args.uncertainty_path)
        
#         except Exception as e:
#             logging.error(f"An unexpected error occurred in calculate_uncertainty_by_grounding: {e}")



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def save_intermediate_data_for_image(image_id, data, save_path):
    """Save all intermediate data for a given image_id to a single .pkl file."""
    try:
        filename = f"{save_path}_{image_id}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Intermediate data for image_id {image_id} saved to {filename}.")
    
    except Exception as e:
        logging.error(f"Error saving intermediate data for image_id {image_id}: {e}")

def calculate_uncertainty_by_grounding_worker(args, keys, results_queue):
    try:
        args.llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        args.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model_id)
        args.llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model_args = ModelArgs(args)
        results = {}
        save_interval = 10

        for i, idx in enumerate(tqdm(keys, desc=f"Process {mp.current_process().pid}")):
            responses = args.responses.get(idx, {}).get('responses', [])
            image_id = args.gqa_data.get(idx, {}).get('imageId')
            intermediate_data_for_image = {}  # Collect all intermediate data for this image_id
            
            for j, response in enumerate(responses):
                try:
                    explanation = response.get('explanation')
                    if not explanation:
                        logging.warning(f"No explanation found for response {j}. Skipping...")
                        continue
                    
                    llama3_response_graph = extract_entities_and_relationships_llama3(model_args, explanation)
                    llama3_response_graph_jsonified = extract_json_from_text(llama3_response_graph)
                    response_triples = extract_triples_from_llama3_json(llama3_response_graph_jsonified)
                    
                    scene_graph = args.scene_graphs_data.get(image_id)
                    
                    if not scene_graph:
                        logging.warning(f"No scene graph found for image ID {image_id}. Skipping...")
                        continue

                    scene_graph_triples = extract_triples_scene_graph(scene_graph)
                    confidence_metric, similarity_matrix = calculate_confidence_metric(model_args, scene_graph_triples, response_triples, len(response_triples))
                    
                    if confidence_metric is not None:
                        response['confidence_metric'] = confidence_metric
                    
                    # Store intermediate data for this response
                    intermediate_data_for_image[j] = {
                        'llama3_response_graph': llama3_response_graph,
                        'response_triples': response_triples,
                        'scene_graph_triples': scene_graph_triples,
                        'similarity_matrix': similarity_matrix
                    }
                
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error for response {j}: {e}")
                except Exception as e:
                    logging.error(f"Error processing response {j}: {e}")

            if idx not in results:
                results[idx] = []
            results[idx].extend(responses)

            # Save intermediate data for this image_id
            save_intermediate_data_for_image(image_id, intermediate_data_for_image, args.uncertainty_path)

            # Save results at every interval
            if (i + 1) % save_interval == 0:
                save_results_if_not_empty(results, args.uncertainty_path)
                results.clear()

        # Save any remaining results after the loop
        save_results_if_not_empty(results, args.uncertainty_path)
        results_queue.put(results)
        
        return results_queue
    
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
            
            results_queue = mp.Queue()
            processes = []

            for i in range(num_processes):
                p = mp.Process(target=calculate_uncertainty_by_grounding_worker, args=(args, chunks[i], results_queue))
                p.start()
                processes.append(p)

            results = {}
            for p in processes:
                p.join()

            while not results_queue.empty():
                result = results_queue.get()
                for idx, responses in result.items():
                    if idx not in results:
                        results[idx] = []
                    results[idx].extend(responses)

            # Final save of all results
            save_results_if_not_empty(results, args.uncertainty_path)
        
        except Exception as e:
            logging.error(f"An unexpected error occurred in calculate_uncertainty_by_grounding: {e}")
    else:
        # Non-parallel execution
        try:
            keys = list(args.responses.keys())[:args.samples]
            results = {}
            results_queue = queue.Queue()
            
            # Directly process all keys using the worker function
            calculate_uncertainty_by_grounding_worker(args, keys, results_queue)

            # Collect results from the results_queue
            while not results_queue.empty():
                result = results_queue.get()
                results.update(result)
    
            # Final save of all results
            save_results_if_not_empty(results, args.uncertainty_path)
        
        except Exception as e:
            logging.error(f"An unexpected error occurred in calculate_uncertainty_by_grounding: {e}")