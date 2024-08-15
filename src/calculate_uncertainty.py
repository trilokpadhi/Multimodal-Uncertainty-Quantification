import torch
import torch.multiprocessing as mp
from tqdm import tqdm 
from .graph import extract_entities_and_relationships_llama3, extract_triples_from_llama3_json, extract_triples_scene_graph
from .utils import extract_json_from_text, ModelArgs
from sentence_transformers import util
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

def calculate_uncertainty_by_self_consistency():
    pass

def preprocess_triple(triple):
    """Normalize the triple by converting to lowercase and joining the elements into a single string."""
    try:
        return ' '.join(word.lower() for word in triple)
    except Exception as e:
        print(f"Error in preprocess_triple: {e}")
        return None

def calculate_confidence_metric(model_args, ground_truth_set, output_set, k=3):
    """Calculate the confidence metric based on the top-K similarities between two sets of triples."""
    
    try:
        ground_truth_list = [preprocess_triple(triple) for triple in ground_truth_set if triple]
        output_list = [preprocess_triple(triple) for triple in output_set if triple]
        
        if not ground_truth_list or not output_list:
            print("Error: One of the input lists is empty or all triples failed to preprocess.")
            return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings_gt = model_args.sentence_transformer.encode(ground_truth_list, convert_to_tensor=True).to(device)
        embeddings_out = model_args.sentence_transformer.encode(output_list, convert_to_tensor=True).to(device)
        
        if embeddings_gt.shape[0] == 0 or embeddings_out.shape[0] == 0:
            print("Error: One of the input tensors is empty.")
            return None
        
        similarity_matrix = util.pytorch_cos_sim(embeddings_gt, embeddings_out).cpu().numpy()
        sorted_similarities = np.sort(similarity_matrix.flatten())[::-1]
        top_k_similarities = sorted_similarities[:k]
        confidence_metric = np.mean(top_k_similarities)
        
        return confidence_metric.item()
    
    except ValueError as ve:
        print(f"ValueError in calculate_confidence_metric: {ve}")
        return None
    
    except TypeError as te:
        print(f"TypeError in calculate_confidence_metric: {te}")
        return None
    
    except Exception as e:
        print(f"An unexpected error occurred in calculate_confidence_metric: {e}")
        return None

def save_intermediate_results(results, save_path):
    """Save results to the specified path."""
    try:
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                existing_results = json.load(f)
        else:
            existing_results = {}

        existing_results.update(results)

        with open(save_path, 'w') as f:
            json.dump(existing_results, f)
        
        print(f"Intermediate results saved to {save_path}.")
    
    except Exception as e:
        print(f"Error saving intermediate results: {e}")

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
        save_interval = 10  # Save every 10 responses

        for i, idx in enumerate(tqdm(keys, desc=f"Process {mp.current_process().pid}")):
            responses = args.responses.get(idx, {}).get('responses', [])
            for j, response in enumerate(responses):
                try:
                    explanation = response.get('explanation')
                    if not explanation:
                        print(f"No explanation found for response {j}. Skipping...")
                        continue

                    llama3_response_graph = extract_entities_and_relationships_llama3(model_args, explanation)
                    llama3_response_graph_jsonified = extract_json_from_text(llama3_response_graph)
                    response_triples = extract_triples_from_llama3_json(llama3_response_graph_jsonified)
                    
                    image_id = args.gqa_data.get(idx, {}).get('imageId')
                    scene_graph = args.scene_graphs_data.get(image_id)
                    
                    if not scene_graph:
                        print(f"No scene graph found for image ID {image_id}. Skipping...")
                        continue

                    scene_graph_triples = extract_triples_scene_graph(scene_graph)
                    confidence_metric = calculate_confidence_metric(model_args, scene_graph_triples, response_triples, len(response_triples))
                    
                    if confidence_metric is not None:
                        response['confidence_metric'] = confidence_metric
                
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error for response {j}: {e}")
                except Exception as e:
                    print(f"Error processing response {j}: {e}")

            if idx not in results:
                results[idx] = []
            results[idx].extend(responses)

            # Save results at every interval
            if (i + 1) % save_interval == 0:
                save_intermediate_results(results, args.uncertainty_path)
                results.clear()  # Clear the results after saving to avoid duplication

        # Save any remaining results after the loop
        if results:
            save_intermediate_results(results, args.uncertainty_path)
            results_queue.put(results)
    
    except Exception as e:
        print(f"An unexpected error occurred in process {mp.current_process().pid}: {e}")

def calculate_uncertainty_by_grounding(args):
    try:
        keys = list(args.responses.keys())[:500]
        num_processes = min(torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(), 6)
        chunk_size = max(1, len(keys) // num_processes)
        chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
        print(f"Divided into {len(chunks)} chunks")
        
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
        save_intermediate_results(results, args.uncertainty_path)
    
    except Exception as e:
        print(f"An unexpected error occurred in calculate_uncertainty_by_grounding: {e}")