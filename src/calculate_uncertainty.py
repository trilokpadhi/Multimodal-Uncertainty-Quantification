import torch
import torch.multiprocessing as mp
from tqdm import tqdm 
from .graph import extract_entities_and_relationships_llama3, extract_triples_from_llama3_json, extract_triples_scene_graph
from .utils import extract_json_from_text, ModelArgs
from sentence_transformers import util
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def calculate_uncertainty_by_self_consistency():
    pass


def preprocess_triple(triple):
    """Normalize the triple by converting to lowercase and joining the elements into a single string."""
    return ' '.join(word.lower() for word in triple)

def calculate_confidence_metric(model_args, ground_truth_set, output_set, k=3):
    """Calculate the confidence metric based on the top-K similarities between two sets of triples."""
    
    # Preprocess triples and convert to strings    
    ground_truth_list = [preprocess_triple(triple) for triple in ground_truth_set]
    print
    output_list = [preprocess_triple(triple) for triple in output_set]
    
    # Batch encode all triples in both sets
    embeddings_gt = model_args.sentence_transformer.encode(ground_truth_list, convert_to_tensor=True)
    embeddings_out = model_args.sentence_transformer.encode(output_list, convert_to_tensor=True)
    
    # Compute the cosine similarity matrix in a vectorized manner
    similarity_matrix = util.pytorch_cos_sim(embeddings_gt, embeddings_out).cpu().numpy()
    
    # Flatten the matrix and sort similarities in descending order
    sorted_similarities = np.sort(similarity_matrix.flatten())[::-1]
    
    # print('sorted_similarities', sorted_similarities)
    # Take the top-K similarities
    top_k_similarities = sorted_similarities[:k]
    
    # Calculate the confidence metric as the average of the top-K similarities
    confidence_metric = np.mean(top_k_similarities)
    
    return confidence_metric.item() # Convert to Python float for JSON serialization

           
def calculate_uncertainty_by_grounding_worker(args, keys, results_queue):
    try:
        # For Graph Extract - Llama 3
        args.llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        args.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model_id)
        args.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        model_args = ModelArgs(args)
        results = {}
        for idx in tqdm(keys, desc=f"Process {mp.current_process().pid}"):
            responses = args.responses[idx]['responses']
            # grounding = response['grounding']
            for i in range(len(responses)):
                explanation = responses[i]['explanation']
                llama3_response_graph = extract_entities_and_relationships_llama3(model_args, explanation)
                llama3_response_graph_jsonified = extract_json_from_text(llama3_response_graph)
                response_triples = extract_triples_from_llama3_json(llama3_response_graph_jsonified)
                image_id = args.gqa_data[idx]['imageId']
                scene_graph = args.scene_graphs_data[image_id]
                scene_graph_triples = extract_triples_scene_graph(scene_graph) # ground truth triples
                confidence_metric = calculate_confidence_metric(model_args, scene_graph_triples, response_triples, len(response_triples))
                responses[i]['confidence_metric'] = confidence_metric
            results[idx] = responses
        print(f"Process {mp.current_process().pid} finished.")
        results_queue.put(results)
    except Exception as e:
        print(f"Error in process {mp.current_process().pid}: {e}")

def calculate_uncertainty_by_grounding(args):
    keys = list(args.responses.keys())
    num_processes = min(torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(), 6)

    # num_processes = 1
    chunk_size = len(keys) // num_processes
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
        if p.is_alive():
            p.join()
        if not results_queue.empty():
            results.update(results_queue.get())
        
    return results  
    
