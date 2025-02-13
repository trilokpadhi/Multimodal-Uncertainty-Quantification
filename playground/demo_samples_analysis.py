# # # import os
# # # import pickle
# # # import pandas as pd
# # # import torch
# # # import numpy as np
# # # from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # # from sentence_transformers import SentenceTransformer
# # # from sklearn.metrics.pairwise import cosine_similarity
# # # from tqdm import tqdm

# # # # Configuration
# # # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # RESPONSE_DIR = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/explanations'
# # # GROUNDING_DIR = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_llama32'

# # # # Load models once
# # # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
# # # deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(DEVICE)
# # # st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # # def batch_entailment_accuracy(responses, answer, tokenizer, model, device):
# # #     """Calculate accuracy using batched entailment checking"""
# # #     if not responses:
# # #         return 0.0
    
# # #     # Batch processing of all responses
# # #     inputs = tokenizer(
# # #         responses,
# # #         [answer] * len(responses),
# # #         padding=True,
# # #         truncation=True,
# # #         max_length=512,
# # #         return_tensors="pt"
# # #     ).to(device)
    
# # #     with torch.no_grad():
# # #         logits = model(**inputs).logits
# # #         preds = torch.argmax(logits, dim=1)
# # #         return (preds == 2).sum().item() / len(responses)

# # # def process_grounding_file(gr_path):
# # #     """Process a single grounding file and return metrics"""
# # #     with open(gr_path, 'rb') as f:
# # #         gr = pickle.load(f)
    
# # #     # Extract question data
# # #     qid = gr['qids'][0]
# # #     question = gr['questions'][0]
# # #     answer = gr['answers'][0]
    
# # #     # Process responses
# # #     responses = [gr[f'response_{k}']['decoded_outputs'].strip() for k in range(20)]
# # #     ground_responses = [gr[f'response_{k}']['llama_32_response'] for k in range(20)]
    
# # #     # Calculate grounding score
# # #     grounding_score = sum(1 for r in ground_responses if 'yes' in r.lower()) / len(ground_responses)
    
# # #     # Calculate consistency
# # #     embeddings = st_model.encode(responses)
# # #     sim_matrix = cosine_similarity(embeddings)
# # #     np.fill_diagonal(sim_matrix, 0)
# # #     consistency = sim_matrix.sum() / (sim_matrix.size - len(responses))
    
# # #     # Calculate accuracy
# # #     accuracy = batch_entailment_accuracy(
# # #         responses, answer, tokenizer, deberta_model, DEVICE
# # #     )
    
# # #     return {
# # #         'question_id': qid,
# # #         'question': question,
# # #         'answer': answer,
# # #         'model_responses': responses,
# # #         'self_consistency_score': consistency,
# # #         'grounding_score': grounding_score,
# # #         'accuracy': accuracy
# # #     }

# # # def main():
# # #     # Get all grounding files
# # #     gr_files = [os.path.join(GROUNDING_DIR, f) for f in os.listdir(GROUNDING_DIR)]
    
# # #     # Process all files with progress bar
# # #     results = []
# # #     for gr_path in tqdm(gr_files, desc="Processing grounding files"):
# # #         try:
# # #             result = process_grounding_file(gr_path)
# # #             results.append(result)
            
# # #             # Print results (optional)
# # #             print(f"\nQuestion ID: {result['question_id']}")
# # #             print(f"Question: {result['question']}")
# # #             print(f"Answer: {result['answer']}")
# # #             print(f"Consistency: {result['self_consistency_score']:.2f}")
# # #             print(f"Grounding: {result['grounding_score']:.2f}")
# # #             print(f"Accuracy: {result['accuracy']:.2f}")
# # #         except Exception as e:
# # #             print(f"Error processing {gr_path}: {str(e)}")
    
# # #     # Create DataFrame
# # #     df = pd.DataFrame(results)[[
# # #         'question_id', 'question', 'answer', 'model_responses',
# # #         'self_consistency_score', 'grounding_score', 'accuracy'
# # #     ]]
    
# # #     # Print summary
# # #     print(f"\nFinal DataFrame Shape: {df.shape}")
# # #     print(f"Average Accuracy: {df.accuracy.mean():.2f}")
# # #     print(f"Average Consistency: {df.self_consistency_score.mean():.2f}")
# # #     print(f"Average Grounding: {df.grounding_score.mean():.2f}")
    
# # #     return df

# # # if __name__ == "__main__":
# # #     df = main()
# # #     df.head()

# # import os
# # import pickle
# # import pandas as pd
# # import torch
# # import numpy as np
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity
# # from tqdm import tqdm
# # from multiprocessing import Pool
# # import torch.nn as nn

# # # Configuration
# # NUM_GPUS = 4
# # RESPONSE_DIR = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/explanations'
# # GROUNDING_DIR = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_llama32'

# # class ParallelModelWrapper:
# #     def __init__(self, gpu_id):
# #         self.device = torch.device(f'cuda:{gpu_id}')
# #         self.gpu_id = gpu_id
        
# #         # Load models with mixed precision
# #         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
# #         self.deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
# #         self.deberta_model = nn.DataParallel(self.deberta_model).half().to(self.device)
        
# #         self.st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# #         self.st_model = self.st_model.half().to(self.device)

# #     def batch_entailment(self, responses, answer):
# #         """Batch entailment with mixed precision"""
# #         if not responses:
# #             return 0.0
            
# #         inputs = self.tokenizer(
# #             responses,
# #             [answer]*len(responses),
# #             padding=True,
# #             truncation=True,
# #             max_length=512,
# #             return_tensors="pt"
# #         ).to(self.device)
        
# #         with torch.cuda.amp.autocast(), torch.no_grad():
# #             logits = self.deberta_model(**inputs).logits
# #             preds = torch.argmax(logits, dim=1)
# #             return (preds == 2).sum().item() / len(responses)

# #     def calculate_embeddings(self, texts):
# #         """Batch embeddings with mixed precision"""
# #         with torch.cuda.amp.autocast(), torch.no_grad():
# #             return self.st_model.encode(texts, convert_to_tensor=True, device=self.device)

# # def process_file(args):
# #     """Process a single file on assigned GPU"""
# #     gpu_id, gr_path = args
# #     torch.cuda.set_device(gpu_id)
# #     model_wrapper = ParallelModelWrapper(gpu_id)
    
# #     try:
# #         with open(gr_path, 'rb') as f:
# #             gr = pickle.load(f)
            
# #         qid = gr['qids'][0]
# #         question = gr['questions'][0]
# #         answer = gr['answers'][0]
        
# #         responses = [gr[f'response_{k}']['decoded_outputs'].strip() for k in range(20)]
# #         ground_responses = [gr[f'response_{k}']['llama_32_response'] for k in range(20)]
        
# #         # Parallel compute grounding score
# #         grounding_score = sum(1 for r in ground_responses if 'yes' in r.lower()) / 20
        
# #         # Parallel compute embeddings and consistency
# #         embeddings = model_wrapper.calculate_embeddings(responses).cpu().numpy()
# #         sim_matrix = cosine_similarity(embeddings)
# #         np.fill_diagonal(sim_matrix, 0)
# #         consistency = sim_matrix.sum() / (sim_matrix.size - 20)
        
# #         # Batch entailment
# #         accuracy = model_wrapper.batch_entailment(responses, answer)
        
# #         return {
# #             'question_id': qid,
# #             'question': question,
# #             'answer': answer,
# #             'model_responses': responses,
# #             'self_consistency_score': consistency,
# #             'grounding_score': grounding_score,
# #             'accuracy': accuracy
# #         }
# #     except Exception as e:
# #         print(f"Error processing {gr_path} on GPU {gpu_id}: {str(e)}")
# #         return None

# # def main():
# #     # Get all grounding files
# #     gr_files = [os.path.join(GROUNDING_DIR, f) for f in os.listdir(GROUNDING_DIR)]
    
# #     # Create GPU assignments (round-robin)
# #     tasks = [(i % NUM_GPUS, f) for i, f in enumerate(gr_files)]
    
# #     # Process files in parallel using multiprocessing
# #     with Pool(processes=NUM_GPUS) as pool:
# #         results = list(tqdm(pool.imap(process_file, tasks), total=len(gr_files), desc="Processing files"))
    
# #     # Filter out failed results
# #     valid_results = [r for r in results if r is not None]
    
# #     # Create DataFrame
# #     df = pd.DataFrame(valid_results)[[
# #         'question_id', 'question', 'answer', 'model_responses',
# #         'self_consistency_score', 'grounding_score', 'accuracy'
# #     ]]
    
# #     # Print summary
# #     print(f"\nProcessed {len(valid_results)}/{len(gr_files)} files")
# #     print(f"Average Accuracy: {df.accuracy.mean():.2f}")
# #     print(f"Average Consistency: {df.self_consistency_score.mean():.2f}")
# #     print(f"Average Grounding: {df.grounding_score.mean():.2f}")
    
# #     return df

# # if __name__ == "__main__":
# #     df = main()
# #     df.head()


# import os
# import pickle
# import pandas as pd
# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm

# import math
# from torch.multiprocessing import Process, Manager

# # Global configuration
# RESPONSE_DIR = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/explanations'
# GROUNDING_DIR = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_llama32'

# ###############################################################################
# #                      Functions and Model Loading                            #
# ###############################################################################

# def load_models(device_id=0):
#     """
#     Loads all models onto the specified GPU (device_id).
#     Returns tokenizer, DeBERTa model, and SentenceTransformer model.
#     """
#     print(f"[Worker {device_id}] Loading models on cuda:{device_id}")
#     device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
#     # Load tokenizer and DeBERTa
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
#     deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)
    
#     # Load SentenceTransformer
#     st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    
#     return tokenizer, deberta_model, st_model

# def batch_entailment_accuracy(responses, answer, tokenizer, model, device):
#     """Calculate accuracy using batched entailment checking."""
#     if not responses:
#         return 0.0
    
#     inputs = tokenizer(
#         responses,
#         [answer] * len(responses),
#         padding=True,
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     ).to(device)
    
#     with torch.no_grad():
#         logits = model(**inputs).logits
#         preds = torch.argmax(logits, dim=1)
#         # 2 corresponds to "entailment" class in DeBERTa MNLI
#         return (preds == 2).sum().item() / len(responses)

# def process_grounding_file(gr_path, tokenizer, deberta_model, st_model, device):
#     """Process a single grounding file and return metrics."""
#     with open(gr_path, 'rb') as f:
#         gr = pickle.load(f)
    
#     # Extract question data
#     qid = gr['qids'][0]
#     question = gr['questions'][0]
#     answer = gr['answers'][0]
    
#     # Process responses
#     responses = [gr[f'response_{k}']['decoded_outputs'].strip() for k in range(20)]
#     ground_responses = [gr[f'response_{k}']['llama_32_response'] for k in range(20)]
    
#     # Calculate grounding score
#     grounding_score = sum('yes' in (r or '').lower() for r in ground_responses) / len(ground_responses)
    
#     # Calculate consistency
#     embeddings = st_model.encode(responses)
#     sim_matrix = cosine_similarity(embeddings)
#     np.fill_diagonal(sim_matrix, 0)
#     consistency = sim_matrix.sum() / (sim_matrix.size - len(responses))
    
#     # Calculate accuracy
#     accuracy = batch_entailment_accuracy(responses, answer, tokenizer, deberta_model, device)
    
#     return {
#         'question_id': qid,
#         'question': question,
#         'answer': answer,
#         'model_responses': responses,
#         'self_consistency_score': consistency,
#         'grounding_score': grounding_score,
#         'accuracy': accuracy
#     }

# ###############################################################################
# #                       Multiprocessing Worker Function                       #
# ###############################################################################

# def worker_process(rank, file_subset, return_dict):
#     """
#     Each worker:
#       1. Loads models onto GPU:rank (if available).
#       2. Processes its subset of files.
#       3. Stores results in the shared dictionary 'return_dict'.
#     """
#     # Pin this process to the appropriate GPU
#     device_id = rank
    
#     # Load models
#     tokenizer, deberta_model, st_model = load_models(device_id)
#     device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
#     worker_results = []
    
#     for gr_path in tqdm(file_subset, desc=f"[Worker {device_id}] Processing"):
#         try:
#             result = process_grounding_file(gr_path, tokenizer, deberta_model, st_model, device)
#             worker_results.append(result)
#         except Exception as e:
#             print(f"[Worker {device_id}] Error processing {gr_path}: {e}")
    
#     # Save the results in the shared dictionary (index by rank)
#     return_dict[rank] = worker_results

# ###############################################################################
# #                              Main Function                                  #
# ###############################################################################

# def main(num_gpus=4):
#     """
#     1. Collect all grounding files.
#     2. Split into chunks for each GPU.
#     3. Spawn processes.
#     4. Combine results into a single DataFrame.
#     """
    
#     # Get all grounding files
#     all_files = [os.path.join(GROUNDING_DIR, f) for f in os.listdir(GROUNDING_DIR) 
#                  if os.path.isfile(os.path.join(GROUNDING_DIR, f))]
    
#     # Sort or shuffle if desired (for balanced distribution)
#     all_files.sort()
    
#     # Split files into chunks for each worker
#     chunk_size = math.ceil(len(all_files) / num_gpus)
#     file_subsets = [all_files[i*chunk_size : (i+1)*chunk_size] for i in range(num_gpus)]
    
#     # Manager to hold results from all workers
#     manager = Manager()
#     return_dict = manager.dict()
    
#     # Spawn processes
#     processes = []
#     for gpu_id in range(num_gpus):
#         subset = file_subsets[gpu_id]
#         p = Process(target=worker_process, args=(gpu_id, subset, return_dict))
#         p.start()
#         processes.append(p)
    
#     # Join processes
#     for p in processes:
#         p.join()
    
#     # Collect final results
#     all_results = []
#     for gpu_id in range(num_gpus):
#         all_results.extend(return_dict.get(gpu_id, []))
    
#     # Create DataFrame
#     df = pd.DataFrame(all_results)[[
#         'question_id', 'question', 'answer', 'model_responses',
#         'self_consistency_score', 'grounding_score', 'accuracy'
#     ]]
    
#     # Print summary
#     print(f"\nFinal DataFrame Shape: {df.shape}")
#     print(f"Average Accuracy: {df['accuracy'].mean():.2f}")
#     print(f"Average Consistency: {df['self_consistency_score'].mean():.2f}")
#     print(f"Average Grounding: {df['grounding_score'].mean():.2f}")
    
#     return df

# if __name__ == "__main__":
#     # Adjust num_gpus based on your system
#     final_df = main(num_gpus=4)
#     print(final_df.head())

import os
import pickle
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from torch.multiprocessing import Process, Manager

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Global config
RESPONSE_DIR = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/explanations'
GROUNDING_DIR = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_llama32'

###############################################################################
#                         Model Loading and Functions                         #
###############################################################################

def load_models(device_id=0):
    """
    Loads the tokenizer, DeBERTa model, and SentenceTransformer model onto `cuda:device_id`.
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[Worker {device_id}] Loading models on {device}")

    # Load tokenizer & DeBERTa (MNLI)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
    deberta_model.to(device)

    # Load SentenceTransformer
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    return tokenizer, deberta_model, st_model, device

def batch_entailment_accuracy(responses, answer, tokenizer, model, device):
    """
    Calculate accuracy (entailment) using batched approach with DeBERTa.
    Class index 2 = "entailment" in the MNLI head.
    """
    if not responses:
        return 0.0
    
    inputs = tokenizer(
        responses,
        [answer] * len(responses),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1)
        return (preds == 2).sum().item() / len(responses)

def process_grounding_file(gr_path, tokenizer, deberta_model, st_model, device):
    """
    Process a single grounding file to compute:
      - self-consistency
      - grounding score
      - accuracy via MNLI
    """
    with open(gr_path, 'rb') as f:
        gr = pickle.load(f)

    # Basic question/answer
    qid = gr['qids'][0]
    question = gr['questions'][0]
    answer = gr['answers'][0]
    image_id = gr['img_ids'][0]
    image_path = gr['image_paths'][0]
    
    # Model responses
    responses = [gr[f'response_{k}']['decoded_outputs'].strip() for k in range(20)]
    ground_responses = [gr[f'response_{k}']['llama_32_response'] for k in range(20)]

    # Grounding score (# of "yes" among 20 responses)
    grounding_score = sum('yes' in (r or '').lower() for r in ground_responses) / len(ground_responses)

    # Self-consistency: average similarity among all responses
    embeddings = st_model.encode(responses)  # returns a CPU or GPU tensor depending on your config
    # If st_model.encode returns a GPU tensor, convert to CPU for sklearn cosine_similarity
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()

    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    # sum of all similarities / (total pairs)
    consistency = sim_matrix.sum() / (sim_matrix.size - len(responses))

    # Accuracy (entailment check)
    accuracy = batch_entailment_accuracy(responses, answer, tokenizer, deberta_model, device)

    return {
        'question_id': qid,
        'question': question,
        'image_id': image_id,
        'image_path': image_path,
        'answer': answer,
        'model_responses': responses,
        'self_consistency_score': consistency,
        'grounding_score': grounding_score,
        'accuracy': accuracy
    }

###############################################################################
#                        Multiprocessing Worker Function                      #
###############################################################################

def worker_process(rank, file_subset, return_dict):
    """
    Each worker will:
    1. Load models on GPU rank (if available).
    2. Process assigned file_subset.
    3. Store results in `return_dict[rank]`.
    """
    # Load models for this worker
    tokenizer, deberta_model, st_model, device = load_models(rank)

    worker_results = []
    
    for gr_path in tqdm(file_subset, desc=f"[Worker {rank}] Processing"):
        try:
            result = process_grounding_file(gr_path, tokenizer, deberta_model, st_model, device)
            worker_results.append(result)
        except Exception as e:
            print(f"[Worker {rank}] Error processing {gr_path}: {e}")

    # Store results in shared dictionary
    return_dict[rank] = worker_results


###############################################################################
#                               Main Function                                 #
###############################################################################

def run_inference(num_gpus=4):
    """
    1. Collect all files in GROUNDING_DIR.
    2. Divide them into subsets for each GPU.
    3. Start processes with 'spawn' to avoid re-initializing CUDA.
    4. Collate and return results in a DataFrame.
    """
    # Gather all grounding files
    all_files = [
        os.path.join(GROUNDING_DIR, f) 
        for f in os.listdir(GROUNDING_DIR) 
        if os.path.isfile(os.path.join(GROUNDING_DIR, f))
    ]
    all_files.sort()

    # Split workload into num_gpus chunks
    chunk_size = math.ceil(len(all_files) / num_gpus)
    file_subsets = [all_files[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]

    # Manager for shared data
    manager = Manager()
    return_dict = manager.dict()

    # Spawn processes
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=worker_process,
            args=(gpu_id, file_subsets[gpu_id], return_dict)
        )
        p.start()
        processes.append(p)

    # Join processes
    for p in processes:
        p.join()

    # Gather all results
    all_results = []
    for gpu_id in range(num_gpus):
        worker_data = return_dict.get(gpu_id, [])
        all_results.extend(worker_data)

    # Create and return DataFrame
    if not all_results:
        print("No results collected. Check for errors.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results, columns=[
        'question_id', 'question', 'answer', 'model_responses', 'image_id', 'image_path',
        'self_consistency_score', 'grounding_score', 'accuracy'
    ])

    print(f"\nFinal DataFrame Shape: {df.shape}")
    print(f"Average Accuracy: {df['accuracy'].mean():.2f}")
    print(f"Average Consistency: {df['self_consistency_score'].mean():.2f}")
    print(f"Average Grounding: {df['grounding_score'].mean():.2f}")

    return df


###############################################################################
#                           Entry Point (spawn)                               #
###############################################################################

if __name__ == "__main__":
    import multiprocessing
    # Use 'spawn' to avoid "Cannot re-initialize CUDA in forked subprocess"
    multiprocessing.set_start_method("spawn", force=True)

    final_df = run_inference(num_gpus=4)
    # lets write the final_df to a csv file
    final_df.to_csv('final_df2.csv', index=False)
    print(final_df.head())