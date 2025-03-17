import torch
import pandas as pd
import math
import ast
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.multiprocessing import Pool, set_start_method, Manager

# Ensure proper multiprocessing start method
set_start_method("spawn", force=True)

# Load model & tokenizer once in the main process
model_name = "microsoft/deberta-base-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define GPU count
num_gpus = min(4, torch.cuda.device_count())  # Use up to 4 GPUs
print(f"Using {num_gpus} GPUs")


def are_semantically_equivalent(respA, respB, question, tokenizer, model, device="cpu"):
    """Checks if respA and respB are mutually entailing given the question context."""
    premise_forward = f"{respA}"
    hypothesis_fwd = f"{respB}"
    premise_backward = f"{respB}"
    hypothesis_bwd = f"{respA}"

    # Forward pass
    inputs_f = tokenizer.encode_plus(premise_forward, hypothesis_fwd, return_tensors="pt", truncation=True)
    inputs_f = {k: v.to(device) for k, v in inputs_f.items()}
    with torch.no_grad():
        logits_f = model(**inputs_f).logits
    label_f = torch.argmax(logits_f, dim=1).item()

    # Backward pass
    inputs_b = tokenizer.encode_plus(premise_backward, hypothesis_bwd, return_tensors="pt", truncation=True)
    inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
    with torch.no_grad():
        logits_b = model(**inputs_b).logits
    label_b = torch.argmax(logits_b, dim=1).item()

    return (label_f == 2) and (label_b == 2)


def cal_entropy_grounding_semantic(row, tokenizer, model, device="cpu"):
    """Compute grounding entropy per row."""
    responses = ast.literal_eval(row["responses"])
    # groundings = ast.literal_eval(row["grounding_llama32_70b_processed"]) 
    groundings = ast.literal_eval(row["grounding_llama32_11b_processed"])
    question = row["question"]
    n = len(responses)

    if n == 0 or len(groundings) != n:
        return 0.0

    adjacency = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if are_semantically_equivalent(responses[i], responses[j], question, tokenizer, model, device=device):
                adjacency[i][j] = adjacency[j][i] = True

    visited = [False] * n
    clusters = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            cluster_members = []
            while stack:
                node = stack.pop()
                cluster_members.append(node)
                for nb in range(n):
                    if adjacency[node][nb] and not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            clusters.append(cluster_members)

    grounding_score_cluster_wise_with_entropy = []
    grounding_score_cluster_wise = []
    for c in clusters:
        cluster_size = len(c)
        yes_count = sum(1 for idx in c if groundings[idx].lower() == "yes")
        p = yes_count / cluster_size

        h_cluster = 0.0 if p in [0, 1] else - (p * math.log2(p) + (1 - p) * math.log2(1 - p))

        grounding_score = yes_count / cluster_size
        # grounding_score_with_entropy.append(grounding_score * h_cluster) 
        grounding_score_cluster_wise_with_entropy.append(grounding_score * ( 1 - h_cluster)) # changed to 1 - h_cluster as it should be confidence in the grounding, and not entropy * grounding score
        grounding_score_cluster_wise.append(grounding_score)
        
    grounding_score_cluster_wise_with_entropy_avg = sum(grounding_score_cluster_wise_with_entropy) / len(grounding_score_cluster_wise_with_entropy) if grounding_score_cluster_wise_with_entropy else 0.0
    
    grounding_score_cluster_wise_avg = sum(grounding_score_cluster_wise) / len(grounding_score_cluster_wise) if grounding_score_cluster_wise else 0.0
    # return sum(grounding_score_with_entropy) / len(grounding_score_with_entropy) if grounding_score_with_entropy else 0.0
    # return sum(grounding_score_cluster_wise) / len(grounding_score_cluster_wise) if grounding_score_cluster_wise else 0.0
    
    return grounding_score_cluster_wise_with_entropy_avg, grounding_score_cluster_wise_avg

def process_batch(df_subset, gpu_id, progress_queue):
    """
    Worker function for parallel processing on a specific GPU.
    Loads model on the assigned GPU and processes the dataframe subset.
    """
    device = f"cuda:{gpu_id}"
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    # results = [] 
    results_with_entropy = []
    results = []
    for _, row in df_subset.iterrows():
        val_with_entropy, val = cal_entropy_grounding_semantic(row, tokenizer, model, device=device)
        
        results_with_entropy.append(val_with_entropy)
        results.append(val)
        
        # Update progress bar
        progress_queue.put(1)

    df_subset["grounding_internal_llama32V_with_cluster_wise_with_entropy"] = results_with_entropy
    df_subset["grounding_internal_llama32V_cluster_wise"] = results
    return df_subset


if __name__ == "__main__":
    tqdm.pandas()

    # Load dataframe
    # df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/vqa/merged_with_grounding_processed.csv")
    df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv")

    # Split dataframe into equal parts for each GPU
    df_splits = [df.iloc[i::num_gpus] for i in range(num_gpus)]

    # Create a shared manager and progress queue
    manager = Manager()
    progress_queue = manager.Queue()

    # Start multiprocessing
    futures = []
    total_rows = len(df)
    
    with Pool(num_gpus) as pool:
        results = []
        for i in range(num_gpus):
            future = pool.apply_async(process_batch, (df_splits[i], i, progress_queue))
            futures.append(future)

        # Show global progress bar
        with tqdm(total=total_rows, desc="Processing Rows") as pbar:
            processed = 0
            while processed < total_rows:
                while not progress_queue.empty():
                    progress_queue.get()
                    processed += 1
                    pbar.update(1)
                time.sleep(0.1)  # Adjust sleep time for efficiency

        # Collect results from workers
        for future in futures:
            results.append(future.get())

    # Combine processed data
    df_parallel = pd.concat(results)

    # Save the output
    df_parallel.to_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_internal_g_feb20.csv", index=False)

    print("✅ Processing complete! Saved results to processed_results.csv") 