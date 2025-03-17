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

# Load tokenizer once in the main process
model_name = "microsoft/deberta-base-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get number of available GPUs
num_gpus = min(torch.cuda.device_count(), 4)  # Use up to 4 GPUs
print(f"✅ Detected {num_gpus} GPU(s). Using {num_gpus} for processing.")

def process_batch(df_subset, gpu_id, progress_queue):
    """Worker function for processing data in parallel."""
    device = f"cuda:{gpu_id}" if gpu_id < torch.cuda.device_count() else "cpu"
    
    print(f"🚀 Worker {gpu_id}: Running on {device}")


    results = []
    results_with_entropy = []
    for _, row in df_subset.iterrows():
        try:
            val, val_with_entropy = calc_external_grounding_uncertainty(row)
            results.append(val)
            results_with_entropy.append(val_with_entropy)
        except Exception as e:
            print(f"❌ Error processing row: {str(e)}")
            results.append(None)  # Prevent complete failure
        
        progress_queue.put(1)  # Update progress

    df_subset = df_subset.copy()
    df_subset["grounding_external_with_entropy"] = results_with_entropy # grounding_internal_llama32V_with_cluster_wise_with_entropy
    df_subset["grounding_external"] = results
    return df_subset

def calc_external_grounding_uncertainty(row):
    """
    Computes uncertainty score based on multiple grounding models.
    """

    try:
        modelA = ast.literal_eval(row["grounding_llama32_11b_processed"])
        modelB = ast.literal_eval(row["grounding_llama32_70b_processed"])
        modelC = ast.literal_eval(row["grounding_qwen_vl_processed"])
        # modelC = ast.literal_eval(row["grounding_qwen_vl_25_processed"])
    except:
        modelA = row["grounding_llama32_11b_processed"]
        modelB = row["grounding_qwen_vl_25_processed"]
        modelB = row["grounding_qwen_vl_processed"]

    if not isinstance(modelA, list) or not isinstance(modelB, list) or not isinstance(modelC, list):
        return 0.0  # Ensure inputs are lists

    num_responses = min(len(modelA), len(modelB), len(modelC))

    if num_responses == 0:
        return 0.0  # No responses

    total_score = 0.0
    total_score_with_entropy = 0.0
    for i in range(num_responses):
        answers = [
            str(modelA[i]).lower().strip(),
            str(modelB[i]).lower().strip(),
            str(modelC[i]).lower().strip()
        ]
        yes_count = sum(ans == "yes" for ans in answers)
        p = yes_count / 3.0

        H = 0.0 if p in [0, 1] else - (p * math.log2(p) + (1 - p) * math.log2(1 - p))
        # resp_score = H * p 
        resp_score = (1- H) * p # changed to 1-H as it should be confidence* confidence score, not entropy * confidence score
        # resp_score = p
        total_score_with_entropy += resp_score
        total_score += p

    total_score_with_entropy_avg = total_score_with_entropy / num_responses
    total_score_avg = total_score / num_responses
    
    return total_score_avg, total_score_with_entropy_avg

if __name__ == "__main__":
    tqdm.pandas()
    df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv")

    # Ensure num_gpus does not exceed number of rows
    num_gpus = min(num_gpus, len(df))  # Prevent assigning too many GPUs
    df_splits = [df.iloc[i::num_gpus] for i in range(num_gpus)]  # Split safely

    # Create a shared manager and progress queue
    manager = Manager()
    progress_queue = manager.Queue()

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
                time.sleep(0.1)

        # Collect results safely
        for future in futures:
            try:
                results.append(future.get())  # Capture errors
            except Exception as e:
                print(f"❌ Error in worker: {str(e)}")

    df_parallel = pd.concat(results, ignore_index=True)  # Prevent duplicate index issues
    df_parallel.to_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_feb_20.csv", index=False)
    print("✅ Processing complete!")