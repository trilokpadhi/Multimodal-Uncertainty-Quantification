# # # import torch
# # # import pandas as pd
# # # import math
# # # import ast
# # # import time
# # # from tqdm import tqdm
# # # from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # # from torch.multiprocessing import Pool, set_start_method, Manager
# # # import math
# # # import ast


# # # # Ensure proper multiprocessing start method
# # # set_start_method("spawn", force=True)

# # # # Load model & tokenizer once in the main process
# # # model_name = "microsoft/deberta-base-mnli"
# # # tokenizer = AutoTokenizer.from_pretrained(model_name)

# # # # Define GPU count
# # # num_gpus = min(4, torch.cuda.device_count())  # Use up to 4 GPUs
# # # print(f"Using {num_gpus} GPUs")


# # # def are_semantically_equivalent(respA, respB, question, tokenizer, model, device="cpu"):
# # #     """Checks if respA and respB are mutually entailing given the question context."""
# # #     premise_forward = f"{respA}"
# # #     hypothesis_fwd = f"{respB}"
# # #     premise_backward = f"{respB}"
# # #     hypothesis_bwd = f"{respA}"

# # #     # Forward pass
# # #     inputs_f = tokenizer.encode_plus(premise_forward, hypothesis_fwd, return_tensors="pt", truncation=True)
# # #     inputs_f = {k: v.to(device) for k, v in inputs_f.items()}
# # #     with torch.no_grad():
# # #         logits_f = model(**inputs_f).logits
# # #     label_f = torch.argmax(logits_f, dim=1).item()

# # #     # Backward pass
# # #     inputs_b = tokenizer.encode_plus(premise_backward, hypothesis_bwd, return_tensors="pt", truncation=True)
# # #     inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
# # #     with torch.no_grad():
# # #         logits_b = model(**inputs_b).logits
# # #     label_b = torch.argmax(logits_b, dim=1).item()

# # #     return (label_f == 2) and (label_b == 2)


# # # def cal_entropy_grounding_semantic(row, tokenizer, model, device="cpu"):
# # #     """Compute grounding entropy per row."""
# # #     responses = ast.literal_eval(row["responses"])
# # #     groundings = ast.literal_eval(row["grounding_llama32_70b_processed"])
# # #     question = row["question"]
# # #     n = len(responses)

# # #     if n == 0 or len(groundings) != n:
# # #         return 0.0

# # #     adjacency = [[False] * n for _ in range(n)]
# # #     for i in range(n):
# # #         for j in range(i + 1, n):
# # #             if are_semantically_equivalent(responses[i], responses[j], question, tokenizer, model, device=device):
# # #                 adjacency[i][j] = adjacency[j][i] = True

# # #     visited = [False] * n
# # #     clusters = []
# # #     for i in range(n):
# # #         if not visited[i]:
# # #             stack = [i]
# # #             visited[i] = True
# # #             cluster_members = []
# # #             while stack:
# # #                 node = stack.pop()
# # #                 cluster_members.append(node)
# # #                 for nb in range(n):
# # #                     if adjacency[node][nb] and not visited[nb]:
# # #                         visited[nb] = True
# # #                         stack.append(nb)
# # #             clusters.append(cluster_members)

# # #     grounding_score_with_entropy = []
# # #     for c in clusters:
# # #         cluster_size = len(c)
# # #         yes_count = sum(1 for idx in c if groundings[idx].lower() == "yes")
# # #         p = yes_count / cluster_size

# # #         h_cluster = 0.0 if p in [0, 1] else - (p * math.log2(p) + (1 - p) * math.log2(1 - p))

# # #         grounding_score = yes_count / cluster_size
# # #         grounding_score_with_entropy.append(grounding_score * h_cluster)

# # #     return sum(grounding_score_with_entropy) / len(grounding_score_with_entropy) if grounding_score_with_entropy else 0.0


# # # def process_batch(df_subset, gpu_id, progress_queue):
# # #     """
# # #     Worker function for parallel processing on a specific GPU.
# # #     Loads model on the assigned GPU and processes the dataframe subset.
# # #     """
# # #     device = f"cuda:{gpu_id}"
# # #     model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# # #     results = []
# # #     for _, row in df_subset.iterrows():
# # #         # result = cal_entropy_grounding_semantic(row, tokenizer, model, device=device)
# # #         result = calc_external_grounding_uncertainty(row)
# # #         results.append(result)
        
# # #         # Update progress bar
# # #         progress_queue.put(1)

# # #     df_subset["grounding_llama32V_with_external_entropy"] = results
# # #     return df_subset





# # # if __name__ == "__main__":
# # #     tqdm.pandas()

# # #     # Load dataframe
# # #     df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv")

# # #     # Split dataframe into equal parts for each GPU
# # #     df_splits = [df.iloc[i::num_gpus] for i in range(num_gpus)]

# # #     # Create a shared manager and progress queue
# # #     manager = Manager()
# # #     progress_queue = manager.Queue()

# # #     # Start multiprocessing
# # #     futures = []
# # #     total_rows = len(df)
    
# # #     with Pool(num_gpus) as pool:
# # #         results = []
# # #         for i in range(num_gpus):
# # #             future = pool.apply_async(process_batch, (df_splits[i], i, progress_queue))
# # #             futures.append(future)

# # #         # Show global progress bar
# # #         with tqdm(total=total_rows, desc="Processing Rows") as pbar:
# # #             processed = 0
# # #             while processed < total_rows:
# # #                 while not progress_queue.empty():
# # #                     progress_queue.get()
# # #                     processed += 1
# # #                     pbar.update(1)
# # #                 time.sleep(0.1)  # Adjust sleep time for efficiency

# # #         # Collect results from workers
# # #         for future in futures:
# # #             results.append(future.get())

# # #     # Combine processed data
# # #     df_parallel = pd.concat(results)

# # #     # Save the output
# # #     df_parallel.to_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_entropy.csv", index=False)

# # #     print("✅ Processing complete! Saved results to processed_results.csv")


# # import torch
# # import pandas as pd
# # import math
# # import ast
# # import time
# # from tqdm import tqdm
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # from torch.multiprocessing import Pool, set_start_method, Manager

# # set_start_method("spawn", force=True)

# # model_name = "microsoft/deberta-base-mnli"
# # tokenizer = AutoTokenizer.from_pretrained(model_name)

# # num_gpus = min(4, torch.cuda.device_count())  # Use up to 4 GPUs
# # print(f"Using {num_gpus} GPUs")

# # # Model initialization function for each worker
# # def init_worker(gpu_id):
# #     """Initialize model once per worker to avoid redundant loading."""
# #     global model
# #     device = f"cuda:{gpu_id}"
# #     model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# # # Safe list conversion
# # def safe_literal_eval(value):
# #     if isinstance(value, str):
# #         try:
# #             return ast.literal_eval(value)
# #         except:
# #             return value
# #     return value

# # def calc_external_grounding_uncertainty(row):
# #     """Computes grounding uncertainty across three models."""
# #     modelA = safe_literal_eval(row["grounding_llama32_11b_processed_score"])
# #     modelB = safe_literal_eval(row["grounding_llama32_70b_processed_score"])
# #     modelC = safe_literal_eval(row["grounding_qwen_vl_processed_score"])

# #     if not (len(modelA) == len(modelB) == len(modelC) > 0):
# #         return 0.0

# #     total_score = 0.0
# #     for i in range(len(modelA)):
# #         answers = [str(modelA[i]).lower().strip(), str(modelB[i]).lower().strip(), str(modelC[i]).lower().strip()]
# #         yes_count = sum(ans == "yes" for ans in answers)
# #         p = yes_count / 3.0

# #         H = 0.0 if p in [0, 1] else - (p * math.log2(p) + (1 - p) * math.log2(1 - p))
# #         total_score += H * p

# #     return total_score / len(modelA)

# # def process_batch(df_subset, gpu_id, progress_queue):
# #     """Worker function for processing data in parallel."""
# #     device = f"cuda:{gpu_id}"
    
# #     results = []
# #     for _, row in df_subset.iterrows():
# #         result = calc_external_grounding_uncertainty(row)
# #         results.append(result)
# #         progress_queue.put(1)

# #     df_result = df_subset.copy()
# #     df_result["grounding_llama32V_with_external_entropy"] = results
# #     return df_result

# # if __name__ == "__main__":
# #     tqdm.pandas()
# #     df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv")
    
# #     df_splits = [df.iloc[i::num_gpus] for i in range(num_gpus)]
# #     manager = Manager()
# #     progress_queue = manager.Queue()

# #     futures = []
# #     total_rows = len(df)

# #     with Pool(num_gpus, initializer=init_worker, initargs=(num_gpus,)) as pool:
# #         results = []
# #         for i in range(num_gpus):
# #             future = pool.apply_async(process_batch, (df_splits[i], i, progress_queue))
# #             futures.append(future)

# #         with tqdm(total=total_rows, desc="Processing Rows") as pbar:
# #             processed = 0
# #             while processed < total_rows:
# #                 while not progress_queue.empty():
# #                     progress_queue.get()
# #                     processed += 1
# #                     pbar.update(1)
# #                 time.sleep(0.1)

# #         for future in futures:
# #             results.append(future.get())

# #     df_parallel = pd.concat(results)
# #     df_parallel.to_csv("/staging/users/tpadhi1/processed_results.csv", index=False)
# #     print("✅ Processing complete!")

# import torch
# import pandas as pd
# import math
# import ast
# import time
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from torch.multiprocessing import Pool, set_start_method, Manager

# set_start_method("spawn", force=True)

# model_name = "microsoft/deberta-base-mnli"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# num_gpus = torch.cuda.device_count()  # Only use available GPUs
# print(f"✅ Detected {num_gpus} GPU(s). Using {num_gpus} for processing.")

# def process_batch(df_subset, gpu_id, progress_queue):
#     """Worker function for processing data in parallel."""
#     device = f"cuda:{gpu_id}"

#     # Validate GPU ID
#     if gpu_id >= torch.cuda.device_count():
#         print(f"❌ Error: Requested GPU {gpu_id}, but only {torch.cuda.device_count()} GPUs available!")
#         return df_subset  # Return empty DataFrame to prevent crashes

#     print(f"🚀 Worker {gpu_id}: Running on {device}")

#     # Load model inside worker
#     model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

#     results = []
#     for _, row in df_subset.iterrows():
#         result = calc_external_grounding_uncertainty(row)
#         results.append(result)
#         progress_queue.put(1)  # Update progress

#     df_subset = df_subset.copy()
#     df_subset["grounding_llama32V_with_external_entropy"] = results
#     return df_subset

# def calc_external_grounding_uncertainty(row):
#     """
#     For each row, we have 20 responses. Each response has 3 grounding model outputs:
#       - grounding_llama32_11b_processed_score
#       - grounding_llama32_70b_processed_score
#       - grounding_qwen_vl_processed_score
#     which are lists of 'yes'/'no' (or possibly other strings).

#     We'll compute for each response i:
#       p_i = (# of 'yes') / 3
#       H_i = -[p_i log2(p_i) + (1-p_i) log2(1-p_i)] if 0 < p_i < 1 else 0
#       response_score_i = H_i * p_i

#     Finally, we average response_score_i over all responses => a single float.

#     Returns that float (overall uncertainty).
#     """

#     # Extract the three columns as lists of length ~20
#     # If your columns are already Python lists, you can skip the ast.literal_eval step
#     # If they're guaranteed to be lists (no string representation), remove the literal_eval usage.
#     try:
#         modelA = ast.literal_eval(row["grounding_llama32_11b_processed_score"])
#         modelB = ast.literal_eval(row["grounding_llama32_70b_processed_score"])
#         modelC = ast.literal_eval(row["grounding_qwen_vl_processed_score"])
#     except:
#         # If already lists, just do
#         modelA = row["grounding_llama32_11b_processed_score"]
#         modelB = row["grounding_llama32_70b_processed_score"]
#         modelC = row["grounding_qwen_vl_processed_score"]

#     # Check lengths
#     nA = len(modelA)
#     nB = len(modelB)
#     nC = len(modelC)
#     if nA == 0 or nB == 0 or nC == 0 or not (nA == nB == nC):
#         return 0.0  # fallback if mismatch

#     num_responses = nA  # expected ~20

#     total_score = 0.0
#     for i in range(num_responses):
#         # Gather the 3 yes/no from each model for response i
#         answers = [
#             str(modelA[i]).lower().strip(),
#             str(modelB[i]).lower().strip(),
#             str(modelC[i]).lower().strip()
#         ]

#         # Count how many are 'yes'
#         yes_count = sum(ans == "yes" for ans in answers)
#         p = yes_count / 3.0

#         # Binary entropy H(p)
#         if p <= 0 or p >= 1:
#             H = 0.0
#         else:
#             H = - (p*math.log2(p) + (1-p)*math.log2(1-p))

#         # Score = H(p) * p
#         resp_score = H * p
#         total_score += resp_score

#     # Average across all responses
#     overall_uncertainty = total_score / num_responses
#     return overall_uncertainty

# if __name__ == "__main__":
#     tqdm.pandas()
#     df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv")

#     # Split dataframe correctly
#     df_splits = [df.iloc[i::num_gpus] for i in range(num_gpus)]

#     # Create a shared manager and progress queue
#     manager = Manager()
#     progress_queue = manager.Queue()

#     futures = []
#     total_rows = len(df)

#     with Pool(num_gpus) as pool:
#         results = []
#         for i in range(num_gpus):
#             future = pool.apply_async(process_batch, (df_splits[i], i, progress_queue))
#             futures.append(future)

#         # Show global progress bar
#         with tqdm(total=total_rows, desc="Processing Rows") as pbar:
#             processed = 0
#             while processed < total_rows:
#                 while not progress_queue.empty():
#                     progress_queue.get()
#                     processed += 1
#                     pbar.update(1)
#                 time.sleep(0.1)

#         for future in futures:
#             results.append(future.get())

#     df_parallel = pd.concat(results)
#     df_parallel.to_csv("/staging/users/tpadhi1/processed_results.csv", index=False)
#     print("✅ Processing complete!")

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
    for _, row in df_subset.iterrows():
        try:
            result = calc_external_grounding_uncertainty(row)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing row: {str(e)}")
            results.append(None)  # Prevent complete failure
        
        progress_queue.put(1)  # Update progress

    df_subset = df_subset.copy()
    df_subset["grounding_llama32V_with_external_entropy"] = results
    return df_subset

def calc_external_grounding_uncertainty(row):
    """
    Computes uncertainty score based on multiple grounding models.
    """

    try:
        modelA = ast.literal_eval(row["grounding_llama32_11b_processed"])
        modelB = ast.literal_eval(row["grounding_llama32_70b_processed"])
        modelC = ast.literal_eval(row["grounding_qwen_vl_processed"])
    except:
        modelA = row["grounding_llama32_11b_processed"]
        modelB = row["grounding_llama32_70b_processed"]
        modelC = row["grounding_qwen_vl_processed"]

    if not isinstance(modelA, list) or not isinstance(modelB, list) or not isinstance(modelC, list):
        return 0.0  # Ensure inputs are lists

    num_responses = min(len(modelA), len(modelB), len(modelC))

    if num_responses == 0:
        return 0.0  # No responses

    total_score = 0.0
    for i in range(num_responses):
        answers = [
            str(modelA[i]).lower().strip(),
            str(modelB[i]).lower().strip(),
            str(modelC[i]).lower().strip()
        ]
        yes_count = sum(ans == "yes" for ans in answers)
        p = yes_count / 3.0

        H = 0.0 if p in [0, 1] else - (p * math.log2(p) + (1 - p) * math.log2(1 - p))
        resp_score = H * p
        total_score += resp_score

    return total_score / num_responses

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
    df_parallel.to_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed_external_g_entropy.csv", index=False)
    print("✅ Processing complete!")