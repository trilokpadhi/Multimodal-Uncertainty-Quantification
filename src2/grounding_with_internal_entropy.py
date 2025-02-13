# # from tqdm import tqdm
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # import ast
# # import math
# # import pandas as pd
# # import torch
# # import torch.nn.functional as F
# # from multiprocessing import Pool

# # def are_semantically_equivalent(respA, respB, question, tokenizer, model, device='cpu'):
# #     """
# #     Checks if respA and respB are mutually entailing given the question context.
# #     Returns True if label=2 in both directions, else False.
# #     """

# #     premise_forward  = f"{respA}"
# #     hypothesis_fwd   = f"{respB}"

# #     premise_backward = f"{respB}"
# #     hypothesis_bwd   = f"{respA}"

# #     # Forward pass
# #     inputs_f = tokenizer.encode_plus(
# #         premise_forward, hypothesis_fwd, 
# #         return_tensors='pt', truncation=True
# #     )
# #     inputs_f = {k: v.to(device) for k, v in inputs_f.items()}
# #     with torch.no_grad():
# #         logits_f = model(**inputs_f).logits
# #     label_f = torch.argmax(logits_f, dim=1).item()  # e.g. 0/1/2

# #     # Backward pass
# #     inputs_b = tokenizer.encode_plus(
# #         premise_backward, hypothesis_bwd, 
# #         return_tensors='pt', truncation=True
# #     )
# #     inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
# #     with torch.no_grad():
# #         logits_b = model(**inputs_b).logits
# #     label_b = torch.argmax(logits_b, dim=1).item()

# #     # Check mutual entailment
# #     return (label_f == 2) and (label_b == 2)


# # def cal_entropy_grounding_semantic(row, tokenizer, model, device='cpu'):
# #     responses = ast.literal_eval(row["responses"])
# #     groundings= ast.literal_eval(row["grounding_llama32_70b_processed"])
# #     question  = row["question"]  # if you stored it in the row
# #     n = len(responses)
# #     if n == 0 or len(groundings) != n:
# #         return 0.0

# #     # Build adjacency with mutual entailment
# #     adjacency = [[False]*n for _ in range(n)]
# #     for i in range(n):
# #         for j in range(i+1, n):
# #             # check if responses i and j are semantically equivalent
# #             if are_semantically_equivalent(
# #                    responses[i], responses[j], 
# #                    question, 
# #                    tokenizer, 
# #                    model, 
# #                    device=device
# #                ):
# #                 adjacency[i][j] = True
# #                 adjacency[j][i] = True

# #     # Then BFS cluster, same logic as before
# #     visited = [False]*n
# #     clusters = []
# #     for i in range(n):
# #         if not visited[i]:
# #             stack = [i]
# #             visited[i] = True
# #             cluster_members = []
# #             while stack:
# #                 node = stack.pop()
# #                 cluster_members.append(node)
# #                 for nb in range(n):
# #                     if adjacency[node][nb] and not visited[nb]:
# #                         visited[nb] = True
# #                         stack.append(nb)
# #             clusters.append(cluster_members)

# #     # Compute grounding entropy
# #     grounding_score_with_entropy = []
# #     for c in clusters:
# #         cluster_size = len(c)
# #         yes_count = sum(1 for idx in c if groundings[idx].lower() == "yes")
# #         p = yes_count / cluster_size

# #         if p == 0 or p == 1:
# #             h_cluster = 0.0
# #         else:
# #             h_cluster = - (p * math.log2(p) + (1 - p)*math.log2(1 - p))

# #         # lets take the grounding score of the cluster, no of yes in the cluster / total no of responses in the cluster
# #         no_of_yes_in_cluster = sum(1 for idx in c if groundings[idx].lower() == "yes")
# #         cluster_size = len(c)
# #         grounding_score = no_of_yes_in_cluster / cluster_size
# #         grounding_score_of_the_cluster = grounding_score * h_cluster
# #         # weighted_sum += grounding_score_of_the_cluster
# #         grounding_score_with_entropy.append(grounding_score_of_the_cluster)
        
# #     return sum(grounding_score_with_entropy) / len(grounding_score_with_entropy)


# # if __name__ == "__main__":

# #     # Enable tqdm for pandas
# #     tqdm.pandas()

# #     df = pd.read_csv('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv')

# #     model_name  = 'microsoft/deberta-base-mnli'
# #     tokenizer = AutoTokenizer.from_pretrained(model_name)
# #     model = AutoModelForSequenceClassification.from_pretrained(model_name)
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     model.to(device)
# #     # df['grounding_llama32V_with_cluster_wise_entropy'] = df.progress_apply(lambda row: cal_entropy_grounding_semantic(row, tokenizer, model, device='cuda'), axis=1)
    
# #     num_gpus = 4
# #     # Function to process a subset of the dataframe on a given GPU
# #     def process_batch(df_subset, gpu_id):
# #         device = f"cuda:{gpu_id}"
# #         model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        
# #         df_subset["grounding_llama32V_with_cluster_wise_entropy"] = df_subset.apply(
# #             lambda row: cal_entropy_grounding_semantic(row, tokenizer, model, device=device), axis=1
# #         )
# #         return df_subset

# #     # Split dataframe into equal parts for each GPU
# #     df_splits = [df.iloc[i::num_gpus] for i in range(num_gpus)]

# #     # Run multiprocessing across GPUs
# #     with Pool(num_gpus) as pool:
# #         results = pool.starmap(process_batch, [(df_splits[i], i) for i in range(num_gpus)])

# #     # Combine processed data
# #     df_parallel = pd.concat(results)

# import torch
# import pandas as pd
# import math
# import ast
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from torch.multiprocessing import Pool, set_start_method
# import time
# from multiprocessing import Manager

# # Ensure proper multiprocessing start method
# set_start_method("spawn", force=True)  # 🔥 Fixes CUDA in multiprocessing issue

# # Load model & tokenizer once in main process
# model_name = "microsoft/deberta-base-mnli"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Define GPU count
# num_gpus = min(4, torch.cuda.device_count())  # Use available GPUs (max 4)
# print(f"Using {num_gpus} GPUs")

# def are_semantically_equivalent(respA, respB, question, tokenizer, model, device="cpu"):
#     """
#     Checks if respA and respB are mutually entailing given the question context.
#     Returns True if label=2 in both directions, else False.
#     """

#     premise_forward = f"{respA}"
#     hypothesis_fwd = f"{respB}"
#     premise_backward = f"{respB}"
#     hypothesis_bwd = f"{respA}"

#     # Forward pass
#     inputs_f = tokenizer.encode_plus(premise_forward, hypothesis_fwd, return_tensors="pt", truncation=True)
#     inputs_f = {k: v.to(device) for k, v in inputs_f.items()}
#     with torch.no_grad():
#         logits_f = model(**inputs_f).logits
#     label_f = torch.argmax(logits_f, dim=1).item()

#     # Backward pass
#     inputs_b = tokenizer.encode_plus(premise_backward, hypothesis_bwd, return_tensors="pt", truncation=True)
#     inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
#     with torch.no_grad():
#         logits_b = model(**inputs_b).logits
#     label_b = torch.argmax(logits_b, dim=1).item()

#     return (label_f == 2) and (label_b == 2)


# def cal_entropy_grounding_semantic(row, tokenizer, model, device="cpu"):
#     """
#     Compute grounding entropy per row.
#     """
#     responses = ast.literal_eval(row["responses"])
#     groundings = ast.literal_eval(row["grounding_llama32_70b_processed"])
#     question = row["question"]
#     n = len(responses)

#     if n == 0 or len(groundings) != n:
#         return 0.0

#     adjacency = [[False] * n for _ in range(n)]
#     for i in range(n):
#         for j in range(i + 1, n):
#             if are_semantically_equivalent(responses[i], responses[j], question, tokenizer, model, device=device):
#                 adjacency[i][j] = adjacency[j][i] = True

#     visited = [False] * n
#     clusters = []
#     for i in range(n):
#         if not visited[i]:
#             stack = [i]
#             visited[i] = True
#             cluster_members = []
#             while stack:
#                 node = stack.pop()
#                 cluster_members.append(node)
#                 for nb in range(n):
#                     if adjacency[node][nb] and not visited[nb]:
#                         visited[nb] = True
#                         stack.append(nb)
#             clusters.append(cluster_members)

#     grounding_score_with_entropy = []
#     for c in clusters:
#         cluster_size = len(c)
#         yes_count = sum(1 for idx in c if groundings[idx].lower() == "yes")
#         p = yes_count / cluster_size

#         h_cluster = 0.0 if p in [0, 1] else - (p * math.log2(p) + (1 - p) * math.log2(1 - p))

#         grounding_score = yes_count / cluster_size
#         grounding_score_with_entropy.append(grounding_score * h_cluster)

#     return sum(grounding_score_with_entropy) / len(grounding_score_with_entropy) if grounding_score_with_entropy else 0.0


# def process_batch(df_subset, gpu_id):
#     """
#     Worker function for parallel processing on a specific GPU.
#     Loads model on the assigned GPU and processes the dataframe subset.
#     """
#     device = f"cuda:{gpu_id}"
#     model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)  # Load model on correct GPU

#     df_subset["grounding_llama32V_with_cluster_wise_entropy"] = df_subset.progress_apply(
#         lambda row: cal_entropy_grounding_semantic(row, tokenizer, model, device=device), axis=1
#     )
#     return df_subset


# if __name__ == "__main__":
#     # tqdm.pandas()
    
#     # # Load dataframe
#     # df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv")

#     # # Split dataframe into equal parts for each GPU
#     # df_splits = [df.iloc[i::num_gpus] for i in range(num_gpus)]

#     # # Run multiprocessing across GPUs
#     # with Pool(num_gpus) as pool:
#     #     results = pool.starmap(process_batch, [(df_splits[i], i) for i in range(num_gpus)])

#     # # Combine processed data
#     # df_parallel = pd.concat(results)

#     # # Save the output
#     # df_parallel.to_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/processed_results.csv", index=False)

#     # print("✅ Processing complete! Saved results to processed_results.csv")
    
#     tqdm.pandas()

#     # Load dataframe
#     df = pd.read_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/merged_with_grounding_processed.csv")

#     # Split dataframe into equal parts for each GPU
#     df_splits = [df.iloc[i::num_gpus] for i in range(num_gpus)]

#     # Create a shared manager and progress queue
#     manager = Manager()
#     progress_queue = manager.Queue()

#     # Start multiprocessing
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
#                 time.sleep(0.1)  # Adjust sleep time for efficiency

#         # Collect results from workers
#         for future in futures:
#             results.append(future.get())

#     # Combine processed data
#     df_parallel = pd.concat(results)

#     # Save the output
#     df_parallel.to_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/processed_results.csv", index=False)

#     print("✅ Processing complete! Saved results to processed_results.csv")

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
    groundings = ast.literal_eval(row["grounding_llama32_70b_processed"])
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

    grounding_score_with_entropy = []
    for c in clusters:
        cluster_size = len(c)
        yes_count = sum(1 for idx in c if groundings[idx].lower() == "yes")
        p = yes_count / cluster_size

        h_cluster = 0.0 if p in [0, 1] else - (p * math.log2(p) + (1 - p) * math.log2(1 - p))

        grounding_score = yes_count / cluster_size
        grounding_score_with_entropy.append(grounding_score * h_cluster)

    return sum(grounding_score_with_entropy) / len(grounding_score_with_entropy) if grounding_score_with_entropy else 0.0


def process_batch(df_subset, gpu_id, progress_queue):
    """
    Worker function for parallel processing on a specific GPU.
    Loads model on the assigned GPU and processes the dataframe subset.
    """
    device = f"cuda:{gpu_id}"
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    results = []
    for _, row in df_subset.iterrows():
        result = cal_entropy_grounding_semantic(row, tokenizer, model, device=device)
        results.append(result)
        
        # Update progress bar
        progress_queue.put(1)

    df_subset["grounding_llama32V_with_cluster_wise_entropy"] = results
    return df_subset


if __name__ == "__main__":
    tqdm.pandas()

    # Load dataframe
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
    df_parallel.to_csv("/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/my_outputs/processed_results.csv", index=False)

    print("✅ Processing complete! Saved results to processed_results.csv")