import json
import torch
import torch.nn.functional as F
import pickle
import torch
import torch.nn.functional as F
import os 


path = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_temp1_topp1_dp_100/results'
sample = 'result_00157461.pkl'

# with open("/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/questions1.2/train_all_questions/train_all_questions_0.json") as file:
#     questions = json.load(file)
    
with open(f'{path}/{sample}', 'rb') as f:
    r = pickle.load(f)

def compute_predictive_entropy(raw_logits, token_ids):
    """
    Compute predictive entropy given raw logits and sequence of token IDs.

    Args:
        raw_logits (list of tensors): List of N tensors, each with shape (1, num_tokens, vocab_size)
        token_ids (list of tensors): List of N tensors, each with shape (num_tokens,)

    Returns:
        entropy (tensor): Predictive entropy, shape (N,)
    """
    # Convert raw logits to log probabilities
    log_probs = []
    for logits in raw_logits:
        log_probs.append(F.log_softmax(logits, dim=-1))

    # Extract log probabilities of observed tokens
    observed_log_probs = []
    for log_prob, token_id in zip(log_probs, token_ids):
        observed_log_prob = log_prob[0, torch.arange(log_prob.shape[1]), token_id]
        observed_log_probs.append(torch.mean(observed_log_prob, dim=1))
        
    print(torch.stack(observed_log_probs).shape)

    # Compute predictive entropy
    entropy = -torch.stack(observed_log_probs).mean(dim=0).squeeze()

    return entropy


raw_logits = r['scores']
token_ids = r['tokens']

print(compute_predictive_entropy(raw_logits, token_ids))

path = "/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_temp1_topp1_dp_100/results"
# Loop through each .pkl file in the directory and compute entropy
for filename in os.listdir(path):
    if filename.endswith('.pkl'):
        filepath = os.path.join(path, filename)
        with open(filepath, 'rb') as f:
            r = pickle.load(f)
        
        raw_logits = r['scores']
        token_ids = r['tokens']
        
        entropy = compute_predictive_entropy(raw_logits, token_ids)
        print(f"Entropy for {filename}: {entropy}")


# # Method 1
# logits = r['scores'][0]
# log_probs = F.log_softmax(logits, dim=-1)
# tokens = r['tokens'][0]
# sequence_log_probs = log_probs[0, torch.arange(49), tokens[0]]
# log_p_s_given_x = torch.sum(sequence_log_probs)
# p_s_given_x = torch.exp(log_p_s_given_x)


# # Method 2 
# def calculate_log_p_s_given_x(logits, token_ids):
#     # Initialize log probability sum
#     log_p_s_given_x = 0
#     # Iterate over each token in the sequence
#     for i in range(len(token_ids)):
#         # Get the log probabilities for the current token
#         log_probabilities = F.log_softmax(logits[:, i, :], dim=1)
        
#         # Get the log probability of the current token given the previous tokens
#         log_p_si_given_s_i = log_probabilities[:, token_ids[i]]
        
#         # Add the log probability to the sum
#         log_p_s_given_x += log_p_si_given_s_i
    
#     return log_p_s_given_x

# logits = r['scores'][0]
# token_ids = r['tokens'][0][0]
# log_p_s_given_x = calculate_log_p_s_given_x(logits, token_ids)
# p_s_given_x = torch.exp(log_p_s_given_x)



# def calculate_log_likelihood(logits, token_ids):
#     """
#     Calculate the log likelihood of a sequence given the model's logits and token IDs.

#     Args:
#     - logits (torch.Tensor): The model's output scores with shape (seq_len, vocab_size).
#     - token_ids (torch.Tensor): The token IDs of the sequence with shape (seq_len,).

#     Returns:
#     - log_likelihood (torch.Tensor): The total log likelihood of the sequence.
#     """

#     # Step 1: Convert logits to log probabilities
#     log_probs = F.log_softmax(logits, dim=-1)

#     # Step 2: Gather the log probabilities corresponding to the token IDs
#     log_likelihoods = log_probs[torch.arange(len(token_ids)), token_ids]

#     # Step 3: Sum the log probabilities to get the total log likelihood of the sequence
#     log_likelihood = torch.sum(log_likelihoods)

#     return log_likelihood

# def calculate_entropy(log_likelihoods):
#     """
#     Calculate the entropy of the sequence by taking the mean of the log likelihoods.

#     Args:
#     - log_likelihoods (torch.Tensor): Log likelihoods of each token in the sequence.

#     Returns:
#     - entropy (torch.Tensor): The entropy of the sequence.
#     """

#     # Step 4: Calculate the entropy as the mean of the log likelihoods
#     entropy = -torch.mean(log_likelihoods)

#     return entropy

# Example usage:
# Assuming we have a sequence of length 5 and a vocabulary size of 32000
# logits = torch.randn(5, 32000)  # Example logits for a sequence of length 5
# token_ids = torch.tensor([15, 102, 453, 2112, 768])  # Example token IDs


