# from transformers import GPT2Tokenizer, AutoModelForCausalLM
# import numpy as np

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
# tokenizer.pad_token_id = tokenizer.eos_token_id
# inputs = tokenizer(["Today is"], return_tensors="pt")

# # Example 1: Print the scores for each token generated with Greedy Search
# outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
# transition_scores = model.compute_transition_scores(
#     outputs.sequences, outputs.scores, normalize_logits=True
# )
# # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
# # encoder-decoder models, like BART or T5.
# input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
# generated_tokens = outputs.sequences[:, input_length:]
# for tok, score in zip(generated_tokens[0], transition_scores[0]):
#     # | token | token string | log probability | probability
#     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

# # Example 2: Reconstruct the sequence scores from Beam Search
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=5,
#     num_beams=4,
#     num_return_sequences=4,
#     return_dict_in_generate=True,
#     output_scores=True,
# )
# transition_scores = model.compute_transition_scores(
#     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
# )
# # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
# # Tip 1: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
# # use case, you might want to recompute it with `normalize_logits=True`.
# # Tip 2: the output length does NOT include the input length
# output_length = np.sum(transition_scores.numpy() < 0, axis=1)
# length_penalty = model.generation_config.length_penalty
# reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
# print(np.allclose(outputs.sequences_scores, reconstructed_scores))

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

from huggingface_hub import login
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

# Authenticate
username = "tpadhi1"
token = "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"
login(token=token)

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# model=LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(device)
# tokenizer= LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# prompt = "Could you give me 3 cities located in Europe ?"

# inputs = tokenizer([prompt], return_tensors="pt").to(device)

# outputs=model.generate(**inputs,return_dict_in_generate=True, output_scores=True,max_new_tokens=75)

# transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

# input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
# generated_tokens = outputs.sequences[:,input_length:]

# for tok, score in zip(generated_tokens[0], transition_scores[0]):
#         # | token | token string | logits | probability
#             print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy(force=True):.4f} | {np.exp(score.numpy(force=True)):.2%}")
            
# # decode the output
# output = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print('Decoded output', output)

import torch
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer

# Assuming you have already loaded your model and tokenizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(device)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

prompt = "Could you give me 3 cities located in Europe?"

# Number of responses to generate
num_responses = 5

def calculate_entropy(probabilities):
    """Calculate the entropy for a set of probabilities."""
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    return entropy.item()

# List to store entropy values
entropies = []

# Generate responses and calculate entropy for each
for _ in range(num_responses):
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = model.generate(**inputs,
                                 return_dict_in_generate=True,
                                 output_scores=True,
                                 max_new_tokens=75,
                                 do_sample=True,  # Ensure sampling to get different responses
                                 temperature=0.7,  # Adjust temperature for more variability
                                 top_p=0.9)

        print(outputs)
        print('-------------------')
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        # # Compute probabilities from logits (scores)
        # scores = torch.stack(outputs.scores, dim=1)  # Shape: (batch_size, sequence_length, vocab_size)
        # probabilities = torch.softmax(scores, dim=-1)  # Convert logits to probabilities

        # # Select probabilities corresponding to the generated tokens
        # token_probs = torch.gather(probabilities, 2, generated_tokens.unsqueeze(-1)).squeeze(-1)

        # # Compute entropy for the generated response
        # entropy = calculate_entropy(token_probs)
        # entropies.append(entropy)
        
        outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        print(outputs)
        


# # Calculate average entropy (uncertainty) over the responses
# average_entropy = np.mean(entropies)

# print(f"Entropy values for each response: {entropies}")
# print(f"Average entropy (Uncertainty): {average_entropy}")