import os
import pickle
import torch
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
from tqdm import tqdm
import itertools

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import ipdb

# Initialize the tokenizer
model_path = "liuhaotian/llava-v1.6-vicuna-7b"
model_base = None
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

def load_results(directory):
    """Load all result files from the specified directory and return a dictionary with result IDs as keys."""
    all_results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            # Extract the result ID from the filename (assuming format is result_id.pkl)
            result_id = filename.split('_')[1].replace('.pkl', '')
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                all_results[result_id] = pickle.load(f)
    return all_results

def replace_infs_with_small_values(tensor, small_value=1e-10):
    """Replace -inf values in a tensor with a small value."""
    tensor = torch.where(tensor == -float('inf'), torch.tensor(small_value, dtype=tensor.dtype), tensor)
    return tensor

# def calculate_accuracy_and_uq(results):
#     """Calculate accuracy, entropy of sequence probabilities, log softmax, and last token probabilities."""
#     accuracies = []
#     entropy_sequence_probs = []
#     log_softmax_entropies = []
#     last_token_entropies = []
#     token_wise_probs_all_responses = []
#     confidences = []

#     for result_id, result in results.items():
#         responses = result['responses']
#         correct_answer = result['answer'].strip().lower()
        
#         # Accuracy Calculation
#         total_correct = 0
#         for resp in responses:
#             try:
#                 answer = resp['answer'].strip().lower()
#                 if answer == correct_answer:
#                     total_correct += 1
#             except Exception as e:
#                 print(f"Error processing 'answer' in {resp}: {e}")

#         accuracy = total_correct / len(responses)
#         accuracies.append(accuracy)
        
#         # Sequence Probabilities Entropy
#         sequence_probs = result['sequence_probabilities']
#         sequence_probs = torch.tensor(sequence_probs)
#         sequence_probs = sequence_probs / sequence_probs.sum()  # Normalize probabilities to sum up to 1
#         entropy_seq_prob = entropy(sequence_probs.numpy())
#         entropy_sequence_probs.append(entropy_seq_prob)
        
#         # Log Softmax Entropy and Last Token Probability Entropy
#         scores = result['scores']
#         tokens_list = result['tokens']  # List of tokens corresponding to each response
#         log_softmax_entropies_per_response = []
#         last_token_entropies_per_response = []
#         token_wise_probs_per_response = []

#         for score, tokens in zip(scores, tokens_list):
#             score = torch.tensor(score).squeeze(0)
#             log_softmax_probs = torch.nn.functional.log_softmax(score, dim=-1)
            
#             # Calculate entropy for log softmax probabilities
#             entropy_log_softmax_probs = entropy(log_softmax_probs.exp().numpy(), axis=-1).mean()
#             log_softmax_entropies_per_response.append(entropy_log_softmax_probs)
            
#             # Calculate entropy for last token probabilities
#             last_token_probs = log_softmax_probs[-1, :].exp()
#             entropy_last_token = entropy(last_token_probs.numpy())
#             last_token_entropies_per_response.append(entropy_last_token)
            
#             # Store token-wise probabilities only for the relevant tokens in each response
#             relevant_probs = log_softmax_probs.exp().detach().numpy()
#             token_probs = relevant_probs[-1, tokens.squeeze(0).tolist()]
#             token_wise_probs_per_response.append(token_probs)
        
#         # Append the average entropy across all responses
#         log_softmax_entropies.append(np.mean(log_softmax_entropies_per_response))
#         last_token_entropies.append(np.mean(last_token_entropies_per_response))
#         token_wise_probs_all_responses.append(token_wise_probs_per_response)

#         # Calculate confidence from log softmax entropy
#         confidence = 1 - np.mean(log_softmax_entropies_per_response)
#         confidences.append(confidence)
    
#     return accuracies, entropy_sequence_probs, log_softmax_entropies, last_token_entropies, sequence_probs, token_wise_probs_all_responses, confidences

def calculate_accuracy_and_uq(results):
    """Calculate accuracy, entropy of sequence probabilities, log softmax, and last token probabilities."""
    accuracies = []
    entropy_sequence_probs = []
    log_softmax_entropies_all = []
    last_token_entropies_all = []
    token_wise_probs_all_responses = []
    confidences = []

    for result_id, result in results.items():
        responses = result['responses']
        correct_answer = result['answer'].strip().lower()
        
        # Accuracy Calculation
        total_correct = 0
        incorrect_responses = 0
        for resp in responses:
            try:
                answer = resp['answer'].strip().lower()
                if answer == correct_answer:
                    total_correct += 1
            except Exception as e:
                print(f"Error processing 'answer' in {resp}: {e}")
                incorrect_responses+=1


        accuracy = total_correct / (len(responses) - incorrect_responses)
        accuracies.append(accuracy)
        
        # Sequence Probabilities Entropy
        sequence_probs = result['sequence_probabilities']
        sequence_probs = torch.tensor(sequence_probs)
        sequence_probs = sequence_probs / sequence_probs.sum()  # Normalize probabilities to sum up to 1
        entropy_seq_prob = entropy(sequence_probs.numpy())
        entropy_sequence_probs.append(entropy_seq_prob)
        
        # Log Softmax Entropy and Last Token Probability Entropy
        log_softmax_entropies_per_response = []
        last_token_entropies_per_response = []
        token_wise_probs_per_response = []

        for score, tokens in zip(result['scores'], result['tokens']):
            score = torch.tensor(score).squeeze(0)
            log_softmax_probs = torch.nn.functional.log_softmax(score, dim=-1)
            
            # Calculate entropy for log softmax probabilities
            # this is over entire vocab
            entropy_log_softmax_probs = entropy(log_softmax_probs.exp().numpy(), axis=-1).mean()
            log_softmax_entropies_per_response.append(entropy_log_softmax_probs)
            
            # Calculate entropy for last token probabilities
            last_token_probs = log_softmax_probs[-1, :].exp()
            entropy_last_token = entropy(last_token_probs.numpy())
            last_token_entropies_per_response.append(entropy_last_token)
            
            # Store token-wise probabilities only for the relevant tokens in each response
            relevant_probs = log_softmax_probs.exp().detach().numpy()
            token_probs = relevant_probs[-1, tokens.squeeze(0).tolist()]
            token_wise_probs_per_response.append(token_probs)
        
        # Append all responses entropies
        log_softmax_entropies_all.append(log_softmax_entropies_per_response)
        last_token_entropies_all.append(last_token_entropies_per_response)
        token_wise_probs_all_responses.append(token_wise_probs_per_response)

        # Calculate confidence from entropy
        confidence = 1 - np.mean(log_softmax_entropies_per_response)
        confidences.append(confidence)
    
    return accuracies, entropy_sequence_probs, log_softmax_entropies_all, last_token_entropies_all, sequence_probs, token_wise_probs_all_responses, confidences

# def generate_html_table(sequence_probs, log_softmax_entropies, last_token_entropies, token_wise_probs_all_responses, tokens_list, full_responses):
#     """Generate an HTML table displaying various metrics for 20 responses."""
#     table_html = "<table border='1'><tr><th>Response #</th><th>Confidence</th><th>Product of Probabilities</th><th>Log Softmax Entropy</th><th>Last Token Probability Entropy</th></tr>"
    
#     for i in range(len(sequence_probs)):
#         # Extract the confidence and any other relevant details from the full response
#         confidence = json.loads(full_responses[i]).get("confidence", "N/A")
#         product_of_probs = np.prod(token_wise_probs_all_responses[i])

#         table_html += f"<tr><td>{i+1}</td><td>{confidence}</td><td>{product_of_probs:.4f}</td><td>{log_softmax_entropies[i]:.4f}</td><td>{last_token_entropies[i]:.4f}</td></tr>"
    
#     table_html += "</table>"
    
#     # Adding Token-wise probabilities for all 20 responses below each row as a separate detailed table (if needed)
#     for response_idx, (token_probs, full_response) in enumerate(zip(token_wise_probs_all_responses, full_responses)):
#         # Display the full response dictionary as a string
#         table_html += f"<h3>Response {response_idx + 1}:</h3><pre>{full_response}</pre>"
#         table_html += "<table border='1'><tr><th>Token ID</th><th>Decoded Token</th><th>Probability</th></tr>"
        
#         flattened_tokens = tokens_list[response_idx].squeeze(0).tolist()  # Flatten token list
#         decoded_tokens = tokenizer.convert_ids_to_tokens(flattened_tokens)  # Decode the token IDs

#         for token_idx, (token_id, token_prob) in enumerate(zip(flattened_tokens, token_probs)):
#             decoded_token = decoded_tokens[token_idx]
#             table_html += f"<tr><td>{token_id}</td><td>{decoded_token}</td><td>{token_prob:.4f}</td></tr>"
        
#         table_html += "</table><br>"
    
#     return table_html

# def generate_html_table(log_softmax_entropies, last_token_entropies, token_wise_probs_all_responses, tokens_list, full_responses):
#     """Generate an HTML table displaying various metrics for 20 responses."""
#     table_html = "<table border='1'><tr><th>Response #</th><th>Sequence Probability</th><th>Log Softmax Entropy</th><th>Last Token Probability Entropy</th></tr>"
    
#     for i in range(len(log_softmax_entropies)):
#         # Extract the confidence and any other relevant details from the full response
#         confidence = json.loads(full_responses[i]).get("confidence", "N/A")
#         product_of_probs = np.prod(token_wise_probs_all_responses[i])

#         table_html += f"<tr><td>{i+1}</td><td>{product_of_probs:.4f}</td><td>{log_softmax_entropies[i]:.4f}</td><td>{last_token_entropies[i]:.4f}</td></tr>"
    
#     table_html += "</table>"
    
#     # Adding Token-wise probabilities for all 20 responses below each row as a separate detailed table (if needed)
#     for response_idx, (token_probs, full_response) in enumerate(zip(token_wise_probs_all_responses, full_responses)):
#         # Display the full response dictionary as a string
#         table_html += f"<h3>Response {response_idx + 1}:</h3><pre>{full_response}</pre>"
#         table_html += "<table border='1'><tr><th>Token ID</th><th>Decoded Token</th><th>Probability</th></tr>"
        
#         flattened_tokens = tokens_list[response_idx].squeeze(0).tolist()  # Flatten token list
#         decoded_tokens = tokenizer.convert_ids_to_tokens(flattened_tokens)  # Decode the token IDs

#         for token_idx, (token_id, token_prob) in enumerate(zip(flattened_tokens, token_probs)):
#             decoded_token = decoded_tokens[token_idx]
#             table_html += f"<tr><td>{token_id}</td><td>{decoded_token}</td><td>{token_prob:.4f}</td></tr>"
        
#         table_html += "</table><br>"
    
#     return table_html

# def generate_html_table(sequence_probs, log_softmax_entropies, last_token_entropies, token_wise_probs_all_responses, tokens_list, full_responses):
#     """Generate an HTML table displaying sequence probabilities, log softmax entropies, last token entropies, and token-wise probabilities for 20 responses."""
#     table_html = "<table border='1'><tr><th>Response #</th><th>Sequence Probability</th><th>Log Softmax Entropy</th><th>Last Token Probability Entropy</th></tr>"
    
#     for i in range(len(sequence_probs)):
#         product_of_probs = np.prod(token_wise_probs_all_responses[i])
#         table_html += f"<tr><td>{i+1}</td><td>{product_of_probs:.4f}</td><td>{log_softmax_entropies[i]:.4f}</td><td>{last_token_entropies[i]:.4f}</td></tr>"
    
#     table_html += "</table><br>"

#     for response_idx, (token_probs, full_response) in enumerate(zip(token_wise_probs_all_responses, full_responses)):
#         table_html += f"<h3>Response {response_idx + 1}:</h3><pre>{full_response}</pre>"
#         table_html += "<table border='1'><tr><th>Token ID</th><th>Decoded Token</th><th>Probability</th></tr>"
        
#         flattened_tokens = tokens_list[response_idx].squeeze(0).tolist()
#         decoded_tokens = tokenizer.convert_ids_to_tokens(flattened_tokens)

#         for token_idx, (token_id, token_prob) in enumerate(zip(flattened_tokens, token_probs)):
#             decoded_token = decoded_tokens[token_idx]
#             table_html += f"<tr><td>{token_id}</td><td>{decoded_token}</td><td>{token_prob:.4f}</td></tr>"
        
#         table_html += "</table><br>"

#     return table_html

def generate_html_table(sequence_probs, log_softmax_entropies, last_token_entropies, token_wise_probs_all_responses, tokens_list, full_responses):
    """Generate an HTML table displaying sequence probabilities, log softmax entropies, last token entropies, and token-wise probabilities for 20 responses."""
    
    # Table for probabilities
    prob_table_html = "<h2>Probabilities and Entropies</h2>"
    prob_table_html += "<table border='1'><tr><th>Response #</th><th>Sequence Probability</th><th>Log Softmax Probability</th><th>Last Token Probability</th></tr>"
    
    for i in range(min(len(sequence_probs), len(log_softmax_entropies), len(last_token_entropies))):
        product_of_probs = np.prod(token_wise_probs_all_responses[i]) if i < len(token_wise_probs_all_responses) else 0
        prob_table_html += f"<tr><td>{i+1}</td><td>{product_of_probs:.4f}</td><td>{log_softmax_entropies[i]:.4f}</td><td>{last_token_entropies[i]:.4f}</td></tr>"
    
    prob_table_html += "</table><br>"

    # Table for entropies only
    entropy_table_html = "<h2>Entropy Details</h2>"
    entropy_table_html += "<table border='1'><tr><th>Response #</th><th>Log Softmax Entropy</th><th>Last Token Entropy</th></tr>"
    
    for i in range(min(len(log_softmax_entropies), len(last_token_entropies))):
        entropy_table_html += f"<tr><td>{i+1}</td><td>{log_softmax_entropies[i]:.4f}</td><td>{last_token_entropies[i]:.4f}</td></tr>"
    
    entropy_table_html += "</table><br>"

    for response_idx, (token_probs, full_response) in enumerate(zip(token_wise_probs_all_responses, full_responses)):
        prob_table_html += f"<h3>Response {response_idx + 1}:</h3><pre>{full_response}</pre>"
        prob_table_html += "<table border='1'><tr><th>Token ID</th><th>Decoded Token</th><th>Probability</th></tr>"
        
        flattened_tokens = tokens_list[response_idx].squeeze(0).tolist()
        decoded_tokens = tokenizer.convert_ids_to_tokens(flattened_tokens)

        for token_idx, (token_id, token_prob) in enumerate(zip(flattened_tokens, token_probs)):
            decoded_token = decoded_tokens[token_idx]
            prob_table_html += f"<tr><td>{token_id}</td><td>{decoded_token}</td><td>{token_prob:.4f}</td></tr>"
        
        prob_table_html += "</table><br>"

    return prob_table_html + entropy_table_html

def plot_image(image, question, responses, table_html_with_info, image_id, output_html_path):
    """
    Plots an image, its question, responses, and displays probabilities in a table, saving the result as an HTML file.

    Parameters:
    - image: A 2D array or an image file path.
    - question: The question associated with the image.
    - responses: List of model responses.
    - table_html_with_info: HTML string for the probabilities table and additional info (e.g., accuracy).
    - image_id: An identifier for the image.
    - output_html_path: The file path to save the HTML output.
    """
    # Create a figure and axes for the image
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the image
    if isinstance(image, str):
        img = plt.imread(image)
    else:
        img = image
    ax.imshow(img)
    ax.set_title(f'Image ID: {image_id}\nQuestion: {question}')
    ax.axis('off')
    
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    # Generate the HTML content
    html_content = f"""
    <html>
    <head><title>Plot for Image ID: {image_id}</title></head>
    <body>
    <h1>Image ID: {image_id}</h1>
    <h2>Question: {question}</h2>
    <h3>Responses:</h3>
    <ul>
    """
    for response in responses:
        html_content += f"<li>{response}</li>"
    
    html_content += f"""
    </ul>
    <h3>Probabilities and Entropies</h3>
    {table_html_with_info}
    <img src="data:image/png;base64,{img_base64}" />
    </body>
    </html>
    """
    
    # Save the HTML content to a file
    with open(output_html_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML file saved to {output_html_path}")

def bin_data(confidences, accuracies, num_bins=10):
    """Bin data by uncertainty/confidence and calculate average accuracy per bin."""
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1

    bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers for plotting
    bin_accuracies = []

    accuracies = np.array(accuracies)  # Convert accuracies to a numpy array for easier indexing

    for i in range(num_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_accuracies.append(np.mean(accuracies[bin_mask]))
        else:
            bin_accuracies.append(0)  # If no data points fall in the bin, set accuracy to 0

    return bin_centers, bin_accuracies, bin_indices

# def main():
#     # Load results
#     results_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_temp1_topp1_dp_100/results'
#     all_results = load_results(results_dir)

#     # Create a subset of the first 2 results for testing/processing
#     all_results_subset = dict(itertools.islice(all_results.items(), 40))

#     with open("/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/questions1.2/train_all_questions/train_all_questions_0.json") as file:
#         questions = json.load(file)

#     # Calculate accuracy and uncertainty measures
#     accuracies, entropy_sequence_probs, log_softmax_entropies, last_token_entropies, _, token_wise_probs_all_responses, confidences = calculate_accuracy_and_uq(all_results_subset)

#     # Bin data and get bin indices
#     bin_centers, bin_accuracies, bin_indices = bin_data(confidences, accuracies)

#     # Create directories for each bin if they don't exist
#     for i in range(len(bin_centers)):
#         os.makedirs(f'bin_{i}', exist_ok=True)

#     # Plot and save the HTML visualizations for the subset of results
#     for i, (result_id, result) in enumerate(tqdm(list(all_results_subset.items()), desc="Processing Results")):
#         image_id = questions[result_id]['imageId']
#         ground_truth_answer = questions[result_id]['answer']
#         ground_truth_full_answer = questions[result_id]['fullAnswer']
#         image_path = f'/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/{image_id}.jpg'
#         question = questions[result_id]['question']  # Access question from the questions dictionary
        
#         # Extract the full response, converting the dictionary to a string for display
#         full_responses = [json.dumps(resp, indent=2) for resp in result['responses']]
    
#         # Generate the HTML table with detailed metrics
#         table_html = generate_html_table(
#             [entropy_sequence_probs[i]],
#             [log_softmax_entropies[i]],
#             [last_token_entropies[i]],
#             token_wise_probs_all_responses[i],
#             result['tokens'],
#             full_responses
#         )
        
#         # Determine which bin this result belongs to
#         bin_idx = bin_indices[i]
#         output_html_path = f"bin_{bin_idx}/output_{image_id}.html"
        
#         plot_image(image_path, question, full_responses, str(accuracies[i]) + ground_truth_answer + ground_truth_full_answer + table_html, image_id, output_html_path)

# if __name__ == "__main__":
#     main()

# def main():
#     # Load results
#     results_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_temp1_topp1_dp_100/results'
#     all_results = load_results(results_dir)

#     # Create a subset of the first 40 results for testing/processing
#     all_results_subset = dict(itertools.islice(all_results.items(), 40))

#     with open("/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/questions1.2/train_all_questions/train_all_questions_0.json") as file:
#         questions = json.load(file)

#     # Calculate accuracy and uncertainty measures
#     accuracies, entropy_sequence_probs, log_softmax_entropies, last_token_entropies, _, token_wise_probs_all_responses, confidences = calculate_accuracy_and_uq(all_results_subset)

#     # Bin data and get bin indices
#     bin_centers, bin_accuracies, bin_indices = bin_data(confidences, accuracies)

#     # Create directories for each bin if they don't exist
#     for i in range(len(bin_centers)):
#         os.makedirs(f'bin_{i}', exist_ok=True)

#     # Plot and save the HTML visualizations for the subset of results
#     for i, (result_id, result) in enumerate(tqdm(list(all_results_subset.items()), desc="Processing Results")):
#         image_id = questions[result_id]['imageId']
#         ground_truth_answer = questions[result_id]['answer']
#         ground_truth_full_answer = questions[result_id]['fullAnswer']
#         image_path = f'/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/{image_id}.jpg'
#         question = questions[result_id]['question']  # Access question from the questions dictionary
        
#         # Extract the full response, converting the dictionary to a string for display
#         full_responses = [json.dumps(resp, indent=2) for resp in result['responses']]
    
#         # Generate the HTML table with detailed metrics for all 20 responses
#         table_html = generate_html_table(
#             entropy_sequence_probs[i],  # Pass the sequence probabilities for all 20 responses
#             log_softmax_entropies[i],  # Pass the log softmax entropies for all 20 responses
#             last_token_entropies[i],  # Pass the last token entropies for all 20 responses
#             token_wise_probs_all_responses[i],  # Pass the token-wise probabilities for all 20 responses
#             result['tokens'],  # Pass the token IDs for all 20 responses
#             full_responses  # Pass the full response dictionaries for all 20 responses
#         )
        
#         # Determine which bin this result belongs to
#         bin_idx = bin_indices[i]
#         output_html_path = f"bin_{bin_idx}/output_{image_id}.html"
        
#         plot_image(image_path, question, full_responses, table_html, image_id, output_html_path)

# if __name__ == "__main__":
#     main()


def main():
    # Load results
    results_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_temp1_topp1_dp_100/results'
    all_results = load_results(results_dir)

    # Create a subset of the first 40 results for testing/processing
    all_results_subset = dict(itertools.islice(all_results.items(), 40))

    with open("/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/questions1.2/train_all_questions/train_all_questions_0.json") as file:
        questions = json.load(file)

    ipdb.set_trace()
    # Calculate accuracy and uncertainty measures
    accuracies, entropy_sequence_probs, log_softmax_entropies_all, last_token_entropies_all, _, token_wise_probs_all_responses, confidences = calculate_accuracy_and_uq(all_results_subset)

    # # Bin data and get bin indices
    # bin_centers, bin_accuracies, bin_indices = bin_data(confidences, accuracies)

    # # Create directories for each bin if they don't exist
    # for i in range(len(bin_centers)):
    #     os.makedirs(f'bin_{i}', exist_ok=True)

    # Plot and save the HTML visualizations for the subset of results
    for i, (result_id, result) in enumerate(tqdm(list(all_results_subset.items()), desc="Processing Results")):
        image_id = questions[result_id]['imageId']
        ground_truth_answer = questions[result_id]['answer']
        ground_truth_full_answer = questions[result_id]['fullAnswer']
        image_path = f'/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/{image_id}.jpg'
        question = questions[result_id]['question']  # Access question from the questions dictionary
        
        # Extract the full response, converting the dictionary to a string for display
        full_responses = [json.dumps(resp, indent=2) for resp in result['responses']]
    
        # Generate the HTML table with detailed metrics for all 20 responses
        table_html = generate_html_table(
            [entropy_sequence_probs[i]] * 20,  # Sequence probabilities for all responses (assuming constant)
            log_softmax_entropies_all[i],  # List of log softmax entropies for 20 responses
            last_token_entropies_all[i],  # List of last token entropies for 20 responses
            token_wise_probs_all_responses[i] if i < len(token_wise_probs_all_responses) else [[] for _ in range(20)],  # Token-wise probabilities
            result['tokens'],  # Token IDs
            full_responses  # Full response dictionaries
        )
        
        # Determine which bin this result belongs to
        # bin_idx = bin_indices[i]
        # output_html_path = f"bin_{bin_idx}/output_{image_id}.html"
        output_html_path = f"output_{image_id}.html"
        
        # Include the ground truth answer and full answer in the plot
        plot_image(image_path, question, full_responses, f"<p><strong>Ground Truth Answer:</strong> {ground_truth_answer}</p>" + 
                   f"<p><strong>Ground Truth Full Answer:</strong> {ground_truth_full_answer} Accuracy {accuracies[i]} </p>" + table_html, image_id, output_html_path)

if __name__ == "__main__":
    main()