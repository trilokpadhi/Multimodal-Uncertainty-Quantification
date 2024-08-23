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

def calculate_accuracy_and_uq(results):
    """Calculate accuracy, entropy of sequence probabilities, log softmax, and last token probabilities."""
    accuracies = []
    entropy_sequence_probs = []
    log_softmax_entropies = []
    last_token_entropies = []

    for result_id, result in results.items():
        responses = result['responses']
        correct_answer = result['answer'].strip().lower()
        
        print('Correct Answer:', correct_answer)
        
        # Accuracy Calculation
        total_correct = 0
        for resp in responses:
            try:
                answer = resp['answer'].strip().lower()
                if answer == correct_answer:
                    total_correct += 1
            except Exception as e:
                print('Error processing', e, 'in', resp)

        print('Number of correct answers:', total_correct)
        accuracy = total_correct / len(responses)
        accuracies.append(accuracy)
        
        # Sequence Probabilities Entropy
        sequence_probs = result['sequence_probabilities']
        entropy_seq_prob = entropy(sequence_probs)
        entropy_sequence_probs.append(entropy_seq_prob)
        
        # Log Softmax Entropy and Last Token Probability Entropy
        scores = result['scores']
        log_softmax_entropies_per_response = []
        last_token_entropies_per_response = []

        for score in scores:
            score = torch.tensor(score).squeeze(0)  # Remove batch dimension if present
            score = replace_infs_with_small_values(score)
            log_softmax_probs = torch.log_softmax(score, dim=-1)
            
            # Calculate entropy for log softmax probabilities
            entropy_log_softmax_probs = entropy(log_softmax_probs.numpy(), axis=-1).mean()
            log_softmax_entropies_per_response.append(entropy_log_softmax_probs)
            
            # Calculate entropy for last token probabilities
            last_token_probs = log_softmax_probs[-1, :].exp()  # Get the probabilities for the last token in the sequence
            entropy_last_token = entropy(last_token_probs.numpy())
            last_token_entropies_per_response.append(entropy_last_token)

        # Append the average entropy across all responses
        log_softmax_entropies.append(np.mean(log_softmax_entropies_per_response))
        last_token_entropies.append(np.mean(last_token_entropies_per_response))
    
    return accuracies, entropy_sequence_probs, log_softmax_entropies, last_token_entropies, sequence_probs, log_softmax_entropies_per_response, last_token_entropies_per_response

def generate_html_table(sequence_probs, log_softmax_entropies, last_token_entropies):
    """Generate an HTML table displaying sequence probabilities, log softmax entropies, and last token entropies."""
    table_html = "<table border='1'><tr><th>Response #</th><th>Sequence Probability</th><th>Log Softmax Entropy</th><th>Last Token Probability Entropy</th></tr>"
    
    for i in range(len(sequence_probs)):
        table_html += f"<tr><td>{i+1}</td><td>{sequence_probs[i]:.4f}</td><td>{log_softmax_entropies[i]:.4f}</td><td>{last_token_entropies[i]:.4f}</td></tr>"
    
    table_html += "</table>"
    return table_html

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

def main():
    # Load results
    results_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_temp1_topp1_dp_100/results'
    all_results = load_results(results_dir)

    with open("/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/questions1.2/train_all_questions/train_all_questions_0.json") as file:
        questions = json.load(file)

    all_results_subset = dict(itertools.islice(all_results.items(), 20))
    # Calculate accuracy and uncertainty measures
    accuracies, entropy_sequence_probs, log_softmax_entropies, last_token_entropies, _, _, _ = calculate_accuracy_and_uq(all_results_subset)

    # Plot and save the HTML visualizations for the first 20 results
    for i, (result_id, result) in enumerate(tqdm(list(all_results_subset.items()), desc="Processing Results")):
        image_id = questions[result_id]['imageId']
        image_path = f'/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/{image_id}.jpg'
        question = questions[result_id]['question']  # Access question from the questions dictionary
        # responses = [resp['answer'] for resp in result['responses']]
        responses = []
        for resp in result['responses']:
            try:
                responses.append(resp['answer'])
            except:
                responses.append(f'No answer could be extracted from {resp}')
        
        # Generate the HTML table
        table_html = generate_html_table(
            [entropy_sequence_probs[i]],  # Pass the sequence probabilities entropy for this specific result
            [log_softmax_entropies[i]],  # Pass the log softmax entropy for this specific result
            [last_token_entropies[i]]  # Pass the last token entropy for this specific result
        )
        
        # Generate additional information for HTML
        accuracy_info = f"<p><strong>Accuracy:</strong> {accuracies[i]:.4f}</p>"
        
        # Save the output HTML file
        output_html_path = f"output_{image_id}.html"  # Name the HTML file based on the result_id
        plot_image(image_path, question, responses, accuracy_info + table_html, image_id, output_html_path)



if __name__ == "__main__":
    main()

