import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from tqdm import tqdm
import pickle
from llava.mm_utils import get_model_name_from_path # type: ignore
from llava.model.builder import load_pretrained_model # type: ignore
import torch.nn.functional as F
import numpy as np

# Load the tokenizer and model for entailment checking
tokenizer_deberta = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
model_deberta = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli")


## Llava tokenizer 
model_path = "liuhaotian/llava-v1.6-vicuna-7b"
model_base = None
model_name = get_model_name_from_path(model_path)
tokenizer_llava, model_llava, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name
)
        
# Define functions
def compute_predictive_entropy(raw_logits, token_ids_list):
    """
    Compute predictive entropy given raw logits and sequence of token IDs.
    Returns:
        entropy: Entropy of 20 responses
        log_likelihoods: Log Likelihood of 20 responses
        log_probs: Log probs of 20 responses token-wise, list of [1, token_length, log probs]
        tokens_list: tokens_list of 20 responses
        decoded_tokens_list: Decoded tokens of 20 responses
    """
    log_likelihoods = []
    log_probs = []
    tokens_list = []
    decoded_tokens_list = []
    
    try:
        for logits, token_ids in zip(raw_logits, token_ids_list):
            try:
                """
                For selected tokens
                """
                # Compute log softmax for all tokens
                log_softmax_for_all_tokens = F.log_softmax(logits, dim=-1)
                
                # Get the start and end index for the specific tokens (IDs 362 and 5527)
                # start index + 1 to not include token 'ation'
                start_index = token_ids[0].tolist().index(362) + 1
                end_index = token_ids[0].tolist().index(5527) 
                
                # Select log probabilities for the given token IDs
                log_softmax_for_selected_tokens = log_softmax_for_all_tokens[:, start_index :end_index, :]
                log_prob = log_softmax_for_selected_tokens[0, torch.arange(log_softmax_for_selected_tokens.shape[1]), token_ids[:, start_index:end_index]]
                
                # Store log probabilities and log likelihoods
                log_probs.append(log_prob.squeeze())  # Mean log probability
                log_likelihoods.append(torch.mean(log_prob))  # Log likelihood for the selected range
                
                # Store token IDs and decoded tokens (you can replace 'convert_ids_to_tokens' with your tokenizer method)
                tokens_list.append(token_ids[:, start_index:end_index].squeeze().tolist())
                decoded_tokens_list.append(tokenizer_llava.convert_ids_to_tokens(token_ids[:, start_index:end_index].squeeze()))
                """
                For All tokens
                """
                # # Compute log softmax for all tokens
                # log_softmax_for_all_tokens = F.log_softmax(logits, dim=-1)
                # log_prob = log_softmax_for_all_tokens[0, torch.arange(log_softmax_for_all_tokens.shape[1]), token_ids]
                
                # # Store log probabilities and log likelihoods
                # log_probs.append(log_prob.squeeze())  # Mean log probability
                # log_likelihoods.append(torch.mean(log_prob))  # Log likelihood for the selected range
                
                # # Store token IDs and decoded tokens (you can replace 'convert_ids_to_tokens' with your tokenizer method)
                # tokens_list.append(token_ids.squeeze().tolist())
                # decoded_tokens_list.append(tokenizer_llava.convert_ids_to_tokens(token_ids.squeeze()))
                
            
            except Exception as e:
                print(f'Error calculating log probs for one set: {e}')
                log_probs.append(None)
                log_likelihoods.append(None)
                tokens_list.append(None)
                decoded_tokens_list.append(None)
        
        # Filter out None values (in case some responses failed)
        log_likelihoods_filtered = [lp for lp in log_likelihoods if lp is not None]
        
        # Calculate entropy
        entropy = -torch.stack(log_likelihoods_filtered).mean(dim=0).squeeze()

        # Return the computed values
        return entropy.item(), log_likelihoods, log_probs, tokens_list, decoded_tokens_list
    
    except Exception as e:
        print(f"Error calculating entropy: {e}")
        return None, None, None, None, None
    

def check_entailment(textA, textB):
    """
    Check whether textB entails textA using a pre-trained model.
    """
    try:
        inputs = tokenizer_deberta.batch_encode_plus(
            batch_text_or_text_pairs=[(textA, textB)],
            add_special_tokens=True, return_tensors="pt",
        )
        logits = model_deberta(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_class_id = torch.argmax(probs).item()
        return predicted_class_id == 0  # 0 indicates entailment
    except Exception as e:
        print(f"Error checking entailment: {e}")
        return False

def calculate_accuracy(ground_truth_answer, responses, question):
    """
    Calculate accuracy based on entailment with ground truth.
    """
    try:
        accuracy_of_responses = []
        entailment_count = 0
        response_not_counted = 0
        for response in responses:
            try:
                model_answer = response['answer']
                if check_entailment(f'{question} {ground_truth_answer}', f'{question} {model_answer}'):
                    entailment_count += 1
                    accuracy_of_responses.append('Entailment')
                else:
                    accuracy_of_responses.append('Contradiction')
            except Exception as e:
                print(f'Error {e} processing response: {response}')
                response_not_counted += 1
                accuracy_of_responses.append(f'Cannot calculate accuracy, Error processing {e}')

        average_accuracy = entailment_count / (len(responses) - response_not_counted)
        return average_accuracy, accuracy_of_responses
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return None

# Load GQA questions
questions_file_path = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/questions1.2/train_all_questions/train_all_questions_0.json'
try:
    with open(questions_file_path, 'r') as f:
        questions_data = json.load(f)
    print('GQA data loaded successfully')
except Exception as e:
    print(f"Error loading questions data: {e}")
    questions_data = {}

# Path to the results
path = "/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_temp1_topp1_dp_100/results"

if __name__=='__main__':
        
    responses_metadata_dict = {}
    
    # Step 1: Loop through each .pkl file and calculate entropy, accuracy, and other values, storing them for later scaling
    for filename in tqdm(os.listdir(path)):
        if filename.endswith('.pkl'):
            filepath = os.path.join(path, filename)
            try:
                with open(filepath, 'rb') as f:
                    r = pickle.load(f)

                question_id = filename.split('_')[1].split('.')[0]
                question = questions_data.get(question_id, {}).get('question', 'Question not found')
                fullanswer = questions_data.get(question_id, {}).get('fullAnswer', 'Answer not found')
                shortanswer = questions_data.get(question_id, {}).get('answer', 'Answer not found')
                
                raw_logits = r['scores']
                token_ids = r['tokens']
                model_responses = r['responses']

                # Calculate entropy and other required values
                entropy, log_likelihoods, log_probs, tokens, decoded_tokens = compute_predictive_entropy(raw_logits, token_ids)
                
                average_accuracy, accuracy_of_responses = calculate_accuracy(fullanswer, model_responses, question)
                                                
                # Store everything in a dictionary
                responses_metadata_per_sample = {
                    "question_id": question_id,
                    "entropy": entropy,
                    "log_likelihoods": log_likelihoods,
                    "log_probs": log_probs,
                    "tokens": tokens,
                    "decoded_tokens": decoded_tokens,
                    "average_accuracy": average_accuracy,
                    "accuracy_of_responses": accuracy_of_responses,
                    "model_responses": model_responses
                }
            
                responses_metadata_dict[filename] = responses_metadata_per_sample                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    # Step 2: Normalize the entropy values between min and max entropy
    # Extract the entropy values from the dictionary
    entropies_list = [metadata['entropy'] for metadata in responses_metadata_dict.values()]
    min_entropy = np.min(entropies_list)
    max_entropy = np.max(entropies_list)

    # Scaling entropy between 0 and 1
    scaled_entropies = [(entropy - min_entropy) / (max_entropy - min_entropy) for entropy in entropies_list]

    # confidences = 1 - scaled_entropies
    # Step 3: Update the dictionary with scaled entropy values
    # Iterate through the dictionary and update each entry
    for i, (filename, metadata) in enumerate(responses_metadata_dict.items()):
        metadata['scaled_entropy'] = scaled_entropies[i]
        metadata['confidence'] = 1.0 - metadata['scaled_entropy']
        
    # Define your confidence bins
    bins = np.linspace(0, 1, 11)  # Create 10 bins between 0 and 1
    bin_indices = np.digitize([metadata['confidence'] for metadata in responses_metadata_dict.values()], bins) - 1  # Get bin indices for each confidence value
    
    # ---------------- Table Generation for response ------------------------------
    # Loop through each .pkl file in the directory and compute entropy and accuracy
    for i, (filename, metadata) in enumerate(responses_metadata_dict.items()):
        
        try:
            question_id = filename.split('_')[1].split('.')[0]
            
            # Get the corresponding question
            question = questions_data.get(question_id, {}).get('question', 'Question not found')
            fullanswer = questions_data.get(question_id, {}).get('fullAnswer', 'Answer not found')
            shortanswer = questions_data.get(question_id, {}).get('answer', 'Answer not found')
            image_id = questions_data.get(question_id, {}).get('imageId', 'Image dosent exist')
            
            # Entropy and log likelihood values
            entropy = metadata['entropy']
            scaled_entropy = metadata['scaled_entropy']
            log_likelihoods = metadata["log_likelihoods"]
            log_probs = metadata["log_probs"]
            tokens = metadata["tokens"]
            decoded_tokens = metadata["decoded_tokens"]
            
            # Accuracy values
            average_accuracy = metadata['average_accuracy']
            accuracy_of_responses = metadata['accuracy_of_responses']

            # # Create a directory to store the images
            # output_dir = "html_plots"
            # os.makedirs(output_dir, exist_ok=True)
            # Determine the bin for the current confidence score
            bin_idx = bin_indices[i]

            # Create a directory to store the images for this bin
            output_dir = f"html_plots/bin_{bin_idx}"
            os.makedirs(output_dir, exist_ok=True)

            # HTML content starts here
            html_content = f"<h1>{question_id}</h1>\n"
            html_content += f"<h2>Question: {question}</h2>\n"
            html_content += f"<h3>Full Answer: {fullanswer}</h3>\n"
            html_content += f"<h3>Short Answer: {shortanswer}</h3>\n"
            html_content += f"<h3>Average Accuracy: {average_accuracy}</h3>\n"
            html_content += f"<h3>Entropy/ Uncertainty: {entropy}</h3>\n"
            html_content += f"<h3>Scaled Entropy/ Uncertainty: {scaled_entropy}</h3>\n"
            html_content += f"<h3> Confidence: {1 - scaled_entropy}</h3>\n"


            # plt.plot(np.arange(10), np.random.rand(10), marker='o', linestyle='-')
            image_dir = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images'
            # image_path = f'{image_dir}/{image_id}.jpg'
            image_path = os.path.join(image_dir, f'{image_id}.jpg')
            image_path_new = os.path.join(output_dir, f"image_{question_id}_{image_id}.jpg")
            # If you want to save the image to a new location
            if not os.path.exists(image_path_new):
                # Load and save the image (assuming it's needed, otherwise just reference the original path)
                img = plt.imread(image_path)
                plt.imshow(img)
                plt.axis('off')  # Turn off axis
                plt.savefig(image_path_new, bbox_inches='tight', pad_inches=0)
                plt.close()

            # Add the image to the HTML content
            html_content += f'<img src="image_{question_id}_{image_id}.jpg" alt="Image">\n'
                
            model_responses = metadata['model_responses']
            if question != 'Question not found':
                # Loop through each model response and add it to the HTML
                for i, response in enumerate(model_responses):
                    html_content += f"<h3>Response {i} {response}</h3>\n"
                    html_content += f"<h3>Accuracy of response {i}: {accuracy_of_responses[i]}</h3>\n"
                    html_content += f"<h3>Log Likelihood of response {i}: {log_likelihoods[i]}</h3>\n"
                    
                    # Create a table plot for token-wise probabilities for this response
                    fig, ax = plt.subplots()
                    ax.axis('tight')
                    ax.axis('off')
                    
                    
                    # Create the table data
                    table_data = [
                        ["Token", "decoded token", "Log Probability", "Probability"],
                        *[[tokens[i][j], decoded_tokens[i][j], f"{log_probs[i][j]:.2f}", f"{np.exp(log_probs[i][j]):.2f}"] for j in range(len(tokens[i]))]
                    ]
                                            
                    # Add the table to the plot with better formatting
                    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.4, 0.2, 0.2])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)  # Adjust cell scaling

                    # Save the table as an image
                    table_path = os.path.join(output_dir, f"table_{question_id}_{i+1}.png")
                    plt.savefig(table_path, bbox_inches='tight', pad_inches=0.1)  # Save with tight bounding box
                    plt.close()
                    
                    # Add the table image to the HTML content
                    # html_content += f'<img src="{table_path}" alt="Table for {response}">\n'
                    html_content += f'<img src="table_{question_id}_{i+1}.png" alt="Table for {response}">\n'

            # Save the complete HTML content to a file
            html_file_path = os.path.join(output_dir, f"question_visualization_{question_id}.html")
            with open(html_file_path, "w") as f:
                f.write(html_content)

            # Debugging info to verify the saved file
            print(f"Saved HTML to {html_file_path}")
                    
        except Exception as e:
            print('Coudnt generate html files due to issue', e)


