# python uq_llava_MMT.py --config-file configs/llava_mmt.yaml --log-file logs/llava_mmt.log
# For running in background: nohup python uq_llava_MMT.py --config-file configs/llava_mmt.yaml --log-file logs/llava_mmt.log > logs/llava_mmt.out 2>&1 &
import argparse
import torch
import yaml
import logging

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import spacy
from grakel import GraphKernel, Graph
import networkx as nx
import numpy as np
import random
import base64
import pandas as pd

import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
import functools

import multiprocessing
from functools import partial
from joblib import Parallel, delayed
import os
import numpy as np
from collections import Counter

from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import torch.multiprocessing as mp

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load SpaCy model
# nlp = spacy.load("en_core_web_sm")

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

# def calculate_entropy(probabilities):
#     """Calculate the entropy for a set of probabilities."""
#     entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
#     return entropy.item()
# def calculate_entropy_from_log_probs(log_probabilities):
#     """
#     Calculate the entropy from log probabilities.

#     Args:
#         log_probabilities (torch.Tensor): Tensor containing log probabilities.

#     Returns:
#         entropy (torch.Tensor): Entropy calculated from the log probabilities.
#     """
#     # Convert log probabilities back to probabilities
#     probabilities = torch.exp(log_probabilities)
    
#     # Calculate entropy
#     entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    
#     return entropy

def calculate_entropy_from_log_probs(log_probs):
    # Avoid -inf by replacing with a very small number
    log_probs = torch.clamp(log_probs, min=-1e6)
    entropy = -torch.mean(log_probs)
    return entropy

def eval_model(model_args, args):
    print(f"Process {mp.current_process().pid}: Starting eval_model")
    # Model
    print(f"Process {mp.current_process().pid}: Disabling torch init")
    disable_torch_init()

    print(f"Process {mp.current_process().pid}: Preparing query")
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if args.model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if args.model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    print(f"Process {mp.current_process().pid}: Determining conversation mode")
    if "llama-2" in args.model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in args.model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in args.model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in args.model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in args.model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    print(f"Process {mp.current_process().pid}: Preparing conversation and prompt")
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # image_files = image_parser(args)
    # images = load_images(image_files)
    print(f"Process {mp.current_process().pid}: Processing images")
    images = [args.image_file]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        args.image_processor,
        args.model.config
    ).to(args.model.device, dtype=torch.float16)

    print('prompt', prompt)
    input_ids = (
        tokenizer_image_token(prompt, args.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    """
    Below code is copied from : https://github.com/haotian-liu/LLaVA/issues/108 
    --------------------------------------------------------------------------
    generation_output = model.generate(
    input_ids,
    images=image_tensor.unsqueeze(0).half().cuda(),
    do_sample=True,
    temperature=0.2,
    max_new_tokens=1024,
    stopping_criteria=[stopping_criteria],
    # add following two lines
    return_dict_in_generate=True,
    output_scores=True)

    input_token_len = input_ids.shape[1]
    output_ids = generation_output.sequences[0, input_token_len:]
    output_scores = generation_output.scores
    -----------------------------------------------------------------------
    
    """
    print(f'Process {mp.current_process().pid}: Running model.generate')
    with torch.inference_mode():
        outputs = args.model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            # for log likelihood
            return_dict_in_generate=True,
            output_scores=True
        )
        
    print(f'Process {mp.current_process().pid}: Model.generate completed')
    
    # return outputs
    scores = torch.stack(outputs.scores, dim=1)
    probabilities = torch.softmax(scores, dim=-1)

    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    log_probabilities = torch.log(probabilities + epsilon)

    log_probabilities_length = log_probabilities.shape[1]
    generated_tokens = outputs.sequences[:, -log_probabilities_length:]

    # Select log probabilities corresponding to the generated tokens
    token_probs = torch.gather(log_probabilities, 2, generated_tokens.unsqueeze(-1)).squeeze(-1)

    # Compute entropy for the generated response
    entropy = calculate_entropy_from_log_probs(token_probs)
    print(f"Entropy: {entropy}")
    print('-'*50)

    decoded_outputs = args.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    print('outputs:', decoded_outputs[0].strip())
    print('-'*50)

    return decoded_outputs[0].strip(), entropy.cuda().item()

def extract_json_from_text(text):
    """Extracts JSON from the given text."""

    start_index = text.find("{")
    end_index = text.rfind("}") + 1
    json_string = text[start_index:end_index]
    
    return json.loads(json_string)

# def generate_responses(args):
#     if args.debug:
#         if hasattr(args, 'keys'):
#             indices = args.keys
#             print(f"Using debug, keys: {indices}")
#         else:
#             image_ids = random.sample(args.gqa_data.keys(), args.num_images)
#     else:
#         # indices = args.filtered_df.index.tolist()
#         # image_ids = args.gqa_data.keys()
#         image_ids = random.sample(args.gqa_data.keys(), 5000) # testing with 5000 images
    
#     results = {}
#     for idx in tqdm(image_ids, desc='Generating responses'):
#         # Get the question and image
#         question = args.gqa_data[idx]['question']
#         prompt = f"""
#         {question}
#         Give your answer in JSON format where the keys are answer(one word answer), explanation( explain your answer), and your confidence(varies between 0 to 1).
#         """
        
#         # prompt = f"{question}"
#         print(prompt)
        
#         # Get the image
#         image_data_root = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/'
#         image_id = args.gqa_data[idx]['imageId']
#         image = Image.open(image_data_root + f'{image_id}.jpg')

#         # Set the arguments
#         # args.image_file = image_file
#         args.image_file = image
#         args.query = prompt
#         args.temperature = args.temperature
#         responses = []
#         for _ in range(args.no_of_responses_each_sample):  # Get n responses
#             response, entropy = eval_model(args)
#             print(response)
#             print('-'*50)
#             if response:
#                 try:
#                     json_string = extract_json_from_text(response)
#                     responses.append(json_string)
#                 except Exception as e:
#                     print(f"Error parsing response: {e}")
#             else:
#                 print("No response was generated.")

#         results[idx] = {
#             'prompt': prompt,
#             'responses': responses,
#             'entropy': entropy,
#             'answer': args.gqa_data[idx]['answer'],
#             'full_answer': args.gqa_data[idx]['fullAnswer']
#         }
    
#     return results

# def generate_responses_worker(args, image_ids, results_queue):
#     results = {}
#     print(f"Process {mp.current_process().pid} started.")
#     for idx in tqdm(image_ids, desc=f'Generating responses on PID {mp.current_process().pid}'):
#         try:
#             question = args.gqa_data[idx]['question']
#             prompt = f"""
#             {question}
#             Give your answer in JSON format where the keys are answer(one word answer), explanation( explain your answer), and your confidence(varies between 0 to 1).
#             """
            
#             print(f"Process {mp.current_process().pid}: {prompt}")
            
#             image_data_root = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/'
#             image_id = args.gqa_data[idx]['imageId']
#             image = Image.open(image_data_root + f'{image_id}.jpg')

#             args.image_file = image
#             args.query = prompt
#             responses = []
#             for _ in range(args.no_of_responses_each_sample):
#                 print(f"Process {mp.current_process().pid}: Running eval_model")
#                 response, entropy = eval_model(args)
#                 print(f"Process {mp.current_process().pid}: Received response: {response}")
#                 print('-'*50)
#                 if response:
#                     try:
#                         json_string = extract_json_from_text(response)
#                         responses.append(json_string)
#                     except Exception as e:
#                         print(f"Error parsing response in process {mp.current_process().pid}: {e}")
#                 else:
#                     print(f"Process {mp.current_process().pid}: No response was generated.")
            
#             results[idx] = {
#                 'prompt': prompt,
#                 'responses': responses,
#                 'entropy': entropy,
#                 'answer': args.gqa_data[idx]['answer'],
#                 'full_answer': args.gqa_data[idx]['fullAnswer']
#             }
        
#         except Exception as e:
#             print(f"Error in process {mp.current_process().pid}: {e}")

#     print(f"Process {mp.current_process().pid} finished.")
#     results_queue.put(results)

# def generate_responses(args):
#     if args.debug:
#         if hasattr(args, 'keys'):
#             image_ids = args.keys
#             print(f"Using debug, keys: {image_ids}")
#         else:
#             image_ids = random.sample(args.gqa_data.keys(), args.num_images)
#     else:
#         image_ids = random.sample(args.gqa_data.keys(), 50)  # testing with 5000 images

#     # Split the image_ids into chunks for each process
#     num_processes = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()
#     chunk_size = len(image_ids) // num_processes
#     image_id_chunks = [image_ids[i:i + chunk_size] for i in range(0, len(image_ids), chunk_size)]

#     # Create a multiprocessing queue to collect results
#     results_queue = mp.Queue()

#     # Start the processes
#     processes = []
#     for i in range(min(num_processes, len(image_id_chunks))):
#         p = mp.Process(target=generate_responses_worker, args=(args, image_id_chunks[i], results_queue))
#         p.start()
#         processes.append(p)

#     # Collect results from all processes
#     results = {}
#     for _ in processes:
#         results.update(results_queue.get())

#     # Ensure all processes have finished execution
#     for p in processes:
#         p.join()

#     return results
 
class ModelArgs:
    def __init__(self, model_path, model_base):
        self.model_path = model_path
        self.model_base = model_base
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name
        )
        self.prompt = None
        self.temperature = None
        self.top_p = None
        self.num_beams = None
        self.max_new_tokens = None
        self.query = None
        self.image_file = None
        
def generate_responses_worker(args, image_ids, results_queue):
    model_args = ModelArgs(args.model_path, args.model_base)
    results = {}
    print(f"Process {mp.current_process().pid} started.")
    for idx in tqdm(image_ids, desc=f'Generating responses on PID {mp.current_process().pid}'):
        try:
            question = args.gqa_data[idx]['question']
            prompt = f"""
            {question}
            Give your answer in JSON format where the keys are answer(one word answer), explanation( explain your answer), and your confidence(varies between 0 to 1).
            """
            
            print(f"Process {mp.current_process().pid}: {prompt}")
            
            image_data_root = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/'
            image_id = args.gqa_data[idx]['imageId']
            image = Image.open(image_data_root + f'{image_id}.jpg')

            model_args.image_file = image
            model_args.query = prompt
            responses = []
            entropy = None
            for _ in range(args.no_of_responses_each_sample):
                print(f"Process {mp.current_process().pid}: Running eval_model")
                response, current_entropy = eval_model(model_args, args)
                print(f"Process {mp.current_process().pid}: Received response: {response}")
                print('-'*50)
                if response:
                    try:
                        json_string = extract_json_from_text(response)
                        responses.append(json_string)
                        if entropy is None:
                            entropy = current_entropy
                    except Exception as e:
                        print(f"Error parsing response in process {mp.current_process().pid}: {e}")
                else:
                    print(f"Process {mp.current_process().pid}: No response was generated.")
            
            results[idx] = {
                'prompt': prompt,
                'responses': responses,
                'entropy': entropy,
                'answer': args.gqa_data[idx]['answer'],
                'full_answer': args.gqa_data[idx]['fullAnswer']
            }
        
        except Exception as e:
            print(f"Error in process {mp.current_process().pid}: {e}")

    print(f"Process {mp.current_process().pid} finished.")
    results_queue.put(results)


        
def generate_responses(args):    
    if args.debug:
        if hasattr(args, 'keys'):
            image_ids = args.keys
            print(f"Using debug, keys: {image_ids}")
        else:
            image_ids = random.sample(list(args.gqa_data.keys()), args.num_images)
    else:
        image_ids = random.sample(list(args.gqa_data.keys()), 50)  # testing with 50 images

    # Split the image_ids into chunks for each process
    # num_processes = min(torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(), 4)  # Limit to 4 processes
    num_processes = 1
    chunk_size = len(image_ids) // num_processes
    image_id_chunks = [image_ids[i:i + chunk_size] for i in range(0, len(image_ids), chunk_size)]

    # Create a multiprocessing queue to collect results
    results_queue = mp.Queue()

    # Start the processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=generate_responses_worker, args=(args, image_id_chunks[i], results_queue))
        p.start()
        processes.append(p)

    # Collect results from all processes
    results = {}
    for _ in processes:
        results.update(results_queue.get())

    # Ensure all processes have finished execution
    for p in processes:
        p.join()

    return results

def calculate_uncertainty_by_self_consistency(args):
    # uncertainty_results = {}
        
    for idx, results in tqdm(args.responses.items(), desc='Calculating uncertainty by self-consistency'):
        if not results['responses']:
            results['consistency'] = None
            continue
        else:
            answers = [response['answer'] for response in results['responses']]
            
            # Count the occurrences of each answer
            answer_counts = Counter(answers)
            
            # Find the proportion of the most common answer
            most_common_answer_count = answer_counts.most_common(1)[0][1]
            total_responses = len(answers)
            
            consistency = most_common_answer_count / total_responses
            uncertainty = 1 - consistency  # Higher consistency means lower uncertainty
            
            results['uncertainty_by_self_consistency'] = uncertainty

        
    return args.responses

def extract_entities_and_relationships_llama3(text, args):
    prompt = f"""
    Extract entities, attributes, and relationships. Provide the output as a JSON object, with the following structure:
    {{
        "entities": [
            {{"type": "whole", "name": "entity1"}},
            {{"type": "whole", "name": "entity2"}},
            ...
        ],
        "attributes": [
            {{"type": "color", "entity": "entity1", "value": "color1"}},
            {{"type": "state", "entity": "entity2", "value": "state1"}},
            {{"type": "material", "entity": "entity1", "value": "material1"}},
            {{"type": "shape", "entity": "entity2", "value": "shape1"}},
            {{"type": "size", "entity": "entity1", "value": "size1"}},
            {{"type": "location", "entity": "entity2", "value": "location1"}},
            {{"type": "quantity", "entity": "entity1", "value": "quantity1"}},
            ...
        ],
        "relations": [
            {{"type": "spatial", "entity1": "entity1", "entity2": "entity2", "description": "relation1"}},
            {{"type": "action", "entity1": "entity2", "entity2": "entity1", "description": "relation2"}},
            ...
        ]
    }}

    Example:
    Text: "A blue motorcycle parked by paint chipped doors."
    Output:
    {{
        "entities": [
            {{"type": "whole", "name": "motorcycle"}},
            {{"type": "whole", "name": "doors"}}
        ],
        "attributes": [
            {{"type": "color", "entity": "motorcycle", "value": "blue"}},
            {{"type": "state", "entity": "doors", "value": "paint chipped"}},
            {{"type": "state", "entity": "motorcycle", "value": "parked"}}
        ],
        "relations": [
            {{"type": "spatial", "entity1": "motorcycle", "entity2": "doors", "description": "next to"}}
        ]
    }}

    Text: "{text}"

    Output:
    """
    messages = [
    {"role": "system", "content": "You are a honest assitant"},
    {"role": "user", "content": prompt},
    ]
    
    # Combine messages into a single string prompt
    # prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    input_ids = args.llama_tokenizer.apply_chat_template(messages,
                                              add_generation_prompt=True,
                                              return_tensors="pt").to(args.llama_model.device)

    # Get the EOS token ID
    terminators = [
        args.llama_tokenizer.eos_token_id,
        args.llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    outputs = args.llama_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    assistant_response = args.llama_tokenizer.decode(response, skip_special_tokens=True)
    
    return assistant_response

import json
def extract_json_from_text(text):
    """Extracts JSON from the given text."""

    start_index = text.find("{")
    end_index = text.rfind("}") + 1
    json_string = text[start_index:end_index]

    return json.loads(json_string)

# Function to construct triples
def construct_triples(data):
    triples = []
    try:
        # Extract entity triples
        for entity in data['entities']:
            name = entity['name']
            entity_type = entity['type']
            triples.append((name, "is_a", entity_type + " entity"))
    except Exception as e:
        print(f"Error forming triples from entities: {e}")


    try:
        # Extract attributes
        for attr in data['attributes']:
            entity = attr['entity']
            predicate = attr['type']
            value = attr['value']
            triples.append((entity, predicate, value))
    except Exception as e:
        print(f"Error forming triples from attributes: {e}")
        
        
    try:
        # Extract relations
        for rel in data['relations']:
            entity1 = rel['entity1']
            predicate = rel['description']
            entity2 = rel['entity2']
            triples.append((entity1, predicate, entity2))
            
    except Exception as e:
        print(f"Error forming triples from relations: {e}")

    return triples

def extract_triples_scene_graph(scene_graph):
    triples = set()  # Use a set to store unique triples
        
    # Iterate over each object in the scene graph
    for obj_id, obj_data in scene_graph['objects'].items():
        head_object = obj_data['name']
        
        # Iterate over each relation the object has with other objects
        if len(obj_data['relations']) > 0:
            for relation in obj_data['relations']:
                tail_object_id = relation['object']
                relationship = relation['name']
                # tail_object = object_dict[tail_object_id]
                # Directly access the tail object name without an intermediate dictionary
                tail_object = scene_graph['objects'][tail_object_id]['name']
                triple = (head_object, f'{relationship}', tail_object)
                triples.add(triple)

        if len(obj_data['attributes']) > 0:
            for attribute in obj_data['attributes']:
                tail_object = attribute
                relationship = 'has_attribute'
                triple = (head_object, relationship, tail_object)
                triples.add(triple)
    return triples

def preprocess_triple(triple):
    """Normalize the triple by converting to lowercase and joining the elements into a single string."""
    return ' '.join(word.lower() for word in triple)

def calculate_confidence_metric(ground_truth_set, output_set, k=3):
    """Calculate the confidence metric based on the top-K similarities between two sets of triples."""
    
    # Preprocess triples and convert to strings
    ground_truth_list = [preprocess_triple(triple) for triple in ground_truth_set]
    output_list = [preprocess_triple(triple) for triple in output_set]
    
    # Batch encode all triples in both sets
    embeddings_gt = model.encode(ground_truth_list, convert_to_tensor=True)
    embeddings_out = model.encode(output_list, convert_to_tensor=True)
    
    # Compute the cosine similarity matrix in a vectorized manner
    similarity_matrix = util.pytorch_cos_sim(embeddings_gt, embeddings_out).cpu().numpy()
    
    # Flatten the matrix and sort similarities in descending order
    sorted_similarities = np.sort(similarity_matrix.flatten())[::-1]
    
    print('sorted_similarities', sorted_similarities)
    # Take the top-K similarities
    top_k_similarities = sorted_similarities[:k]
    
    # Calculate the confidence metric as the average of the top-K similarities
    confidence_metric = np.mean(top_k_similarities)
    
    return confidence_metric


def calculate_uncertainty_by_grounding(args):
    
    for idx, results in tqdm(args.responses.items(), desc='Calculating uncertainty by grounding'):
        responses = results['responses']
        explanations = []
        for s in responses:
            try:
                explanations.append(s['explanation'])
            except Exception as e:
                print(f"Error: {e}")
        for explanation in explanations:
            llama3_graph = extract_entities_and_relationships_llama3(explanation, args)
            json_graph = extract_json_from_text(llama3_graph)
            triples = construct_triples(json_graph) # triples from the explanation of LLM
            gt_scene_graph = args.scene_graphs[idx] # ground truth scene graph
            ground_truth_triples = extract_triples_scene_graph(gt_scene_graph) # triples from the ground truth scene graph
            confidence_grounding = calculate_confidence_metric(ground_truth_triples, triples)
            results['confidence_grounding'] = confidence_grounding
            results['llama3_graph'] = llama3_graph
            print(f"Confidence by grounding: {confidence_grounding}")
        
    return args.responses
            

def load_args_from_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return argparse.Namespace(**config)

def is_serializable(obj):
    """
    Check if an object is serializable.
    """
    try:
        yaml.dump(obj)
        return True
    except (TypeError, ValueError):
        return False
    
def log_args(args):
    # with open(log_file, 'w') as file:
    #     yaml.dump(vars(args), file)
    # args_dict = vars(args)
    # clean_args_dict = {k: v for k, v in args_dict.items() if is_serializable(v)}
    # print(args_dict)  # Debug print to check the contents
    # with open(args.log_file, 'w') as file:
    #     yaml.dump(args_dict, file)

    args_dict = vars(args)
    clean_args_dict = {}
    for k, v in args_dict.items():
        if is_serializable(v):
            clean_args_dict[k] = v
        else:
            clean_args_dict[k] = str(v)  # Convert non-serializable objects to strings
    with open(args.log_file, 'w') as file:
        yaml.dump(clean_args_dict, file)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", help="Path to the config file.")
    # parser.add_argument("--log-file", help="Path to the log file.")
    args = parser.parse_args()

    
    # # Load arguments from config file
    config_args = load_args_from_config(args.config_file)
    
    # Update args with loaded config_args
    args.__dict__.update(config_args.__dict__)


    if args.get_response:        
        # dataset
        if args.dataset == 'mmt_bench':
            df = pd.read_csv(args.data_path, sep = '\t')
            args.filtered_df = df[df['category'].str.contains(args.category, case=False)]
        elif args.dataset == 'gqa':
            with open(args.data_path, 'r') as file:
                data = json.load(file)
                args.gqa_data = data
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")
        
        args.responses = generate_responses(args)

        # Save the responses 
        with open(args.responses_path, 'w') as file:
            json.dump(args.responses, file)
        
        print("Responses saved successfully.")
    
    else:
        print("No response generated since args.get_response is set to False.")

    if args.get_uncertainty:
        # if args.responses is None:
        if not hasattr(args, 'responses'):
            print("Loading responses from file.")
            with open(args.responses_path, 'r') as file:
                args.responses = json.load(file)

        if not hasattr(args, 'filtered_df'):
            if args.dataset == 'mmt_bench':
                df = pd.read_csv(args.data_path, sep = '\t')
                args.filtered_df = df[df['category'].str.contains(args.category, case=False)]
            elif args.dataset == 'gqa':
                with open(args.data_path, 'r') as file:
                    data = json.load(file)
                    args.gqa_data = data
            else:
                raise ValueError(f"Invalid dataset: {args.dataset}")
        # For Graph Extract - Llama 3
        args.llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        args.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model_id)
        args.llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )            
        
        # Load scene graph data for GQA
        scene_graphs_file = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/sceneGraphs/train_sceneGraphs.json'
        
        # Sentence Transformer model
        args.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        with open(scene_graphs_file, 'r') as file:
            scene_graphs = json.load(file)
            args.scene_graphs = scene_graphs
        responses_with_uncertainty = calculate_uncertainty_by_grounding(args)

    #     # Embedding model for nodes of graphs
    #     args.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #     if args.uncertainty_method == 'node_and_structural_similarity':
    #         responses_with_uncertainty = quantify_uncertainty_from_image_captions_with_node_and_structural_simlarity(args)
    #     elif args.uncertainty_method == 'vec_similarity_bigram_overlap':
    #         responses_with_uncertainty = quantify_uncertainty_from_image_captions_with_vec_similarity_bigram_overlap(args)
    #     elif args.uncertainty_method == 'self_consistency':
    #         responses_with_uncertainty = calculate_uncertainty_by_self_consistency(args)
    #     elif args.uncertainty_method == 'log_likelihood':
    #         responses_with_uncertainty = quantify_uncertainty_log_likelihood(args.responses)
    #     else:
    #         raise ValueError(f"Invalid uncertainty method: {args.uncertainty_method}")
            
    #     # Save the results
    #     with open(args.output_path, 'w') as file:
    #         json.dump(responses_with_uncertainty, file)

    #     print("Uncertainty Results saved successfully.")

    # # Log the arguments
    # log_args(args)
