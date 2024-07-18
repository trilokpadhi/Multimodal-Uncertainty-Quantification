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
from transformers import AutoProcessor, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
import functools

import multiprocessing
from functools import partial
from joblib import Parallel, delayed
import os
import numpy as np
import copy

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

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

def eval_model(args):
    # Model
    disable_torch_init()

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

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # image_files = image_parser(args)
    # images = load_images(image_files)
    images = [args.image_file]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        args.image_processor,
        args.model.config
    ).to(args.model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, args.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = args.model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = args.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)
    return outputs

# Function to decode and save image
def decode_and_save_image(image_data_base64, image_id):
    image_data = base64.b64decode(image_data_base64)
    image = Image.open(BytesIO(image_data))
    image_file = f'decoded_image_{image_id}.png'
    image.save(image_file)
    return image_file

# Function to extract entities and relationships using SpaCy
def extract_entities_and_relationships(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    relationships = []
    for token in doc:
        if token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ in ["nsubj", "dobj", "prep"]:
                    relationships.append((token.text, child.text))
    if not entities:
        for chunk in doc.noun_chunks:
            entities.append((chunk.text, 'NOUN_CHUNK'))
    return entities, relationships

# Function to construct graph from entities and relationships
def construct_graph(entities, relationships):
    G = nx.DiGraph()
    for entity, label in entities:
        G.add_node(entity, label=label)
    for subj, obj in relationships:
        if not G.has_node(subj):
            G.add_node(subj, label="unknown")
        if not G.has_node(obj):
            G.add_node(obj, label="unknown")
        G.add_edge(subj, obj)
    return G

# Assuming `eval_model` and `Args` are already defined as in previous examples
def generate_responses(args):
    if args.debug:
        indices = random.sample(args.filtered_df.index.tolist(), args.num_images)
    else:
        indices = args.filtered_df.index.tolist()
    
    results = {}
    for idx in tqdm(indices):
        with pd.option_context('display.max_colwidth', None):
            question = args.filtered_df.loc[idx, 'question']
            choice_a = args.filtered_df.loc[idx, 'A']
            choice_b = args.filtered_df.loc[idx, 'B']
            choice_c = args.filtered_df.loc[idx, 'C']
            choice_d = args.filtered_df.loc[idx, 'D']

            prompt = f"""
            {question}
            A. {choice_a}
            B. {choice_b}
            C. {choice_c}
            D. {choice_d}
            
            Give your answer in JSON format where the keys are answer(one or more options above), explanation( explain your answer), and confidence(varies between 0 to 1).
            
            """
            print(prompt)
        
        # Decode the image
        image_data_base64 = args.filtered_df.loc[idx, 'image']
        image_data = base64.b64decode(image_data_base64)
        image = Image.open(BytesIO(image_data))

        # Set the arguments
        # args.image_file = image_file
        args.image_file = image
        args.query = prompt
        args.temperature = args.temperature
        responses = []
        for _ in range(args.no_of_responses_each_sample):  # Get n responses
            response = eval_model(args)
            print(response)
            print('-'*50)
            if response:
                try:
                    """
                    For responses as below (Perfect JSON format):
                    {
                        "answer": "B",
                        "explanation": "The sandwich is on the foil, which is a common way to wrap food to keep it fresh and prevent it from sticking to surfaces. The foil is not on the lunch, nor is the lunch on the foil. The sandwich is the main subject of the image, and it is placed on the foil, which is a piece of paper or aluminum that is commonly used for wrapping food.",
                        "confidence": 0.9
                    }
                    """
                    responses.append(json.loads(response))
                except Exception as e:
                    try:
                        """
                        For responses as below:
                        ```json
                        {
                        "answer": "C",
                        "explanation": "The woman is wearing blue jeans. The jeans are a common type of clothing item that is typically worn as pants. They are a popular choice for casual wear and are often associated with a relaxed, comfortable style. The jeans in the image are likely a key feature of the woman's outfit, as they are a common and recognizable type of clothing.",
                        "confidence": 0.9
                        }
                        ```
                        """
                        json_string = response.strip().removeprefix('```json').removesuffix('```').strip()
                        responses.append(json.loads(json_string))
                    except Exception as e:
                        print(f"Error parsing response: {e}")

            else:
                print("No response was generated.")

        results[idx] = {
            'prompt': prompt,
            'responses': responses,
        }
    
    return results

def node_similarity(embedding_model, node1, node2):
    """
    Calculate similarity between two nodes based on their vector representations.
    """
    vec1 = embedding_model.encode([node1])[0]
    vec2 = embedding_model.encode([node2])[0]
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def encode_nodes(graph, embedding_model, vector_dict):
    nodes_to_encode = [node for node in graph.nodes if node not in vector_dict]
    if nodes_to_encode:
        new_vectors = embedding_model.encode(nodes_to_encode)
        vector_dict.update(zip(nodes_to_encode, new_vectors))
    
    for node in graph.nodes:
        graph.nodes[node]['vector'] = vector_dict[node]

def graph_similarity_kernel(embedding_model, graph1, graph2, device):
    """
    Kernel function to compute similarity between two graphs based on node similarity and graph structure.
    """

    vectors1 = []
    vectors2 = []

    # Precompute vectors for all unique nodes
    all_nodes = set(graph1.nodes) | set(graph2.nodes)
    vector_dict = {}

    # Encode in batches
    batch_size = 1000  # Adjust this based on your model's capabilities and memory constraints
    for i in range(0, len(all_nodes), batch_size):
        batch = list(all_nodes)[i:i+batch_size]
        batch_vectors = embedding_model.encode(batch, device=device)
        vector_dict.update(zip(batch, batch_vectors))

    # Apply vectors to graphs
    encode_nodes(graph1, embedding_model, vector_dict)
    encode_nodes(graph2, embedding_model, vector_dict)

    # Extract vectors
    vectors1 = [graph1.nodes[node]['vector'] for node in graph1.nodes]
    vectors2 = [graph2.nodes[node]['vector'] for node in graph2.nodes]

    # Compute cosine similarity between all pairs of vectors
    similarity_matrix = cosine_similarity(vectors1, vectors2)

    # Take the average similarity
    node_similarity = np.mean(similarity_matrix)

    # Compare graph structures (you can adjust this part)
    # structure_similarity = 1 if len(graph1) == len(graph2) else 0
    structure_similarity = graph_edit_distance_with_node_similarity(embedding_model, graph1, graph2)

    # Combine node and structure similarity (you can adjust the weights)
    return 0.5 * node_similarity + 0.5 * structure_similarity

def graph_edit_distance_with_node_similarity(embedding_model, G1, G2):
    """
    Calculate a modified Graph Edit Distance that incorporates node similarity.
    """
    # Node substitution cost
    node_subst_cost = np.zeros((len(G1), len(G2)))
    for i, n1 in enumerate(G1.nodes(data=True)):
        for j, n2 in enumerate(G2.nodes(data=True)):
            node_subst_cost[i, j] = 1 - node_similarity(embedding_model, n1[1], n2[1])

    # Edge substitution cost
    edge_subst_cost = 1.0

    # Node insertion/deletion cost
    node_ins_del_cost = 1.0

    # Edge insertion/deletion cost
    edge_ins_del_cost = 1.0

    # Calculate costs
    cost_matrix = node_subst_cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment_cost = cost_matrix[row_ind, col_ind].sum()

    # Count unmatched nodes
    unmatched_nodes = abs(len(G1) - len(G2))

    # Count edge differences
    G1_edges = set(G1.edges())
    G2_edges = set((row_ind[i], col_ind[i]) for i in range(min(len(G1), len(G2))) if (row_ind[i], col_ind[i]) in G2.edges())
    edge_diff = len(G1_edges.symmetric_difference(G2_edges))

    total_cost = (assignment_cost + 
                  node_ins_del_cost * unmatched_nodes + 
                  edge_ins_del_cost * edge_diff)

    max_cost = max(len(G1), len(G2)) * (node_ins_del_cost + edge_ins_del_cost * max(G1.number_of_edges(), G2.number_of_edges()))
    similarity = 1 - (total_cost / max_cost)

    return similarity

# def quantify_uncertainty_from_image_captions_with_node_and_structural_simlarity(args):
#     """
#     Here we use node and structural similarity as features to compute the similarity between two graphs, and then use these similarities to compute the uncertainty.
#     """

#     for idx, result in tqdm(args.responses.items(), desc='Calculating uncertainty'):
#         responses = result['responses']
        
#         # get the dataframe and the image from the index
#         image_data_base64 = args.filtered_df.loc[int(idx), 'image']
#         image_data = base64.b64decode(image_data_base64)
#         image = Image.open(BytesIO(image_data))
        
#         # get caption from florence-large
#         task_prompt = '<CAPTION>'
#         caption = get_caption(args, task_prompt, image)  # proxy for ground truth

#         graphs = []

#         all_sentences = [caption[task_prompt]] + [s['explanation'] for s in responses]
#         for sentence in all_sentences:
#             entities, relationships = extract_entities_and_relationships(sentence)
#             G = construct_graph(entities, relationships)
#             graphs.append(G)    

#         n = len(graphs)
#         K = np.zeros((n, n))
#         for i in range(n):
#             for j in range(i, n):
#                 K[i, j] = graph_similarity_kernel(args, graphs[i], graphs[j])
#                 K[j, i] = K[i, j]   
#         # Parallel computation of kernel values
        
#         embedding_model = args.embedding_model
#         results = Parallel(n_jobs=4)(delayed(compute_kernel)(embedding_model, graphs, i, j) for i in range(n) for j in range(i, n))

        
#         for i, j, value in results:
#             K[i, j] = value
#             K[j, i] = value


#         alpha = 0.5
#         beta = 0.5
#         ground_truth_index = 0
#         similarities_within_group = K[1:, 1:][np.triu_indices(n-1, k=1)]  # similarities within group of responses
#         similarities_with_ground_truth = K[ground_truth_index, 1:]  # similarities with ground truth
#         avg_similarity_within_group = np.mean(similarities_within_group)
#         avg_similarity_with_ground_truth = np.mean(similarities_with_ground_truth)
#         uncertainty = alpha * (1 - avg_similarity_within_group) + beta * (1 - avg_similarity_with_ground_truth)
#         result['uncertainty'] = uncertainty

#     return args.responses

import torch
import torch.multiprocessing as mp
from functools import partial

import torch
import torch.multiprocessing as mp
from functools import partial
import os

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
from functools import partial

def process_chunk(chunk, embedding_model, graphs, device):
    results = []
    for i, j in chunk:
        with torch.no_grad():
            value = graph_similarity_kernel(embedding_model, graphs[i], graphs[j], device)
        results.append((i, j, value))
    return results

def worker_function(gpu_id, chunks, embedding_model, graphs):
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    embedding_model = embedding_model.to(device)
    results = []
    for chunk in chunks:
        results.extend(process_chunk(chunk, embedding_model, graphs, device))
    return results

# def quantify_uncertainty_from_image_captions_with_node_and_structural_simlarity(args):
#     num_gpus = torch.cuda.device_count()
#     print(f"Number of available GPUs: {num_gpus}")

#     # Create a pool of processes, one for each GPU
#     mp.set_start_method('spawn', force=True)
#     pool = mp.Pool(6) # To run on 6 GPUs

#     # Load the embedding model (it will be moved to specific GPUs in worker_function)
#     embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#     for idx, result in tqdm(args.responses.items(), desc='Calculating uncertainty'):
#         responses = result['responses']
        
#         image_data_base64 = args.filtered_df.loc[int(idx), 'image']
#         image_data = base64.b64decode(image_data_base64)
#         image = Image.open(BytesIO(image_data))
        
#         task_prompt = '<CAPTION>'
#         caption = get_caption(args, task_prompt, image)

#         graphs = []
#         all_sentences = [caption[task_prompt]] + [s['explanation'] for s in responses]
#         for sentence in all_sentences:
#             entities, relationships = extract_entities_and_relationships(sentence)
#             G = construct_graph(entities, relationships)
#             graphs.append(G)    

#         n = len(graphs)
#         K = np.zeros((n, n))

#         # Prepare all pairs
#         all_pairs = [(i, j) for i in range(n) for j in range(i, n)]
        
#         # Divide work among GPUs
#         chunk_size = 100  # Adjust based on your GPU memory
#         chunks = [all_pairs[i:i+chunk_size] for i in range(0, len(all_pairs), chunk_size)]
#         chunks_per_gpu = [chunks[i::num_gpus] for i in range(num_gpus)]

#         # Process chunks on multiple GPUs
#         worker_func = partial(worker_function, embedding_model=embedding_model, graphs=graphs)
#         all_results = pool.starmap(worker_func, enumerate(chunks_per_gpu))

#         # Combine results
#         for results in all_results:
#             for i, j, value in results:
#                 K[i, j] = value
#                 K[j, i] = value

#         alpha = 0.5
#         beta = 0.5
#         ground_truth_index = 0
#         similarities_within_group = K[1:, 1:][np.triu_indices(n-1, k=1)]
#         similarities_with_ground_truth = K[ground_truth_index, 1:]
#         avg_similarity_within_group = np.mean(similarities_within_group)
#         avg_similarity_with_ground_truth = np.mean(similarities_with_ground_truth)
#         uncertainty = alpha * (1 - avg_similarity_within_group) + beta * (1 - avg_similarity_with_ground_truth)
#         result['uncertainty'] = uncertainty

#     pool.close()
#     pool.join()

#     return args.responses

def process_graphs_on_gpu(gpu_id, graphs, embedding_model):
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    embedding_model = embedding_model.to(device)
    
    n = len(graphs)
    K = torch.zeros((n, n), device=device)
    
    for i in range(n):
        for j in range(i, n):
            with torch.no_grad():
                value = graph_similarity_kernel(embedding_model, graphs[i], graphs[j], device)
            K[i, j] = value
            K[j, i] = value
    
    return K.cpu().numpy()

def quantify_uncertainty_from_image_captions_with_node_and_structural_simlarity(args):
    num_gpus = min(3, torch.cuda.device_count())  # Use up to 3 GPUs
    print(f"Number of GPUs being used: {num_gpus}")

    # Create a pool of processes, one for each GPU
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(num_gpus)

    # Load the embedding model (it will be moved to specific GPUs in worker_function)
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    for idx, result in tqdm(args.responses.items(), desc='Calculating uncertainty'):
        responses = result['responses']
        
        image_data_base64 = args.filtered_df.loc[int(idx), 'image']
        image_data = base64.b64decode(image_data_base64)
        image = Image.open(BytesIO(image_data))
        
        task_prompt = '<CAPTION>'
        caption = get_caption(args, task_prompt, image)

        graphs = []
        all_sentences = [caption[task_prompt]] + [s['explanation'] for s in responses]
        for sentence in all_sentences:
            entities, relationships = extract_entities_and_relationships(sentence)
            G = construct_graph(entities, relationships)
            graphs.append(G)    

        n = len(graphs)
        
        # Divide work among GPUs
        graphs_per_gpu = [graphs[i::num_gpus] for i in range(num_gpus)]

        # Process graphs on multiple GPUs
        worker_func = partial(process_graphs_on_gpu, embedding_model=embedding_model)
        K_parts = pool.starmap(worker_func, enumerate(graphs_per_gpu))

        # Combine results
        K = np.zeros((n, n))
        for i, K_part in enumerate(K_parts):
            K[i::num_gpus, i::num_gpus] = K_part

        alpha = 0.5
        beta = 0.5
        ground_truth_index = 0
        similarities_within_group = K[1:, 1:][np.triu_indices(n-1, k=1)]
        similarities_with_ground_truth = K[ground_truth_index, 1:]
        avg_similarity_within_group = np.mean(similarities_within_group)
        avg_similarity_with_ground_truth = np.mean(similarities_with_ground_truth)
        uncertainty = alpha * (1 - avg_similarity_within_group) + beta * (1 - avg_similarity_with_ground_truth)
        result['uncertainty'] = uncertainty

    pool.close()
    pool.join()

    return args.responses


def compute_kernel(embedding_model, graphs, i, j, device):
    value = graph_similarity_kernel(embedding_model, graphs[i], graphs[j], device)
    return (i, j, value)
        
def get_caption(args, task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = args.caption_model_processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = args.caption_model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = args.caption_model_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = args.caption_model_processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer


def load_args_from_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return argparse.Namespace(**config)

        
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
        args.model_name = get_model_name_from_path(args.model_path)
        args.tokenizer, args.model, args.image_processor, args.context_len = load_pretrained_model(
            args.model_path, args.model_base, args.model_name
        )
        
        # generate results and calculate uncertainty
        df = pd.read_csv(args.data_path, sep = '\t')
        args.filtered_df = df[df['category'].str.contains(args.category, case=False)]
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
            df = pd.read_csv(args.data_path, sep = '\t')
            args.filtered_df = df[df['category'].str.contains(args.category, case=False)]

        # Uncertainty quantification
        # Load the florence-large model 
        args.caption_model_id = 'microsoft/Florence-2-large'
        args.caption_model = AutoModelForCausalLM.from_pretrained(args.caption_model_id, trust_remote_code=True).eval().cuda()
        args.caption_model_processor = AutoProcessor.from_pretrained(args.caption_model_id, trust_remote_code=True)

        # Start multiprocessing 
        # multiprocessing.set_start_method('spawn', force=True)

        # Embedding model for nodes of graphs
        args.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        if args.uncertainty_method == 'node_and_structural_similarity':
            responses_with_uncertainty = quantify_uncertainty_from_image_captions_with_node_and_structural_simlarity(args)
        elif args.uncertainty_method == 'vec_similarity_bigram_overlap':
            # responses_with_uncertainty = quantify_uncertainty_from_image_captions_with_vec_similarity_bigram_overlap(args)
            raise NotImplementedError("Method not implemented yet.")
        else:
            raise ValueError(f"Invalid uncertainty method: {args.uncertainty_method}")
            
        # Save the results
        with open(args.output_path, 'w') as file:
            json.dump(responses_with_uncertainty, file)

        print("Uncertainty Results saved successfully.")
