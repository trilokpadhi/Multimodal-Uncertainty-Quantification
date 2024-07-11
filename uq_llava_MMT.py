# python uq_llava_MMT.py --config-file configs/llava_mmt.yaml --log-file logs/llava_mmt.log

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

# Function to convert NetworkX graphs to GraKeL graphs
def convert_to_grakel_graphs(graphs):
    grakel_graphs = []
    for G in graphs:
        if len(G.nodes) > 0 and len(G.edges) > 0:
            node_labels = {i: G.nodes[node]['label'] for i, node in enumerate(G.nodes)}
            edges = [(list(G.nodes).index(u), list(G.nodes).index(v)) for u, v in G.edges()]
            grakel_graphs.append(Graph(edges, node_labels=node_labels))

    return grakel_graphs

# Function to compute Weisfeiler-Lehman kernel and calculate uncertainty
def calculate_uncertainty(graphs):
    grakel_graphs = convert_to_grakel_graphs(graphs)
    
    if not grakel_graphs:
        print("No valid graphs were generated.")
        return None

    gk = GraphKernel(kernel={"name": "weisfeiler_lehman"}, normalize=True)
    
    try:
        K = gk.fit_transform(grakel_graphs)
        pairwise_distances = 1 - K
        uncertainty = np.mean(pairwise_distances)
    except Exception as e:
        print(f"Error during kernel computation: {e}")
        uncertainty = None
    
    return uncertainty

# Assuming `eval_model` and `Args` are already defined as in previous examples
def generate_responses_and_calculate_uncertainty(filtered_df_hallucination, num_images=1):
    indices = random.sample(filtered_df_hallucination.index.tolist(), 5)
    
    # indices = filtered_df_hallucination.index.tolist()
    results = {}
    
    for idx in tqdm(indices):
        with pd.option_context('display.max_colwidth', None):
            question = filtered_df_hallucination.loc[idx, 'question']
            choice_a = filtered_df_hallucination.loc[idx, 'A']
            choice_b = filtered_df_hallucination.loc[idx, 'B']
            choice_c = filtered_df_hallucination.loc[idx, 'C']
            choice_d = filtered_df_hallucination.loc[idx, 'D']

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
        image_data_base64 = filtered_df_hallucination.loc[idx, 'image']
        image_data = base64.b64decode(image_data_base64)
        image = Image.open(BytesIO(image_data))

        # Set the arguments
        # args.image_file = image_file
        args.image_file = image
        args.query = prompt
        args.temperature = args.temperature
        responses = []
        for _ in range(args.no_of_responses_each_sample):  # Get 30 responses
            response = eval_model(args)
            print(response)
            print('-'*50)
            if response:
                responses.append(json.loads(response))
            else:
                print("No response was generated.")
                
        
        # Extract entities and relationships from responses
        graphs = []
        for response in responses:
            print(response)
            # print(response['explanation'])
            if response['explanation']:
                entities, relationships = extract_entities_and_relationships(response['explanation'])
                G = construct_graph(entities, relationships)
                graphs.append(G)
        
        # Calculate uncertainty
        uncertainty = calculate_uncertainty(graphs)
        
        if uncertainty is None:
            print(f"Uncertainty could not be calculated for index {idx}.")
        
        results[idx] = {
            'prompt': prompt,
            'responses': responses,
            'uncertainty': uncertainty
        }
    
    return results
    #     with pd.option_context('display.max_colwidth', None):
    #         question = filtered_df_hallucination.loc[2791, 'question']
    #         choice_a = filtered_df_hallucination.loc[2791, 'A']
    #         choice_b = filtered_df_hallucination.loc[2791, 'B']
    #         choice_c = filtered_df_hallucination.loc[2791, 'C']
    #         choice_d = filtered_df_hallucination.loc[2791, 'D']

    #         prompt = f"""
    #         {question}
    #         A. {choice_a}
    #         B. {choice_b}
    #         C. {choice_c}
    #         D. {choice_d}
            
    #         Give your answer in JSON format where the keys are answer(one or more options above), explanation( explain your answer), and confidence(varies between 0 to 1).
            
    #         """
    #         print(prompt)
        
    #     # Decode the image
    #     image_data_base64 = filtered_df_hallucination.loc[idx, 'image']
    #     image_data = base64.b64decode(image_data_base64)
    #     image = Image.open(BytesIO(image_data))

    #     # Set the arguments
    #     # args.image_file = image_file
    #     args.image_file = image
    #     args.query = prompt
    #     args.temperature = 0.1
    #     responses = []
    #     for _ in range(args.no_of_responses_each_sample):  # Get 30 responses
    #         response = eval_model(args)
    #         print(response)
    #         print('-'*50)
    #         if response:
    #             responses.append(json.loads(response))
    #         else:
    #             print("No response was generated.")
                
        
    #     # Extract entities and relationships from responses
    #     graphs = []
    #     for response in responses:
    #         print(response)
    #         # print(response['explanation'])
    #         if response['explanation']:
    #             entities, relationships = extract_entities_and_relationships(response['explanation'])
    #             G = construct_graph(entities, relationships)
    #             graphs.append(G)
        
    #     # Calculate uncertainty
    #     uncertainty = calculate_uncertainty(graphs)
        
    #     if uncertainty is None:
    #         print(f"Uncertainty could not be calculated for index {idx}.")
        
    #     results[idx] = {
    #         'index': idx,
    #         'prompt': prompt,
    #         'responses': responses,
    #         'uncertainty': uncertainty
    #     }
    
    # return results

def load_args_from_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return argparse.Namespace(**config)

def log_args(args, log_file):
    with open(log_file, 'w') as file:
        yaml.dump(vars(args), file)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="Path to the config file.")
    parser.add_argument("--log-file", help="Path to the log file.")
    # parser.add_argument("--config-file", type=str, default="facebook/opt-350m")
    # parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()
    # # Load arguments from config file
    config_args = load_args_from_config(args.config_file)
    # Update args with loaded config_args
    args.__dict__.update(config_args.__dict__)
    args.model_name = get_model_name_from_path(args.model_path)
    args.tokenizer, args.model, args.image_processor, args.context_len = load_pretrained_model(
        args.model_path, args.model_base, args.model_name
    )
    
    # generate results and calculate uncertainty
    df = pd.read_csv(args.data_path, sep = '\t')
    filtered_df = df[df['category'].str.contains('hallucination', case=False)]
    results = generate_responses_and_calculate_uncertainty(filtered_df, num_images=1)
    
    # Save the results
    with open(args.output_path, 'w') as file:
        json.dump(results, file)

    # Log the arguments
    log_args(config_args, args.log_file)