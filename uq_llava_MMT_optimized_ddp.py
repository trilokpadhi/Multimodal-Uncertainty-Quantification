import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
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
from collections import Counter

# Disable tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")


class GraphDataset(Dataset):
    def __init__(self, responses, filtered_df):
        self.responses = responses
        self.filtered_df = filtered_df
        self.dataset_keys = list(responses.keys())

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        # result = self.responses[idx]
        result = self.responses[self.dataset_keys[idx]]
        image_data_base64 = self.filtered_df.loc[int(self.dataset_keys[int(idx)]), 'image']
        image_data = base64.b64decode(image_data_base64)
        # image = Image.open(BytesIO(image_data))
        return result, image_data

class GraphSimilarityModel(nn.Module):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def forward(self, graphs):
        similarities = []
        for i in range(len(graphs)):
            for j in range(i+1, len(graphs)):
                similarity = self.graph_similarity_kernel(graphs[i], graphs[j])
                similarities.append(similarity)
        return torch.tensor(similarities)

    def graph_similarity_kernel(self, graph1, graph2):
        vectors1 = self.get_node_vectors(graph1)
        vectors2 = self.get_node_vectors(graph2)

        similarity_matrix = cosine_similarity(vectors1, vectors2)
        node_similarity = np.mean(similarity_matrix)

        structure_similarity = self.graph_edit_distance_with_node_similarity(graph1, graph2)

        return 0.5 * node_similarity + 0.5 * structure_similarity

    def get_node_vectors(self, graph):
        nodes = list(graph.nodes())
        vectors = self.embedding_model.encode(nodes)
        return vectors

    def graph_edit_distance_with_node_similarity(self, G1, G2):
        node_subst_cost = np.zeros((len(G1), len(G2)))
        for i, n1 in enumerate(G1.nodes()):
            for j, n2 in enumerate(G2.nodes()):
                node_subst_cost[i, j] = 1 - cosine_similarity([self.embedding_model.encode([n1])], [self.embedding_model.encode([n2])])[0][0]

        cost_matrix = node_subst_cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_cost = cost_matrix[row_ind, col_ind].sum()

        unmatched_nodes = abs(len(G1) - len(G2))
        G1_edges = set(G1.edges())
        G2_edges = set((row_ind[i], col_ind[i]) for i in range(min(len(G1), len(G2))) if (row_ind[i], col_ind[i]) in G2.edges())
        edge_diff = len(G1_edges.symmetric_difference(G2_edges))

        total_cost = assignment_cost + unmatched_nodes + edge_diff
        max_cost = max(len(G1), len(G2)) * (1 + max(G1.number_of_edges(), G2.number_of_edges()))
        similarity = 1 - (total_cost / max_cost)

        return similarity

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

import os

def set_distributed_env(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

def main(rank, world_size, args):
    set_distributed_env(rank, world_size)
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    dataset = GraphDataset(args.responses, args.filtered_df)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    embedding_model = args.embedding_model.to(rank)
    model = GraphSimilarityModel(embedding_model)
    model = DDP(model, device_ids=[rank])

    for idx, (result, image_data) in enumerate(tqdm(dataloader, desc='Calculating uncertainty')):
        result = result[0]
        image = Image.open(BytesIO(image_data[0]))
        # image = Image.open(BytesIO(image_data))
        task_prompt = '<CAPTION>'
        caption = get_caption(args, task_prompt, image)

        graphs = []
        all_sentences = [caption[task_prompt]]
        for s in result['responses']:
            try:
                all_sentences.append(s['explanation'])
            except Exception as e:
                print(f"An error occurred while processing response: {e}. Skipping this response.")

        for i, sentence in enumerate(all_sentences):
            try:
                entities, relationships = extract_entities_and_relationships(sentence)
                G = construct_graph(entities, relationships)
                graphs.append(G)
                
                if i == 0:
                    result['ground_truth_graph'] = G
                else:
                    result['responses'][i - 1]['graph'] = G
            except Exception as e:
                print(f"Error constructing graph: {e}")
                if i > 0:
                    result['responses'][i - 1]['graph'] = None

        similarities = model(graphs)
        
        n = len(graphs)
        K = torch.zeros((n, n), device=rank)
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                K[i, j] = K[j, i] = similarities[k]
                k += 1

        alpha = 0.5
        beta = 0.5
        ground_truth_index = 0
        similarities_within_group = K[1:, 1:].triu(diagonal=1)
        similarities_with_ground_truth = K[ground_truth_index, 1:]
        avg_similarity_within_group = similarities_within_group.mean().item()
        avg_similarity_with_ground_truth = similarities_with_ground_truth.mean().item()
        uncertainty = alpha * (1 - avg_similarity_within_group) + beta * (1 - avg_similarity_with_ground_truth)

        result['avg_similarity_within_group'] = avg_similarity_within_group
        result['avg_similarity_with_ground_truth'] = avg_similarity_with_ground_truth
        result['uncertainty'] = uncertainty.item()

        args.responses[idx] = result

    cleanup()

def load_args_from_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return argparse.Namespace(**config)

if __name__ == "__main__":

    # Read arguments from command line
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
    
        args.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)