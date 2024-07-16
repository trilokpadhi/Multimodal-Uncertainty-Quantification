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
def calculate_uncertainty_graph_kernel(graphs):
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

# Function to quantify uncertainty
def quantify_uncertainty(results):
    for idx, result in results.items():
        responses = result['responses']
        
        # Extract entities and relationships from responses
        graphs = []
        for response in responses:
            if 'explanation' in response and response['explanation']:
                try:
                    entities, relationships = extract_entities_and_relationships(response['explanation'])
                    G = construct_graph(entities, relationships)
                    graphs.append(G)
                except Exception as e:
                    print(f"Error extracting entities and relationships: {e}")
        
        # Calculate uncertainty
        try:
            uncertainty = calculate_uncertainty_graph_kernel(graphs)
        except Exception as e:
            print(f"Error calculating uncertainty: {e}")
            uncertainty = None
        
        result['uncertainty'] = uncertainty                           
    
    return results

def extract_sentence_features(sentence):
    doc = nlp(sentence)
    key_words = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
    return doc.vector, key_words, [token.pos_ for token in doc]


def custom_kernel_s3(feat1, feat2):
    vec1, words1, _ = feat1
    vec2, words2, _ = feat2

    vec_similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    word_overlap = len(set(words1) & set(words2)) / max(len(set(words1) | set(words2)), 1)

    # Calculate bigram overlap
    bigrams1 = set(zip(words1[:-1], words1[1:]))
    bigrams2 = set(zip(words2[:-1], words2[1:]))
    bigram_overlap = len(bigrams1 & bigrams2) / max(len(bigrams1 | bigrams2), 1)

    if np.allclose(vec1, vec2, atol=1e-6) and words1 == words2:
        return 1.0

    return 0.5 * vec_similarity + 0.25 * word_overlap + 0.25 * bigram_overlap

def custom_kernel(graph1, graph2):
    # Compare node vectors
    vectors1 = np.array([node['vector'] for node in graph1[0].values()])
    vectors2 = np.array([node['vector'] for node in graph2[0].values()])

    # Compute cosine similarity between all pairs of vectors
    similarity_matrix = cosine_similarity(vectors1, vectors2)

    # Take the average similarity
    node_similarity = np.mean(similarity_matrix)

    # Compare graph structures (you can adjust this part)
    structure_similarity = 1 if len(graph1[1]) == len(graph2[1]) else 0

    # Combine node and structure similarity (you can adjust the weights)
    return 0.5 * node_similarity + 0.5 * structure_similarity


def quantify_uncertainty_from_image_captions_with_node_and_structural_simlarity(args):
    """
    Here we use node and structural similarity as features to compute the similarity between two graphs, and then use these similarities to compute the uncertainty.
    
    """
    for idx, result in args.responses.items():
        responses = result['responses']
        
        # get the dataframe and the image from the index
        image_data_base64 = args.filtered_df.loc[idx, 'image']
        image_data = base64.b64decode(image_data_base64)
        image = Image.open(BytesIO(image_data))
        
        # get caption from florence-large
        task_prompt = '<CAPTION>'
        caption = get_caption(args, task_prompt, image)  # proxy for ground truth

        graphs = []

        all_sentences = [caption[task_prompt]] + [s['explanation'] for s in responses]
        for sentence in all_sentences:
            entities, relationships = extract_entities_and_relationships(sentence)
            G = construct_graph(entities, relationships)
            graphs.append(G)    

        n = len(graphs)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = custom_kernel([graphs[i], graphs[j]])
                K[j, i] = K[i, j]    
        alpha = 0.5
        beta = 0.5
        ground_truth_index = 0
        similarities_within_group = K[1:, 1:][np.triu_indices(n-1, k=1)]  # similarities within group of responses
        similarities_with_ground_truth = K[ground_truth_index, 1:]  # similarities with ground truth
        avg_similarity_within_group = np.mean(similarities_within_group)
        avg_similarity_with_ground_truth = np.mean(similarities_with_ground_truth)
        uncertainty = alpha * (1 - avg_similarity_within_group) + beta * (1 - avg_similarity_with_ground_truth)
        result['uncertainty'] = uncertainty

    return args.responses

def quantify_uncertainty_from_image_captions_with_vec_similarity_bigram_overlap(args):
    """
    here we use vector similarity, word overlap, and bigram overlap as features to compute the similarity between two sentences, and then use these similarities to compute the uncertainty
    
    """
    for idx, result in args.responses.items():
        responses = result['responses']
        
        # get the dataframe and the image from the index
        image_data_base64 = args.filtered_df.loc[idx, 'image']
        image_data = base64.b64decode(image_data_base64)
        image = Image.open(BytesIO(image_data))
        
        # get caption from florence-large
        task_prompt = '<CAPTION>'
        caption = get_caption(args, task_prompt, image)  # proxy for ground truth
        ground_truth_features = extract_sentence_features(caption[task_prompt])
        
        # Extract features for each response explanation
        features = [ground_truth_features] + [extract_sentence_features(s['explanation']) for s in responses]

        alpha = 0.5
        beta = 0.5
        n = len(features)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                K[i, j] = custom_kernel_s3(features[i], features[j])
                K[j, i] = K[i, j]

        similarities_within_group = K[1:, 1:][np.triu_indices(n-1, k=1)]  # similarities within group of responses
        similarities_with_ground_truth = K[0, 1:]  # similarities with ground truth

        avg_similarity_within_group = np.mean(similarities_within_group)
        avg_similarity_with_ground_truth = np.mean(similarities_with_ground_truth)

        uncertainty = alpha * (1 - avg_similarity_within_group) + beta * (1 - avg_similarity_with_ground_truth)

        result['uncertainty'] = uncertainty

    return args.responses


        
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

def log_args(args, log_file):
    with open(log_file, 'w') as file:
        yaml.dump(vars(args), file)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="Path to the config file.")
    parser.add_argument("--log-file", help="Path to the log file.")
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
    args.filtered_df = df[df['category'].str.contains(args.category, case=False)]
    args.responses = generate_responses(args)
    
    # Load the florence-large model
    args.caption_model_id = 'microsoft/Florence-2-large'
    args.caption_model = AutoModelForCausalLM.from_pretrained(args.caption_model_id, trust_remote_code=True).eval().cuda()
    args.caption_model_processor = AutoProcessor.from_pretrained(args.caption_model_id, trust_remote_code=True)
    
    if args.uncertainty_method == 'node_and_structural_similarity':
        responses_with_uncertainty = quantify_uncertainty_from_image_captions_with_node_and_structural_simlarity(args)
    elif args.uncertainty_method == 'vec_similarity_bigram_overlap':
        responses_with_uncertainty = quantify_uncertainty_from_image_captions_with_vec_similarity_bigram_overlap(args)
    else:
        raise ValueError(f"Invalid uncertainty method: {args.uncertainty_method}")
        
    # Save the results
    with open(args.output_path, 'w') as file:
        json.dump(responses_with_uncertainty, file)

    # Log the arguments
    log_args(config_args, args.log_file)