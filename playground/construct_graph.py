import spacy
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from huggingface_hub import HfApi, login

# Authenticate
username = "tpadhi1"
token = "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"
login(token=token)

nlp = spacy.load("en_core_web_sm")

def extract_entities_and_relationships(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    relationships = []
    for token in doc:
        if token.dep_ in ["nsubj", "dobj"] and token.head.pos_ == "VERB":
            subject = token.text
            object = token.head.text
            relationships.append((subject, object))
    return entities, relationships

# Function to construct graph from entities and relationships
def construct_graph(entities, relationships):
    G = nx.DiGraph()
    for entity, label in entities:
        print('Entity :', entity,'Label:', label)
        G.add_node(entity, label=label)
    for subj, obj in relationships:
        print('Subject:', subj, 'Object:', obj)
        if not G.has_node(subj):
            G.add_node(subj, label="unknown")
        if not G.has_node(obj):
            G.add_node(obj, label="unknown")
        G.add_edge(subj, obj)
    return G

# Load Llama 2 model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def extract_entities_and_relationships_using_llama2(text):

    # Prompt Style 1
    # prompt = """
    # Analyze the following text and extract concepts:
    # Text: {}
    # Return the results as a JSON object with the following structure:
    # {{'concepts': ''}}
    # """

    # Prompt Style 2
    prompt = f"""
    Analyze the following text and extract entities, attributes, and relationships. Provide the output as a JSON object with the following structure:

    {{
        "entities": [
            {{"type": "whole", "name": "motorcycle"}},
            {{"type": "whole", "name": "doors"}}
        ],
        "attributes": [
            {{"type": "color", "entity": "motorcycle", "value": "blue"}},
            {{"type": "state", "entity": "doors", "value": "painted"}},
            {{"type": "state", "entity": "door paint", "value": "chipped"}},
            {{"type": "state", "entity": "motorcycle", "value": "parked"}}
        ],
        "relations": [
            {{"type": "spatial", "entity1": "motorcycle", "entity2": "doors", "description": "next to"}}
        ]
    }}

    Entity Types:
    - Whole: Complete objects
    - Part: Components of entities

    Attribute Types:
    - State: Current condition or status
    - Color: Visual color
    - Type: Specific category or classification
    - Material: Substance the entity is made of
    - Count: Numerical quantity
    - Texture: Surface feel or appearance
    - Text: Visible text or writing
    - Shape: Geometric form
    - Size: Relative or absolute dimensions

    Relationship Types:
    - Spatial: Physical positioning (e.g., next to, above, inside)
    - Action: Dynamic interactions (e.g., kicks, holds, pushes)

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
            {{"type": "state", "entity": "doors", "value": "painted"}},
            {{"type": "state", "entity": "door paint", "value": "chipped"}},
            {{"type": "state", "entity": "motorcycle", "value": "parked"}}
        ],
        "relations": [
            {{"type": "spatial", "entity1": "motorcycle", "entity2": "doors", "description": "next to"}}
            
    """
   
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_new_tokens = 50, repetition_penalty=1.1, temperature=0.01)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    print('Response:')
    print('-'*50)
    print(response)
    print('-'*50)

# Usage with our examples
response_json_path = "/home/ubuntu/Multimodal-Uncertainty-Quantification/debug_bin_9_10_llava_mmt_output_uq_io_graph_similarity_temp_1_topp_1.json"
with open(response_json_path, 'r') as f:
    response_json = json.load(f)
    for key, value in response_json.items():
        print('Key:', key)
        responses = value['responses']
        for response in responses:
            try:
                print('Explanation:', response['explanation'])
                text = response['explanation']
                print('Text:', text)
                # entities, relationships = extract_entities_and_relationships_using_llama2(text)
                extract_entities_and_relationships_using_llama2(text)
                # print('Entities:', entities)
                # print('Relationships', relationships)
            except Exception as e:
                print('Error:', e)
        break
