import spacy
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

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
model_name = "meta-llama/Llama-2-7b-chat-hf"  # or whichever version you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def extract_entities_and_relationships_using_llama2(text):
    prompt = f"""
    Extract entities and relationships from the following text. 
    Provide the output as a JSON object with two lists: 'entities' and 'relationships'.
    Each entity should be a list with the entity text and its label.
    Each relationship should be a list with the subject, predicate, and object.

    Text: {text}

    Output:
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=1000)
    
    response = tokenizer.decode(outputs[0])
    
    # Extract JSON from response
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    json_str = response[json_start:json_end]
    
    try:
        result = json.loads(json_str)
        return result['entities'], result['relationships']
    except json.JSONDecodeError:
        print("Failed to parse JSON from model output")
        return [], []
    


# Usage
text = "My name is John. I live in London. I work at Google."
entities, relationships = extract_entities_and_relationships_using_llama2(text)
print('Entities:', entities)
print('Relationships', relationships)
graph = construct_graph(entities, relationships)
