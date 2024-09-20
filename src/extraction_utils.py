from transformers import AutoTokenizer, AutoModelForCausalLM, LlavaForConditionalGeneration, AutoProcessor
from data_utils import get_dataloader
from inference_utils import generate_explanations_MM
from tqdm import tqdm
from src_old.graph import extract_triples_from_llama3_json
import torch
import json
import logging
def extract_json_from_text(text):
    """Extracts JSON from the given text."""

    start_index = text.find("{")
    end_index = text.rfind("}") + 1
    json_string = text[start_index:end_index]
    
    return json.loads(json_string)

def extract_triples(sample_responses):
    llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    # get the keys which statrt with response_i
    # count the keys which start with response_i in the sample_responses
    
    for key in sample_responses.keys():
        try:
            if not key.startswith("response"):
                continue
            else:
                response = sample_responses.get(key)['decoded_outputs']
                try:
                    response_jsonified = extract_json_from_text(response)
                    sample_responses[key]['response_json'] = response_jsonified
                except json.JSONDecodeError as e:
                    sample_responses[key]['error_json_extraction'] = f"JSON parsing error: {e}"
                try:
                    explanation = response_jsonified.get("explanation")
                except AttributeError as e:
                    sample_responses[key]['error_explanation_extraction'] = f"Attribute error: {e}"
                llama3_response_graph = extract_entities_and_relationships_llama3( llama_model, llama_tokenizer, explanation)
                llama3_response_graph_jsonified = extract_json_from_text(llama3_response_graph)
                response_triples = extract_triples_from_llama3_json(llama3_response_graph_jsonified)
                sample_responses[key]['response_triples'] = response_triples
        except Exception as e:
            # logging.error(f"JSON parsing error for response {j}: {e}")
            sample_responses[key]['error_json_extraction'] = f"JSON parsing error: {e}"   

    return sample_responses
    
def extract_entities_and_relationships_llama3(llama_model, llama_tokenizer, text):
    """Extract entities, attributes, and relationships from a given text using the LLAMA-3 model."""
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
    # prompt = f"Extract all possible triples from the given {text}."
    # prompt = f"Extract all possible entities from the given {text} in the following JSON format: {{'entities': ['entity1', 'entity2', ...]}}"
    messages = [
    {"role": "system", "content": "You are a honest assitant"},
    {"role": "user", "content": prompt},
    ]
    
    # Combine messages into a single string prompt
    # prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    input_ids = llama_tokenizer.apply_chat_template(messages,
                                              add_generation_prompt=True,
                                              return_tensors="pt").to(llama_model.device)

    # Get the EOS token ID
    terminators = [
        llama_tokenizer.eos_token_id,
        llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    outputs = llama_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    assistant_response = llama_tokenizer.decode(response, skip_special_tokens=True)
    
    return assistant_response

def calculate_top_k_similarity():
    pass


if __name__ == "__main__":
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    dataloader = get_dataloader('/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA', 'json')
    sample = next(iter(dataloader))
    rank = 0
    config = {
        'mm_model': {
            'model_path': 'liuhaotian/llava-v1.6-vicuna-7b',
            'temperature': 0.5,
            'top_p': 0.95,
            'num_beams': 5,
            'max_new_tokens': 100
        }
    }
    sample_explanations = generate_explanations_MM(model, processor, sample, rank, config['mm_model'])
    
    
    # test code for triples extraction from LLAMA-3
    sample_triples = extract_triples(sample_explanations)
    