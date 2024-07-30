from transformers import pipeline
import torch
from huggingface_hub import login
import json

# Authenticate
username = "tpadhi1"
token = "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"
login(token=token)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=0,  # 'cuda' for GPU, 'cpu' for CPU
)

def extract_entities_and_relationships_using_llama3(text, pipe):
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
            ...
        ],
        "relations": [
            {{"type": "spatial", "entity1": "entity1", "entity2": "entity2", "description": "relation1"}},
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
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    # Get the EOS token ID
    eos_token_id = pipe.tokenizer.eos_token_id

    outputs = pipe(
        prompt,
        max_new_tokens=256,
        eos_token_id=eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    assistant_response = outputs[0]["generated_text"]
    
    print('-'*5)
    print(assistant_response)
    
    return assistant_response

# Read the response JSON
response_json_path = "/home/ubuntu/Multimodal-Uncertainty-Quantification/debug_responses/debug_bin_9_10_llava_mmt_output_uq_io_graph_similarity_temp_1_topp_1.json"
with open(response_json_path, 'r') as f:
    response_json = json.load(f)
    with open('response_llama3.txt', 'w') as output_file:
        loop_count = 0
        for key, value in response_json.items():
            if loop_count >= 1:
                break
            print('Key:', key)
            responses = value['responses']
            for response in responses:
                try:
                    print('Explanation:', response['explanation'])
                    text = response['explanation']
                    print('Text:', text)
                    output_file.write('Explanation: ' + text + '\n')    
                    output_file.write('--' * 50 + '\n')
                    # Get the NuExtract prediction
                    prediction = extract_entities_and_relationships_using_llama3(text, pipe)
                    output_file.write(prediction + '\n')
                except Exception as e:
                    print('Error:', e)
            loop_count += 1