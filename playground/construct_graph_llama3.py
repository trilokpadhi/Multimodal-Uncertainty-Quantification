from transformers import pipeline
import torch
from huggingface_hub import login
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

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

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
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

    input_ids = tokenizer.apply_chat_template(messages,
                                              add_generation_prompt=True,
                                              return_tensors="pt").to(model.device)

    # Get the EOS token ID
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
    )
    
    print('-'*100)
    # print(assistant_response)
    response = outputs[0][input_ids.shape[-1]:]
    assistant_response = tokenizer.decode(response, skip_special_tokens=True)
    
    return assistant_response

import json
def extract_json_from_text(text):
    """Extracts JSON from the given text."""

    start_index = text.find("{")
    end_index = text.rfind("}") + 1
    json_string = text[start_index:end_index]

    return json.loads(json_string)


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
                    json_from_prediction = extract_json_from_text(prediction)
                    print('Prediction:', json_from_prediction)
                    output_file.write('Prediction: ' + json.dumps(json_from_prediction, indent=4) + '\n')
                    # output_file.write(prediction + '\n')
                    break
                except Exception as e:
                    print('Error:', e)
            # loop_count += 1
            break
        







"""
Testing LLama3 response time with and without using pipeline

"""
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import time

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# inp = tokenizer("Who was first man on moon", return_tensors = "pt")
# print(inp)
# start = time.time()
# output = model.generate(**inp, max_new_tokens=500, top_p=0.95, temperature=0.1)
# end = time.time()
# print("Output >>> " + tokenizer.decode(output[0], skip_special_tokens=True))
# print("Time taken without using pipe: ", end-start)


# from transformers import pipeline   
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     device=0,  # 'cuda' for GPU, 'cpu' for CPU
# )

# messages = [
#     {"role": "system", "content": "You are a honest assitant"},
#     {"role": "user", "content": "Who was first man on moon?"},
#     ]
    
# # Combine messages into a single string prompt
# prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

# # Get the EOS token ID
# eos_token_id = pipe.tokenizer.eos_token_id

# start = time.time()
# outputs = pipe(
#     prompt,
#     max_new_tokens=256,
#     eos_token_id=eos_token_id,
#     do_sample=True,
#     temperature=0.01,
#     top_p=0.9,
# )
# end = time.time()
# assistant_response = outputs[0]["generated_text"]
# print("Time taken using pipe: ", end-start)
# print('-'*5)
# print(assistant_response)

