import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def predict_NuExtract(model, tokenizer, text, schema, example=["", "", ""]):
    schema = json.dumps(json.loads(schema), indent=4)
    input_llm =  "<|input|>\n### Template:\n" +  schema + "\n"
    for i in example:
      if i != "":
          input_llm += "### Example:\n"+ json.dumps(json.loads(i), indent=4)+"\n"
    
    input_llm +=  "### Text:\n"+text +"\n<|output|>\n"
    input_ids = tokenizer(input_llm, return_tensors="pt",truncation = True, max_length=4000).to("cuda")

    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    return output.split("<|output|>")[1].split("<|end-output|>")[0]


# We recommend using bf16 as it results in negligable performance loss
# for large use "numind/NuExtract-large"
# for medium use "numind/NuExtract"
# for small use "numind/NuExtract-tiny"
# model_id = "numind/NuExtract-large"
model_id = "numind/NuExtract"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    device_map=None,
)
# model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype="auto", 
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# model.to("cuda")

model.eval()

# text = """We introduce Mistral 7B, a 7–billion-parameter language model engineered for
# superior performance and efficiency. Mistral 7B outperforms the best open 13B
# model (Llama 2) across all evaluated benchmarks, and the best released 34B
# model (Llama 1) in reasoning, mathematics, and code generation. Our model
# leverages grouped-query attention (GQA) for faster inference, coupled with sliding
# window attention (SWA) to effectively handle sequences of arbitrary length with a
# reduced inference cost. We also provide a model fine-tuned to follow instructions,
# Mistral 7B – Instruct, that surpasses Llama 2 13B – chat model both on human and
# automated benchmarks. Our models are released under the Apache 2.0 license.
# Code: https://github.com/mistralai/mistral-src
# Webpage: https://mistral.ai/news/announcing-mistral-7b/"""

# text = """The caption describes a vehicle that is participating in the Mongol Rally with a bathtub on top. This is the most accurately depicted in the image, as it shows a car with an object placed on its roof, which is a tub. The text also references the Mongol Rally '09, which is consistent with the timestamp on the photo. No other choice quite accurately captures all elements shown in the image."""
text = "The man is wearing a pair of shorts in the image. The shorts are beige and have a grey pattern."
# schema = """{
#     "Entities": "",
#     "Relations": ""
# }"""

schema ="""    {
        "entities": [
            {"type": "", "name": "", "value": ""}
        ],
        "attributes": [
            {"type": "", 
            "entity": "", 
            "value": ""}  
        ],
        "relations": []
    }
"""
# schema = """{
#     "Entities": {
#         "Name": "",
#         "Attributes": {
#             "Type": "",
#             "Color": "",
#             "Size": ""
#         },
#         "Relations": {
#             "Action": ""
#         }
#     }
# }"""

# schema = """{
#     "Model": {
#         "Name": "",
#         "Number of parameters": "",
#         "Number of max token": "",
#         "Architecture": []
#     },
#     "Usage": {
#         "Use case": [],
#         "Licence": ""
#     }
# }"""

prediction = predict_NuExtract(model, tokenizer, text, schema, example=["","",""])
print(prediction)


