import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm


# Initialize model and processor
# model_id_llava = "llava-hf/llava-1.5-7b-hf"
model_id_llava = "microsoft/llava-med-v1.5-mistral-7b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_llava = LlavaForConditionalGeneration.from_pretrained(model_id_llava).to(device)
processor_llava = AutoProcessor.from_pretrained(model_id_llava)


ICL_prompt = """ USER: Answer the questions. Here are few examples:
Question: What is the color of the object?
Answer: The color of the object is red.
Question: What are the people doing ?
Answer: The people in the image are playing soccer.
Question: What animal is in the image?
Answer: The animal is a cat.

<image>
What is the man carrying?
ASSISTANT: """

NO_ICL_prompt = """ USER:   <image>
What is the man carrying?
ASSISTANT: """

# Parameters for inference
params = {
    'no_of_responses_sampled_per_image': 20,  # Set to 1 for simplicity
    'temperature': 1.0,
    'top_p': 0.9,
    'num_beams': 5,
    'max_new_tokens': 50
}


image_path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/VQA/train2014/COCO_train2014_000000382316.jpg'
raw_image = Image.open(image_path)

# I have to test 4 cases, 
# 1. ICL prompt with temperature 1.0
# 2. ICL prompt with temperature 0.5
# 3. NO ICL prompt with temperature 1.0
# 4. NO ICL prompt with temperature 0.5

# # ICL prompt with temperature 1.0
# print('*'*50)
# print("ICL prompt with temperature 1.0")
# print('*'*50)

# inputs = processor_llava(
#     images=raw_image, 
#     text=ICL_prompt, 
#     return_tensors="pt", 
#     padding=True, 
#     truncation=True
# ).to(device)

# # for i in range(20):
# for i in tqdm(range(20)):
#     outputs = model_llava.generate(
#         **inputs,
#         do_sample=True,
#         temperature=1.0,
#         top_p=0.9,
#         num_beams=5,
#         max_new_tokens=50,
#         use_cache=False,
#         return_dict_in_generate=True,
#         output_scores=True
#     )

#     generated_token_ids = outputs.sequences[0]
#     decoded_output = processor_llava.decode(generated_token_ids, skip_special_tokens=True)
#     print("="*50)
#     print("Generated response:", decoded_output)
#     print("="*50)

# # ICL prompt with temperature 0.5
# print('*'*50)
# print("ICL prompt with temperature 0.5")
# print('*'*50)

# inputs = processor_llava(
#     images=raw_image, 
#     text=ICL_prompt, 
#     return_tensors="pt", 
#     padding=True, 
#     truncation=True
# ).to(device)

# # for i in range(20):
# for i in tqdm(range(20)):
#     outputs = model_llava.generate(
#         **inputs,
#         do_sample=True,
#         temperature=0.5,
#         top_p=0.9,
#         num_beams=5,
#         max_new_tokens=50,
#         use_cache=False,
#         return_dict_in_generate=True,
#         output_scores=True
#     )

#     generated_token_ids = outputs.sequences[0]
#     decoded_output = processor_llava.decode(generated_token_ids, skip_special_tokens=True)
#     # print("Generated response:", decoded_output)
#     print("="*50)
#     print("Generated response:", decoded_output)
#     print("="*50)


# # NO ICL prompt with temperature 1.0
# print('*'*50)
# print("NO ICL prompt with temperature 1.0")
# print('*'*50)

# inputs = processor_llava(
#     images=raw_image, 
#     text=NO_ICL_prompt, 
#     return_tensors="pt", 
#     padding=True, 
#     truncation=True
# ).to(device)

# # for i in range(20):
# for i in tqdm(range(20)):
#     outputs = model_llava.generate(
#         **inputs,
#         do_sample=True,
#         temperature=1.0,
#         top_p=0.9,
#         num_beams=5,
#         max_new_tokens=50,
#         use_cache=False,
#         return_dict_in_generate=True,
#         output_scores=True
#     )

#     generated_token_ids = outputs.sequences[0]
#     decoded_output = processor_llava.decode(generated_token_ids, skip_special_tokens=True)
#     # print("Generated response:", decoded_output)
#     print("="*50)
#     print("Generated response:", decoded_output)
#     print("="*50)


# # NO ICL prompt with temperature 0.5
# print('*'*50)
# print("NO ICL prompt with temperature 0.5")
# print('*'*50)

# inputs = processor_llava(
#     images=raw_image, 
#     text=NO_ICL_prompt, 
#     return_tensors="pt", 
#     padding=True, 
#     truncation=True
# ).to(device)

# # for i in range(20):
# for i in tqdm(range(20)):
#     outputs = model_llava.generate(
#         **inputs,
#         do_sample=True,
#         temperature=0.5,
#         top_p=0.9,
#         num_beams=5,
#         max_new_tokens=50,
#         use_cache=False,
#         return_dict_in_generate=True,
#         output_scores=True
#     )

#     generated_token_ids = outputs.sequences[0]
#     decoded_output = processor_llava.decode(generated_token_ids, skip_special_tokens=True)
#     # print("Generated response:", decoded_output)
#     print("="*50)
#     print("Generated response:", decoded_output)
#     print("="*50)

## ok so the above seems to work, 
import sys
sys.path.append('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/src2')
from data_utils import VQADataset, vqa_collate_fn
from torch.utils.data import DataLoader
from inference_utils import generate_explanations_MM

config_data = {
    "dataset": "vqa",
    "root_dir": "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/VQA",
    "image_dir": "train2014",
    "annotation_file": "v1/annotations/mscoco_train2014_annotations_HIGH_agreement_question_types_GROUND.json",
    "question_file": "v1/questions/OpenEnded_mscoco_train2014_questions_RANDOMSEED_42_COUNT_2000_CURRDATE_20241228_170848.json",
    "dataset_type": "json",
    "samples": 2000
}


# Create dataloader
dataloader = DataLoader(VQADataset(config_data), batch_size=1, collate_fn=vqa_collate_fn)

# Get a sample
sample = next(iter(dataloader))
rank = 0

params = {
    'inference_batch_size': 4,  # Not used since batching is removed
    'no_of_responses_sampled_per_image': 20,
    'temperature': 1.0,
    'top_p': 0.9,
    'num_beams': 5,
    'max_new_tokens': 50
}
config_logging = {
    'explanation_dir': '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/explanations_test'
}

sample_output_path, sample_output = generate_explanations_MM(model_llava, processor_llava, sample, rank, params, config_logging)

print("Sample output path:", sample_output_path)
print("Sample output:", sample_output)

params = {
    'inference_batch_size': 4,  # Not used since batching is removed
    'no_of_responses_sampled_per_image': 20,
    'temperature': 0.5,
    'top_p': 0.9,
    'num_beams': 5,
    'max_new_tokens': 50
}

sample_output_path, sample_output = generate_explanations_MM(model_llava, processor_llava, sample, rank, params, config_logging)

print("Sample output path:", sample_output_path)
print("Sample output:", sample_output)


