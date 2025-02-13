# import time
# import torch
# from transformers import AutoProcessor, MllamaForConditionalGeneration
# from huggingface_hub import login
# from PIL import Image
# import requests

# # ----------------------------
# # 1. SETUP
# # ----------------------------

# # Login to HuggingFace
# key = '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'
# login(key)

# # Load model and processor
# model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# processor = AutoProcessor.from_pretrained(model_id)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# messages = [
#     {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": "Can you please describe this image in just one sentence?"}
#     ]}
# ]

# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

# # Prepare inputs
# def prepare_inputs(model, processor, image, input_text):
#     inputs = processor(
#         image,
#         input_text,
#         add_special_tokens=False,
#         return_tensors="pt",
#     ).to(model.device)
#     return inputs

# # ----------------------------
# # 2. INFERENCE TEST FUNCTION
# # ----------------------------

# def test_inference_time(model, inputs, max_new_tokens=70):
#     torch.cuda.empty_cache()  # Clear GPU cache
#     start_time = time.time()
#     output = model.generate(**inputs, max_new_tokens=max_new_tokens)
#     end_time = time.time()
#     return end_time - start_time, output

# # ----------------------------
# # 3. OPTIMIZATION STRATEGIES
# # ----------------------------

# strategies = {}

# # 1. Baseline (Default Settings)
# def baseline():
#     model = MllamaForConditionalGeneration.from_pretrained(
#         model_id,
#         torch_dtype=torch.float32  # Default precision
#     ).to("cuda")
#     inputs = prepare_inputs(model, processor, image, input_text)
#     time_taken, output = test_inference_time(model, inputs)
#     strategies["Baseline"] = time_taken
#     return output


# # 2. FP16 Precision
# def fp16():
#     model = MllamaForConditionalGeneration.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16  # FP16 Precision
#     ).to("cuda")
#     inputs = prepare_inputs(model, processor, image, input_text)
#     time_taken, output = test_inference_time(model, inputs)
#     strategies["FP16"] = time_taken
#     return output


# # 3. BF16 Precision
# def bf16():
#     model = MllamaForConditionalGeneration.from_pretrained(
#         model_id,
#         torch_dtype=torch.bfloat16  # BF16 Precision
#     ).to("cuda")
#     inputs = prepare_inputs(model, processor, image, input_text)
#     time_taken, output = test_inference_time(model, inputs)
#     strategies["BF16"] = time_taken
#     return output


# # 4. Cache Optimization
# def cache_optimized():
#     model = MllamaForConditionalGeneration.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         # use_cache=True  # Cache optimization
#     ).to("cuda")
#     inputs = prepare_inputs(model, processor, image, input_text)
#     time_taken, output = test_inference_time(model, inputs)
#     strategies["Cache_Optimized"] = time_taken
#     return output


# # 5. Torch Compilation
# def torch_compile():
#     model = MllamaForConditionalGeneration.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16
#     ).to("cuda")
#     model = torch.compile(model)  # PyTorch 2.0 Compilation
#     inputs = prepare_inputs(model, processor, image, input_text)
#     time_taken, output = test_inference_time(model, inputs)
#     strategies["Torch_Compile"] = time_taken
#     return output


# # ----------------------------
# # 4. RUN TESTS
# # ----------------------------
# print("Testing inference times...")

# outputs = {}
# # outputs["Baseline"] = baseline()
# outputs["FP16"] = fp16()
# outputs["BF16"] = bf16()
# outputs["Cache_Optimized"] = cache_optimized()
# outputs["Torch_Compile"] = torch_compile()

# # ----------------------------
# # 5. RESULTS
# # ----------------------------

# # Print and compare inference times
# print("\nInference Times:")
# for strategy, time_taken in strategies.items():
#     print(f"{strategy}: {time_taken:.2f} seconds")

# # Get the best strategy
# best_strategy = min(strategies, key=strategies.get)
# print(f"\nBest Strategy: {best_strategy} with {strategies[best_strategy]:.2f} seconds")

# # Display output for the best strategy
# print("\nOutput for Best Strategy:")
# print(processor.decode(outputs[best_strategy][0]))

import time
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from huggingface_hub import login
from PIL import Image
import requests

# ----------------------------
# 1. SETUP
# ----------------------------

# Login to HuggingFace
key = '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'
login(key)

# Model and Processor Setup
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

# Test Image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Test Prompt
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Can you please describe this image in just one sentence?"}
    ]}
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)


# Prepare Inputs
def prepare_inputs(model, processor, image, input_text):
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    return inputs


# ----------------------------
# 2. INFERENCE TEST FUNCTION
# ----------------------------

def test_inference_time(model, inputs, max_new_tokens=70):
    torch.cuda.empty_cache()  # Clear GPU cache

    # Measure Inference Time
    start_time = time.time()
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end_time = time.time()
    inference_time = end_time - start_time

    # Measure Decoding Time
    decode_start = time.time()
    decoded_output = processor.decode(output[0][inputs["input_ids"].shape[-1]:])
    decode_end = time.time()
    decoding_time = decode_end - decode_start

    return inference_time, decoding_time, decoded_output


# ----------------------------
# 3. OPTIMIZATION STRATEGIES
# ----------------------------

strategies = {}

# 1. Baseline (Default Settings)
def baseline():
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32  # Default precision
    ).to("cuda")
    inputs = prepare_inputs(model, processor, image, input_text)
    inf_time, dec_time, output = test_inference_time(model, inputs)
    strategies["Baseline"] = {"inference": inf_time, "decoding": dec_time}
    return output


# 2. FP16 Precision
def fp16():
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16  # FP16 Precision
    ).to("cuda")
    inputs = prepare_inputs(model, processor, image, input_text)
    inf_time, dec_time, output = test_inference_time(model, inputs)
    strategies["FP16"] = {"inference": inf_time, "decoding": dec_time}
    return output


# 3. BF16 Precision
def bf16():
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16  # BF16 Precision
    ).to("cuda")
    inputs = prepare_inputs(model, processor, image, input_text)
    inf_time, dec_time, output = test_inference_time(model, inputs)
    strategies["BF16"] = {"inference": inf_time, "decoding": dec_time}
    return output


# 4. Cache Optimization
def cache_optimized():
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        # use_cache=True  # Cache optimization
    ).to("cuda")
    inputs = prepare_inputs(model, processor, image, input_text)
    inf_time, dec_time, output = test_inference_time(model, inputs)
    strategies["Cache_Optimized"] = {"inference": inf_time, "decoding": dec_time}
    return output


# 5. Torch Compilation
def torch_compile():
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    model = torch.compile(model)  # PyTorch 2.0 Compilation
    inputs = prepare_inputs(model, processor, image, input_text)
    inf_time, dec_time, output = test_inference_time(model, inputs)
    strategies["Torch_Compile"] = {"inference": inf_time, "decoding": dec_time}
    return output


# ----------------------------
# 4. RUN TESTS
# ----------------------------
print("Testing inference and decoding times...")

outputs = {}
# outputs["Baseline"] = baseline()
outputs["FP16"] = fp16()
outputs["BF16"] = bf16()
outputs["Cache_Optimized"] = cache_optimized()
outputs["Torch_Compile"] = torch_compile()

# ----------------------------
# 5. RESULTS
# ----------------------------

# Print and compare inference + decoding times
print("\nInference and Decoding Times:")
for strategy, times in strategies.items():
    print(f"{strategy}: Inference: {times['inference']:.2f}s, Decoding: {times['decoding']:.2f}s")

# Get the best strategy based on combined time
best_strategy = min(strategies, key=lambda x: strategies[x]['inference'] + strategies[x]['decoding'])
best_time = strategies[best_strategy]['inference'] + strategies[best_strategy]['decoding']

print(f"\nBest Strategy: {best_strategy} with Total Time: {best_time:.2f}s")

# Display output for the best strategy
print("\nOutput for Best Strategy:")
print(outputs[best_strategy])