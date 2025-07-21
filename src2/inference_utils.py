import torch
import torch
from PIL import Image
import requests
from transformers import pipeline
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import os
import torch
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from typing import Union, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForMaskGeneration, pipeline, GenerationConfig
from tqdm import tqdm
import json
import psutil
import torch
import GPUtil
from data_utils import get_gqa_dataloader
from qwen_vl_utils import process_vision_info
# llama2 for full sentence generation if the model gives one word response 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded
import time
import random

############### Uncomment below imports if you are using Llava-med ## 
# import sys
# sys.path.append('/home/ubuntu/trilok/LLaVA-Med/llava')
# from llava.mm_utils import tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.mm_utils import KeywordsStoppingCriteria

######## Llama2 model for full sentence generation ########
# Load the tokenizer and model from Hugging Face
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with "Llama-2-13b-hf" or "Llama-2-70b-hf" for larger models
# tokenizer_llama2 = AutoTokenizer.from_pretrained(model_name)
# Set the pad_token to eos_token to avoid padding errors
# tokenizer_llama2.pad_token = tokenizer_llama2.eos_token
# model_llama2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
# model_llama2 = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map=aa

# Define the custom EOS sequences
custom_eos_sequences = ["\n", "."]
    
def log_memory_usage(rank):
    # Log GPU memory usage
    print(f"GPU {rank} - Current Memory Allocated: {torch.cuda.memory_allocated(f'cuda:{rank}') / (1024 ** 2):.2f} MB")
    print(f"GPU {rank} - Max Memory Allocated: {torch.cuda.max_memory_allocated(f'cuda:{rank}') / (1024 ** 2):.2f} MB")
    
    # Log CPU memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"CPU Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")

    # Log overall GPU memory usage using GPUtil
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id} - Memory Free: {gpu.memoryFree}MB, Memory Used: {gpu.memoryUsed}MB, Memory Total: {gpu.memoryTotal}MB")
        
# from src.extraction_utils import extract_json_from_text
def extract_json_from_text(text):
    """Extracts JSON from the given text."""

    start_index = text.find("{")
    end_index = text.rfind("}") + 1
    json_string = text[start_index:end_index]
    
    return json.loads(json_string)

def save_results(results, directory, filename):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'Results saved to {file_path}')
    
########### Below implementation is for generating explanations using Llava model without LLama 2 completions ###########
# def generate_explanations_MM(model, processor, sample, rank, params, config_logging):
def generate_explanations_MM(model, processor, sample, rank, config): # pass config instead of params and config_logging
    torch_device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    # model = model.half().to(torch_device)
    # raw_image = Image.open(sample['image_paths'][0])
    question_id = sample['question_ids'][0]
    raw_image = sample['images'][0]  # Assuming images are already loaded in the sample
    question = sample['questions'][0]
    total = config['mm_model']['no_of_responses_sampled_per_image']
    if os.path.exists(os.path.join(config['logging']['explanation_dir'], f"explanations_{question_id}.pkl")):
        print(f"Explanations for question {question_id} already exist... Skipping...")
        return f"explanations_{question_id}.pkl"

    print(f"Generating explanations for question {question_id} using model type: {config['mm_model']['model_type']}")

    total = config['mm_model']['no_of_responses_sampled_per_image']
    if config['mm_model']['model_type'] == 'phi':
        generation_config = GenerationConfig.from_pretrained(config['mm_model']['model_id'])
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        prompts = [f"{user_prompt}<|image_1|>{question}{prompt_suffix}{assistant_prompt}"] * total
        inputs = processor(text=prompts, images=[raw_image]*total, return_tensors='pt', padding=True).to(torch_device)

    elif config['mm_model']['model_type'] == 'qwen':
        messages = [{"role": "user", "content": [{"type": "image", "image": raw_image}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text]*total, images=image_inputs*total, padding=True, return_tensors="pt").to(torch_device)

    elif config['mm_model']['model_type'] == 'llava':
        prompts = [f"USER: <image>\n{question}\nASSISTANT:"] * total
        inputs = processor(text=prompts, images=[raw_image]*total, return_tensors="pt", padding=True).to(torch_device)

    else:
        raise ValueError(f"Unsupported model_type: {config['mm_model']['model_type']}")

    if config['mm_model']['model_type'] in['llava', 'qwen']:
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=config['mm_model']['temperature'],
            top_p=config['mm_model']['top_p'],
            num_beams=config['mm_model']['num_beams'],
            max_new_tokens=config['mm_model']['max_new_tokens'],
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    elif config['mm_model']['model_type'] == 'phi': # is there any issue here? Answer: Yes, the syntax is incorrect.

        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=config['mm_model']['temperature'],
            top_p=config['mm_model']['top_p'],
            num_beams=config['mm_model']['num_beams'],
            max_new_tokens=config['mm_model']['max_new_tokens'],
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=generation_config,
            num_logits_to_keep=0, # Disable logits for efficiency
        )
        
    else:
        raise ValueError(f"Unsupported model_type: {config['mm_model']['model_type']}")

    for i in range(total):
        generated_token_ids = outputs.sequences[i, inputs['input_ids'].shape[-1]:]
        decoded_outputs = processor.decode(generated_token_ids, skip_special_tokens=True).strip().split('\n')[0]
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

        sample[f'response_{i}'] = {
            'prompt': prompts[i],
            'transition_scores': transition_scores[i].cpu().numpy(),
            'generated_token_ids': generated_token_ids.cpu().numpy(),
            'decoded_outputs': decoded_outputs
        }

    del inputs, outputs, generated_token_ids, decoded_outputs, transition_scores
    torch.cuda.empty_cache()

    save_results(sample, config['logging']['explanation_dir'], f"explanations_{question_id}.pkl")
    print(f'Explanations generated and saved to file f"explanations_{question_id}.pkl"')
    return f"explanations_{question_id}.pkl"


###############################################################################
# We define a new function `generate_explanations_MM_med` 
# that is specialized for your LLaVA-med model. 
# It's similar to your original `generate_explanations_MM`, 
# but we adapt the prompt + generate logic to your medical model.
###############################################################################
def generate_explanations_MM_med(model, tokenizer, image_processor, sample, rank, params, config_logging):
    """
    Generate multiple responses for a single sample (image+question)
    using a medical LLaVA-like model.

    model: The loaded LLaVA-med model
    tokenizer: The associated tokenizer
    image_processor: A CLIP-like image processor
    sample: A batch from the SLAKE dataloader (dict)
    rank: which GPU
    params: generation config
    config_logging: logging config with directories
    """

    # 1) Device
    torch_device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

    # 2) Identify sample info
    question_id = sample['qids'][0]
    prompt = sample['promptified_questions'][0]  # the pre-made prompt text
    image_path = sample['image_paths'][0]

    # 3) Check if we already have a saved explanation
    explanation_file = os.path.join(config_logging['explanation_dir'], f"explanations_{question_id}.pkl")
    if os.path.exists(explanation_file):
        print(f"Explanations for question {question_id} already exist... Skipping...")
        return f"explanations_{question_id}.pkl"

    print(f"Generating explanations for question {question_id}")

    # 4) Load the image
    raw_image = Image.open(image_path).convert("RGB")

    # 5) Build batch of images & prompts
    total = params['no_of_responses_sampled_per_image']
    # prompts = question  # single prompt

    # 6) Preprocess images
    # Preprocess the single image
    image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
    images = image_tensor.unsqueeze(0).half().cuda()
    images = images.to(torch_device, dtype=torch.float16)
    
    replace_token = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, 'mm_use_im_start_end', False):
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    print(f"The final prompt: {prompt}")
    num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
    image_args = {"images": images}
    
    ## Preprocess text
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch_device)

    ### Model-specific parameters
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
    do_sample = True if temperature > 0.001 else False

    # 6) Stopping criteria    
    stop_str = params.get("stop_str", "###")
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

    # 7) Construct text input for each item
    # Because this is LLaVA-med, we likely do something like <image> token in the prompt
    # or <im_start><image><im_end>. We'll assume you have a single special token <image>.
    # Adjust if your model expects something else (like <im_start><image><im_end>).
    # We'll simply pass them in a loop, generating one at a time (or do a batched approach).
    

    for i in range(total):
        # Single prompt
        # The question might already have <image> inside it, or you insert it:
        # e.g.: question_text = question + "\n<image>"
        # Let's assume promptified_questions includes <image> 
        # or the special tokens as needed.
        
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                # images=single_image,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_beams=params['num_beams'],
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=[stopping_criteria],
                **image_args
            )

        # decode
        # generated_ids = outputs.sequences[0, input_ids.shape[1]:]  # remove prompt portion
        # dont remove prompt portion, as the model dosent output the prompt
        generated_ids = outputs.sequences[0]
        decoded_outputs = tokenizer.decode(generated_ids, skip_special_tokens=True)
        decoded_outputs = decoded_outputs.strip().split('\n')[0]
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        # store results
        sample[f'response_{i}'] = {
            'prompt': prompt,
            'generated_token_ids': generated_ids.cpu().numpy(),
            'decoded_outputs': decoded_outputs,
            'transition_scores': transition_scores.cpu().numpy()
        }

        # this will 
        # generated_token_ids = outputs.sequences[i, inputs['input_ids'].shape[-1]:]
        # decoded_outputs = processor.decode(generated_token_ids, skip_special_tokens=True).strip().split('\n')[0]
        # transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

        # sample[f'response_{i}'] = {
        #     'prompt': prompts[i],
        #     'transition_scores': transition_scores[i].cpu().numpy(),
        #     'generated_token_ids': generated_token_ids.cpu().numpy(),
        #     'decoded_outputs': decoded_outputs
        # }
        
        # clear GPU memory after each step
        del outputs, generated_ids
        torch.cuda.empty_cache()

    # 8) save
    save_results(sample, config_logging['explanation_dir'], f"explanations_{question_id}.pkl")
    print(f'Explanations generated and saved => explanations_{question_id}.pkl')

    return f"explanations_{question_id}.pkl"

def call_gemini_with_retries(model, contents, generation_config=None, candidate_count=1, max_retries=3):
    for attempt in range(max_retries):
        try:
            return model.generate_content(
                contents=contents,
                generation_config=generation_config,
                candidate_count=candidate_count
            )
        except ResourceExhausted as e:
            # 429 Too Many Requests
            wait_time = 2**attempt + random.random()
            print(f"[Retry {attempt+1}/{max_retries}] Rate-limited. Sleeping {wait_time:.1f}s...")
            time.sleep(wait_time)
        except DeadlineExceeded as e:
            # Timed out
            wait_time = 2**attempt + random.random()
            print(f"[Retry {attempt+1}/{max_retries}] Request timed out. Sleeping {wait_time:.1f}s...")
            time.sleep(wait_time)
    raise RuntimeError("Max retries exceeded for Gemini request")

def generate_explanations_MM_gemini(model, sample, rank, params, config_logging):
    """
    Generate explanations for a multimodal (image + text) question using Google's Gemini Model.

    :param model: An instance of genai.GenerativeModel (e.g. genai.GenerativeModel(model_id)).
    :param sample: Dictionary containing image paths, question IDs, promptified questions, etc.
    :param rank: Device rank or process rank (to keep consistent with your existing structure).
    :param params: Dictionary of generation parameters (e.g. temperature, top_p, etc.).
    :param config_logging: Dictionary of logging/config information, including explanation_dir.
    :return: The filename of the saved explanations pkl file.
    """

    # Identify question / image
    # question_id = sample['question_ids'][0] # for vqa
    question_id = sample['qids'][0] # for slake 
    # image_path = sample['image_paths'][0] # for vqa
    image_path = sample['image_paths'][0].replace('/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/Slake1.0/imgs', '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/Slake1.0/imgs') # for slake 

    generation_config = {
    "max_output_tokens": 128,  # Adjust the number of tokens as needed
    "temperature": 0.7,        # Optional: Controls the randomness of the output
    "top_p": 0.9               # Optional: Controls the diversity of the output
}
    
    # Check if results already exist
    explanation_filename = f"explanations_{question_id}.pkl"
    explanation_filepath = os.path.join(config_logging['explanation_dir'], explanation_filename)

    if os.path.exists(explanation_filepath):
        print(f"Explanations for question {question_id} already exist... Skipping...")
        return explanation_filename

    print(f"Generating explanations for question {question_id}")

    # Number of responses to sample for each image
    total = params.get('no_of_responses_sampled_per_image', 1)

    # Read image in binary
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    # The base prompt or question
    base_prompt = sample['promptified_questions'][0]

    # Generate multiple responses using Gemini
    for i in range(total):
        # Adjust prompt if needed, or use the same prompt for each sample
        prompt = base_prompt

        # Make the request to Gemini
        response = model.generate_content(
            contents=[
                {"mime_type": "image/jpeg", "data": image_data},   # or "image/png" if PNG
                {"text": prompt}
            ],
            generation_config=generation_config,
            # candidate_count=1
            # You can add more parameters if needed, e.g. candidate_count, etc.
        )
        # (CHANGES) Handle blocked or empty responses
        if not response.candidates:
            # The content was blocked or no candidate was returned
            print(prompt)
            print('-'*50)
            print(f"BLOCKED_BY_CONTENT_POLICY: reason={response.prompt_feedback.block_reason}")
            output_text = f"BLOCKED_BY_CONTENT_POLICY: reason={response.prompt_feedback.block_reason}"
        else:
            # Safe to call response.text now, or directly get from first candidate
            # `response.text` is a convenience for a single candidate
            # or you can do: output_text = response.candidates[0]['output']
            # but response.text should work since there is exactly 1 candidate
            output_text = response.text

        # If you want only the first line, you can do something like:
        output_text_first_line = output_text.split('\n')[0]

        sample[f'response_{i}'] = {
            'prompt': prompt,
            'decoded_outputs': output_text_first_line
        }
        

    # Save the results to .pkl
    save_results(sample, config_logging['explanation_dir'], explanation_filename)
    print(f'Explanations generated and saved to file "{explanation_filename}"')

    return explanation_filename

    
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
        
def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    
def annotate(image: Union[Image.Image, np.ndarray], detections: List[DetectionResult]) -> Image.Image:
    image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = detection.box
        draw.rectangle(box.xyxy, outline="red", width=3)
        draw.text((box.xmin, box.ymin), f"{detection.label} {detection.score:.2f}", fill="red")
    return image

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float,
    object_detector: pipeline
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """

    labels = [label if label.endswith(".") else label+"." for label in labels]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

# called as segment(image, detections, True, segmentator, processor, rank)
def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool,
    segmentator: AutoModelForMaskGeneration,
    processor: AutoProcessor,
    rank: int = 0
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(f'cuda:{rank}')
    inputs = {k: v.half().to(f'cuda:{rank}') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs['original_sizes'].int(),
        reshaped_input_sizes=inputs['reshaped_input_sizes'].int(),
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    del outputs, masks, inputs
    return detection_results

# generate_grounded_segmentation(sample_explanations, threshold=config['grounding']['threshold'],object_detector=object_detector, segmentator=segmentator, processor=processor, rank=rank)

def generate_grounded_segmentation(
        sample_explanations_file_name,
        threshold,
        object_detector, 
        segmentator, 
        processor,
        rank, config_logging) -> Tuple[np.ndarray, List[DetectionResult]]:
    
    sample_explanations_file_path = os.path.join(config_logging['explanation_dir'], sample_explanations_file_name)
    sample_explanations = pickle.load(open(sample_explanations_file_path, 'rb'))
    question_id = sample_explanations['question_ids'][0]
    sample_grounding_file_path = os.path.join(config_logging['grounding_dir_with_gdsam'], f"grounding_{question_id}.pkl")
    # check if the grounding file already exists
    if os.path.exists(sample_grounding_file_path):
        print(f"Grounding scores for question {question_id} already exist at {sample_grounding_file_path}. Skipping...")
        return
    else:
        print(f"Generating grounding scores for question {question_id}")
        # for key in tqdm(sample_explanations.keys(), desc=f"Generating Grounding Scores on GPU {rank}", total=len(sample_explanations)):
        sample_grounding = {}
        sample_grounding['question_id'] = question_id
        for key in sample_explanations.keys():
            try:
                if not key.startswith("response"):
                    continue
                else:
                    sample_grounding[key] = {}
                    # check if decoded_outputs_llama2 is present, if not use decoded_outputs
                    response = sample_explanations[key].get('decoded_outputs_llama2', sample_explanations[key]['decoded_outputs'])
                    sample_grounding[key]['decoded_outputs'] = response
                image = sample_explanations['image_paths'][0]
                # reponse_jsonified = extract_json_from_text(response) # remove json extraction, as we are not prompting the model to get json response
                # labels = [reponse_jsonified.get('explanation')]
                labels = [response]
                if isinstance(image, str):
                    image = load_image(image)
                try:
                    detections = detect(image, labels, threshold, object_detector)
                    # grounding from grounding dino 
                    sample_grounding[key]['grounding_with_gd_score'] = detections[0].score
                except Exception as e:
                    print('-'*50)
                    print(f"Detection error for response {key}: {e}")
                    # sample_explanations[key]['error_detection'] = f"Detection error: {e}"
                    sample_grounding[key]['error_detection'] = f"Detection error: {e}"
                    continue
                try:
                    detections = segment(image, detections, True,
                                        segmentator, processor, rank)
                    # sample_explanations[key]['detections'] = detections
                    # sample_grounding[key]['detections_gdsam'] = detections[0]
                    # sample_explanations[key]['image'] = np.array(image)
                    # sample_explanations[key]['grounding_score'] = detections[0].score
                    sample_grounding[key]['grounding_with_gd_sam_score'] = detections[0].score
                except Exception as e:
                    print('-'*50)
                    print(f"Segmentation error for response {key}: {e}")
                    # sample_explanations[key]['error_segmentation'] = f"Segmentation error: {e}"
                    sample_grounding[key]['error_segmentation'] = f"Segmentation error: {e}"
                    continue

                
                # Clear GPU cache to free memory
                del detections
                torch.cuda.empty_cache()
            except Exception as e:
                print('-'*50)
                print(f"Grounding error for response {key}: {e}")
                sample_explanations[key]['error_grounding'] = f"Grounding error: {e}"


        # save the results to a file
        # save_results(sample_explanations, config_logging['grounding_dir'], f"grounding_{sample_explanations['question_ids'][0]}.pkl")
        save_results(sample_grounding, config_logging['grounding_dir_with_gdsam'], f"grounding_{sample_grounding['question_id']}.pkl")
        print(f'Grounding scores generated and saved to file f"grounding_{sample_explanations["question_ids"][0]}.pkl"') 

def generate_grounding_with_llama3_2(
        file,
        model,
        processor,
        rank,
        config_logging):
    
    # Load explanation file
    sample_explanations_file_path = os.path.join(config_logging['explanation_dir'], file)
    sample_explanations = pickle.load(open(sample_explanations_file_path, 'rb'))
    # question_id = sample_explanations['question_ids'][0]
    question_id = sample_explanations['qids'][0]

    # Output path
    # sample_explanations_with_grounding_file_path = os.path.join(config_logging['grounding_dir_with_llama32_90B'], f"grounding_{question_id}.pkl")
    sample_explanations_with_grounding_file_path = os.path.join(config_logging['grounding_dir_with_llama32_11B'], f"grounding_{question_id}.pkl")

    # Create directory if it doesn't exist
    # os.makedirs(config_logging['grounding_dir_with_llama32_90B'], exist_ok=True)
    os.makedirs(config_logging['grounding_dir_with_llama32_11B'], exist_ok=True)
    # Skip if already processed
    if os.path.exists(sample_explanations_with_grounding_file_path):
        print(f"Grounding scores for question {question_id} already exist. Skipping...")
        return

    sample_explanations_with_grounding = sample_explanations.copy()
    # sample_explanations_with_grounding['question_id'] = question_id

    image_path = sample_explanations['image_paths'][0]
    # image_path = sample_explanations['image_paths'][0].replace('/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/Slake1.0/imgs', '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/Slake1.0/imgs')
    image = Image.open(image_path).convert("RGB")
                
    # Process responses
    for key in sample_explanations.keys():
        if not key.startswith("response"):
            continue

        try:
            response_text = sample_explanations[key].get('decoded_outputs', None)
            if response_text is None:
                print(f"Skipping response {key} as no decoded outputs found.")
                continue

            # Llama 3.2 Vision Prompt
            GROUNDING_PROMPT = (
                "I have an image and a short statement that describes a specific part of the image. "
                "Your job is to verify if this statement accurately reflects what is shown in the image.\n\n"
                "Image: <Attached above>\n"
                f"Statement: \"{response_text}\"\n\n"
                "Instructions: Respond with only one word — either \"Yes\" if the statement is correct, "
                "\"No\" if the statement is incorrect, or \"Not sure\" if you are uncertain. "
                "Do not provide any additional explanations."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Attach the image
                        {"type": "text", "text": GROUNDING_PROMPT}
                    ]
                }
            ]

            # Prepare inputs
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)

            # Generate output
            output = model.generate(**inputs, max_new_tokens=70)
            result = processor.decode(output[0][inputs["input_ids"].shape[-1]:])

            # Save the result
            sample_explanations_with_grounding[key]["llama_32_response"] = result
            print(f"Rank {rank} - Result: {result}")

            # Clear GPU cache
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in processing {key}: {e}")
            sample_explanations_with_grounding[key] = {"error": str(e)}

    # Save results
    with open(sample_explanations_with_grounding_file_path, 'wb') as f:
        pickle.dump(sample_explanations_with_grounding, f)

    print(f"Saved grounding results for {question_id} to {sample_explanations_with_grounding_file_path}")
    
def generate_grounding_with_gemini(
    file,
    gemini_model,
    rank,
    config_logging
):
    """
    Use Gemini to check if each response in the explanation file correctly describes the image.

    :param file: The name of the .pkl file containing explanations (e.g., "explanations_123.pkl").
    :param gemini_model: An instance of genai.GenerativeModel (e.g., genai.GenerativeModel(model_id)).
    :param rank: The current worker/process rank (useful for logging). 
    :param config_logging: Dictionary with paths such as explanation_dir and grounding_dir.
    """
    # 1. Load the explanation file
    explanation_path = os.path.join(config_logging['explanation_dir'], file)
    with open(explanation_path, 'rb') as f:
        sample_explanations = pickle.load(f)

    question_id = sample_explanations['question_ids'][0] # for vqa
    # question_id = sample_explanations['qids'][0] # for slake 

    # 2. Build output path for grounding results
    # Adjust which directory you use based on your config.
    grounding_dir = config_logging['grounding_dir_with_gemini']  # or another path if you prefer
    os.makedirs(grounding_dir, exist_ok=True)

    grounding_path = os.path.join(grounding_dir, f"grounding_{question_id}.pkl")

    # Skip if already done
    if os.path.exists(grounding_path):
        print(f"Grounding scores for question {question_id} already exist. Skipping...")
        return

    # Copy the entire dictionary so we can add the grounding results
    sample_explanations_with_grounding = sample_explanations.copy()

    # 3. Read the image
    image_path = sample_explanations['image_paths'][0] # for vqa
    # image_path = sample_explanations['image_paths'][0].replace('/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/Slake1.0/imgs', '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/Slake1.0/imgs') # for slake 
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    # 4. Iterate over each response key
    for key in sample_explanations.keys():
        if not key.startswith("response"):
            continue

        response_text = sample_explanations[key].get('decoded_outputs', None)
        if not response_text:
            print(f"Rank {rank}: Skipping {key} - no decoded outputs found.")
            continue

        # The prompt that we feed Gemini
        GROUNDING_PROMPT = (
            "I have an image and a short statement that describes a specific part of the image. "
            "Your job is to verify if this statement accurately reflects what is shown in the image.\n\n"
            "Image: <The attached image>\n"
            f"Statement: \"{response_text}\"\n\n"
            "Instructions: Respond with only one word — \"Yes\" if the statement is correct, "
            "\"No\" if the statement is incorrect, or \"Not sure\" if you are uncertain. "
            "Do not provide any additional explanations."
        )

        # 5. Call Gemini
        try:
            # A minimal generation config
            generation_config = {
                "max_output_tokens": 50,
                "temperature": 0.0,
                "top_p": 1.0,
            }

            # Make the request
            response = gemini_model.generate_content(
                contents=[
                    {"mime_type": "image/jpeg", "data": image_data},
                    {"text": GROUNDING_PROMPT}
                ],
                generation_config=generation_config,
                # candidate_count=1  # Single answer 
            )

            # Check if blocked or empty
            if not response.candidates:
                grounding_result = f"BLOCKED: reason={response.prompt_feedback.block_reason}"
            else:
                grounding_result = response.text  # Single candidate

            sample_explanations_with_grounding[key]["gemini_grounding_response"] = grounding_result
            print(f"Rank {rank} - {key} => {grounding_result}")

        except (ResourceExhausted, DeadlineExceeded) as e:
            # Rate-limiting or timeouts
            print(f"Rank {rank} - {key} => Rate limit or timeout error: {str(e)}")
            sample_explanations_with_grounding[key]["gemini_grounding_response"] = f"API_ERROR: {str(e)}"
        except Exception as e:
            print(f"Rank {rank} - {key} => Error: {str(e)}")
            sample_explanations_with_grounding[key]["gemini_grounding_response"] = f"ERROR: {str(e)}"

        # Optional: if you still want to reduce GPU memory usage in a multi-GPU environment
        torch.cuda.empty_cache()

    # 6. Save the updated dictionary
    with open(grounding_path, 'wb') as f:
        pickle.dump(sample_explanations_with_grounding, f)

    print(f"Saved Gemini grounding results for question {question_id} to {grounding_path}")
    
def generate_grounding_with_qwen_vl(
        file,
        model,
        processor,
        rank,
        config_logging):
    
    # Load explanation file
    sample_explanations_file_path = os.path.join(config_logging['explanation_dir'], file)
    sample_explanations = pickle.load(open(sample_explanations_file_path, 'rb'))
    # question_id = sample_explanations['question_ids'][0]
    question_id = sample_explanations['qids'][0]

    # Output path
    # sample_explanations_with_grounding_file_path = os.path.join(config_logging['grounding_dir_with_qwen_vl'], f"grounding_{question_id}.pkl")
    sample_explanations_with_grounding_file_path = os.path.join(config_logging['grounding_dir_with_qwen_vl_25_7B'], f"grounding_{question_id}.pkl")

    # Create directory if it doesn't exist
    # os.makedirs(config_logging['grounding_dir_with_qwen_vl'], exist_ok=True)
    os.makedirs(config_logging['grounding_dir_with_qwen_vl_25_7B'], exist_ok=True)
    # Skip if already processed
    if os.path.exists(sample_explanations_with_grounding_file_path):
        print(f"Grounding scores for question {question_id} already exist. Skipping...")
        return

    sample_explanations_with_grounding = sample_explanations.copy()
    # sample_explanations_with_grounding['question_id'] = question_id

    image_path = sample_explanations['image_paths'][0]
    # image_path = sample_explanations['image_paths'][0].replace('/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/Slake1.0/imgs', '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/Slake1.0/imgs')
    image = Image.open(image_path).convert("RGB")
                
    # Process responses
    for key in sample_explanations.keys():
        if not key.startswith("response"):
            continue

        try:
            response_text = sample_explanations[key].get('decoded_outputs', None)
            if response_text is None:
                print(f"Skipping response {key} as no decoded outputs found.")
                continue

            # Llama 3.2 Vision Prompt
            GROUNDING_PROMPT = (
                "I have an image and a short statement that describes a specific part of the image. "
                "Your job is to verify if this statement accurately reflects what is shown in the image.\n\n"
                "Image: <Attached above>\n"
                f"Statement: \"{response_text}\"\n\n"
                "Instructions: Respond with only one word — either \"Yes\" if the statement is correct, "
                "\"No\" if the statement is incorrect, or \"Not sure\" if you are uncertain. "
                "Do not provide any additional explanations."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": image_path},  # Attach the image
                        {"type": "text", "text": GROUNDING_PROMPT}
                    ]
                }
            ]

            # Prepare inputs
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text = [input_text],
                images = image_inputs,
                videos = None,
                # add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)

            # Generate output
            output = model.generate(**inputs, max_new_tokens=70)
            result = processor.decode(output[0][inputs["input_ids"].shape[-1]:])

            # Save the result
            sample_explanations_with_grounding[key]["qwen_vl_response"] = result
            print(f"Rank {rank} - Result: {result}")

            # Clear GPU cache
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in processing {key}: {e}")
            sample_explanations_with_grounding[key] = {"error": str(e)}

    # Save results
    with open(sample_explanations_with_grounding_file_path, 'wb') as f:
        pickle.dump(sample_explanations_with_grounding, f)

    print(f"Saved grounding results for {question_id} to {sample_explanations_with_grounding_file_path}")

### Grounding with BioMedCLIP
def generate_biomedclip_scores(
    sample_explanations_file_name: str,
    model,               # your open_clip model object (BioMedCLIP)
    preprocess,          # your open_clip preprocess function
    tokenizer,           # your open_clip tokenizer
    device: str,
    rank: int,
    config_logging: dict
):
    """
    Replaces "grounded segmentation" with BioMedCLIP text-image similarity scoring.
    For each response in the sample .pkl, we compute a CLIP-like similarity score.
    
    sample_explanations_file_name:  The .pkl filename (e.g. "explanations_12345.pkl")
    model:                          The open_clip model loaded from 'create_model_from_pretrained(...)'
    preprocess:                     The open_clip image pre-processing function
    tokenizer:                      The open_clip tokenizer for text
    device:                         e.g. "cuda:0"
    rank:                           GPU rank (for logging)
    config_logging:                 dict with your logging settings (e.g. "explanation_dir" and "grounding_dir")
    """
    # 1) Load the LLaVA explanations .pkl
    sample_explanations_file_path = os.path.join(
        config_logging['explanation_dir'],
        sample_explanations_file_name
    )
    with open(sample_explanations_file_path, 'rb') as f:
        sample_explanations = pickle.load(f)

    # 2) Check if we already have a "BioMedCLIP" score file
    # question_id = sample_explanations['question_ids'][0]  # or 'qids'[0] if you store it differently
    question_id = sample_explanations['qids'][0]
    sample_clip_file_path = os.path.join(
        config_logging['grounding_dir_with_biomedclip'],
        f"clipscore_{question_id}.pkl"
    )
    if os.path.exists(sample_clip_file_path):
        print(f"BioMedCLIP scores for question {question_id} already exist at {sample_clip_file_path}. Skipping...")
        return

    print(f"[Rank {rank}] Computing BioMedCLIP scores for question_id {question_id}...")

    # 3) Build a dict to store results
    sample_clip_scores = {"question_id": question_id}

    # 4) Load the image
    image_path = sample_explanations['image_paths'][0].replace('/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/', '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/')
    if not os.path.exists(image_path):
        print(f"[Warning] Image path does not exist: {image_path}")
        return
    image = Image.open(image_path).convert("RGB")

    # 5) Preprocess the image for BioMedCLIP
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # 6) For each response, compute CLIP similarity
    #    We'll store them in sample_clip_scores.
    for key in sample_explanations.keys():
        if key.startswith("response"):
            response_text = sample_explanations[key].get(
                'decoded_outputs_llama2',
                sample_explanations[key]['decoded_outputs']
            )
            # If no text, skip
            if not response_text or not isinstance(response_text, str):
                continue

            # Tokenize text
            text_tokens = tokenizer(response_text, context_length=256).to(device)

            with torch.no_grad():
                # image_features: [batch_size, dim]
                # text_features:  [batch_size, dim]
                # logit_scale:    scalar
                image_features, text_features, logit_scale = model(image_tensor, text_tokens)
                # similarity shape: [1] (scalar)
                similarity_score = (logit_scale * image_features @ text_features.T).squeeze()
                similarity_score = similarity_score.item()

            # Store the similarity in our dict
            sample_clip_scores.setdefault(key, {})
            sample_clip_scores[key]['decoded_outputs'] = response_text
            sample_clip_scores[key]['biomedclip_score'] = similarity_score

    # 7) Save the results to a new .pkl
    #    e.g. "clipscore_12345.pkl"
    os.makedirs(config_logging['grounding_dir'], exist_ok=True)
    clip_out_path = os.path.join(config_logging['grounding_dir'], f"clipscore_{question_id}.pkl")
    with open(clip_out_path, 'wb') as f:
        pickle.dump(sample_clip_scores, f)

    print(f"[Rank {rank}] BioMedCLIP scores saved => {clip_out_path}")
    
if __name__ == "__main__":
    # test code for explanation generation
    
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    from PIL import Image
    import numpy as np
    import requests
    import matplotlib.pyplot as plt
    import cv2


    # ----------------------------
    # 1. DEVICE SETUP
    # ----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----------------------------
    # 2. LOAD MODELS
    # ----------------------------

    # Grounding DINO (Object Detection)
    detector_model_id = 'IDEA-Research/grounding-dino-tiny'
    detector_model = AutoModel.from_pretrained(detector_model_id).to(device)
    detector_tokenizer = AutoTokenizer.from_pretrained(detector_model_id)
    print("Grounding DINO model loaded.")

    # SAM (Segmentation)
    segmenter_model_id = 'facebook/sam-vit-base'
    segmenter_model = AutoModel.from_pretrained(segmenter_model_id).to(device)
    segmenter_processor = AutoProcessor.from_pretrained(segmenter_model_id)
    print("SAM model loaded.")

    config = {
        'mm_model': {
            'model_path': "llava-hf/llava-1.5-7b-hf",
            'temperature': 0.5,
            'top_p': 0.95,
            'num_beams': 5,
            'max_new_tokens': 100,
            'no_of_responses_sampled_per_image': 5
        },
        'grounding': {
            'detector_id': 'IDEA-Research/grounding-dino-tiny',
            'threshold': 0.01
        },
        'segmenter_id': 'facebook/sam-vit-base'
    }
    
    log_memory_usage(0)
    print('Loading MM model')
    
    model = LlavaForConditionalGeneration.from_pretrained(config['mm_model']['model_path'])
    processor = AutoProcessor.from_pretrained(config['mm_model']['model_path'])
    dataloader = get_gqa_dataloader('/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA', 'json', 0, 1)
    
    print('Loading data')
    log_memory_usage(0)
    sample = next(iter(dataloader))
    rank = 0
    
    # check explanation generation for a single sample from the MM model
    sample_explanations = generate_explanations_MM(model, processor, sample, rank, config['mm_model'])
    print('Explanation generation unit test successful')
    log_memory_usage(0)
    print('Genrated explanations for a single sample')
    
    del model, processor
    torch.cuda.empty_cache()
    
    log_memory_usage(0)
    print('Cleared memory')
    
    rank = 1
    # load models for grounding and segmentation
    object_detector = pipeline(model=config['grounding']['detector_id'], task="zero-shot-object-detection", device=f'cuda:{rank}')
    object_detector.model = object_detector.model.half().to(device=f'cuda:{rank}')
    segmentator = AutoModelForMaskGeneration.from_pretrained(config['segmenter_id']).half().to(device=f'cuda:{rank}')
    processor = AutoProcessor.from_pretrained(config['segmenter_id'])
    
    log_memory_usage(0)
    print('Loaded models for grounding and segmentation')
    
    # check grounding for a single sample
    samples_with_grounding = generate_grounded_segmentation(
        sample_explanations,
        threshold=config['grounding']['threshold'],object_detector=object_detector, segmentator=segmentator, processor=processor, rank=rank)
    
    log_memory_usage(0)
    print('Generated grounding for a single sample')
    print('Grounding unit test successful')    
    
    
    
    
    
#####################################################################
# Custom stopping criteria for Llama2 model
#####################################################################
# class CustomStoppingCriteria(StoppingCriteria):
#     def __init__(self, tokenizer, eos_tokens):
#         self.tokenizer = tokenizer
#         self.eos_token_ids = [self.tokenizer.encode(eos, add_special_tokens=False)[0] for eos in eos_tokens]

#     def __call__(self, input_ids, scores, **kwargs):
#         # Check if any of the last generated tokens match the custom EOS tokens
#         return input_ids[0, -1].item() in self.eos_token_ids

# class CustomStoppingCriteria(StoppingCriteria):
#     def __init__(self, tokenizer, eos_sequences):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.eos_sequences_ids = [self.tokenizer.encode(eos, add_special_tokens=False) for eos in eos_sequences]
#         self.max_eos_length = max(len(seq) for seq in self.eos_sequences_ids)

#     def __call__(self, input_ids, scores, **kwargs):
#         for eos_seq in self.eos_sequences_ids:
#             if len(input_ids[0]) >= len(eos_seq):
#                 if input_ids[0][-len(eos_seq):].tolist() == eos_seq:
#                     print(f"Stopping generation: Detected EOS sequence '{self.tokenizer.decode(eos_seq)}'")
#                     return True
#         return False
# class CustomStoppingCriteria(StoppingCriteria):
#     def __init__(self, tokenizer, eos_sequences):
#         super().__init__()
#         self.tokenizer = tokenizer
#         # Tokenize each EOS sequence into a list of token IDs
#         self.eos_sequences_ids = [self.tokenizer.encode(eos, add_special_tokens=False) for eos in eos_sequences]
#         # Determine the maximum length among all EOS sequences
#         self.max_eos_length = max(len(seq) for seq in self.eos_sequences_ids)

#     def __call__(self, input_ids, scores, **kwargs):
#         # Iterate through each EOS sequence
#         for eos_seq in self.eos_sequences_ids:
#             seq_length = len(eos_seq)
#             if len(input_ids[0]) >= seq_length:
#                 # Extract the last 'n' tokens where 'n' is the length of the EOS sequence
#                 last_tokens = input_ids[0][-seq_length:].tolist()
#                 if last_tokens == eos_seq:
#                     eos_decoded = self.tokenizer.decode(eos_seq, skip_special_tokens=True)
#                     print(f"Stopping generation: Detected EOS sequence '{eos_decoded}'")
#                     return True
#         return False


##################################################################### 
# Below implementation is for generating explanations using Llava model with LLama 2 completions 
#####################################################################
# def generate_explanations_MM(
#     model_llava,
#     processor_llava,
#     sample,
#     rank,
#     params,
#     config_logging
# ):
#     # 1) Set up device + half precision for Llava
#     torch_device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
#     model_llava = model_llava.half().to(torch_device)

#     question_id = sample['question_ids'][0]
#     question_text = sample['questions'][0]  # The raw question text
#     raw_image = Image.open(sample['image_paths'][0])

#     # 2) Early exit if explanations file already exists
#     explanation_file = os.path.join(config_logging['explanation_dir'], f"explanations_{question_id}.pkl")
#     if os.path.exists(explanation_file):
#         print(f"Explanations for question {question_id} exist at {explanation_file} ... Skipping...")
#         return f"explanations_{question_id}.pkl"

#     # 3) Build a batch of identical prompts/images for multiple sampling
#     repeated_prompts = [sample['promptified_questions'][0]] * params['no_of_responses_sampled_per_image']
#     repeated_images = [raw_image] * params['no_of_responses_sampled_per_image']

#     # 4) Prepare inputs for Llava
#     # from transformers import StoppingCriteriaList
#     # stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(processor_llava.tokenizer, custom_eos_sequences)])

#     inputs = processor_llava(
#         images=repeated_images,
#         text=repeated_prompts,
#         return_tensors="pt"
#     ).to(torch_device, torch.float16)

#     print(f"Generating explanations (Llava) for question_id {question_id} ...")
#     outputs = model_llava.generate(
#         input_ids=inputs["input_ids"],
#         pixel_values=inputs["pixel_values"],
#         do_sample=True,
#         temperature=params['temperature'],
#         top_p=params['top_p'],
#         num_beams=params['num_beams'],
#         max_new_tokens=params['max_new_tokens'],
#         use_cache=False,
#         return_dict_in_generate=True,
#         output_scores=True,
#         # stopping_criteria=stopping_criteria
#     )

#     # 5) Decode each of the 20 responses
#     #    NOTE: outputs.sequences is shape [batch_size, seq_len].
#     #    We'll store them first in a temporary list.
#     batch_size = params['no_of_responses_sampled_per_image']
#     transition_scores = model_llava.compute_transition_scores(
#         outputs.sequences, outputs.scores, normalize_logits=True
#     )
#     transition_scores = transition_scores.cpu().numpy()

#     raw_decoded_responses = []
#     for i in range(batch_size):
#         # For each item i, slice out the newly generated tokens
#         generated_token_ids = outputs.sequences[i, inputs['input_ids'].shape[-1]:]
#         decoded = processor_llava.decode(generated_token_ids, skip_special_tokens=True)
#         # Truncate to first newline
#         # decoded = decoded.split('\n')[0].strip()
#         decoded = decoded.strip().split('\n')[0]
#         raw_decoded_responses.append(decoded)

#     # 6) Identify which are "short" => fewer than 4 words
#     #    We'll do a single batch call to Llama2 for these short answers.
#     short_indices = []
#     short_answers = []
#     for i, resp in enumerate(raw_decoded_responses):
#         word_count = len(resp.split())
#         if 0 < word_count < 4:
#             short_indices.append(i)
#             short_answers.append(resp)

#     # 7) If any short answers exist, run Llama2 in batch
#     expansions = [None] * batch_size  # Will hold final expansions for short answers
#     raw_llama2_responses = [None] * batch_size  # Will hold raw Llama2 responses for short answers
#     for i in range(batch_size):
#         # Default to None
#         expansions[i] = None
#         raw_llama2_responses[i] = None

#     if len(short_answers) > 0:
#         # Build a prompt for each short answer
#         llama2_prompts = []
#         for short_ans in short_answers:
#             prompt_for_llama2 = f"""
#             QUESTION: What animal is shown?
#             MODEL: dog
#             LLAMA2: The animal shown is a dog.
#             QUESTION: What is the girl eating?
#             MODEL: apple
#             LLAMA2: The girl is eating an apple.
#             QUESTION: {question_text}
#             MODEL: {short_ans}
#             LLAMA2:
#             """
#             llama2_prompts.append(prompt_for_llama2)

#         # Send them in a batch to Llama2
#         torch_device_llama2 = f"cuda:{rank}"  # or f"cuda:{rank}" if you prefer
#         model_llama2.to(torch_device_llama2)

#         tokenized_llama2 = tokenizer_llama2(
#             llama2_prompts, 
#             return_tensors="pt", 
#             padding=True,
#             truncation=True
#         ).to(torch_device_llama2)

#         stopping_criteria_llama2 = StoppingCriteriaList([
#             CustomStoppingCriteria(tokenizer_llama2, custom_eos_sequences)
#         ])

#         with torch.no_grad():
#             outputs_llama2 = model_llama2.generate(
#                 **tokenized_llama2,
#                 max_new_tokens=50,
#                 temperature=0.1,
#                 num_beams=2,
#                 top_p=0.9,
#                 stopping_criteria=stopping_criteria_llama2
#             )

#         # Decode each expanded answer
#         # NOTE: outputs_llama2 is shape [batch_size_of_short_answers, seq_len].
#         # We'll map them back to the correct indices.
#         for idx_in_batch, i_response in enumerate(short_indices):
#             generated_ids = outputs_llama2[idx_in_batch, tokenized_llama2['input_ids'].shape[-1]:]
#             expanded_text = tokenizer_llama2.decode(generated_ids, skip_special_tokens=True)
#             # Truncate to the first newline (if present)
#             # expanded_text = expanded_text.split('\n')[0].strip()
#             expanded_text = expanded_text.strip().split('\n')[0]
#             expansions[i_response] = expanded_text
#             raw_llama2_responses[i_response] = expanded_text

#     # 8) Now expansions[] holds the final text for all 20 responses
#     #    Let's store them in sample.
#     for i in range(batch_size):
#         sample[f"response_{i}"] = {
#             "prompt": sample['promptified_questions'][0],
#             "transition_scores": transition_scores[i],
#             "decoded_outputs": expansions[i] if expansions[i] is not None else raw_decoded_responses[i],
#             "raw_decoded_outputs_llava": raw_decoded_responses[i],
#             "raw_decoded_outputs_llama2": raw_llama2_responses[i]
#         }

#     # 9) Save to disk
#     save_results(sample, config_logging['explanation_dir'], f"explanations_{question_id}.pkl")
#     print(f'Explanations (with Llama2 expansions) saved => explanations_{question_id}.pkl')

#     # 10) Cleanup
#     del inputs, outputs
#     torch.cuda.empty_cache()
#     return f"explanations_{question_id}.pkl"