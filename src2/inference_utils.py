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
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForMaskGeneration, pipeline
from tqdm import tqdm
import json
import psutil
import torch
import GPUtil

# llama2 for full sentence generation if the model gives one word response 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


######## Llama2 model for full sentence generation ########
# Load the tokenizer and model from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with "Llama-2-13b-hf" or "Llama-2-70b-hf" for larger models
tokenizer_llama2 = AutoTokenizer.from_pretrained(model_name)
# Set the pad_token to eos_token to avoid padding errors
tokenizer_llama2.pad_token = tokenizer_llama2.eos_token
model_llama2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
# model_llama2 = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map=aa

# Define the custom EOS sequences
custom_eos_sequences = ["\n", "."]

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
class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, eos_sequences):
        super().__init__()
        self.tokenizer = tokenizer
        # Tokenize each EOS sequence into a list of token IDs
        self.eos_sequences_ids = [self.tokenizer.encode(eos, add_special_tokens=False) for eos in eos_sequences]
        # Determine the maximum length among all EOS sequences
        self.max_eos_length = max(len(seq) for seq in self.eos_sequences_ids)

    def __call__(self, input_ids, scores, **kwargs):
        # Iterate through each EOS sequence
        for eos_seq in self.eos_sequences_ids:
            seq_length = len(eos_seq)
            if len(input_ids[0]) >= seq_length:
                # Extract the last 'n' tokens where 'n' is the length of the EOS sequence
                last_tokens = input_ids[0][-seq_length:].tolist()
                if last_tokens == eos_seq:
                    eos_decoded = self.tokenizer.decode(eos_seq, skip_special_tokens=True)
                    print(f"Stopping generation: Detected EOS sequence '{eos_decoded}'")
                    return True
        return False
    
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
def generate_explanations_MM(model, processor, sample, rank, params, config_logging):
    torch_device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    # model = model.half().to(torch_device)
    raw_image = Image.open(sample['image_paths'][0])
    question_id = sample['question_ids'][0]

    if os.path.exists(os.path.join(config_logging['explanation_dir'], f"explanations_{question_id}.pkl")):
        print(f"Explanations for question {question_id} already exist... Skipping...")
        return f"explanations_{question_id}.pkl"

    print(f"Generating explanations for question {question_id}")

    batch_size = params.get('inference_batch_size', max(1, params['no_of_responses_sampled_per_image'] // 2))
    total = params['no_of_responses_sampled_per_image']
    prompts = [sample['promptified_questions'][0]] * total
    images = [raw_image] * total

    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    offset = 0
    for pbatch, ibatch in zip(chunk(prompts, batch_size), chunk(images, batch_size)):
        # stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(processor.tokenizer, custom_eos_sequences)])
        # inputs = processor(images=ibatch, text=pbatch, return_tensors="pt").to(torch_device, torch.float16)
        inputs = processor(images=ibatch, text=pbatch, return_tensors="pt").to(torch_device)

        outputs = model.generate(
            input_ids=inputs['input_ids'],
            pixel_values=inputs['pixel_values'],
            do_sample=True,
            temperature=params['temperature'],
            top_p=params['top_p'],
            num_beams=params['num_beams'],
            max_new_tokens=params['max_new_tokens'],
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
            # stopping_criteria=stopping_criteria
        )

        for i in range(len(pbatch)):
            generated_token_ids = outputs.sequences[i, inputs['input_ids'].shape[-1]:]
            # take all the generated tokens, and not just the new ones
            # generated_token_ids = outputs.sequences[i]
            # decoded_outputs = processor.decode(generated_token_ids, skip_special_tokens=True).split('\n')[0]  I shouldnt split on newline as the model can generate newline and then generate the response, this is the reason why we are getting no response
            decoded_outputs = processor.decode(generated_token_ids, skip_special_tokens=True)
            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

            sample[f'response_{offset + i}'] = {
                'prompt': pbatch[i],
                'transition_scores': transition_scores[i].cpu().numpy(),
                'generated_token_ids': generated_token_ids.cpu().numpy(),
                'decoded_outputs': decoded_outputs
            }

        offset += len(pbatch)
        del inputs, outputs, generated_token_ids, decoded_outputs, transition_scores
        torch.cuda.empty_cache()

    save_results(sample, config_logging['explanation_dir'], f"explanations_{question_id}.pkl")
    print(f'Explanations generated and saved to file f"explanations_{question_id}.pkl"')
    return f"explanations_{question_id}.pkl"

##### Below implementation is for generating explanations using Llava model with LLama 2 completions #####
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
    sample_grounding_file_path = os.path.join(config_logging['grounding_dir'], f"grounding_{question_id}.pkl")
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
        save_results(sample_grounding, config_logging['grounding_dir'], f"grounding_{sample_grounding['question_id']}.pkl")
        print(f'Grounding scores generated and saved to file f"grounding_{sample_explanations["question_ids"][0]}.pkl"') 



if __name__ == "__main__":
    # test code for explanation generation
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
    dataloader = get_dataloader_gqa('/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA', 'json', 0, 1)
    
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
    
    
    
    
    
    
  