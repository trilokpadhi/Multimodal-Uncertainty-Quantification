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
from data_utils import get_dataloader
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForMaskGeneration, pipeline
from tqdm import tqdm
import json
import psutil
import torch
import GPUtil

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

def generate_explanations_MM(model, processor, sample, rank, params):
    # Assign the model and inputs to the correct device based on rank
    torch_device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    model = model.half().to(torch_device)  # Move model to half precision and to correct GPU
    prompt = sample['promptified_questions'][0]
    raw_image = Image.open(sample['image_paths'][0])
    
    # Process inputs with half precision
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(torch_device, torch.float16)
    
    # for i in range(20):
    for i in range(params['no_of_responses_sampled_per_image']):
        outputs = model.generate(
            input_ids=inputs['input_ids'],        # Text tokens
            pixel_values=inputs['pixel_values'],  # Image tokens
            do_sample=True,                      # Enable sampling
            temperature=params['temperature'],    # Sampling temperature
            top_p=params['top_p'],                # Top-p sampling
            num_beams=params['num_beams'],        # Beam search
            max_new_tokens=params['max_new_tokens'],
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Process the generated output, remove the input token ids 
        generated_token_ids = outputs.sequences[:, inputs['input_ids'].shape[-1]:] 
        decoded_outputs = processor.decode(generated_token_ids.flatten(), skip_special_tokens=True)
        
        # Store results in sample dict
        sample[f'response_{i}'] = {}
        sample[f'response_{i}']['prompt'] = prompt
        sample[f'response_{i}']['outputs'] = outputs
        sample[f'response_{i}']['generated_token_ids'] = generated_token_ids
        sample[f'response_{i}']['decoded_outputs'] = decoded_outputs

        # Clear GPU memory after each step
        del outputs, generated_token_ids, decoded_outputs
        torch.cuda.empty_cache()

    return sample


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
        sample_explanations,
        threshold,
        object_detector, 
        segmentator, 
        processor,
        rank) -> Tuple[np.ndarray, List[DetectionResult]]:
    
    for key in tqdm(sample_explanations.keys(), desc=f"Generating Grounding Scores on GPU {rank}", total=len(sample_explanations)):
        try:
            if not key.startswith("response"):
                continue
            else:
                response = sample_explanations.get(key)['decoded_outputs']
            image = sample_explanations['image_paths'][0]
            reponse_jsonified = extract_json_from_text(
                response)
            labels = [reponse_jsonified.get('explanation')]
            if isinstance(image, str):
                image = load_image(image)
            try:
                detections = detect(image, labels, threshold, object_detector)
            except Exception as e:
                print('-'*50)
                print(f"Detection error for response {key}: {e}")
                sample_explanations[key]['error_detection'] = f"Detection error: {e}"
                continue
            try:
                detections = segment(image, detections, True,
                                    segmentator, processor, rank)
                sample_explanations[key]['detections'] = detections
                sample_explanations[key]['image'] = np.array(image)
                sample_explanations[key]['grounding_score'] = detections[0].score
            except Exception as e:
                print('-'*50)
                print(f"Segmentation error for response {key}: {e}")
                sample_explanations[key]['error_segmentation'] = f"Segmentation error: {e}"
                continue

            
            # Clear GPU cache to free memory
            del detections
            torch.cuda.empty_cache()
        except Exception as e:
            print('-'*50)
            print(f"Grounding error for response {key}: {e}")
            sample_explanations[key]['error_grounding'] = f"Grounding error: {e}"

    return sample_explanations



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
    dataloader = get_dataloader('/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA', 'json', 0, 1)
    
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
    
    
    
    
    
    
  