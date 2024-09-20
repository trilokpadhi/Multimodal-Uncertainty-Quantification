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
from typing import Union, List, Optional
import numpy as np
from dataclasses import dataclass


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
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return np.array(image), results

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
    
# Example usage:
path = '/home/ubuntu/Multimodal-Uncertainty-Quantification/runs/llava_gqa_llama3_sg_grounding_temp_1_topp_1_dp_1000.pkl_1159364.pkl'
image_id = '1159364'
with open(path, 'rb') as f:
    data = pickle.load(f)
    
tuples = [('tables', 'state', 'no'), ('plates', 'state', 'no'), ('tables', 'in the picture', 'plates')]
labels = [f'{t[0]} {t[1]} {t[2]}' for t in tuples]
    
# image_path = os.path.join('/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/images', '1159364' + '.jpg')

image_path = '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/images/2402219.jpg'

# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# labels = ["a cat.", "a remote control."]
threshold = 0.6

detector_id = "IDEA-Research/grounding-dino-tiny"

# Load image
image = load_image(image_path)

# Run detection
image_array, detections = detect(image, labels, threshold, detector_id)

plot_detections(image_array, detections, "bus_new.png")

# # Print detection results
# print("Detection results:")
# for detection in detections:
#     print(detection)