
# batch inference of llava
# https://github.com/haotian-liu/LLaVA/issues/709

from transformers import (
    AutoProcessor,
    LlavaConfig,
    LlavaForConditionalGeneration,
    is_torch_available,
    is_vision_available, 
    pipeline
)
import torch

def load_model_llava(model_id, rank):
    torch.cuda.set_device(rank)  # Set the current GPU based on rank
    model = LlavaForConditionalGeneration.from_pretrained(model_id).to(f'cuda:{rank}')
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    return model, processor
    

def load_model_dino(detector_id, rank):
    object_detector = 
    return object_detector

    
    


