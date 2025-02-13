# llava_inference.py
import argparse
import os
import yaml
import torch
import torch.distributed as dist
from tqdm import tqdm

# add the path to the sys.path
import sys
sys.path.append('/home/ubuntu/trilok/LLaVA-Med/llava')

from data_utils import get_slake_dataloader
from inference_utils import generate_explanations_MM_med
from transformers import LlavaForConditionalGeneration, AutoProcessor
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image

import torch

######################################################################
# The main inference pipeline, analogous to `llava_inference_pipeline`
######################################################################
def llava_med_inference_pipeline(config, rank, world_size):
    # 1) Set device
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # 2) Build dataloader for SLAKE
    dataloader = get_slake_dataloader(config['data'], rank=rank, world_size=world_size)

    # 3) Load LLaVA-med model
    # disable_torch_init()
    model_path = config['mm_model']['model_path']
    model_base = config['mm_model']['model_base']
    model_name = config['mm_model']['model_name']
    # The second arg is base model (could be None or a path).
    # The third arg is the short name of the model
    # We specify device here or do model.to(device) afterward
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base, model_name, device=device)

    # 4) Generate responses for each sample
    num_samples = len(dataloader)
    for sample in tqdm(dataloader, total=num_samples, desc=f"Rank {rank} - LLaVA-Med"):
        generate_explanations_MM_med(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            sample=sample,
            rank=rank,
            params=config['mm_model'],
            config_logging=config['logging']
        )

    print(f"Rank {rank}: LLaVA-Med inference completed.")
    
# def llava_inference_pipeline(config, rank, world_size):
#     torch.cuda.set_device(rank)
#     device = f"cuda:{rank}"

#     # 1. Get subset of data for this rank
#     # dataloader = get_vqa_dataloader(config['data'], rank=rank, world_size=world_size)
#     # dataloader = get_gqa_dataloader(config['data'], rank=rank, world_size=world_size)
#     # get vqa dataloader
#     if config['data']['dataset'] == 'slake':
#         dataloader = get_slake_dataloader(config['data'], rank=rank, world_size=world_size)

#     # 2. Load LLaVA
#     model_id_llava = config['mm_model']['model_path']
#     model_llava = LlavaForConditionalGeneration.from_pretrained(model_id_llava).to(device)
#     processor_llava = AutoProcessor.from_pretrained(model_id_llava)

#     # 3. Generate LLaVA responses for each sample
#     num_samples = len(dataloader)
#     for sample in tqdm(dataloader, total=num_samples, desc=f"Rank {rank} - LLaVA"):
#         generate_explanations_MM(
#             model_llava,
#             processor_llava,
#             sample,
#             rank=rank,
#             params=config['mm_model'],
#             config_logging=config['logging']
#         )
#     print(f"Rank {rank}: LLaVA inference completed.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # llava_inference_pipeline(config, rank, world_size)
    llava_med_inference_pipeline(config, rank, world_size)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()