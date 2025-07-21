# llava_inference.py
import argparse
import os
import yaml
import torch
import torch.distributed as dist
from tqdm import tqdm

from data_utils import get_vqa_dataloader, get_gqa_dataloader
from inference_utils import generate_explanations_MM
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def llava_inference_pipeline(config, rank, world_size):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # 1. Get subset of data for this rank
    # dataloader = get_vqa_dataloader(config['data'], rank=rank, world_size=world_size)
    # dataloader = get_gqa_dataloader(config['data'], rank=rank, world_size=world_size)
    # get vqa dataloader
    if config['data']['dataset'] == 'vqa':
        dataloader = get_vqa_dataloader(config['data'], rank=rank, world_size=world_size)
    elif config['data']['dataset'] == 'gqa':
        dataloader = get_gqa_dataloader(config['data'], rank=rank, world_size=world_size)
    else:
        raise ValueError("Invalid dataset specified in the config file.")
    

    # 2. Load LLaVA
    model_id = config['mm_model']['model_id']
    model_type = config['mm_model']['model_type']

    if model_type == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
    
    elif model_type == 'phi':
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        
    elif model_type == 'qwen':
        from transformers import QwenForConditionalGeneration, AutoProcessor
        model = QwenForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(model_id)

    # 3. Generate LLaVA responses for each sample
    num_samples = len(dataloader)
    for sample in tqdm(dataloader, total=num_samples, desc=f"Rank {rank} - LLaVA"):
        generate_explanations_MM(
            model,
            processor,
            sample,
            rank=rank,
            config=config
            # params=config['mm_model'],
            # config_logging=config['logging']
        )
    print(f"Rank {rank}: LLaVA inference completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    llava_inference_pipeline(config, rank, world_size)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()