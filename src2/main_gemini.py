import argparse
import os
import yaml
import torch
import torch.distributed as dist
from tqdm import tqdm

# Data loaders (reuse your existing code or adapt as needed)
from data_utils import get_vqa_dataloader, get_gqa_dataloader, get_slake_dataloader

# Gemini inference utility
from inference_utils import generate_explanations_MM_gemini

# Google Generative AI
import google.generativeai as genai
from dotenv import load_dotenv

def gemini_inference_pipeline(config, rank, world_size):
    """
    A pipeline that loads the dataset, the Gemini model (via google.generativeai),
    and then generates explanations for each sample using generate_explanations_MM_gemini.
    """
    # If you want multiple GPU processes to share the dataset:
    torch.cuda.set_device(rank)

    # 1. Get subset of data for this rank
    if config['data']['dataset'] == 'vqa':
        dataloader = get_vqa_dataloader(config['data'], rank=rank, world_size=world_size)
    elif config['data']['dataset'] == 'gqa':
        dataloader = get_gqa_dataloader(config['data'], rank=rank, world_size=world_size)
    elif config['data']['dataset'] == 'slake':
        dataloader = get_slake_dataloader(config['data'], rank=rank, world_size=world_size)
    else:
        raise ValueError("Invalid dataset specified in the config file.")

    # 2. Load Gemini
    # Make sure to have .env with GEMINI_API_KEY=your_key or set it via environment variable
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your environment.")

    genai.configure(api_key=api_key)

    # If your config uses 'mm_model' block, you might reuse 'model_path' or add a new field:
    # For example, you could do:
    # gemini_model_id = config['mm_model'].get('gemini_model_id', 'gemini-2.0-flash')
    # or read directly from 'model_path' if you store 'gemini-2.0-flash' there:
    gemini_model_id = config['mm_model'].get('gemini_model_path', 'gemini-2.0-flash')

    # Initialize Gemini model
    gemini_model = genai.GenerativeModel(gemini_model_id)

    # 3. Generate responses for each sample
    num_samples = len(dataloader)
    for sample in tqdm(dataloader, total=num_samples, desc=f"Rank {rank} - Gemini"):
        generate_explanations_MM_gemini(
            model=gemini_model,
            sample=sample,
            rank=rank,
            params=config['mm_model'],
            config_logging=config['logging']
        )

    print(f"Rank {rank}: Gemini inference completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    # Initialize distributed process group (optional if you truly want to run distributed)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Run the pipeline
    gemini_inference_pipeline(config, rank, world_size)

    # Synchronize and clean up
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()