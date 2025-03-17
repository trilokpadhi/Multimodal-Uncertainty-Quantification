# main_grounding_gemini.py

import os
import logging
import argparse
import yaml
import torch
import torch.distributed as dist
from tqdm import tqdm

# If your function is in inference_utils.py, import it here
from inference_utils import generate_grounding_with_gemini

import pickle
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------------------------------------------------------
# Logging setup
logging.basicConfig(filename='inference_log.log', level=logging.INFO)

def gemini_grounding_inference_pipeline(config, rank, world_size):
    """
    A pipeline that:
      - Reads the environment and config
      - Loads all .pkl files in explanation_dir
      - Partitions them among ranks
      - Calls generate_grounding_with_gemini for each
    """
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # 1. Load the environment for Gemini
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in the environment.")
    genai.configure(api_key=gemini_api_key)

    # 2. Instantiate the Gemini model
    # If you store a model ID in your config, e.g. config['mm_model']['model_path'],
    # use it, otherwise default to 'gemini-2.0-flash'
    gemini_model_id = config.get('gemini_model_path', 'gemini-2.0-flash')
    gemini_model = genai.GenerativeModel(gemini_model_id)

    # 3. Collect all explanation files from config['logging']['explanation_dir']
    explanation_dir = config['logging']['explanation_dir']
    all_files = [f for f in os.listdir(explanation_dir) if f.endswith(".pkl")]
    all_files = sorted(all_files)

    # Partition the list of files among ranks for distributed processing
    subset_files = all_files[rank::world_size]

    # 4. Iterate over the subset for each rank
    for file in tqdm(subset_files, desc=f"Rank {rank} processing"):
        path = os.path.join(explanation_dir, file)
        # Just a small check in case of concurrency
        if not os.path.isfile(path):
            continue

        with open(path, 'rb') as f:
            sample = pickle.load(f)
        question_id = sample['question_ids'][0] # for vqa
        # question_id = sample['qids'][0] # for slake
        print(f"Rank {rank} - Processing question {question_id} with Gemini")

        # 5. Generate grounding with Gemini
        generate_grounding_with_gemini(
            file=file,
            gemini_model=gemini_model,
            rank=rank,
            config_logging=config['logging']
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--no_distributed', action='store_true', help='Run without distributed mode')
    args = parser.parse_args()

    if args.no_distributed:
        # Single-process version
        config = yaml.safe_load(open(args.config))
        gemini_grounding_inference_pipeline(config, rank=0, world_size=1)
    else:
        # 1. Initialize the process group
        dist.init_process_group(backend="nccl")

        # 2. Determine the local rank and world size
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 3. Load the YAML config
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)

        # 4. Run the distributed pipeline
        gemini_grounding_inference_pipeline(config, rank, world_size)

        # 5. Barrier + Cleanup
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()