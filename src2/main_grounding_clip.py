# main.py
import os
import logging
import argparse
import yaml
import torch
import torch.distributed as dist
from tqdm import tqdm
import pickle
from huggingface_hub import login
from transformers import CLIPModel, CLIPProcessor
from inference_utils import generate_clip_scores

# ------------------------------------------------------------------
# Logging setup
logging.basicConfig(filename='inference_log.log', level=logging.INFO)
key = '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'
login(key)

#################################################################################
# Updated grounding_inference_pipeline for CLIP
#################################################################################
def grounding_inference_pipeline(config, rank, world_size):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    
    # Load the CLIP model and processor from transformers
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device).eval()

    # 2) Get the full list of .pkl files and partition by rank
    all_files = sorted(os.listdir(config['logging']['explanation_dir']))
    subset_files = all_files[rank::world_size]

    # 3) Process each file assigned to this rank
    for file in tqdm(subset_files, desc=f"Rank {rank} processing"):
        path = os.path.join(config['logging']['explanation_dir'], file)
        with open(path, 'rb') as f:
            sample = pickle.load(f)

        # Use the first question id from the sample for logging
        # question_id = sample['qids'][0] 
        question_id = sample['question_ids'][0] # for vqa

        print(f"Rank {rank} - Processing question {question_id}")

        # 4) Compute CLIP scores for the sample explanations using the new function
        generate_clip_scores(
            sample_explanations_file_name=file,
            model=model,
            processor=processor,
            device=device,
            rank=rank,
            config_logging=config['logging']
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--no_distributed', action='store_true', help='Run without distributed mode')
    args = parser.parse_args()

    if args.no_distributed:
        grounding_inference_pipeline(yaml.safe_load(open(args.config)), rank=0, world_size=1)
    else:
        # Initialize the process group for distributed processing
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        with open(args.config, "r") as file:
            config = yaml.safe_load(file)

        grounding_inference_pipeline(config, rank, world_size)
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
