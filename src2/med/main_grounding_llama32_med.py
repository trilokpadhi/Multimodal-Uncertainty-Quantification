# main.py
import os
import logging
import argparse
import yaml
import torch
import torch.distributed as dist
from tqdm import tqdm

# Import your existing utilities
# from src2.data_utils import get_vqa_dataloader   # or get_dataloader if needed
from inference_utils import generate_grounded_segmentation, generate_grounding_with_llama3_2
from transformers import pipeline, AutoProcessor, LlavaForConditionalGeneration, AutoModelForMaskGeneration, MllamaForConditionalGeneration
import pickle
from huggingface_hub import login

# ------------------------------------------------------------------
# Logging setup
logging.basicConfig(filename='inference_log.log', level=logging.INFO)
key = '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'
login(key)

#################################################################################
# Updated version of the grounding_inference_pipeline function with files split over ranks, when running in distributed mode
#################################################################################
def grounding_inference_pipeline(config, rank, world_size):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    model_id = config['grounding']['llama32_model_path']
    model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # 2) Get the full list of .pkl files, then create a subset for each rank
    all_files = os.listdir(config['logging']['explanation_dir'])
    all_files = sorted(all_files)  # Not strictly necessary, but can be helpful for consistency

    # Partition the files by rank using the stride approach
    subset_files = all_files[rank::world_size]
    # subset_files = all_files

    # 3) Iterate over the subset of files for this rank
    for file in tqdm(subset_files, desc=f"Rank {rank} processing"):
        path = os.path.join(config['logging']['explanation_dir'], file)
        with open(path, 'rb') as f:
            sample = pickle.load(f)

        # question_id = sample['question_ids'][0]
        question_id = sample['qids'][0]
        print(f"Rank {rank} - Processing question {question_id}")

        # 4) Grounding with GroundingDINO + SAM
        # generate_grounded_segmentation(
        #     file,
        #     threshold=config['grounding']['threshold'],
        #     object_detector=object_detector,
        #     segmentator=segmentator,
        #     processor=processor_sam,
        #     rank=rank,
        #     config_logging=config['logging']
        # )
        # 4) Process grounding with Llama 3.2 Vision
        
        generate_grounding_with_llama3_2(
            file=file,
            model=model,
            processor=processor,
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
        return
    else:
        # 1. Initialize the process group
        dist.init_process_group(backend="nccl")

        # 2. Determine the local rank and world size
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 3. Load the YAML config
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)

        # 4. Run the distributed inference pipeline
        grounding_inference_pipeline(config, rank, world_size)

        # 5. Barrier + Cleanup
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
    