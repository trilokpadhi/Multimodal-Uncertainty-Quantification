# main.py
import os
import logging
import argparse
import yaml
import torch
import torch.distributed as dist
from tqdm import tqdm

# Import your existing utilities
from src2.data_utils import get_vqa_dataloader   # or get_dataloader if needed
from inference_utils import generate_explanations_MM, generate_grounded_segmentation
from transformers import pipeline, AutoProcessor, LlavaForConditionalGeneration, AutoModelForMaskGeneration

# ------------------------------------------------------------------
# Logging setup
logging.basicConfig(filename='inference_log.log', level=logging.INFO)

def inference_pipeline_distributed(config, rank, world_size):
    """
    Runs the entire inference pipeline on a subset of data for the given rank.
    """

    # 1. Set CUDA device
    torch.cuda.set_device(rank)

    # 2. Prepare the dataloader
    #    We assume get_vqa_dataloader can accept `rank` and `world_size` to split data.
    #    If not, you can manually split your dataset outside and only load the subset here.
    dataloader = get_vqa_dataloader(config['data'], rank=rank, world_size=world_size)

    # 3. Load all models onto this GPU
    device = f"cuda:{rank}"
    model_id_llava = config['mm_model']['model_path']
    model_llava = LlavaForConditionalGeneration.from_pretrained(model_id_llava).to(device)
    processor_llava = AutoProcessor.from_pretrained(model_id_llava)

    object_detector = pipeline(
        model=config['grounding']['detector_id'],
        task="zero-shot-object-detection",
        device=device
    )
    # Convert OD model to half precision
    object_detector.model = object_detector.model.half().to(device)

    segmentator = AutoModelForMaskGeneration.from_pretrained(config['grounding']['segmenter_id']).half().to(device)
    processor_sam = AutoProcessor.from_pretrained(config['grounding']['segmenter_id'])

    # 4. Iterate over your local subset of data
    num_samples = config['data']['samples']
    for idx, sample in enumerate(tqdm(dataloader, total=num_samples, desc=f"Rank {rank} processing")):
        # Step 1: Run LLaVA
        question_id = sample['question_ids'][0]
        print(f"Rank {rank} - Processing question {question_id}")

        # Generate multiple responses with LLaVA
        model_responses_with_explanations = generate_explanations_MM(
            model_llava,
            processor_llava,
            sample,
            rank=rank,
            params=config['mm_model'],
            config_logging=config['logging']
        )

        # Step 2: Grounding with GroundingDINO + SAM
        generate_grounded_segmentation(
            model_responses_with_explanations,
            threshold=config['grounding']['threshold'],
            object_detector=object_detector,
            segmentator=segmentator,
            processor=processor_sam,
            rank=rank,
            config_logging=config['logging']
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # 1. Initialize the process group
    dist.init_process_group(backend="nccl")

    # 2. Determine the local rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 3. Load the YAML config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # 4. Run the distributed inference pipeline
    inference_pipeline_distributed(config, rank, world_size)

    # 5. Barrier + Cleanup
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()