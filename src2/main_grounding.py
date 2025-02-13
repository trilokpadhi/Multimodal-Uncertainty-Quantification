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
from inference_utils import generate_grounded_segmentation
from transformers import pipeline, AutoProcessor, LlavaForConditionalGeneration, AutoModelForMaskGeneration
import pickle

# ------------------------------------------------------------------
# Logging setup
logging.basicConfig(filename='inference_log.log', level=logging.INFO)


#################################################################################
# Updated version of the grounding_inference_pipeline function with files split over ranks, when running in distributed mode
#################################################################################
def grounding_inference_pipeline(config, rank, world_size):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # 1) Load the object detection & segmentation models
    object_detector = pipeline(
        model=config['grounding']['detector_id'],
        task="zero-shot-object-detection",
        device=device)
    object_detector.model = object_detector.model.half().to(device)
    
    segmentator = AutoModelForMaskGeneration.from_pretrained(config['grounding']['segmenter_id']).half().to(device)
    processor_sam = AutoProcessor.from_pretrained(config['grounding']['segmenter_id'])

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

        question_id = sample['question_ids'][0]
        print(f"Rank {rank} - Processing question {question_id}")

        # 4) Grounding with GroundingDINO + SAM
        generate_grounded_segmentation(
            file,
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
    
# ###############################################################
# # Updated version of the grounding_inference_pipeline function for debugging
# Note: this was used to debug the code, to know that the groundign model was corrupted and was fixed by creating a new environment, llava2, by installing everything from the scratch
# ###############################################################
# def grounding_inference_pipeline(config, rank, world_size):
#     # Set device
#     torch.cuda.set_device(rank)
#     device = f"cuda:{rank}"

#     print(f"Using device: {device}")

#     # Load object detection model
#     print("Loading object detector...")
#     object_detector = pipeline(
#         model=config['grounding']['detector_id'],
#         task="zero-shot-object-detection",
#         device=device)
#     object_detector.model = object_detector.model.half().to(device)
#     print("Object Detector Model Loaded Successfully.")

#     # Load segmentation model
#     print("Loading segmentator...")
#     segmentator = AutoModelForMaskGeneration.from_pretrained(config['grounding']['segmenter_id']).half().to(device)
#     processor_sam = AutoProcessor.from_pretrained(config['grounding']['segmenter_id'])
#     print("Segmentation Model Loaded Successfully.")

#     # Fetch files
#     print("Fetching explanation files...")
#     all_files = os.listdir(config['logging']['explanation_dir'])
#     all_files = sorted(all_files)
#     print(f"Total files: {len(all_files)}")

#     # Partition files by rank
#     subset_files = all_files[rank::world_size]
#     print(f"Subset for rank {rank}: {subset_files}")

#     # Process each file
#     for file in tqdm(subset_files, desc=f"Rank {rank} processing"):
#         path = os.path.join(config['logging']['explanation_dir'], file)
#         print(f"Processing file: {path}")
#         with open(path, 'rb') as f:
#             sample = pickle.load(f)

#         # Extract question ID
#         question_id = sample['question_ids'][0]
#         print(f"Rank {rank} - Processing question {question_id}")

#         # Perform grounding
#         generate_grounded_segmentation(
#             file,
#             threshold=config['grounding']['threshold'],
#             object_detector=object_detector,
#             segmentator=segmentator,
#             processor=processor_sam,
#             rank=rank,
#             config_logging=config['logging']
#         )
#     print(f"Rank {rank} processing completed.")
    
    
######################################################################
# Below is the original version of the grounding_inference_pipeline function, before the update to split files over ranks, which dose not work in distributed mode
# def grounding_inference_pipeline(config, rank, world_size):
#     """
#     Runs the entire inference pipeline on a subset of data for the given rank.
#     """

#     # 1. Set CUDA device
#     torch.cuda.set_device(rank)

#     # 2. Prepare the dataloader
#     #    We assume get_vqa_dataloader can accept `rank` and `world_size` to split data.
#     #    If not, you can manually split your dataset outside and only load the subset here.
#     # dataloader = get_vqa_dataloader(config['data'], rank=rank, world_size=world_size)

#     # 3. Load all models onto this GPU
#     device = f"cuda:{rank}"
    
#     # model_id_llava = config['mm_model']['model_path']
#     # model_llava = LlavaForConditionalGeneration.from_pretrained(model_id_llava).to(device)
#     # processor_llava = AutoProcessor.from_pretrained(model_id_llava)

#     object_detector = pipeline(
#         model=config['grounding']['detector_id'],
#         task="zero-shot-object-detection",
#         device=device
#     )
    
#     # Convert OD model to half precision
#     object_detector.model = object_detector.model.half().to(device)

#     segmentator = AutoModelForMaskGeneration.from_pretrained(config['grounding']['segmenter_id']).half().to(device)
#     processor_sam = AutoProcessor.from_pretrained(config['grounding']['segmenter_id'])

#     # 4. Iterate over your local subset of data
#     files = os.listdir(config['logging']['explanation_dir'])
    
#     for file in tqdm(files, desc=f"Rank {rank} processing"):
#         # read the sample from the explanation directory, its a pickle file 
#         path = os.path.join(config['logging']['explanation_dir'], file)
#         with open(path, 'rb') as f:
#             sample = pickle.load(f)
        
#         # Step 1: Run LLaVA
#         question_id = sample['question_ids'][0]
#         print(f"Rank {rank} - Processing question {question_id}")

#         # Generate multiple responses with LLaVA
#         # model_responses_with_explanations = generate_explanations_MM(
#         #     model_llava,
#         #     processor_llava,
#         #     sample,
#         #     rank=rank,
#         #     params=config['mm_model'],
#         #     config_logging=config['logging']
#         # )

#         # Step 2: Grounding with GroundingDINO + SAM
#         generate_grounded_segmentation(
#             file,
#             threshold=config['grounding']['threshold'],
#             object_detector=object_detector,
#             segmentator=segmentator,
#             processor=processor_sam,
#             rank=rank,
#             config_logging=config['logging']
#         )
