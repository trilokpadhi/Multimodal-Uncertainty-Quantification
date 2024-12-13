import os
import torch
import torch.distributed as dist
from transformers import LlavaForConditionalGeneration, AutoProcessor, pipeline, AutoModelForMaskGeneration
from datasets import load_dataset
import pickle
from tqdm import tqdm
import yaml
import argparse
import os
from inference_utils import generate_explanations_MM, generate_grounded_segmentation
from extraction_utils import extract_triples
from data_utils import get_dataloader
import traceback

def setup_distributed():
    try:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist_backend = 'nccl'

        dist.init_process_group(backend=dist_backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Rank {rank}: Distributed process group initialized on device {device}.")
        return rank, world_size, local_rank, device
    except Exception as e:
        print(f"Failed to initialize distributed environment: {str(e)}")
        raise

def cleanup_distributed():
    torch.distributed.barrier()
    try:
        dist.destroy_process_group()
        print("Distributed process group destroyed.")
    except Exception as e:
        print(f"Error during distributed cleanup: {str(e)}")

def load_models(config, rank):
    model_id_llava = config['mm_model']['model_path']
    model_llava = LlavaForConditionalGeneration.from_pretrained(model_id_llava).to(f'cuda:{rank}')
    processor_llava = AutoProcessor.from_pretrained(model_id_llava)
    
    object_detector = pipeline(model=config['grounding']['detector_id'], task="zero-shot-object-detection", device=f'cuda:{rank}')
    object_detector.model = object_detector.model.half().to(device=f'cuda:{rank}')
    segmentator = AutoModelForMaskGeneration.from_pretrained(config['grounding']['segmenter_id']).half().to(device=f'cuda:{rank}')
    processor = AutoProcessor.from_pretrained(config['grounding']['segmenter_id'])
    
    return model_llava, processor_llava, object_detector, segmentator, processor

def inference(rank, world_size, config):
    try:
        # Load data
        # dataset = load_dataset(config['data']['dataset_name'])['test']
        # dataset_shard = dataset.shard(num_shards=world_size, index=rank)
        # dataset = GQADataset(config)
        print(f"Rank {rank}: Starting inference with world size {world_size}")
        torch.distributed.barrier()
        dataloader = get_dataloader(config['data'], rank, world_size)
        print(f"Rank {rank}: Dataloader length: {len(dataloader)}")

        
        # Load models
        model_llava, processor_llava, object_detector, segmentator, processor = load_models(config, rank)
        
        for idx, sample in enumerate(tqdm(dataloader, desc=f"Rank {rank} Processing")):
            print(f"Rank {rank}: Processing sample {idx}")
            # Step 1: Run MM model (LLaVA)
            model_responses_file = generate_explanations_MM(model_llava, processor_llava, sample, rank, config['mm_model'], config['logging'])
            
            # Step 2: Extract triples (if configured)
            if config['triple_extraction']:
                model_responses = extract_triples(model_responses)
            
            # Step 3: Grounding with DINO
            generate_grounded_segmentation(model_responses_file, config['grounding']['threshold'],object_detector, segmentator, processor, rank, config['logging'])
            
        
    except Exception as e:
        print(f"Rank {rank}: Error during inference - {str(e)}")
        traceback.print_exc()
        raise

def main(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    try:
        rank, world_size, local_rank, device = setup_distributed()
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return
    
    try:
        inference(rank, world_size, config)
    except Exception as e:
        print(f"Inference failed: {str(e)}")
    
    cleanup_distributed()

if __name__ == "__main__":
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    main(args.config)