# Imports
import os
import logging
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Custom imports (these will depend on your specific code)
from extraction_utils import extract_triples
from data_utils import get_dataloader
from inference_utils import generate_explanations_MM, generate_grounded_segmentation

from transformers import pipeline, AutoProcessor, LlavaForConditionalGeneration, AutoModelForMaskGeneration
import pickle
# Logging setup
logging.basicConfig(filename='inference_log.log', level=logging.INFO)

# Distributed setup
def setup(rank, world_size):
    # Set MASTER_ADDR and MASTER_PORT environment variables
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Inference Pipeline
def inference_pipeline(rank, world_size, config):
    setup(rank, world_size)

    # Load data based on dataset type (json/pandas)
    dataloader = get_dataloader(config['data']['data_path'], config['data']['dataset_type'])
    
    # Load models
    # model_llava, processor_llava = load_model_llava(config['mm_model']['model_path'], rank)
    model_id_llava = config['mm_model']['model_path']
    model_llava = LlavaForConditionalGeneration.from_pretrained(model_id_llava).to(f'cuda:{rank}')
    processor_llava = AutoProcessor.from_pretrained(model_id_llava)
    object_detector = pipeline(model=config['grounding']['detector_id'], task="zero-shot-object-detection", device=f'cuda:{rank}')
    segmentator = AutoModelForMaskGeneration.from_pretrained(config['grounding']['segmenter_id']).to(device=f'cuda:{rank}')
    processor = AutoProcessor.from_pretrained(config['grounding']['segmenter_id'])
    num_samples = config['samples']

    for idx, sample in enumerate(dataloader):
        if idx >= num_samples:
            break

        # Step 1: Run MM model (LLaVA)
        question_id = sample['question_ids'][0]
        model_responses_with_explanations = generate_explanations_MM(model_llava, processor_llava,
                                                       sample, rank, config['mm_model'])
        # Ensure that the explanation directory exists
        explanation_dir = config['logging']['explanation_dir']
        os.makedirs(explanation_dir, exist_ok=True)
        explanation_file_path = f"{explanation_dir}/explanations_{rank}_{idx}_{question_id}.pkl"
        with open(explanation_file_path, 'wb') as f:
            pickle.dump(model_responses_with_explanations, f)
        print(f'Explanations for question {question_id} saved to {explanation_file_path}')
        
        # Step 2: Extract triples
        triple_extraction = config['triple_extraction']
        if triple_extraction:
            model_responses_with_explanations_triples = extract_triples(model_responses_with_explanations)
        else:
            print('Skipping triple extraction, since config.triple_extraction is set to', config['triple_extraction'])

        # Step 3: Grounding with DINO
        model_responses_with_explanations_grounding = generate_grounded_segmentation(
            model_responses_with_explanations,
            threshold=config['grounding']['threshold'],
            object_detector=object_detector, segmentator=segmentator, processor=processor, rank=rank)
        # Ensure that the explanation directory exists
        grounding_dir = config['logging']['grounding_dir']
        os.makedirs(grounding_dir, exist_ok=True)
        grounding_file_path = f"{grounding_dir}/grounding_{rank}_{idx}_{question_id}.pkl"
        with open(grounding_file_path, 'wb') as f:
            pickle.dump(model_responses_with_explanations_grounding, f)
        print(f'groundings for question {question_id} saved to {grounding_file_path}')

    cleanup()

# Main function
def main():
    # Load YAML config
    with open("/home/ubuntu/Multimodal-Uncertainty-Quantification/configs/llava_gqa2.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set world size to 1 for single GPU debugging
    world_size = torch.cuda.device_count()
    # world_size = 1

    # For single GPU, directly call inference_pipeline without spawning processes
    if world_size == 1:
        inference_pipeline(rank=0, world_size=world_size, config=config)
    else:
        # For multi-GPU (distributed), use mp.spawn
        mp.spawn(inference_pipeline, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()