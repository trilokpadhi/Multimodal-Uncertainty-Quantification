# Imports
import os
import logging
import yaml
import torch
from extraction_utils import extract_triples
from data_utils import get_dataloader
from inference_utils import generate_explanations_MM, generate_grounded_segmentation
from transformers import pipeline, AutoProcessor, LlavaForConditionalGeneration, AutoModelForMaskGeneration
import pickle
from tqdm import tqdm
import argparse

# Logging setup
logging.basicConfig(filename='inference_log.log', level=logging.INFO)

# Inference Pipeline (serial version)
def inference_pipeline(config):
    # Load data
    dataloader = get_dataloader(config['data'], rank=0, world_size=1)
    
    # Load models
    model_id_llava = config['mm_model']['model_path']
    model_llava = LlavaForConditionalGeneration.from_pretrained(model_id_llava).to('cuda')
    processor_llava = AutoProcessor.from_pretrained(model_id_llava)
    
    object_detector = pipeline(model=config['grounding']['detector_id'], task="zero-shot-object-detection", device='cuda')
    object_detector.model = object_detector.model.half().to('cuda')
    segmentator = AutoModelForMaskGeneration.from_pretrained(config['grounding']['segmenter_id']).half().to('cuda')
    processor = AutoProcessor.from_pretrained(config['grounding']['segmenter_id'])
    num_samples = config['data']['samples'] 

    for sample in tqdm(dataloader, total=num_samples, desc="Processing samples"):

        # Step 1: Run MM model (LLaVA)
        question_id = sample['question_ids'][0]
        
        print(f"Processing question {question_id}")
        # model_responses_with_explanations = generate_explanations_MM(model_llava, processor_llava, sample, config['mm_model'])
        model_responses_with_explanations = generate_explanations_MM(model_llava, processor_llava, sample, rank=0, params=config['mm_model'], config_logging=config['logging'])
        
        # # Ensure that the explanation directory exists
        # explanation_dir = config['logging']['explanation_dir']
        # os.makedirs(explanation_dir, exist_ok=True)
        # explanation_file_path = f"{explanation_dir}/explanations_{idx}_{question_id}.pkl"
        # with open(explanation_file_path, 'wb') as f:
        #     pickle.dump(model_responses_with_explanations, f)
        # print(f'Explanations for question {question_id} saved to {explanation_file_path}')

        # Step 3: Grounding with DINO
        generate_grounded_segmentation(model_responses_with_explanations,threshold=config['grounding']['threshold'],
                                        object_detector=object_detector, segmentator=segmentator, processor=processor, 
                                        rank=0, config_logging=config['logging'])
        
        # Ensure that the grounding directory exists
        # grounding_dir = config['logging']['grounding_dir']
        # os.makedirs(grounding_dir, exist_ok=True)
        # grounding_file_path = f"{grounding_dir}/grounding_{idx}_{question_id}.pkl"
        # with open(grounding_file_path, 'wb') as f:
        #     pickle.dump(model_responses_with_explanations_grounding, f)
        # print(f'Groundings for question {question_id} saved to {grounding_file_path}')

# Main function
def main(args):
    # Load YAML config
    config_path = args.config 
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Run the pipeline serially
    inference_pipeline(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    main(args)