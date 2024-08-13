# For running in baxkground, we need to use the following command:
# nohup python -m src.main --config-file configs/llava_gqa.yaml > logs/llava_mmt.out 2>&1 &
import argparse
import json
import pandas as pd
from .generate_responses import generate_responses
from .calculate_uncertainty import calculate_uncertainty_by_grounding
from .utils import load_args_from_config
import torch.multiprocessing as mp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", help="Path to the config file.")
    args = parser.parse_args()
    
    # Load arguments from config file and update the args object
    config_args = load_args_from_config(args.config_file)
    args.__dict__.update(config_args.__dict__)

    # dataset_type: json or dataframe
    if args.dataset_type.lower() == 'dataframe':
        df = pd.read_csv(args.data_path, sep = '\t')
        args.filtered_df = df[df['category'].str.contains(args.category, case=False)]
    elif args.dataset_type.lower() == 'json': 
        with open(args.data_path, 'r') as file:
            data = json.load(file)
            args.gqa_data = data
            scene_graphs_file_path = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/sceneGraphs/train_sceneGraphs.json'
            with open(scene_graphs_file_path, 'r') as f:
                args.scene_graphs_data = json.load(f)
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")
        
    mp.set_start_method('spawn')
    if args.get_response:        
        
        args.responses = generate_responses(args)
        with open(args.responses_path, 'w') as file:
            json.dump(args.responses, file)
        print("Responses saved successfully.")
    
    else:
        print("No response generated since args.get_response is set to False.") 

    if args.get_uncertainty:
        if not hasattr(args, 'responses'):
            print("Loading responses from file.")
            with open(args.responses_path, 'r') as file:
                args.responses = json.load(file)
                uncertainty = calculate_uncertainty_by_grounding(args)
                with open(args.uncertainty_path, 'w') as file:
                    json.dump(uncertainty, file)
    else:
        print("No uncertainty calculated since args.get_uncertainty is set to False.")

if __name__ == "__main__":
    main()