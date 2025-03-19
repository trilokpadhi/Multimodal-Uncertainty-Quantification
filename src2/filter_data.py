"""
To run the script, use the following command:
python src/filter_data.py --dataset vqa --sample_size 1000 --filter_type 'random'
"""

import json
import random
from tqdm import tqdm
from datetime import datetime
import argparse
import os


def load_json_file(file_path):
    """Load JSON data from the given file path."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json_file(data, file_path):
    """Save JSON data to the given file path."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def sample_data(data, sample_size):
    """Randomly sample data from the dataset."""
    if isinstance(data, list):
        if len(data) <= sample_size:
            return data
        return random.sample(data, sample_size)
    elif isinstance(data, dict):
        if len(data) <= sample_size:
            return data
        return random.sample(list(data.items()), sample_size)
    else:
        raise ValueError("Unsupported data type. Data should be a list or a dictionary.")

def main():
    
    """
    ======== Randomly Sample Data from a JSON File ========
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Filter and sample VQA/GQA datasets.")
    parser.add_argument('--dataset', type=str, required=True, choices=['vqa', 'gqa'], help="Specify the dataset to process: 'vqa' or 'gqa'")
    parser.add_argument('--sample_size', type=int, default=2000, help="Number of samples to extract")
    parser.add_argument('--filter_type', type=str, default='random', choices=['random', 'grounded'], help="Specify the type of filtering to apply: 'random' or 'grounded'")
    args = parser.parse_args()
            
    
    if args.dataset == 'vqa':
        # Configuration for sampling VQA dataset
        # questions_file = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/VQA/v1/questions/OpenEnded_mscoco_test-dev2015_questions.json' # original questions file path for v1
        root_path = 'datasets_/VQA/v2/'
        questions_file_path = root_path + 'questions/v2_OpenEnded_mscoco_val2014_questions.json' # original questions file path for v2
        annotations_file_path = root_path + 'annotations/v2_mscoco_val2014_annotations.json' # original annotations file path for v2
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load dataset
        questions = load_json_file(questions_file_path)
        annotations = load_json_file(annotations_file_path)

        if args.filter_type == 'grounded':
            # Configuration for filtering VQA annotations
            annotations_json_path = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/annotations/v2_mscoco_train2014_annotations.json'
            annotations_grounded_json_path = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/annotations/v2_mscoco_train2014_annotations_grounded.json'
            questions_json_path = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions.json'
            questions_grounded_json_path = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions_grounded.json'
            
            question_types_to_ground = [
                'which', 'what is the name', 'how', 'what color', 'what', 'what room is', 'what number is', 'what is', 
                'what is this', 'what is the man', 'what does the', 'where is the', 'what time', 'what is on the', 
                'what are', 'what animal is', 'what is the person', 'what is the', 'what kind of', 'what are the'
            ]

            # Load annotations
            annotations = load_json_file(annotations_json_path)
            
            # Filter annotations based on question types
            annotations_to_ground = [
                annot for annot in tqdm(annotations['annotations'], desc="Filtering annotations")
                if annot['question_type'] in question_types_to_ground
            ]
            
            # Save filtered annotations
            save_json_file({
                'info': annotations['info'],
                'license': annotations['license'],
                'data_type': annotations['data_type'],
                'annotations': annotations_to_ground
            }, annotations_grounded_json_path)
            
            print(f"Filtered annotations based on question types that can be grounded and saved to {annotations_grounded_json_path}")

            # Load questions
            questions = load_json_file(questions_json_path)
            
            # Get question ids from filtered annotations
            question_ids = {annot['question_id'] for annot in annotations_to_ground}
            
            # Filter questions based on question ids
            questions_to_ground = [
                question for question in tqdm(questions['questions'], desc="Filtering questions")
                if question['question_id'] in question_ids
            ]
            
            # Save filtered questions
            save_json_file({
                'info': questions['info'],
                'task_type': questions['task_type'],
                'data_type': questions['data_type'],
                'license': questions['license'],
                'data_subtype': questions['data_subtype'],
                'questions': questions_to_ground
            }, questions_grounded_json_path)
            
            print(f"Filtered questions based on question types that can be grounded and saved to {questions_grounded_json_path}")

        elif args.filter_type == 'random':
            
            print(f"Sampling {args.sample_size} entries from the dataset...")
            
            output_questions_file_path = os.path.join(root_path, f'questions/v2_OpenEnded_mscoco_val2014_questions_{args.sample_size}_{current_time}.json')
            output_annotations_file_path = os.path.join(root_path, f'annotations/v2_mscoco_val2014_annotations_{args.sample_size}_{current_time}.json')
    
            # Sample the dataset
            sampled_data = sample_data(questions['questions'], args.sample_size)
            
            # Save the sampled dataset
            save_json_file({
                'info': questions['info'],
                'task_type': questions['task_type'],
                'data_type': questions['data_type'],
                'license': questions['license'],
                'data_subtype': questions['data_subtype'],
                'questions': sampled_data
            }, output_questions_file_path)
            
            # also filter annotations based on the sampled questions
            question_ids = {question['question_id'] for question in sampled_data}
            sampled_annotations = [
                annot for annot in tqdm(annotations['annotations'], desc="Filtering annotations")
                if annot['question_id'] in question_ids
            ]
            
            # Determine the most common answer for each annotation
            for annot in sampled_annotations:
                answer_counts = {}
                for answer in annot['answers']:
                    if answer['answer'] in answer_counts:
                        answer_counts[answer['answer']] += 1
                    else:
                        answer_counts[answer['answer']] = 1
                most_common_answer = max(answer_counts, key=answer_counts.get)
                annot['most_common_answer'] = most_common_answer
            
            # Save the sampled annotations
            save_json_file({
                'info': annotations['info'],
                'license': annotations['license'],
                'data_subtype': annotations['data_subtype'],
                'annotations': sampled_annotations,
                'data_type': annotations['data_type']
            }, output_annotations_file_path)

            print(f"Sampled Questions {args.sample_size} entries and saved to {output_questions_file_path}")
            print(f"Sampled Annotations {args.sample_size} entries and saved to {output_annotations_file_path}")
        
        else:
            raise ValueError("Unsupported filter type. Please choose either 'random' or 'grounded'.")
    
    elif args.dataset == 'gqa':
        # Add GQA dataset processing logic here
        print("GQA dataset processing is not implemented yet.")

if __name__ == '__main__':
    main()