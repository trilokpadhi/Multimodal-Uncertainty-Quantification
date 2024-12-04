import json
import random

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
    for GQA dataset
    """
    # questions_file = '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/questions1.2/train_all_questions/train_all_questions_0.json'
    # sample_size = 10000  # Change this value to sample a different number of questions
    # output_file = '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/questions1.2/train_all_questions/train_all_questions_0_random_filtered_10000.json'
    
    """
    for VQA dataset
    """
    questions_file = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions.json'
    sample_size = 1000  # Change this value to sample a different number of questions
    output_file = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions_filtered_1000.json'

    # Load dataset
    # dataset = load_json_file(questions_file) # for GQA dataset
    full_dataset = load_json_file(questions_file) # for VQA dataset
    dataset = full_dataset['questions']

    # Sample the dataset
    sampled_data = sample_data(dataset, sample_size)

    # Convert the sampled data back to dictionary (from list of tuples)
    # sampled_dict = dict(sampled_data)

    # Save the sampled dataset
    # save_json_file(sampled_dict, output_file) # for GQA dataset
    # save the list of dictionaries to a json file
    # for VQA dataset I would prefer the same format as the original file which is data.keys()
    save_json_file({'info': full_dataset['info'], 'task_type': full_dataset['task_type'], 'data_type': full_dataset['data_type'], 'license': full_dataset['license'], 'data_subtype': full_dataset['data_subtype'], 'questions': sampled_data}, output_file)

    print(f"Sampled {sample_size} entries and saved to {output_file}")

if __name__ == '__main__':
    main()