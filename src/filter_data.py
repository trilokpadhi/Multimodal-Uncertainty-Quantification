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
    if len(data) <= sample_size:
        return data
    return random.sample(list(data.items()), sample_size)

def main():
    questions_file = '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/questions1.2/train_all_questions/train_all_questions_0.json'
    sample_size = 100  # Change this value to sample a different number of questions
    output_file = 'short_dataset.json'

    # Load dataset
    dataset = load_json_file(questions_file)

    # Sample the dataset
    sampled_data = sample_data(dataset, sample_size)

    # Convert the sampled data back to dictionary (from list of tuples)
    sampled_dict = dict(sampled_data)

    # Save the sampled dataset
    save_json_file(sampled_dict, output_file)

    print(f"Sampled {sample_size} entries and saved to {output_file}")

if __name__ == '__main__':
    main()