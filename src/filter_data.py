import json
import random
from tqdm import tqdm
from datetime import datetime

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
    """
    for GQA dataset
    """
    # questions_file = '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/questions1.2/train_all_questions/train_all_questions_0.json'
    # sample_size = 10000  # Change this value to sample a different number of questions
    # output_file = '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/questions1.2/train_all_questions/train_all_questions_0_random_filtered_10000.json'
    
    """
    for VQA dataset
    """
    # question file path with grounding : /home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions_grounded.json
    # question file path without grounding : /home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions.json
    questions_file = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions_grounded.json'
    sample_size = 2000  # Change this value to sample a different number of questions
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions_grounded_{sample_size}_{current_time}.json'

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
    
    # """
    # === Filter based on question type for VQA Dataset ===
    # """
    
    # # question types in the VQA dataset : just for reference - total count = 65
    # all_question_types = {'is that a', 'which', 'what is the name', 'are there any', 'can you', 'how', 'are they', 'what color', 'what', 'why is the', 'is this person', 'what room is', 'what number is', 'is this a', 'is there a', 'has', 'is the', 'what is the color of the', 'are', 'how many people are in', 'what is', 'what is this', 'none of the above', 'what is the man', 'do you', 'who is', 'are these', 'what sport is', 'what type of', 'do', 'what brand', 'is there', 'where are the', 'what is in the', 'is the woman', 'was', 'what does the', 'why', 'where is the', 'is he', 'how many people are', 'is this', 'what time', 'what is on the', 'what color are the', 'what color is', 'is it', 'is this an', 'what are', 'what animal is', 'what is the person', 'could', 'what is the', 'is', 'does the', 'what kind of', 'what color is the', 'what are the', 'is the person', 'what is the woman', 'how many', 'are the', 'is the man', 'are there', 'does this'}

    # # questions that can be grounded : by Trilok
    # question_types_to_ground = ['which', 'what is the name', 'how', 'what color', 'what', 'what room is', 'what number is', 'what is', 'what is this', 'what is the man', 'what does the', 'where is the', 'what time', 'what is on the', 'what are', 'what animal is', 'what is the person' , 'what is the', 'what kind of', 'what are the']
    
    # # load the annotations file
    # annotations_json_path = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/annotations/v2_mscoco_train2014_annotations.json'
    # annotations = load_json_file(annotations_json_path)
    
    # # filter the annotations based on the question types that can be grounded
    # annotations_to_ground = []
    # for annot in tqdm(annotations['annotations'], desc="Filtering annotations"):
    #     if annot['question_type'] in question_types_to_ground:
    #         # save it to a different list
    #         annotations_to_ground.append(annot)
            
    # # save the filtered annotations to a new file
    # annotations_grounded_json_path = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/annotations/v2_mscoco_train2014_annotations_grounded.json'
    
    # save_json_file({'info': annotations['info'], 'license': annotations['license'], 'data_type': annotations['data_type'], 'annotations': annotations_to_ground}, annotations_grounded_json_path)
    
    # print(f"Filtered annotations based on question types that can be grounded and saved to {annotations_grounded_json_path}")
            
    # # get the question file filtered 
    # ## each annotation from key 'annotations' from annotation file is of the below format
    # # {'question_type': 'what is in the', 'multiple_choice_answer': 'frisbee', 'answers': [{'answer': 'frisbee', 'answer_confidence': 'yes', 'answer_id': 1}, {'answer': 'frisbee', 'answer_confidence': 'yes', 'answer_id': 2}, {'answer': 'white frisbee', 'answer_confidence': 'yes', 'answer_id': 3}, {'answer': 'frisbie', 'answer_confidence': 'yes', 'answer_id': 4}, {'answer': 'frisbee', 'answer_confidence': 'yes', 'answer_id': 5}, {'answer': 'frisbee', 'answer_confidence': 'yes', 'answer_id': 6}, {'answer': 'frisbee', 'answer_confidence': 'maybe', 'answer_id': 7}, {'answer': 'frisbee', 'answer_confidence': 'yes', 'answer_id': 8}, {'answer': 'flying disc', 'answer_confidence': 'yes', 'answer_id': 9}, {'answer': 'frisbee', 'answer_confidence': 'yes', 'answer_id': 10}], 'image_id': 524291, 'answer_type': 'other', 'question_id': 524291000}
    # # It has the question_id, take these question_ids and get the questions from the question file

    # # load the questions file
    # questions_json_path = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions.json'
    # questions = load_json_file(questions_json_path)
    
    # # get the question ids from the annotations file
    # question_ids = [annot['question_id'] for annot in annotations_to_ground]
    # # filter the questions based on the question ids
    # questions_to_ground = []
    # for question in tqdm(questions['questions'], desc="Filtering questions"):
    #     if question['question_id'] in question_ids:
    #         questions_to_ground.append(question)
            
    # # save the filtered questions to a new file
    # questions_grounded_json_path = '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data/questions/v2_OpenEnded_mscoco_train2014_questions_grounded.json'
    
    # save_json_file({'info': questions['info'], 'task_type': questions['task_type'], 'data_type': questions['data_type'], 'license': questions['license'], 'data_subtype': questions['data_subtype'], 'questions': questions_to_ground}, questions_grounded_json_path)
    # print(f"Filtered questions based on question types that can be grounded and saved to {questions_grounded_json_path}")

if __name__ == '__main__':
    main()