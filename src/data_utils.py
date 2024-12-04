import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pickle
from torch.utils.data.distributed import DistributedSampler

# dataloader for GQA
def get_dataloader(config, rank, world_size):
    """
    Load data based on dataset type (json/pandas)
    """
    dataset_type = config['dataset_type']
    print(f"Dataset type: {dataset_type}")
    if dataset_type.lower() == 'dataframe':
        raise NotImplementedError("Dataloader for dataframe not implemented yet.")
    elif dataset_type.lower() == 'json': 
        dataset = GQADataset(config)
        # Create a distributed sampler to split the dataset across GPUs
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

        return DataLoader(dataset, batch_size=1, collate_fn=collate_fn, sampler=sampler)

    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    
class GQADataset:
    def __init__(self, config):
        self.root = config['root_dir']
        self.image_data_root = os.path.join(self.root, config['image_dir'])
        self.questions_data_root = os.path.join(self.root, config['question_file'])
        self.question_data = json.load(open(self.questions_data_root, 'r'))
        # select only 10 questions for testing
        # self.question_data = {k: self.question_data[k] for k in list(self.question_data.keys())[:20]}
        print(f"Number of questions: {len(self.question_data)}")
        self.question_ids = list(self.question_data.keys())

    def __len__(self):
        return len(self.question_data.items())

    def __getitem__(self, idx):
        question_id = self.question_ids[idx]
        self.question = self.question_data[question_id]['question']
        self.promptified_question = self.promptify(self.question)
        self.answer = self.question_data[question_id]['answer']
        self.full_answer = self.question_data[question_id]['fullAnswer']
        self.image_id = self.question_data[question_id]['imageId']
        # self.image = Image.open(self.image_data_root + '/' + f'{self.image_id}.jpg')
        return {
            'question_id': question_id,
            'question': self.question,
            'promptified_question': self.promptified_question,
            'answer': self.answer,
            'full_answer': self.full_answer,
            'image_id': self.image_id,
            'image_path': self.image_data_root + '/' + f'{self.image_id}.jpg',
            # 'image': self.image
        }
    
    def promptify(self, question):
        # return f"""USER:<image>
        # {question}
        # Give your answer in JSON format where the keys are answer(one word answer), explanation( explain your answer in one short sentence), and your confidence(varies between 0 to 1).
        # """
        few_shot_examples = """ Question: What is the color of the object?
        Answer: The color of the object is red.
        Question: Is the color of the object red?
        Answer: Yes, the color of the object is red.
        Question: Do you see clocks in the image?
        Answer: No, I do not see any clocks in the image.
        """
        prompt = f"""USER: Answer the questions, Here are few examples: 
        {few_shot_examples}
        <image>
        {question}
        """
        return prompt
    


    
        
# write a collate function to collate the samples
def collate_fn(batch):
    question_ids = []
    questions = []
    promptified_questions = []
    answers = []
    full_answers = []
    image_ids = []
    image_paths = []
    # images = []
    for sample in batch:
        question_ids.append(sample['question_id'])
        questions.append(sample['question'])
        promptified_questions.append(sample['promptified_question'])
        answers.append(sample['answer'])
        full_answers.append(sample['full_answer'])
        image_ids.append(sample['image_id'])
        image_paths.append(sample['image_path'])
        # images.append(sample['image'])
    
    return {
        'question_ids': question_ids,
        'questions': questions,
        'promptified_questions': promptified_questions,
        'answers': answers,
        'full_answers': full_answers,
        'image_ids': image_ids,
        'image_paths': image_paths,
        # 'images': images
    }
    

# Define the VQA Dataset
class VQADataset(Dataset):
    def __init__(self, config):
        self.root = config['root_dir']
        self.image_data_root = os.path.join(self.root, config['image_dir'])
        self.annotation_file = os.path.join(self.root, config['annotation_file'])
        self.question_file = os.path.join(self.root, config['question_file'])

        # Load VQA annotations and questions
        self.annotations = json.load(open(self.annotation_file, 'r'))['annotations']
        self.questions = json.load(open(self.question_file, 'r'))['questions'] # so now a smaller sampled version of this dataset, is in the form of a json file which is a list of dictionaries, each dictionary has a question_id, question, image_id, and answer, so how would this change ? Answer: The question_id is the key in the dictionary, so we can directly access the question, image_id, and answer using the question_id as the key.
        print(f"Number of annotations: {len(self.annotations)}")
        print(f"Number of questions: {len(self.questions)}")

        # # Create question and answer mappings
        # self.qa_map = {ann['question_id']: ann for ann in self.annotations}
        sampled_question_ids = {ques['question_id'] for ques in self.questions}
        self.qa_map = {ann['question_id']: ann for ann in self.annotations if ann['question_id'] in sampled_question_ids}
        self.question_map = {ques['question_id']: ques for ques in self.questions}

        self.question_ids = list(self.qa_map.keys())

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, idx):
        question_id = self.question_ids[idx]
        question_entry = self.question_map[question_id]
        annotation_entry = self.qa_map[question_id]

        question = question_entry['question']
        # the filename is of the format COCO_train2014_000000000025.jpg
        # image_id = annotation_entry['image_id'] 
        image_id = f'COCO_train2014_{annotation_entry["image_id"]:012d}'
        answer = annotation_entry['answers'][0]['answer']  # Assuming a single answer is used
        all_answers = [ans['answer'] for ans in annotation_entry['answers']]

        promptified_question = self.promptify(question)

        return {
            'question_id': question_id,
            'question': question,
            'promptified_question': promptified_question,
            'answer': answer,
            'all_answers': all_answers,
            'image_id': image_id,
            'image_path': os.path.join(self.image_data_root, f'{image_id}.jpg'),
        }

    def promptify(self, question):
        # Example prompt with few-shot examples
        few_shot_examples = """Question: What is the color of the object?
        Answer: The color of the object is red.
        Question: Is the object big or small?
        Answer: The object is big.
        Question: Do you see people in the image?
        Answer: Yes, there are people in the image.
        """
        prompt = f"""USER: Answer the questions based on the examples provided:
        {few_shot_examples}
        <image>
        {question}
        """
        return prompt

# Define the collate function
def vqa_collate_fn(batch):
    question_ids = []
    questions = []
    promptified_questions = []
    answers = []
    all_answers = []
    image_ids = []
    image_paths = []

    for sample in batch:
        question_ids.append(sample['question_id'])
        questions.append(sample['question'])
        promptified_questions.append(sample['promptified_question'])
        answers.append(sample['answer'])
        all_answers.append(sample['all_answers'])
        image_ids.append(sample['image_id'])
        image_paths.append(sample['image_path'])

    return {
        'question_ids': question_ids,
        'questions': questions,
        'promptified_questions': promptified_questions,
        'answers': answers,
        'all_answers': all_answers,
        'image_ids': image_ids,
        'image_paths': image_paths,
    }

# DataLoader function for VQA
def get_vqa_dataloader(config, rank, world_size):
    """
    Load the VQA dataset and create a DataLoader.
    """
    dataset_type = config['dataset_type']
    print(f"Dataset type: {dataset_type}")
    
    if dataset_type.lower() == 'json':
        dataset = VQADataset(config)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        return DataLoader(dataset, batch_size=1, collate_fn=vqa_collate_fn, sampler=sampler)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    
    
if __name__ == "__main__":
    # root = '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/'
    # config = {'dataset_type': 'json', 'root_dir': '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/', 'image_dir': 'images', 'question_file': 'questions1.2/train_all_questions/train_all_questions_0_random_filtered_100.json'}
    '''
    for GQA dataset
    '''
    # config = {'dataset_type': 'json', 'root_dir': '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/', 'image_dir': 'images', 'question_file': 'questions1.2/train_all_questions/train_all_questions_0_true_filtered_100.json'}
    # dataset_type = 'json'
    # dataloader = get_dataloader(config, 0, 1)
    # for idx, sample in enumerate(dataloader):
    #     print(f"Sample {idx}: {sample}")
    #     if idx >= 5:
    #         break
    '''
    for VQA dataset
    '''
    config = {
    'dataset_type': 'json',
    'root_dir': '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data',
    'image_dir': 'train2014',
    'annotation_file': 'annotations/v2_mscoco_train2014_annotations.json',
    'question_file': 'questions/v2_OpenEnded_mscoco_train2014_questions_filtered_1000.json',
}

    rank = 0  # Current GPU rank
    world_size = 1  # Total number of GPUs

    vqa_dataloader = get_vqa_dataloader(config, rank, world_size)
    # Example loop to process the VQA dataloader
    for batch in vqa_dataloader:
        print("Batch keys:", batch.keys())
        print("Questions:", batch['questions'])
        print("Answers:", batch['answers'])
        print("Image paths:", batch['image_paths'])
        
        # check if the image paths are valid
        for image_path in batch['image_paths']:
            assert os.path.exists(image_path), f"Invalid image path: {image_path}"
        break  # Remove this for processing the entire dataset
        