import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pickle
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, load_from_disk

# dataloader for GQA
# Below is deprecated code for GQA dataloader, use get_gqa_dataloader instead

from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader, DistributedSampler

def gqa_collate_fn(batch):
    """
    A simple collate function that batches raw data without any processing.
    """
    return {
        'question_ids': [item['id'] for item in batch],
        'images': [item['image'] for item in batch],
        'questions': [item['question'] for item in batch],
        'answers': [item['fullAnswer'] for item in batch],
        'image_ids': [item['imageId'] for item in batch]
    }

def get_gqa_dataloader(config, batch_size, rank, world_size):
    """
    Creates a DataLoader that yields batches of raw, unprocessed data.
    """
    # dataset = load_dataset(config['root_dir'])
    dataset = load_from_disk(config['data']['root_dir'])
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=gqa_collate_fn,
    )
    
    return dataloader

"""
This is depracted code for GQA dataset, we directly use the Hugging Face dataset 
"""
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
# This is deprecated code for GQA collate function, use gqa_collate_fn instead
def gqa_collate_fn_old(batch):
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
        # image_id = f'COCO_train2014_{annotation_entry["image_id"]:012d}' 
        image_id = f'COCO_val2014_{annotation_entry["image_id"]:012d}'
        # answer = annotation_entry['answers'][0]['answer']  # Assuming a single answer is used
        answer = annotation_entry['most_common_answer']
        all_answers = [ans['answer'] for ans in annotation_entry['answers']]

        promptified_question = self.promptify(question)

        return {
            'question_id': question_id,
            'question': question,
            'promptified_question': promptified_question,
            'answer': answer, # most common answer
            'all_answers': all_answers,
            'image_id': image_id,
            'image_path': os.path.join(self.image_data_root, f'{image_id}.jpg'),
        }

    def promptify(self, question):
        # Example prompt with few-shot examples
        few_shot_examples = """Question: What is the color of the object?
        Answer: The color of the object is red.
        Question: What are the people doing ?
        Answer: The people in the image are playing soccer.
        Question: What animal is in the image?
        Answer: The animal is a cat.
        """
        prompt = f"""USER: Answer the questions. Here are few examples:
        {few_shot_examples}
        <image>
        {question}
        ASSISTANT: 
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
    
    dataset = VQADataset(config)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    return DataLoader(dataset, batch_size=1, collate_fn=vqa_collate_fn, sampler=sampler)
    
    
class SLAKEDataset(Dataset):
    def __init__(self, config):
        """
        config is expected to contain:
          - 'split' (str): "train", "validation", or "test"
          - 'root_dir' (str): local path to the SLAKE images directory
              e.g. '/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/Slake1.0/imgs'
                  
        Example     
        config = {
        'split': 'test',
        'root_dir': '/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/Slake1.0',
        'image_dir': 'imgs',
        'question_file': 'filtered_slake_train',
        # potentially other fields
        }
        The dataset was filtered by using 'what' questions, using the train split.
        """
        # self.split = config.get('split', 'train')  # e.g. 'train', 'validation', 'test'
        self.root_dir = config.get('root_dir', './SLAKE/Slake1.0/')
        self.img_dir = os.path.join(self.root_dir, config.get('images_dir', 'imgs'))
        # Load the entire SLAKE dataset from Hugging Face
        # This downloads it if not cached locally
        # dataset = load_dataset("BoKelvin/SLAKE")
        dataset = load_from_disk(os.path.join(self.root_dir, config.get('question_file', 'filtered_slake_train')))
        
        # # Select the split
        # if self.split not in dataset:
        #     raise ValueError(f"Invalid split '{self.split}'. Must be one of: {list(dataset.keys())}")
        
        # self.data = dataset[self.split]  # e.g. dataset['train']
        self.data = dataset
        print(f"SLAKE loaded with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        """
        The sample typically has keys:
          'img_name', 'location', 'answer', 'modality', 'base_type',
          'answer_type', 'question', 'qid', 'content_type', 'triple',
          'img_id', 'q_lang'
        Example:
          {
            'img_name': 'xmlab1/source.jpg',
            'location': 'Abdomen',
            'answer': 'MRI',
            'modality': 'MRI',
            'base_type': 'vqa',
            'answer_type': 'OPEN',
            'question': 'What modality is used to take this image?',
            'qid': 0,
            'content_type': 'Modality',
            'triple': ['vhead', '_', '_'],
            'img_id': 1,
            'q_lang': 'en'
          }
        """

        # Create a prompt for the question if desired
        promptified_question = self.promptify(sample['question'])

        # Build the return dictionary
        item = {
            'img_name': sample['img_name'],
            'location': sample['location'],
            'answer': sample['answer'],
            'modality': sample['modality'],
            'base_type': sample['base_type'],
            'answer_type': sample['answer_type'],
            'question': sample['question'],
            'qid': sample['qid'],
            'content_type': sample['content_type'],
            'triple': sample['triple'],
            'img_id': sample['img_id'],
            'q_lang': sample['q_lang'],
            'promptified_question': promptified_question,
            # Construct the path to the local image
            'image_path': os.path.join(self.root_dir, self.img_dir, sample['img_name'])
        }
        return item

    def promptify(self, question):
        """
        Optionally create a 'few-shot' prompt or simply return the question.
        You can customize this as needed.
        """
        few_shot_examples = """Question: What modality is used to take this image?
        Answer: The modality used to take this image is MRI.
        Question: Question: Which part of the body does this image belong to?
        Answer: This image belongs to the abdomen.
        Question: What is the main organ in the image?
        Answer: The main organ in the image is Lung and Spinal Cord.
        """
        # Combine with user question
        prompt = (
            f"{few_shot_examples}\n"
            f"Question: {question}\n"
            f" <image> \n"
            f"Answer:"
        )
        
        return prompt
    
def slake_collate_fn(batch):
    img_names = []
    locations = []
    answers = []
    modalities = []
    base_types = []
    answer_types = []
    questions = []
    qids = []
    content_types = []
    triples = []
    img_ids = []
    q_langs = []
    promptified_questions = []
    image_paths = []

    for sample in batch:
        img_names.append(sample['img_name'])
        locations.append(sample['location'])
        answers.append(sample['answer'])
        modalities.append(sample['modality'])
        base_types.append(sample['base_type'])
        answer_types.append(sample['answer_type'])
        questions.append(sample['question'])
        qids.append(sample['qid'])
        content_types.append(sample['content_type'])
        triples.append(sample['triple'])
        img_ids.append(sample['img_id'])
        q_langs.append(sample['q_lang'])
        promptified_questions.append(sample['promptified_question'])
        image_paths.append(sample['image_path'])

    return {
        'img_names': img_names,
        'locations': locations,
        'answers': answers,
        'modalities': modalities,
        'base_types': base_types,
        'answer_types': answer_types,
        'questions': questions,
        'qids': qids,
        'content_types': content_types,
        'triples': triples,
        'img_ids': img_ids,
        'q_langs': q_langs,
        'promptified_questions': promptified_questions,
        'image_paths': image_paths,
    }
    
def get_slake_dataloader(config, rank, world_size):
    """
    Creates a SLAKE dataloader for the specified split,
    using a DistributedSampler if in distributed mode.
    
    config should contain keys like:
      - 'split': 'train' or 'validation' or 'test'
      - 'root_dir': path to local SLAKE images
      - anything else needed for your environment
    """
    dataset = SLAKEDataset(config)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,                # or whatever you prefer
        sampler=sampler,
        collate_fn=slake_collate_fn  # from above
    )
    return dataloader

if __name__ == "__main__":
#     # root = '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/'
#     # config = {'dataset_type': 'json', 'root_dir': '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/', 'image_dir': 'images', 'question_file': 'questions1.2/train_all_questions/train_all_questions_0_random_filtered_100.json'}
#     '''
#     for GQA dataset
#     '''
#     # config = {'dataset_type': 'json', 'root_dir': '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/', 'image_dir': 'images', 'question_file': 'questions1.2/train_all_questions/train_all_questions_0_true_filtered_100.json'}
#     # dataset_type = 'json'
#     # dataloader = get_dataloader(config, 0, 1)
#     # for idx, sample in enumerate(dataloader):
#     #     print(f"Sample {idx}: {sample}")
#     #     if idx >= 5:
#     #         break
#     '''
#     for VQA dataset
#     '''
#     config = {
#     'dataset_type': 'json',
#     'root_dir': '/home/ubuntu/trilok/Multimodal-Uncertainty-Quantification/dataset/VQA/data',
#     'image_dir': 'train2014',
#     'annotation_file': 'annotations/v2_mscoco_train2014_annotations.json',
#     'question_file': 'questions/v2_OpenEnded_mscoco_train2014_questions_filtered_1000.json',
# }

#     rank = 0  # Current GPU rank
#     world_size = 1  # Total number of GPUs

#     vqa_dataloader = get_vqa_dataloader(config, rank, world_size)
#     # Example loop to process the VQA dataloader
#     for batch in vqa_dataloader:
#         print("Batch keys:", batch.keys())
#         print("Questions:", batch['questions'])
#         print("Answers:", batch['answers'])
#         print("Image paths:", batch['image_paths'])
        
#         # check if the image paths are valid
#         for image_path in batch['image_paths']:
#             assert os.path.exists(image_path), f"Invalid image path: {image_path}"
#         break  # Remove this for processing the entire dataset
    """
    for SLAKE dataset
    """
    config = {
        'split': 'test',
        'root_dir': '/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/Slake1.0',
        'image_dir': 'imgs',
        'question_file': 'filtered_slake_train',
        # potentially other fields
    }
    rank, world_size = 0, 1  # Single GPU or CPU test

    loader = get_slake_dataloader(config, rank, world_size)
    for batch_idx, batch in enumerate(loader):
        print(f"Batch {batch_idx}:\n", batch)