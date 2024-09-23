import json
import pandas as pd
from torch.utils.data import DataLoader
import os
from PIL import Image
import pickle
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(config, rank, world_size):
    """
    Load data based on dataset type (json/pandas)
    """
    dataset_type = config['dataset_type']
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
        return f"""USER:<image>
        {question}
        Give your answer in JSON format where the keys are answer(one word answer), explanation( explain your answer in one short sentence), and your confidence(varies between 0 to 1).
        """
    
    def write_to_pkl(self, data, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        
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
    


if __name__ == "__main__":
    # root = '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/'
    # config = {'dataset_type': 'json', 'root_dir': '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/', 'image_dir': 'images', 'question_file': 'questions1.2/train_all_questions/train_all_questions_0_random_filtered_100.json'}
    config = {'dataset_type': 'json', 'root_dir': '/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/GQA/', 'image_dir': 'images', 'question_file': 'questions1.2/train_all_questions/train_all_questions_0_true_filtered_100.json'}
    dataset_type = 'json'
    dataloader = get_dataloader(config, 0, 1)
    for idx, sample in enumerate(dataloader):
        print(f"Sample {idx}: {sample}")
        if idx >= 5:
            break