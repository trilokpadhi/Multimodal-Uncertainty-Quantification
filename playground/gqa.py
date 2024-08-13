import glob
import json
import pickle
import random
from PIL import Image
from IPython.display import display, display_markdown

def get_file_names():
    file_names = glob.glob("/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/questions1.2/train_all_questions/*.json")
    return file_names

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_pickle(data, filename):
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(data, f)

# Make pickle data file
file_names = get_file_names()
file_names = [file_names[0]]

pickle_list = []

for file_name in file_names:
    print(file_name)
    data = load_json(file_name)

    for line in data:
        pickle_list.append([data[line]["answer"], data[line]["imageId"], data[line]["question"]])
write_pickle(pickle_list, "test")

# Function to load and display images
def display_images_and_questions(samples, image_dir='/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/'):
    for sample in samples:
        answer, image_id, question = sample
        image_path = f"{image_dir}{image_id}.jpg"  # Assuming images are in JPG format
        try:
            image = Image.open(image_path)
            display(image)
            display_markdown(f"**Question**: {question}\n\n**Answer**: {answer}", raw=True)
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")

# Randomly sample some images and their questions
num_samples = 5
random_samples = random.sample(pickle_list, num_samples)
display_images_and_questions(random_samples)
