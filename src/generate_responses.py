from utils import ModelArgs, extract_json_from_text, eval_model
from PIL import Image
from tqdm import tqdm
import random
import torch.multiprocessing as mp
import torch
import queue
import pickle
import os

def generate_responses_worker(args, image_ids, output_dir):
    model_args = ModelArgs(args)
    print(f"Process {mp.current_process().pid} started.")
    for idx in tqdm(image_ids, desc=f'Generating responses on PID {mp.current_process().pid}'):
        try:
            question = args.gqa_data[idx]['question']
            prompt = f"""
            {question}
            Give your answer in JSON format where the keys are answer(one word answer), explanation( explain your answer), and your confidence(varies between 0 to 1).
            """
            
            image_data_root = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/'
            image_id = args.gqa_data[idx]['imageId']
            image = Image.open(image_data_root + f'{image_id}.jpg')

            model_args.image_file = image
            model_args.query = prompt
            responses = []
            all_scores = []
            all_tokens = []
            all_probabilities = []
            for _ in range(args.no_of_responses_each_sample):
                response, (sequence_probability, scores, tokens) = eval_model(model_args)
                if response:
                    try:
                        json_string = extract_json_from_text(response)
                        responses.append(json_string)
                        all_probabilities.append(sequence_probability)
                        all_scores.append(scores.cpu())  # Move to CPU to save memory if necessary
                        all_tokens.append(tokens.cpu())
                    except Exception as e:
                        print(f"Error parsing response in process {mp.current_process().pid}: {e}")
                else:
                    print(f"Process {mp.current_process().pid}: No response was generated.")
            
            result = {
                'responses': responses,
                'answer': args.gqa_data[idx]['answer'],
                'full_answer': args.gqa_data[idx]['fullAnswer'],
                'sequence_probabilities': all_probabilities,
                'scores': all_scores,
                'tokens': all_tokens,
            }

            # Save individual result to a pickle file named after the idx
            output_file = os.path.join(output_dir, f'result_{idx}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"Result for idx {idx} saved to {output_file}.")

        except Exception as e:
            print(f"Error in process {mp.current_process().pid} for idx {idx}: {e}")

    print(f"Process {mp.current_process().pid} finished.")


def generate_responses(args):
    if args.debug:
        image_ids = random.sample(list(args.gqa_data.keys()), args.samples)
        output_dir = os.path.join(args.responses_path, 'debug_results')
        os.makedirs(output_dir, exist_ok=True)
        generate_responses_worker(args, image_ids, output_dir)
        return {}
    else:
        image_ids = random.sample(list(args.gqa_data.keys()), args.samples)
        num_processes = min(torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(), 6)
        chunk_size = len(image_ids) // num_processes
        image_id_chunks = [image_ids[i:i + chunk_size] for i in range(0, len(image_ids), chunk_size)]
        output_dir = os.path.join(args.responses_path, 'results')
        os.makedirs(output_dir, exist_ok=True)
        processes = []

        for i in range(num_processes):
            p = mp.Process(target=generate_responses_worker, args=(args, image_id_chunks[i], output_dir))
            p.start()
            processes.append(p)

        # Ensure all processes have finished execution
        for p in processes:
            p.join()

        print("All processes have finished. Results saved in individual pickle files.")