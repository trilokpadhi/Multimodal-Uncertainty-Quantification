from .utils import ModelArgs, extract_json_from_text, eval_model
from PIL import Image
from tqdm import tqdm
import random
import torch.multiprocessing as mp
import torch

def generate_responses_worker(args, image_ids, results_queue):
    model_args = ModelArgs(args)
    results = {}
    print(f"Process {mp.current_process().pid} started.")
    for idx in tqdm(image_ids, desc=f'Generating responses on PID {mp.current_process().pid}'):
        try:
            question = args.gqa_data[idx]['question']
            prompt = f"""
            {question}
            Give your answer in JSON format where the keys are answer(one word answer), explanation( explain your answer), and your confidence(varies between 0 to 1).
            """
            
            print(f"Process {mp.current_process().pid}: {prompt}")
            image_data_root = '/home/ubuntu/Multimodal-Uncertainty-Quantification/datasets_/GQA/images/'
            image_id = args.gqa_data[idx]['imageId']
            image = Image.open(image_data_root + f'{image_id}.jpg')

            model_args.image_file = image
            model_args.query = prompt
            responses = []
            entropy = None
            for _ in range(args.no_of_responses_each_sample):
                print(f"Process {mp.current_process().pid}: Running eval_model")
                response, entropy = eval_model(model_args)
                print(f"Process {mp.current_process().pid}: Received response: {response}")
                print('-'*50)
                if response:
                    try:
                        json_string = extract_json_from_text(response)
                        json_string['entropy'] = entropy
                        responses.append(json_string)
                    except Exception as e:
                        print(f"Error parsing response in process {mp.current_process().pid}: {e}")
                else:
                    print(f"Process {mp.current_process().pid}: No response was generated.")
            
            results[idx] = {
                'responses': responses,
                'answer': args.gqa_data[idx]['answer'],
                'full_answer': args.gqa_data[idx]['fullAnswer']
            }
        
        except Exception as e:
            print(f"Error in process {mp.current_process().pid}: {e}")

    print(f"Process {mp.current_process().pid} finished.")
    results_queue.put(results)


def generate_responses(args):    
    if args.debug:
        if hasattr(args, 'keys'):
            image_ids = args.keys
            print(f"Using debug, keys: {image_ids}")
        else:
            image_ids = random.sample(list(args.gqa_data.keys()), args.samples)
            
        # Non-parallel execution
        results = {}
        results_queue = []

        # Use the same worker function directly
        generate_responses_worker(args, image_ids, results_queue)

        # Collect results from the results_queue
        for result in results_queue:
            results.update(result)

        return results
    else:
        image_ids = random.sample(list(args.gqa_data.keys()), args.samples)
        # Split the image_ids into chunks for each process
        num_processes = min(torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(), 4)  # Limit to 4 processes
        chunk_size = len(image_ids) // num_processes
        image_id_chunks = [image_ids[i:i + chunk_size] for i in range(0, len(image_ids), chunk_size)]

        # Set the start method to 'spawn' for CUDA compatibility
        # mp.set_start_method('spawn') -> moved to main.py as this is again used in calculate_uncertainty.py

        # Create a multiprocessing queue to collect results
        results_queue = mp.Queue()

        # Start the processes
        processes = []
        for i in range(num_processes):
            p = mp.Process(target=generate_responses_worker, args=(args, image_id_chunks[i], results_queue))
            p.start()
            processes.append(p)

        # Collect results from all processes
        results = {}
        for _ in processes:
            results.update(results_queue.get())

        # Ensure all processes have finished execution
        for p in processes:
            p.join()

        return results