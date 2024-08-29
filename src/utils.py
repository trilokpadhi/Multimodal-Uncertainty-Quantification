
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import torch.multiprocessing as mp
import torch
import regex as re
import yaml
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# from .calculate_uncertainty import calculate_entropy_from_log_probs
from sentence_transformers import SentenceTransformer

def calculate_entropy_from_log_probs(log_probs):
    # Avoid -inf by replacing with a very small number
    log_probs = torch.clamp(log_probs, min=-1e6)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs)
    return entropy

class modeModelArgs:
    def __init__(self, args):
        self.model_path = args.model_path
        self.model_base = args.model_base
        self.model_name = get_model_name_from_path(args.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            args.model_path, args.model_base, self.model_name
        )
        self.conv_mode = args.conv_mode
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_new_tokens = args.max_new_tokens
        self.query = None
        self.image_file = None
        
        # for llama3
        self.llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                                                torch_dtype=torch.bfloat16,
                                                                device_map="auto")
        
        
        # for sentence transformer
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
class DataArgs:
    def __init__(self, args):
        self.dataset_type = args.dataset_type
        self.data_path = args.data_path
        self.category = args.category
        self.responses = None
        self.responses_path = args.responses_path
        self.get_response = args.get_response
        self.gqa_data = args.gqa_data
        self.filtered_df = None
        self.no_of_responses_each_sample = args.no_of_responses_each_sample
        
        
def eval_model(model_args):
    print(f"Process {mp.current_process().pid}: Starting eval_model")
    # Model
    disable_torch_init()

    print(f"Process {mp.current_process().pid}: Preparing query")
    qs = model_args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model_args.model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_args.model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    print(f"Process {mp.current_process().pid}: Determining conversation mode")
    if "llama-2" in model_args.model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_args.model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_args.model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_args.model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_args.model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if model_args.conv_mode is not None and conv_mode != model_args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, model_args.conv_mode, model_args.conv_mode
            )
        )
    else:
        model_args.conv_mode = conv_mode

    print(f"Process {mp.current_process().pid}: Preparing conversation and prompt")
    conv = conv_templates[model_args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print(f"Process {mp.current_process().pid}: Processing images")
    images = [model_args.image_file]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        model_args.image_processor,
        model_args.model.config
    ).to(model_args.model.device, dtype=torch.float16)

    print('prompt', prompt)
    input_ids = (
        tokenizer_image_token(prompt, model_args.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    print(f'Process {mp.current_process().pid}: Running model.generate')
    with torch.inference_mode():
        outputs = model_args.model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if model_args.temperature > 0 else False,
            temperature=model_args.temperature,
            top_p=model_args.top_p,
            num_beams=model_args.num_beams,
            max_new_tokens=model_args.max_new_tokens,
            use_cache=True,
            # for log likelihood
            return_dict_in_generate=True,
            output_scores=True
        )

    # """
    # Code to calculate the entropy of the sequence
    # """
    # print(f'Process {mp.current_process().pid}: Model.generate completed')

    # scores = torch.stack(outputs.scores, dim=1)
    # probabilities = torch.softmax(scores, dim=-1)

    # epsilon = 1e-10
    # log_probabilities = torch.log(probabilities + epsilon)

    # log_probabilities_length = log_probabilities.shape[1]
    # generated_tokens = outputs.sequences[:, -log_probabilities_length:]

    # token_probs = torch.gather(log_probabilities, 2, generated_tokens.unsqueeze(-1)).squeeze(-1)

    # entropy = calculate_entropy_from_log_probs(token_probs)
    # print(f"Entropy: {entropy}")
    # print('-'*50)

    decoded_outputs = model_args.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    print('outputs:', decoded_outputs[0].strip())
    print('-'*50)
    
    # Stack the scores (logits) for each token
    scores = torch.stack(outputs.scores, dim=1)

    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(scores, dim=-1)

    # # Gather the probabilities of the generated tokens
    probabilities_length = probabilities.shape[1]
    generated_tokens = outputs.sequences[:, -probabilities_length:]

    # Select the probabilities corresponding to the generated tokens
    token_probs = torch.gather(probabilities, 2, generated_tokens.unsqueeze(-1)).squeeze(-1)

    # Calculate the conditional probability of the entire sequence
    sequence_probabilities = token_probs.prod(dim=1)

    return decoded_outputs[0].strip(), (sequence_probabilities.cpu().item(), scores, generated_tokens)


def load_args_from_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return argparse.Namespace(**config)

def extract_json_from_text(text):
    """Extracts JSON from the given text."""

    start_index = text.find("{")
    end_index = text.rfind("}") + 1
    json_string = text[start_index:end_index]
    
    return json.loads(json_string)