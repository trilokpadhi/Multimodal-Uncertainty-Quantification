# python eval_ferret.py --model-path checkpoints/ferret-13b --model-base checkpoints/vicuna-13b-v1.3/ --model-name ferret --load-8bit --prompt "Write a caption for this <image> " --images ferret/serve/examples/bathroom.jpg
import torch
import argparse
# from ferret.utils import load_pretrained_model, load_image_from_base64, process_images
# from ferret.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
# from ferret.mm_utils import tokenizer_image_token

from ferret.model.builder import load_pretrained_model
from ferret.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from tr
from PIL import Image
class FerretModelInference:
    def __init__(self, model_path, model_base=None, model_name=None, load_8bit=False, load_4bit=False, image_w=336, image_h=336, keep_aspect_ratio=False):
        self.image_w = image_w
        self.image_h = image_h
        self.keep_aspect_ratio = keep_aspect_ratio
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, load_8bit, load_4bit
        )
        self.is_multimodal = 'llava' in model_name.lower() or 'ferret' in model_name.lower()

    @torch.inference_mode()
    def generate(self, prompt, images=None, max_new_tokens=256, temperature=1.0, top_p=1.0, stop_str=None):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError("Number of images does not match number of <image> tokens in prompt")
            
            # images = [load_image_from_base64(image) for image in images]
            images = Image.open(images[0]) 
            if self.keep_aspect_ratio:
                images = process_images(images, image_processor, model.config)
            else:
                images = image_processor(images, return_tensors='pt', do_resize=True, do_center_crop=False, size=[self.image_h, self.image_w])['pixel_values']
            
            if isinstance(images, list):
                images = [image.to(model.device, dtype=torch.float16) for image in images]
            else:
                images = images.to(model.device, dtype=torch.float16)

            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
        output_ids = list(input_ids)
        pred_ids = []

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        past_key_values = None
        for _ in range(max_new_tokens):
            if past_key_values is None:
                out = model(torch.as_tensor([input_ids]).cuda(), use_cache=True, images=images)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device="cuda")
                out = model(input_ids=torch.as_tensor([[token]], device="cuda"), use_cache=True, attention_mask=attention_mask, past_key_values=past_key_values, images=images)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            pred_ids.append(token)

            if stop_str is not None and token == tokenizer(stop_str).input_ids[0]:
                break
            elif token == tokenizer.eos_token_id:
                break

        output = tokenizer.decode(pred_ids, skip_special_tokens=True)
        return output

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model if applicable")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model in 8-bit precision")
    parser.add_argument("--load-4bit", action="store_true", help="Load the model in 4-bit precision")
    parser.add_argument("--keep-aspect-ratio", action="store_true", help="Keep the aspect ratio for image processing")
    parser.add_argument("--image-w", type=int, default=336, help="Width of the images for multimodal models")
    parser.add_argument("--image-h", type=int, default=336, help="Height of the images for multimodal models")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value")
    parser.add_argument("--stop-str", type=str, help="Stop string for early termination")
    parser.add_argument("--images", type=str, nargs='*', help="Base64 encoded images for multimodal models")

    args = parser.parse_args()

    inference = FerretModelInference(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=args.model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        image_w=args.image_w,
        image_h=args.image_h,
        keep_aspect_ratio=args.keep_aspect_ratio
    )

    result = inference.generate(
        prompt=args.prompt,
        images=args.images,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_str=args.stop_str
    )
    print(f"Generated text: {result}")