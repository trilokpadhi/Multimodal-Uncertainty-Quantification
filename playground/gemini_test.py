from google import genai
from google.genai import types
import os
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key) trilok thre

response = client.models.generate_content(
    model='gemini-1.5-flash',
    contents="Who won Wimbledon this year?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            google_search=types.GoogleSearchRetrieval
        )]
    )
)
print(response)
    
# import os
# from dotenv import load_dotenv
# import google.generativeai as genai
# from PIL import Image

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=api_key)

# # Set the generation configuration with max_output_tokens
# generation_config = {
#     "max_output_tokens": 50,  # Adjust the number of tokens as needed
#     "temperature": 0.7,        # Optional: Controls the randomness of the output
#     "top_p": 0.9               # Optional: Controls the diversity of the output
# }

# # Model selection
# model_id = "gemini-2.0-flash" 
# # model_id = "gemini-2.0-flash-thinking-exp"  # not working
# # Load the multimodal model
# model = genai.GenerativeModel(model_id)

# # Open and read the image file
# with open('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/2d_calibrated_reliability.png', "rb") as img_file:
#     image = img_file.read()

# # Construct multimodal input
# contents = [
#     {"mime_type": "image/jpeg", "data": image},
#     {"text": "Describe the scene in the image."}
# ]

# # Estimate input token count
# token_count = model.count_tokens(contents)
# input_tokens = token_count.total_tokens

# # Generate response
# response = model.generate_content(
#     contents=contents,
#     generation_config=generation_config
# )

# print(response)
# # Estimate output token count
# output_tokens = len(response.text.split())  # Approximate based on word count

# # Pricing details (based on Gemini 2.0 Flash pricing)
# input_cost_per_million = 0.10  # $ per 1M input tokens
# output_cost_per_million = 0.40  # $ per 1M output tokens

# # Cost calculation
# input_cost = (input_tokens / 1_000_000) * input_cost_per_million
# output_cost = (output_tokens / 1_000_000) * output_cost_per_million
# total_cost = input_cost + output_cost

# # Print results
# print(f"Input Tokens: {input_tokens}")
# print(f"Output Tokens: {output_tokens}")
# print(f"Estimated Cost: ${total_cost:.6f}")
# print(response.text.split('\n')[0])


# # # import os
# # # from dotenv import load_dotenv
# # # import google.generativeai as genai
# # # from google.genai.model import ImagePrompt
# # # # Load environment variables
# # # load_dotenv()

# # # import base64

# # # # Load your image
# # # image_path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/2d_calibrated_reliability.png'
# # # with open(image_path, 'rb') as image_file:
# # #     image_data = image_file.read()

# # # # Encode the image to base64
# # # encoded_image = base64.b64encode(image_data).decode('utf-8')

# # # image_prompt = ImagePrompt(image_bytes=encoded_image)
# # # api_key = os.getenv("GEMINI_API_KEY")
# # # genai.configure(api_key=api_key)
# # # # https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro
# # # # model = genai.GenerativeModel(model_name="gemini-exp-1121")
# # # model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
# # # # model = genai.GenerativeModel("gemini-1.5-flash")
# # # response = model.generate_content("Write a story about a magic backpack.", image_prompts=[image_prompt])
# # # print(response.text)

# # import os
# # from dotenv import load_dotenv
# # import google.generativeai as genai
# # from PIL import Image

# # # Load environment variables
# # load_dotenv()
# # api_key = os.getenv("GEMINI_API_KEY")
# # genai.configure(api_key=api_key)

# # # Set the generation configuration with max_output_tokens
# # generation_config = {
# #     "max_output_tokens": 50,  # Adjust the number of tokens as needed
# #     "temperature": 0.7,        # Optional: Controls the randomness of the output
# #     "top_p": 0.9               # Optional: Controls the diversity of the output
# # }


# # # model_id = "gemini-1.5-pro-latest" 
# # model_id = "gemini-2.0-flash"
# # # Load the multimodal model
# # model = genai.GenerativeModel(model_id)

# # # Open and read the image file
# # with open('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/2d_calibrated_reliability.png', "rb") as img_file:
# #     image = img_file.read()

# # # Construct multimodal input
# # response = model.generate_content(
# #     contents=[
# #         {"mime_type": "image/jpeg", "data": image},
# #         {"text": "Describe the scene in the image."}
# #     ],
# #     generation_config=generation_config
# # )

# # print(response.text.split('\n')[0])


