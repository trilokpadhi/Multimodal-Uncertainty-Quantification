# Uncertainty Quantification of Multimodal models



## Models 
- [LLaVa](https://github.com/haotian-liu/LLaVA/tree/main)
- [CogVLM](https://github.com/THUDM/CogVLM.git)
- [CogVLM2](https://github.com/THUDM/CogVLM2.git) - GPT4V-level open-source multi-modal model based on Llama3-8B

- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4.git)

- [CoDi-2](https://github.com/microsoft/i-Code/tree/main/CoDi-2) - In-Context, Interleaved, and Interactive Any-to-Any Generation

- [Paligemma](https://huggingface.co/google/paligemma-3b-pt-224) [How to finetune Paligemma](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/finetune_paligemma.ipynb#scrollTo=RWOdf_fw2SAO)
- [Blip3 (X-gen MM)]()

## Datasets 
- [MMT-Bench](https://github.com/OpenGVLab/MMT-Bench)
- [MM-Bench](https://github.com/open-compass/MMBench)
- [MLLM-Bench](https://github.com/FreedomIntelligence/MLLM-Bench/)
- []


### Useful Commands 
Llava Inference 
```bash
python llava/eval/run_llava.py --model-path "liuhaotian/llava-v1.6-vicuna-7b"  --image-file /home/ubuntu/Trinity/playground/LLaVA/images/demo_cli.gif --query 'Write a caption' 
```

### Issues & Resolutions
- Authorization Error: To access paligemma create an access tokens [here](https://huggingface.co/settings/tokens) with write access.

