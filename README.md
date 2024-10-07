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


### Useful Commands 
Llava Inference 
```bash
python llava/eval/run_llava.py --model-path "liuhaotian/llava-v1.6-vicuna-7b"  --image-file /home/ubuntu/Trinity/playground/LLaVA/images/demo_cli.gif --query 'Write a caption' 
```

### Issues & Resolutions
- Authorization Error: To access paligemma create an access tokens [here](https://huggingface.co/settings/tokens) with write access.

- RuntimeError: Failed to import transformers.models.llama.modeling_llama 
```bash 
Traceback (most recent call last):
  File "/home/ubuntu/Multimodal-Uncertainty-Quantification/playground/construct_graph.py", line 44, in <module>
    model = AutoModelForCausalLM.from_pretrained(model_name)
  File "/home/ubuntu/miniconda3/envs/llava/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 563, in from_pretrained
    model_class = _get_model_class(config, cls._model_mapping)
  File "/home/ubuntu/miniconda3/envs/llava/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 384, in _get_model_class
    supported_models = model_mapping[type(config)]
  File "/home/ubuntu/miniconda3/envs/llava/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 735, in __getitem__
    return self._load_attr_from_module(model_type, model_name)
  File "/home/ubuntu/miniconda3/envs/llava/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 749, in _load_attr_from_module
    return getattribute_from_module(self._modules[module_name], attr)
  File "/home/ubuntu/miniconda3/envs/llava/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 693, in getattribute_from_module
    if hasattr(module, attr):
  File "/home/ubuntu/miniconda3/envs/llava/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1576, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/home/ubuntu/miniconda3/envs/llava/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1588, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.models.llama.modeling_llama because of the following error (look up to see its traceback):
/home/ubuntu/miniconda3/envs/llava/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops5zeros4callEN3c108ArrayRefINS2_6SymIntEEENS2_8optionalINS2_10ScalarTypeEEENS6_INS2_6LayoutEEENS6_INS2_6DeviceEEENS6_IbEE
```
- Resolution: 
```bash
pip install transformers==4.33.0
```
- How to get log probs from LLava [here](https://github.com/haotian-liu/LLaVA/issues/108)

- To get process killed: 
```bash
ps aux | grep "python -c" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep "/opt/conda/envs/llava/bin/python" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```
```bash
sudo lsof -i :29502 # check the process running on port 29502, below command will kill all
for pid in $(ps -ef | grep 'main_distributed.py' | grep -v grep | awk '{print $2}'); do
    sudo kill -9 $pid
done
```
- To run on specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5 nohup python -m src.main --config-file configs/llava_gqa.yaml > llava_gqa.out 2>&1 &
```
- Run with nohup 
```bash
nohup python src/main_distributed.py --config /home/ubuntu/Multimodal-Uncertainty-Quantification/configs/llava_gqa_random_100.yaml > output_random_100.log 2>&1 &
```
- To check the log of the process, This will show you the last few lines of the log and update as new output is appended.
```bash
tail -f output_random_100.log
```

#### Results Directory
```plaintext
runs/
├── llava_gqa_sum/
│   ├── explanations/
│   │   └── 678.pkl
│   │       └── Each explanations_0_0_181060668.pkl file contains:
│   │           ├── question_ids: IDs for each question processed.
│   │           ├── questions: Original questions associated with each sample.
│   │           ├── promptified_questions: Questions reformulated for prompt use.
│   │           ├── answers: Short-form answers provided by the model.
│   │           ├── full_answers: Detailed answers generated by the model.
│   │           ├── image_ids: IDs for associated images.
│   │           ├── image_paths: File paths to the associated images.
│   │           ├── response_0 to response_19: Model responses for different trials or settings.
│   │           └──Each response contains:
│   │               ├── prompt: Prompt used for the model.
│   │               ├── outputs: Its the output of model.generate() function, which has scores, logits, and generated tokens.
│   |               ├── |-- scores is a tuple of length of the generated tokens and each score is of the shape torch.Size([1, 32064])
│   │               ├── generated_token_ids: Generated token IDs.
│   │               ├── decoded_outputs: Decoced generated tokens. 
│   ├── grounding/
│   │   └── (Files related to grounding information.)
│   └── uncertainty/
│       └── (Files related to uncertainty analysis.)
```