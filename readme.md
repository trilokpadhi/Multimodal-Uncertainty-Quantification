
# Multimodal Uncertainty Quantification 

**Update as of March 17**

This repository contains various scripts and utilities for multimodal uncertainty quantification. Below is a brief description of each file in the `src2` directory.

## Files

- `data_utils.py`: Utility functions for data processing and manipulation.
- `generate_metrics.py`: Main script for generating various metrics including baseline results, accuracy, and grounding.
- `grounding_with_external_entropy.py`: Script for grounding with external entropy.
- `grounding_with_internal_entropy.py`: Script for grounding with internal entropy.
- `inference_llava.py`: Inference script for LLAVA model.
- `inference_utils.py`: Utility functions for inference.
- `main.py`: Main entry point for running different experiments.
- `main_gemini.py`: Main script for GEMINI experiments.
- `main_grounding.py`: Main script for grounding experiments.
- `main_grounding_biomedclip_med.py`: Grounding script for BiomedCLIP model on medical data.
- `main_grounding_gdsam.py`: Grounding script for GDSAM model.
- `main_grounding_gemini_med.py`: Grounding script for GEMINI model on medical data.
- `main_grounding_llama32.py`: Grounding script for LLAMA32 model.
- `main_grounding_llama32_med.py`: Grounding script for LLAMA32 model on medical data.
- `main_grounding_qwen_VL.py`: Grounding script for QWEN VL model.
- `main_grounding_qwen_VL_med.py`: Grounding script for QWEN VL model on medical data.
- `main_llava.py`: Main script for LLAVA experiments.
- `main_llava_med.py`: Main script for LLAVA experiments on medical data.
- `main_reliability_diagram.py`: Script for generating reliability diagrams.
- `main_reliability_diagram2.py`: Another script for generating reliability diagrams with different settings.
- `main_reliability_diagram_final_metric_march12.py`: Final script for generating reliability diagrams with specific metrics (as of March 12).
- `main_reliability_diagram_with_variance.py`: Script for generating reliability diagrams with variance.
- `plot_calib_baselines_with_grounding.py`: Script for plotting calibration baselines with grounding.
- `plot_calib_baselines_with_grounding_debug.py`: Debug version of the script for plotting calibration baselines with grounding.

## Usage

Each script can be run individually depending on the experiment or analysis you want to perform. For example, to generate metrics, you can run:

```sh
python [generate_metrics.py]
```

## VQA Dataset Overview
```
VQA/
├── v1/
│   ├── annotations/
│   │   ├── Annotations_Train_mscoco.zip
│   │   ├── Annotations_Val_mscoco.zip
│   │   ├── mscoco_train2014_annotations.json
│   │   ├── mscoco_val2014_annotations.json
│   ├── questions/
│   │   ├── MultipleChoice_mscoco_test2015_questions.json
│   │   ├── OpenEnded_mscoco_train2014_questions.json
│   │   ├── Questions_Test_mscoco.zip
│   │   ├── Questions_Train_mscoco.zip
│   │   ├── Questions_Val_mscoco.zip
│   ├── train2014.zip
│   ├── wget-log
├── v2/
│   ├── annotations/
│   │   ├── v2_Annotations_Train_mscoco.zip
│   │   ├── v2_Annotations_Val_mscoco.zip
│   │   ├── v2_mscoco_train2014_annotations.json
│   │   ├── v2_mscoco_val2014_annotations.json
│   ├── questions/
│   │   ├── v2_OpenEnded_mscoco_test2015_questions.json
│   │   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   │   ├── v2_Questions_Test_mscoco.zip
│   │   ├── v2_Questions_Train_mscoco.zip
│   │   ├── v2_Questions_Val_mscoco.zip
```

### Key Differences Between V1 and V2

| Feature            | VQA v1.0                                      | VQA v2.0                                                      |
|--------------------|-----------------------------------------------|---------------------------------------------------------------|
| **Image Sources**  | MS COCO and abstract images.                  | Only MS COCO (no abstract images).                            |
| **Question Types** | Open-ended and multiple-choice tasks.         | Only open-ended tasks.                                        |
| **Dataset Size**   | Smaller dataset with 614,163 questions.       | Larger dataset with 1,105,904 questions.                      |
| **Bias Reduction** | Prone to language bias; answers may rely on question patterns. | Balanced dataset; each question pairs with multiple images for visual reasoning. |
| **Abstract Images**| Includes abstract cartoon-like images for experiments. | No abstract images; focuses on real-world MS COCO images.     |


