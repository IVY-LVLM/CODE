# CODE: Contrasting Self-generated Description to Combat Hallucination in Large Multi-modal Models [[Project](https://github.com/IVY-LVLM/CODE/)][[arXiv](https://arxiv.org/abs/2406.01920)]

Official implementation of ['CODE: Contrasting Self-generated Description to Combat Hallucination in Large Multi-modal Models'](https://arxiv.org/abs/2406.01920).
![image](https://github.com/user-attachments/assets/19a834f2-f4da-47ae-bf19-2d9fb6e19fa9)


## :page_facing_up: Table of contents

- [Abstract](#pencil2-abstract)
- [Environment Setup](#eyes-environment-setup)
- [Default Setting](#clap-default-setting)
- [Project Structure](#house-project-structure)
- [Evaluate Models on Benchmarks](#hammer-evaluate-models-on-benchmarks)
- [Download Datasets](#arrow_down-download-datasets)

## :pencil2: Abstract
Large Multi-modal Models (LMMs) have recently demonstrated remarkable abilities in visual context understanding and coherent response generation. However, alongside these advancements, the issue of hallucinations has emerged as a significant challenge, producing erroneous responses that are unrelated to the visual contents. In this paper, we introduce a novel contrastive-based decoding method, COuntering DEscription Contrastive Decoding (CODE), which leverages self-generated descriptions as contrasting references during the decoding phase of LMMs to address hallucination issues. CODE utilizes the comprehensive descriptions from model itself as visual counterpart to correct and improve response alignment with actual visual content. By dynamically adjusting the information flow and distribution of next-token predictions in the LMM's vocabulary, CODE enhances the coherence and informativeness of generated responses. Extensive experiments demonstrate that our method significantly reduces hallucinations and improves cross-modal consistency across various benchmarks and cutting-edge LMMs. Our method provides a simple yet effective decoding strategy that can be integrated to existing LMM frameworks without additional training.

## :eyes: Environment Setup

```bash
conda create -n code -y python=3.9
conda activate code

# install packaging, pytorch
pip3 install packaging torch torchvision torchaudio

# install dependencies
pip install -r requirements.txt
pip install -e transformers
```

## :clap: Default Setting

Before executing the code, you must complete the YAML file below by specifying the folder paths and API keys.

``` yaml
# default_settings.yaml
settings:
  log_folder: <LOG FOLDER>
  data_folder: <DATA FOLDER>
  openai_api_key: <OPENAI API KEY>
```

## :house: Project Structure
Here is the project structure.

The project structure primarily includes four directories: benchmarks, file_utils, models, and tools. The file evaluate.py is used to perform evaluations on benchmarks, while generate_counterfactual_keywords_gpt4v.py is designated for generating counterfactual keywords using gpt4v.

```
.
├── benchmarks                   # 6 Evaluation Benchmarks (+Chair)
│   ├── __init__.py             
│   ├── base_eval_dataset.py
│   ├── coco-chair.py
│   ├── llavabench.py
│   ├── llava-qa90.py
│   ├── mmhalbench.py
│   ├── mmvp.py
│   ├── pope.py
│   └── realworld-qa.py
├── file_utils
│   ├── __pycache__
│   └── result_file_manage.py
├── huggingface_file           # modified huggingface code
│   └── modules
├── models                     # 6 Models
│   ├── __init__.py
│   ├── base_model.py
│   ├── contrastive_decoding
│   ├── emu2-chat.py
│   ├── internlm-xc2.py
│   ├── intern-vl.py
│   ├── llava-model-hf.py
│   ├── llava-next.py
│   └── yi-vl.py
├── default_settings.yaml
├── evaluate.py
├── README.md
├── requirements.txt
└── transformers               # modified transformers
    ├── README.md
    ├── setup.py
    └── src
```

## :white_check_mark: Benchmark Folder Structure

You must first prepare the benchmark dataset. According to the folder structure provided, please make sure to place the image files in the designated directories.

```
.
├── llavabench
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── llava-qa90
│   ├── 000000020650.jpg
│   ├── 000000034096.jpg
│   └── ...
├── mmhalbench
│   ├── 10172500456_1f40b6bd38_o.jpg
│   ├── 11715451803_24861529ab_o.jpg
│   └── ...
├── mmvp
│   └── MMVP Images
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
├── realworldqa
│   ├── annotations.json
│   └── images
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
└── pope
    ├── COCO_val2014_000000001171.jpg
    ├── COCO_val2014_000000003845.jpg
    └── ...
```

## :hammer: Evaluate Models on Benchmarks

1. Run the evaluation code
```bash
# activate the environment
conda activate CODE

# evaluate <model_name> on <benchmark_name> with CODE DECODING 
python evaluate.py --models <model_name> --datasets <benchmark_name> --alt-text --contrastive --cd_alpha <cd_alpha>
```

2. Select Counterfactual keyword file

The list of log files will be displayed. 
You can start the evaluation from the beginning by selecting the new result file. If you select the existing file, you can continue evaluation.
```
<<<<<=====Result file=====>>>>>
Here's the list of result files for the given model and benchmark: 
1. New result file
2. llavabench_emu2-chat_results_0731_v1.jsonl
Enter the number of your selection(1-2): 
```

3. Check the results

The results will be revealed in the console or you can check the log files from the log directory.
In default_settings.yaml, you can designate log_folder.

## :arrow_down: Download Datasets

- [POPE](https://github.com/RUCAIBox/POPE)
- [MMVP](https://huggingface.co/datasets/MMVP/MMVP)
- [LLaVA-Bench(In-the-Wild)](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild)
- [MMHalBench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench)
- [LLaVA-QA90](https://github.com/llava-rlhf/LLaVA-RLHF/tree/main/Eval/llava)
- [Realworld-QA](https://huggingface.co/datasets/xai-org/RealworldQA)
