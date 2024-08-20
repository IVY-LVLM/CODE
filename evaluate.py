import sys
import argparse
import os

import yaml
import contextlib

from tools.read_yaml import *

sys.path.append(os.getcwd())

from benchmarks.base_eval_dataset import load_dataset
from file_utils.result_file_manage import ResultFileManager
from models.base_model import AVAILABLE_MODELS, load_model
from transformers import set_seed

set_seed(555)

def get_info(info):
    if "name" not in info:
        raise ValueError("Model name is not specified.")
    name = info["name"]
    # info.pop("name")
    return name, info


def load_models(model_infos):
    for model_info in model_infos:
        name, info = get_info(model_info)
        model = load_model(name, info)
        yield model


def load_datasets(dataset_infos):
    for dataset_info in dataset_infos:
        name, info = get_info(dataset_info)
        dataset = load_dataset(name, info)
        yield dataset


class DualOutput:
    def __init__(self, file, stdout):
        self.file = file
        self.stdout = stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--models",
        type=str,
        nargs="?",
        help="lmm model to use",
        default='llama_adapter,uniter,uniter_large',
    )
    args.add_argument(
        "--datasets",
        type=str,
        nargs="?",
        help="dataset to use",
        default='vqa2,mscoco',
    )
    args.add_argument( 
        "--contrastive",
        action="store_true",
        help="State if you want use contrastive decoding. (Must be used with --alt-text option)",
        default=False,
    )
    args.add_argument( 
        "--alt-text",
        action="store_true",
        help="State if you want use alt-text instead of image.",
        default=False,
    )
    args.add_argument( 
        "--excel",
        action="store_true",
        help="Excel on/off",
        default=False,
    )
    args.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
    )
    args.add_argument( 
        "--temperature",
        type=float,
        default=0.0,
    )
    args.add_argument( 
        "--num_beams",
        type=int,
        default=1,
    )
    args.add_argument( 
        "--do_sample",
        action="store_true",
        default=False,
    )
    args.add_argument( 
        "--top_p",
        type=float,
        default=1.0,
    )
    args.add_argument( 
        "--opera_decoding",
        action="store_true",
        default=False,
    )
    args.add_argument( 
        "--vcd_decoding",
        action="store_true",
        default=False,
    )
    
    args.add_argument( 
        "--cd_alpha",
        type=float,
        default=0.7,
    )

    args.add_argument( 
        "--continue_file",
        type=str,
        default="",
    )
    args.add_argument( 
        "--new_file",
        action="store_true",
        default=False,
    )


    phrased_args = args.parse_args()

    model_names = phrased_args.models.split(",")
    model_infos = [{"name": name, "temperature": phrased_args.temperature, "max_new_tokens": phrased_args.max_new_tokens, "contrastive": phrased_args.contrastive, "alt_text": phrased_args.alt_text, "excel": phrased_args.excel, "num_beams": phrased_args.num_beams, "do_sample": phrased_args.do_sample, "top_p": phrased_args.top_p, "cd_alpha": phrased_args.cd_alpha, "opera_decoding": phrased_args.opera_decoding, "vcd_decoding": phrased_args.vcd_decoding} for name in model_names]

    dataset_infos = [{"name": dataset_name} for dataset_name in phrased_args.datasets.split(",")]

    if phrased_args.contrastive:
        assert phrased_args.alt_text, "Argument CONTRASTIVE MUST be used with alt-text"

    if phrased_args.vcd_decoding:
        assert phrased_args.do_sample, "VCD decoding MUST be used with do_sample"

    if not os.path.exists(get_log_folder()):
        os.makedirs(get_log_folder())

    for model_info in model_infos:
        name = model_info["name"]
        print("\nMODEL INFO:", model_info)
        print("-" * 80)
        dataset_count = 0
        for data_idx, dataset_info in enumerate(dataset_infos):
            dataset_name, _dataset_info = get_info(dataset_info)

            result_file_manager = ResultFileManager(model_info["name"], dataset_name, phrased_args.continue_file, phrased_args.new_file)

            model = load_model(model_info["name"], model_info)
            dataset = load_dataset(dataset_name, _dataset_info)
            model_info['name'] = name

            dataset_count += 1
            print('MODEL:', model.name, 'TEMPERATURE:', model.temperature, 'MAX_NEW_TOKENS:', model.max_new_tokens, 'NUM_BEAMS:', model.num_beams, 'TOP_P:', model.top_p, 'DO_SAMPLE:', model.do_sample, 'CONTRASTIVE', model.contrastive, 'OPERA', model.opera_decoding)
            print(f"\nDATASET: {dataset.name}")
            print("-" * 20)

            dataset.evaluate(model, result_file_manager)  # Assuming this function now prints results directly.
            print()

        print("-" * 80)
        print(f"Total Datasets Evaluated: {dataset_count}\n")

    print("=" * 80)

# python evaluate.py --models llava --datasets mmbench

