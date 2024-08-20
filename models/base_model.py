import pathlib
import sys
from .contrastive_decoding.cd_greedy import picd_greedy
from .contrastive_decoding.cd_search import picd_search
from .contrastive_decoding.cd_sample import picd_sampling
from .vcd_utils.vcd_sample import evolve_vcd_sampling

cur_path = sys.path[0]
sys.path.append("../..")
wp_cur_dir = pathlib.Path(cur_path)
sys.path.append(cur_path)
sys.path.append(str(wp_cur_dir.parent))
sys.path.append(str(wp_cur_dir.parent.parent))
print(sys.path)

from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict

import importlib

AVAILABLE_MODELS: Dict[str, str] = {
    "llava-model-hf": "LLaVA_Model_HF",
    "llava-next": "LLaVA_Next",
    "yi-vl": "YiVL",
    "emu2-chat": "Emu2Chat",
    "intern-vl": "InternVL",
    "internlm-xc2": "InternLM_XC2",
}

class BaseModel(ABC):
    def __init__(self, model_name: str, model_path: str, *, max_batch_size: int = 1, contrastive: bool = False, alt_text: bool = False,  excel: bool = False, opera_decoding = False, vcd_decoding = False, cd_alpha = None):
        self.name = model_name
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.contrastive = contrastive
        self.alt_text = alt_text
        self.excel = excel
        self.opera_decoding = opera_decoding
        self.vcd_decoding = vcd_decoding
        self.cd_alpha = cd_alpha
        if self.alt_text:
            picd_greedy()
            picd_search()
            picd_sampling()

        if self.vcd_decoding:
            evolve_vcd_sampling()

    @abstractmethod
    def generate(self, **kwargs):
        pass

    @abstractmethod
    def eval_forward(self, **kwargs):
        pass
    
    @abstractmethod
    def get_coco_caption_prompt(self):
        return "Please describe this image in detail."

def load_model(model_name: str, model_args: Dict[str, str]) -> BaseModel:
    assert model_name in AVAILABLE_MODELS, f"{model_name} is not an available model."
    module_path = "models." + model_name
    model_formal_name = AVAILABLE_MODELS[model_name]
    imported_module = importlib.import_module(module_path)
    model_class = getattr(imported_module, model_formal_name)
    print(f"Imported class: {model_class}")
    model_args.pop("name")
    return model_class(**model_args)
