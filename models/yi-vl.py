import os
import io
import base64
import pickle
import numpy as np
import requests

import torch

from tools.read_yaml import get_hf_home
from .vcd_utils.vcd_add_noise import add_diffusion_noise

from .base_model import BaseModel
from models.llava_yi.conversation import conv_templates
from models.llava_yi.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    load_pretrained_model,
    process_images,
    tokenizer_image_token,
)
from transformers import GenerationConfig
from models.llava_yi.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from PIL import Image
from io import BytesIO
import warnings
from accelerate import infer_auto_device_map
warnings.filterwarnings("ignore")

default_path = os.path.join(get_hf_home(), "hub/Yi-VL-34B")

def get_pil_image(raw_image_data) -> Image.Image:
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data

    elif isinstance(raw_image_data, dict) and "bytes" in raw_image_data:
        return Image.open(io.BytesIO(raw_image_data["bytes"]))

    elif isinstance(raw_image_data, str):  # Assuming this is a base64 encoded string
        image_bytes = base64.b64decode(raw_image_data)
        return Image.open(io.BytesIO(image_bytes))

    else:
        raise ValueError("Unsupported image data format")

class YiVL(BaseModel):
    def __init__(self, temperature, max_new_tokens, num_beams, do_sample, top_p, opera_decoding, vcd_decoding, model_name: str = "yi-vl", model_path: str = default_path, conv_mode: str = "mm_default", contrastive: bool = False, alt_text: bool = False, excel: bool = False,cd_alpha=None):
        super().__init__(model_name=model_name, model_path=model_path, contrastive=contrastive, alt_text=alt_text, excel=excel, opera_decoding=opera_decoding, vcd_decoding=vcd_decoding,cd_alpha=cd_alpha)
        model_path = os.path.expanduser(model_path)
        key_info["model_path"] = model_path
        get_model_name_from_path(model_path)
        if opera_decoding:
            device_map = YiVL.get_device_map(model_path)
            tokenizer, model, image_processor, _ = load_pretrained_model(model_path, device_map=device_map)
        else:
            tokenizer, model, image_processor, _ = load_pretrained_model(model_path)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_beams = num_beams
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            opera_decoding=self.opera_decoding,
            image_token_id=IMAGE_TOKEN_INDEX
        )

        self.greedy_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            num_beams=1,
            opera_decoding=False,
            image_token_id=IMAGE_TOKEN_INDEX
        )
        self.psuedoimageprompt = "Provide a detailed description of the image, covering all visible elements and their interactions, so as to thoroughly answer any potential questions about the image."#"Describe the image with enough detail to answer any questions about the given image."

    def _generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, alt_text=None, contrastive=False, cd_version=None, make_alttext=False):
        raw_image_data = get_pil_image(raw_image_data)
        raw_image_data = raw_image_data.convert("RGB")
        image_tensor = process_images([raw_image_data], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.bfloat16)
    
        prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompts_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep
        
        outputs = self.generate_sentence(image_tensor, stop_str, prompt, alt_text=alt_text, contrastive=contrastive, cd_version=cd_version, make_alttext=make_alttext)
        
        return outputs
    
    def generate_sentence(self, image_tensor, stop_str, prompt, alt_text=None, contrastive=False, cd_version=None, noise_step=500, make_alttext=False):
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        input_dict = {}
        if alt_text:
            input_dict['instead_pixel_values'] = self.tokenizer(alt_text, return_tensors="pt").input_ids.to("cuda:0")
            if contrastive:
                input_dict['contrastive'] = True
                input_dict['cd_version'] = cd_version
        else:
            input_dict = None

        if self.vcd_decoding:
            image_tensor_cd = add_diffusion_noise(image_tensor, noise_step)
        else:
            image_tensor_cd = None
        
        if make_alttext:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    generation_config=self.greedy_config,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    input_dict=input_dict,
                    image_cd = image_tensor_cd,
                )
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    generation_config=self.generation_config,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    input_dict=input_dict,
                    image_cd = image_tensor_cd,
                    cd_alpha = self.cd_alpha
                )
            
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        return outputs

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, ):
        if self.alt_text:
            alt_text = self._generate(self.psuedoimageprompt, raw_image_data, dataset_name, image_path, make_alttext=True)
            
            return self._generate(text_prompt, raw_image_data, dataset_name, image_path, alt_text=alt_text, contrastive=self.contrastive) 
        else:
            return self._generate(text_prompt, raw_image_data, dataset_name, image_path)

    def eval_forward(self, text_prompt: str, image_path: str):
        # Similar to the Idefics' eval_forward but adapted for QwenVL
        pass

    def get_coco_caption_prompt(self):
        return "Please describe this image in detail."
    
    @staticmethod
    def get_device_map(model_path):
        gpu_number = torch.cuda.device_count()
        
        if os.path.exists('.device_map')==False:
            os.mkdir('.device_map')

        filename = f"./.device_map/Yi-VL_gpu{gpu_number}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                device_map = pickle.load(f)
                return device_map
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path)
        max_memory = {}
        memory_per_gpu = str(int(120/gpu_number - 1))
        for i in range(gpu_number):
            max_memory[i] = f"{memory_per_gpu}GiB"
        max_memory[0] = "1GiB"
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["CLIPEncoderLayer", "LlamaDecoderLayer"])
        with open(filename, 'wb') as f:
            pickle.dump(device_map, f)
        del model
        return device_map