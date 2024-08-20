import os
import io
import base64
import pickle
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationConfig
from PIL import Image

from .vcd_utils.vcd_add_noise import add_diffusion_noise

from tools.read_yaml import get_data_folder
from .base_model import BaseModel
from accelerate import infer_auto_device_map
import warnings
warnings.filterwarnings("ignore")

default_path = "BAAI/Emu2-Chat"
default_image_path = get_data_folder()

class Emu2Chat(BaseModel):
    def __init__(self, temperature, max_new_tokens, num_beams, do_sample, top_p, opera_decoding, vcd_decoding, model_name: str = "emu2-chat", model_path: str = default_path, contrastive: bool = False, alt_text: bool = False, excel: bool = False, cd_alpha = None):
        super().__init__(model_name=model_name, model_path=model_path, contrastive=contrastive, alt_text=alt_text, excel=excel, opera_decoding=opera_decoding, vcd_decoding=vcd_decoding, cd_alpha=cd_alpha)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision='20ea30b04f8fee599cf97535e655c200df728501')
        device_map = Emu2Chat.get_device_map()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, revision='20ea30b04f8fee599cf97535e655c200df728501').eval()
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
            img_start_id=self.tokenizer.convert_tokens_to_ids('[IMG]'),
            img_end_id=self.tokenizer.convert_tokens_to_ids('[/IMG]')
        )
        self.greedy_condfig = GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    num_beams=1,
                    opera_decoding=False,
                    img_start_id=self.tokenizer.convert_tokens_to_ids('[IMG]'),
                    img_end_id=self.tokenizer.convert_tokens_to_ids('[/IMG]')
                )
        
        self.psuedoimageprompt = "Provide a detailed description of the image, covering all visible elements and their interactions, so as to thoroughly answer any potential questions about the image."#"Describe the image with enough detail to answer any questions about the given image."

    def _generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, alt_text=None, contrastive=False, cd_version=None, make_alttext=False):
        raw_image_data = os.path.join(default_image_path, dataset_name, image_path)
        
        image = Image.open(raw_image_data).convert('RGB')

        inputs = self.model.build_input_ids(
            text=['[<IMG_PLH>]'+text_prompt],
            tokenizer=self.tokenizer,
            image=[image]
        )

        input_dict = {}
        if alt_text:
            input_dict['instead_pixel_values'] = self.tokenizer(alt_text, return_tensors="pt").input_ids.to("cuda:0")
            if contrastive:
                input_dict['contrastive'] = True
                input_dict['cd_version'] = cd_version
        elif self.opera_decoding:
            input_dict['selected'] = True
        else:
            input_dict = None

        if self.vcd_decoding:
            image_cd = add_diffusion_noise(inputs['image'])
            image_cd = image_cd.to(torch.bfloat16)
        else:
            image_cd = None

        if make_alttext:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image=inputs["image"].to(torch.bfloat16),
                    length_penalty=-1,
                    num_beams=1,
                    max_new_tokens=512,
                    do_sample=False,
                    top_p=1.0,
                    temperature=0.0,
                    generation_config=self.greedy_condfig,
                    input_dict=input_dict,
                    image_cd = image_cd,
                    )
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image=inputs["image"].to(torch.bfloat16),
                    length_penalty=-1,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    generation_config=self.generation_config,
                    input_dict=input_dict,
                    image_cd = image_cd,
                    cd_alpha = self.cd_alpha
                    )

        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return response

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, ):
        if self.alt_text:
            alt_text = self._generate(self.psuedoimageprompt, raw_image_data, dataset_name, image_path, make_alttext=True)
            
            return self._generate(text_prompt, raw_image_data, dataset_name, image_path, alt_text=alt_text, contrastive=self.contrastive) 
        else:
            return self._generate(text_prompt, raw_image_data, dataset_name, image_path)
        
    def eval_forward(self, text_prompt: str, image_path: str):
        pass

    def get_coco_caption_prompt(self):
        return "Please describe this image in detail."

    @staticmethod
    def get_device_map():
        gpu_number = torch.cuda.device_count()
        
        if os.path.exists('.device_map')==False:
            os.mkdir('.device_map')

        filename = f"./.device_map/Emu2-chat_gpu{gpu_number}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                device_map = pickle.load(f)
                return device_map
        model = AutoModelForCausalLM.from_pretrained(default_path, device_map="auto", trust_remote_code=True, revision='20ea30b04f8fee599cf97535e655c200df728501').eval()
        max_memory = {}
        memory_per_gpu = str(int(150/gpu_number))
        for i in range(gpu_number):
            max_memory[i] = f"{memory_per_gpu}GiB"
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=['Block','LlamaDecoderLayer'])
        device_map['model.decoder.lm.lm_head'] = 0
        with open(filename, 'wb') as f:
            pickle.dump(device_map, f)
        del model
        return device_map
