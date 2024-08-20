import pickle
import torch
import os
import torchvision.transforms as T

from tools.read_yaml import get_data_folder
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .base_model import BaseModel
from PIL import Image
from accelerate import infer_auto_device_map

import warnings
warnings.filterwarnings("ignore")

default_model_path = "internlm/internlm-xcomposer2-vl-7b"
default_image_path = get_data_folder()

class InternLM_XC2(BaseModel):
    def __init__(self, temperature, max_new_tokens, num_beams, do_sample, top_p, opera_decoding, vcd_decoding, model_name: str = "internlm-xc2", model_path: str = default_model_path, contrastive: bool = False, alt_text: bool = False,  excel: bool = False, cd_alpha =None):
        super().__init__(model_name=model_name, model_path=model_path, contrastive=contrastive, alt_text=alt_text, excel=excel, opera_decoding=opera_decoding, vcd_decoding=vcd_decoding, cd_alpha = cd_alpha)
        torch.set_grad_enabled(False)
        self.tokenizer = AutoTokenizer.from_pretrained(default_model_path, trust_remote_code=True, revision='c67bd06390dbe068a582c6561570725b1289a7c5')
        device_map = InternLM_XC2.get_device_map()
        self.model = AutoModelForCausalLM.from_pretrained(
            default_model_path, 
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True, 
            revision='c67bd06390dbe068a582c6561570725b1289a7c5').eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_beams = num_beams
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            top_p=self.top_p,
            opera_decoding=self.opera_decoding
        )
        self.greedy_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.0,
            num_beams=1,
            do_sample=False,
            top_p=1.0,
            opera_decoding=False
        )


        self.psuedoimageprompt = "Provide a detailed description of the image, covering all visible elements and their interactions, so as to thoroughly answer any potential questions about the image."#"Describe the image with enough detail to answer any questions about the given image."

    def _generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, alt_text=None, contrastive=False, cd_version=None, make_alttext=False,):
        raw_image_data = os.path.join(default_image_path, dataset_name, image_path)        
        query = '<ImageHere>' + text_prompt

        input_dict = {}
        if alt_text:
            input_dict['instead_pixel_values'] = self.tokenizer(alt_text, return_tensors="pt").input_ids.to("cuda:0")
            if contrastive:
                input_dict['contrastive'] = True
                input_dict['cd_version'] = cd_version
        else:
            input_dict = None

        if make_alttext:
            with torch.cuda.amp.autocast():
                response, _ = self.model.chat(self.tokenizer, 
                                            query=query, 
                                            image=raw_image_data, 
                                            max_new_tokens = self.max_new_tokens, 
                                            num_beams = 1, 
                                            temperature = 0.0, 
                                            do_sample=False, 
                                            top_p=1.0, 
                                            history=[], 
                                            generation_config=self.greedy_config, 
                                            input_dict=input_dict,
                                            vcd_decoding=self.vcd_decoding,
                                            )
        else:
            with torch.cuda.amp.autocast():
                response, _ = self.model.chat(self.tokenizer, 
                                            query=query, 
                                            image=raw_image_data, 
                                            max_new_tokens = self.max_new_tokens, 
                                            num_beams = self.num_beams, 
                                            temperature = self.temperature, 
                                            do_sample=self.do_sample, 
                                            top_p=self.top_p, history=[], 
                                            generation_config=self.generation_config, 
                                            input_dict=input_dict,
                                            vcd_decoding=self.vcd_decoding,
                                            cd_alpha=self.cd_alpha
                                            )

        return response

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, ):
        if self.alt_text:
            alt_text = self._generate(self.psuedoimageprompt, raw_image_data, dataset_name, image_path, make_alttext=True)
            
            return self._generate(text_prompt, raw_image_data, dataset_name, image_path, alt_text=alt_text, contrastive=self.contrastive) 
        else:
            return self._generate(text_prompt, raw_image_data, dataset_name, image_path)
        
    def eval_forward(self, text_prompt: str, raw_image_data: str):
        pass

    def get_coco_caption_prompt(self):
        return "Please describe this image in detail."

    @staticmethod
    def get_device_map():
        gpu_number = torch.cuda.device_count()
        
        if os.path.exists('.device_map')==False:
            os.mkdir('.device_map')

        filename = f"./.device_map/InternLM-XC2_gpu{gpu_number}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                device_map = pickle.load(f)
                return device_map
        model = AutoModelForCausalLM.from_pretrained(default_model_path, device_map="auto", trust_remote_code=True, revision='c67bd06390dbe068a582c6561570725b1289a7c5').eval()
        max_memory = {}
        memory_per_gpu = str(int(50/gpu_number))
        for i in range(gpu_number):
            max_memory[i] = f"{memory_per_gpu}GiB"
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=['InternLM2DecoderLayer'])
        device_map['vision_proj'] = 0
        with open(filename, 'wb') as f:
            pickle.dump(device_map, f)
        del model
        return device_map