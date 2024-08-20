import pickle
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, GenerationConfig
import torch
from .base_model import BaseModel
from .vcd_utils.vcd_add_noise import add_diffusion_noise

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from accelerate import infer_auto_device_map

default_model_path = "llava-hf/llava-v1.6-34b-hf"
import warnings
warnings.filterwarnings("ignore")

class LLaVA_Next(BaseModel):
    def __init__(self, temperature, max_new_tokens, num_beams, do_sample, top_p, opera_decoding, vcd_decoding, model_name: str = "llava-next-34B", model_path: str = default_model_path, contrastive: bool = False, alt_text: bool = False, excel: bool = False,cd_alpha=None):
        super().__init__(model_name=model_name, model_path=model_path, contrastive=contrastive, alt_text=alt_text, excel=excel, opera_decoding=opera_decoding, vcd_decoding=vcd_decoding,cd_alpha=cd_alpha)
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
        if opera_decoding:
            device_map = LLaVA_Next.get_device_map()
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-34b-hf", 
            torch_dtype=torch.float16, 
            device_map = device_map,
            low_cpu_mem_usage=True)
        else:
            device_map = "auto"
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-34b-hf", 
                torch_dtype=torch.float16, 
                device_map = device_map,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            )

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_beams = num_beams
        
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            output_scores=False,
            output_logits=False,
            return_dict_in_generate=True,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            opera_decoding=self.opera_decoding,
            image_token_id=self.processor.tokenizer.convert_tokens_to_ids("<image>")
        )
        self.greedy_config = GenerationConfig(
            max_new_tokens=512,
            output_scores=False,
            output_logits=False,
            return_dict_in_generate=True,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            num_beams=1,
            opera_decoding=False,
            image_token_id=self.processor.tokenizer.convert_tokens_to_ids("<image>")
        )

        self.system_prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n" + "<USERINPUT>" + "<|im_end|><|im_start|>assistant\n"
        self.psuedoimageprompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n" + "Provide a detailed description of the image, covering all visible elements and their interactions, so as to thoroughly answer any potential questions about the image." + "<|im_end|><|im_start|>assistant\n"
        self.cache_folder = '/mnt/ssd2/lmm/pope/at_cache'

    def generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, noise_step = 500):
        
        image = raw_image_data
        inputs = self.processor(self.system_prompt.replace("<USERINPUT>", text_prompt), image, return_tensors="pt").to("cuda:0")

        if self.alt_text:
            alt_text = self.cache_exist(image_path)
            
            if alt_text is None:
                pseudo_inputs = self.processor(self.psuedoimageprompt, image, return_tensors="pt").to("cuda:0")
                ## Generate Image-alt text
                img_pseudo_dict = self.model.generate(**pseudo_inputs, generation_config=self.greedy_config)
                alt_text = self.processor.decode(img_pseudo_dict['sequences'][0, pseudo_inputs.input_ids.size(1):], skip_special_tokens=True)
            
                ## Generate output for pseudo image input and question
                inputs['instead_pixel_values'] = self.processor.tokenizer(alt_text, return_tensors="pt").input_ids.to("cuda:0")
            else:
                inputs['instead_pixel_values'] = alt_text

            if self.contrastive:
                inputs['contrastive'] = True
            if self.excel:
                inputs['processor'] = self.processor
                inputs['raw_image_data'] = image
            
            output_dict = self.model.generate(**inputs, generation_config=self.generation_config, cd_alpha=self.cd_alpha)
            out_sequence = self.processor.decode(output_dict['sequences'][0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

            return out_sequence
        elif self.vcd_decoding:
            # Generate output for original input and question
            image_tensor_cd = add_diffusion_noise(inputs['pixel_values'], noise_step)
            inputs['image_cd'] = image_tensor_cd
            output_dict = self.model.generate(**inputs, generation_config=self.generation_config)
            out_sequence = self.processor.decode(output_dict['sequences'][0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

            return out_sequence
        else:
            output_dict = self.model.generate(**inputs, generation_config=self.generation_config)
            out_sequence = self.processor.decode(output_dict['sequences'][0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

            return out_sequence

    def cache_exist(self, image_path):
        p = os.path.join(self.cache_folder, image_path[:-3] + 'pt')
        if os.path.exists(p):
            return torch.load(p)
        else:
            return None

    def eval_forward(self, text_prompt: str, raw_image_data: str):
        pass

    def get_coco_caption_prompt(self):
        return "Please describe this image in detail."
    
    @staticmethod
    def get_device_map():
        gpu_number = torch.cuda.device_count()
        
        if os.path.exists('.device_map')==False:
            os.mkdir('.device_map')

        filename = f"./.device_map/LLaVA-Next_gpu{gpu_number}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                device_map = pickle.load(f)
                return device_map
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-34b-hf", 
            torch_dtype=torch.float16, 
            device_map = "auto",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"
            ).eval()
        max_memory = {}
        memory_per_gpu = str(int(120/gpu_number - 1))
        for i in range(gpu_number):
            max_memory[i] = f"{memory_per_gpu}GiB"
        max_memory[0] = "1GiB"
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"])
        with open(filename, 'wb') as f:
            pickle.dump(device_map, f)
        del model
        return device_map