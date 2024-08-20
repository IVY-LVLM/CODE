import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, GenerationConfig

from .base_model import BaseModel
from .vcd_utils.vcd_add_noise import add_diffusion_noise

from models.llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.conversation import conv_templates, SeparatorStyle
from models.llava.mm_utils import KeywordsStoppingCriteria
import warnings
warnings.filterwarnings("ignore")

default_model_path = "llava-hf/llava-1.5-13b-hf"

class LLaVA_Model_HF(BaseModel):
    def __init__(self, temperature, max_new_tokens, num_beams, do_sample, top_p, opera_decoding, vcd_decoding, model_path: str = default_model_path, model_base: str = None, model_name: str = "llava-v1.5", conv_mode: str = "llava_v1", contrastive: bool = False, alt_text: bool = False,  excel: bool = False,cd_alpha=None):
        super().__init__(model_name=model_name, model_path=model_path, contrastive=contrastive, alt_text=alt_text, excel=excel, opera_decoding=opera_decoding, vcd_decoding=vcd_decoding, cd_alpha=cd_alpha)
        self.processor = AutoProcessor.from_pretrained(default_model_path)
        if opera_decoding:
            self.model = LlavaForConditionalGeneration.from_pretrained(default_model_path, device_map="auto", attn_implementation="eager").eval()
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(default_model_path, device_map="auto").eval()
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
            image_token_id=self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN),
        )
        self.greedy_condfig = GenerationConfig(
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            num_beams=1,
            opera_decoding=False,
            image_token_id=self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN),
        )
        
        self.psuedoimageprompt = "Provide a detailed description of the image, covering all visible elements and their interactions, so as to thoroughly answer any potential questions about the image."#"Describe the image with enough detail to answer any questions about the given image."

    def _generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, alt_text=None, contrastive=False, cd_version=None, make_alttext=False):       
        prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + text_prompt

        conv = conv_templates[self.conv_mode].copy()

        if make_alttext:
            prompts_input = prompts_input
        else:
            prompts_input = prompts_input + ' ' + conv.get_llava_response_prompt(dataset_name)

        conv.append_message(conv.roles[0], prompts_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
        outputs = self.generate_sentence(raw_image_data, stop_str, prompt, alt_text=alt_text, contrastive=contrastive, cd_version=cd_version, make_alttext=make_alttext)

        return outputs

    def generate_sentence(self, raw_image_data, stop_str, prompt, alt_text=None, contrastive=False, cd_version=None, make_alttext=False):
        input_ids = self.processor(images=raw_image_data, text=prompt, return_tensors="pt").to('cuda')
        
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.processor.tokenizer, input_ids['input_ids'])

        if alt_text:
            input_ids['instead_pixel_values'] = self.processor.tokenizer(alt_text, return_tensors="pt").input_ids.to("cuda:0")
        if contrastive:
            input_ids['contrastive'] = True
        if cd_version:
            input_ids['cd_version'] = cd_version
        if self.vcd_decoding:
            # Generate output for original input and question
            image_tensor_cd = add_diffusion_noise(input_ids['pixel_values'])
            input_ids['image_cd'] = image_tensor_cd

        if make_alttext:
            with torch.inference_mode():
                output_ids = self.model.generate(
                                                **input_ids,
                                                generation_config=self.greedy_condfig,
                                                use_cache=True,
                                                stopping_criteria=[stopping_criteria])
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(
                                                **input_ids,
                                                generation_config=self.generation_config,
                                                use_cache=True,
                                                stopping_criteria=[stopping_criteria],
                                                cd_alpha=self.cd_alpha
                                                )
        
        outputs = self.processor.batch_decode(output_ids[:, input_ids['input_ids'].shape[1]:], skip_special_tokens=True)[0]

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



    def eval_forward(self, text_prompt: str, raw_image_data: str):
        pass

    def get_coco_caption_prompt(self):
        return "Please describe this image in detail."