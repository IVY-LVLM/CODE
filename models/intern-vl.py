import torch
import os
import torchvision.transforms as T

from tools.read_yaml import get_data_folder
from PIL import Image
from transformers import AutoTokenizer, AutoModel, GenerationConfig
from torchvision.transforms.functional import InterpolationMode
from .base_model import BaseModel

from .vcd_utils.vcd_add_noise import add_diffusion_noise

import warnings
warnings.filterwarnings("ignore")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

default_model_path = "OpenGVLab/InternVL-Chat-V1-5"
default_image_path = get_data_folder()

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InternVL(BaseModel):
    def __init__(self, temperature, max_new_tokens, num_beams, do_sample, top_p, opera_decoding, vcd_decoding, model_name: str = "intern-vl", model_path: str = default_model_path, contrastive: bool = False,  alt_text: bool = False,  excel: bool = False, cd_alpha=None):
        super().__init__(model_name=model_name, model_path=model_path, contrastive=contrastive, alt_text=alt_text, excel=excel, opera_decoding=opera_decoding, vcd_decoding=vcd_decoding, cd_alpha=cd_alpha)
        self.tokenizer = AutoTokenizer.from_pretrained(default_model_path, trust_remote_code=True, revision='c1987c574e0010d8104c545131f97beeffb96a73')
        if opera_decoding:
            self.model = AutoModel.from_pretrained(
            default_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            attn_implementation="eager",
            use_flash_attention_2=False,
            revision='c1987c574e0010d8104c545131f97beeffb96a73',
            ).eval()
        else:
            self.model = AutoModel.from_pretrained(
            default_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            revision='c1987c574e0010d8104c545131f97beeffb96a73',
            device_map='auto').eval()
        

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_beams = num_beams
        
        self.generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            top_p=self.top_p,
            opera_decoding=self.opera_decoding,
            img_start_id=self.tokenizer.convert_tokens_to_ids('<img>'),
            img_end_id=self.tokenizer.convert_tokens_to_ids('</img>')
        )

        self.greedy_config = dict(
            max_new_tokens=512,
            temperature=0.0,
            num_beams=1,
            do_sample=False,
            top_p=1.0,
            opera_decoding=False,
            img_start_id=self.tokenizer.convert_tokens_to_ids('<img>'),
            img_end_id=self.tokenizer.convert_tokens_to_ids('</img>')
        )
        
        
        self.psuedoimageprompt = "Provide a detailed description of the image, covering all visible elements and their interactions, so as to thoroughly answer any potential questions about the image."#"Describe the image with enough detail to answer any questions about the given image."
            
    def _generate(self, text_prompt: str, raw_image_data: str, dataset_name: str = None, image_path: str = None, alt_text=None, contrastive=False, cd_version=None, make_alttext=False):
        raw_image_data = os.path.join(default_image_path, dataset_name, image_path)
        pixel_values = load_image(raw_image_data, max_num=6).to(torch.bfloat16).cuda()
        
        query = text_prompt

        input_dict = {}
        if alt_text:
            input_dict['instead_pixel_values'] = self.tokenizer(alt_text, return_tensors="pt").input_ids.to("cuda:0")
            if contrastive:
                input_dict['contrastive'] = True
                input_dict['cd_version'] = cd_version
        elif self.opera_decoding:
            input_dict['selected'] = None

        else:
            input_dict = None
            
        if self.vcd_decoding:
            image_cd = add_diffusion_noise(pixel_values)
        else:
            image_cd = None

        if make_alttext:
            response = self.model.chat(tokenizer=self.tokenizer, pixel_values=pixel_values, question=query, generation_config=self.greedy_config, input_dict=input_dict, image_cd=image_cd)
        else:        
            response = self.model.chat(tokenizer=self.tokenizer, pixel_values=pixel_values, question=query, generation_config=self.generation_config, input_dict=input_dict, image_cd=image_cd, cd_alpha=self.cd_alpha)
                
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
